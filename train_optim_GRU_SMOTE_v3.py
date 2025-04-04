import json
import os
import time
import random
import shutil
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm
from rich import print

# Try to import imblearn for SMOTE (Synthetic Minority Oversampling Technique) if available
try:
    from imblearn.over_sampling import SMOTE

    SMOTE_AVAILABLE = True
    print("[green]SMOTE is available for oversampling")
except ImportError:
    from imblearn.over_sampling import SMOTE

    SMOTE_AVAILABLE = False
    print("[yellow]SMOTE not available. Install with 'pip install imblearn' for improved class balancing")

# Check if CUDA is available and set memory allocation configuration
if torch.cuda.is_available():
    # Enable memory efficient features
    torch.cuda.set_per_process_memory_fraction(0.8)  # Limit to 80% of GPU memory
    # Set PyTorch to allocate memory more efficiently
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[yellow]Using device: {DEVICE}")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"[yellow]GPU: {torch.cuda.get_device_name(0)}")
    print(f"[yellow]Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Hyperparameters - adjusted based on feature analysis
BATCH_SIZE = 8  # Moderate batch size for better gradient estimates
EPOCHS = 100
NUM_WORKERS = 4
PATIENCE = 15
DROPOUT = 0.5
HIDDEN_DIM = 256
INPUT_DIM = 14  # Basic trajectory features + derived features from feature analysis
GLOBAL_DIM = 10  # Top global metrics from feature analysis
USE_SMOTE = True
NUM_FOLDS = 5  # Number of folds for cross-validation


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        weight = self.class_weights.to(inputs.device) if self.class_weights is not None else None
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=weight)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()


class EarlyStopping:
    def __init__(self, patience=10, verbose=True, delta=0.001):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_val_loss = None
        self.best_f1 = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss, val_f1):
        if (self.best_val_loss is None and self.best_f1 is None or val_loss < self.best_val_loss - self.delta
                and val_f1 > self.best_f1 + self.delta):
            if self.verbose and self.best_f1 is not None and self.best_val_loss is not None:
                print(f"[green]EarlyStopping: \nF1 improved from {self.best_f1:.4f} to {val_f1:.4f}"
                      f"\nVal Loss improved from {self.best_val_loss:.4f} to {val_loss:.4f}")
            self.best_val_loss = val_loss
            self.best_f1 = val_f1
            self.counter = 0
            return True  # Improvement
        else:
            self.counter += 1
            if self.verbose:
                print(f"[red]EarlyStopping: No improvement. Patience {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return False  # No improvement


def apply_smote(x_train, y_train):
    """Apply SMOTE (Synthetic Minority Oversampling Technique) to balance the training data"""
    if not SMOTE_AVAILABLE:
        print("[yellow]SMOTE not available, skipping oversampling")
        return x_train, y_train

    print("[green]Applying SMOTE oversampling...")
    # Print initial class distribution
    print(f"Before SMOTE: Class distribution {Counter(y_train)}")

    # Create and configure SMOTE
    smote = SMOTE(random_state=42, k_neighbors=min(5, Counter(y_train).most_common()[-1][1] - 1))

    try:
        x_resampled, y_resampled = smote.fit_resample(x_train, y_train)
        print(f"After SMOTE: Class distribution {Counter(y_resampled)}")
        return x_resampled, y_resampled
    except ValueError as val_err:
        print(f"[red]SMOTE error: {val_err}")
        print("[yellow]Falling back to original data")
        return x_train, y_train


class MouseMovementDataset(Dataset):
    def __init__(self, data_folder, global_means=None, global_stds=None):
        self.samples = []
        self.class_counts = {0: 0, 1: 0}  # Track class distribution

        for root, _, files in os.walk(data_folder):
            # Set label based on folder name
            label = 1 if any(term in root.lower() for term in ["deceitful", "deceptive"]) else 0
            json_files = [f for f in files if f.endswith(".json")]
            self.class_counts[label] += len(json_files)

            for file in json_files:
                file_path = os.path.join(root, file)
                self.samples.append((file_path, label))

        self.global_means = global_means
        self.global_stds = global_stds

        if global_means is not None and global_stds is not None:
            print(f"Dataset class distribution: {self.class_counts}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as exc:
            print(f"Error loading {path}: {exc}")
            # Return empty data as fallback
            return (torch.zeros(1, INPUT_DIM, dtype=torch.float32),
                    torch.zeros(GLOBAL_DIM, dtype=torch.float32), torch.tensor(label), 1)

        # Extract trajectory and metrics
        traj = data.get("trajectory", [])
        metrics = data.get("trajectory_metrics", {})

        # Extract top global features based on feature analysis
        # Convert all values to float and handle missing values
        g = np.zeros(GLOBAL_DIM, dtype=np.float32)

        # Map top metrics from feature analysis to indices
        global_feature_keys = [
            'decision_path_efficiency',  # High importance
            'final_decision_path_efficiency',
            'total_time',
            'direction_changes',  # Important
            'pause_count',  # Important
            'hesitation_time',
            'time_to_first_movement',
            'hesitation_count',
            'hover_time',
            'total_pause_time'  # Important
        ]

        # Fill in available metrics
        for i, key in enumerate(global_feature_keys):
            try:
                g[i] = float(metrics.get(key, 0))
            except (ValueError, TypeError):
                g[i] = 0.0

        # Get screen dimensions for normalization
        max_x, max_y = 1920, 1080  # Default screen resolution
        max_v, max_a, max_j, max_c = 1000, 1000, 1000, 2 * np.pi

        if traj:
            max_x = max(p.get('x', 0) for p in traj) or max_x
            max_y = max(p.get('y', 0) for p in traj) or max_y
            max_v = max(abs(p.get('velocity', 0)) for p in traj) or max_v
            max_a = max(abs(p.get('acceleration', 0)) for p in traj) or max_a

        # Extract feature sequence with focus on important features
        features = []
        prev_velocity = 0
        prev_accel = 0

        timestamps = [p.get('timestamp', 0) for p in traj] if traj else [0]
        min_time = min(timestamps)
        max_time = max(timestamps) if len(timestamps) > 1 else min_time + 1

        for i, p in enumerate(traj):
            # Basic positional features
            x = p.get('x', 0) / max_x  # Normalize x to [0,1]
            y = p.get('y', 0) / max_y  # Normalize y to [0,1]

            # Velocity and acceleration (key features from analysis)
            velocity = p.get('velocity', 0) / max_v
            acceleration = p.get('acceleration', 0) / max_a

            # Important derived features from analysis
            accel_change = acceleration - prev_accel if i > 0 else 0
            velocity_change = velocity - prev_velocity if i > 0 else 0

            # Movement smoothness related features (important from analysis)
            jerk = p.get('jerk', 0) / max_j if max_j > 0 else 0
            curvature = p.get('curvature', 0) / max_c if max_c > 0 else 0

            # Direction features
            dx = p.get('dx', 0)
            dy = p.get('dy', 0)

            # Temporal features
            norm_time = (p.get('timestamp', 0) - min_time) / (max_time - min_time) if max_time > min_time else 0

            # Direction change detection (1 if direction changed)
            dir_change = 0
            if i > 0:
                prev_dx = traj[i - 1].get('dx', 0)
                prev_dy = traj[i - 1].get('dy', 0)
                if (dx * prev_dx < 0) or (dy * prev_dy < 0):
                    dir_change = 1

            # Pauses (1 if velocity very low)
            is_paused = 1 if abs(velocity) < 0.01 else 0

            # Store previous values for change calculation
            prev_velocity = velocity
            prev_accel = acceleration

            # Combine the most discriminative features based on feature analysis
            feature_vector = [
                x, y,  # Basic position
                velocity,  # High importance
                acceleration,  # High importance
                accel_change,  # Highest importance
                velocity_change,  # High importance
                jerk,  # High importance
                curvature,
                dx, dy,
                dir_change,  # Important
                is_paused,  # Important
                norm_time,
                1.0 if p.get('click', 0) == 1 else 0.0
            ]

            features.append(feature_vector)

        # Handle empty trajectory case
        if not features:
            features = [[0] * INPUT_DIM]

        features = np.array(features, dtype=np.float32)

        # Normalize global features
        if self.global_means is not None and self.global_stds is not None:
            g = (g - self.global_means) / (self.global_stds + 1e-8)

        return (torch.tensor(features, dtype=torch.float32),
                torch.tensor(g, dtype=torch.float32), torch.tensor(label), len(features))


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    seqs, globals_, labels, lengths = zip(*batch)
    padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    return padded, torch.stack(globals_), torch.tensor(labels), torch.tensor(lengths)


class DeceptionNet(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, global_dim=GLOBAL_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()

        # 1D CNN feature extraction optimized for key features
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(DROPOUT / 2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(DROPOUT / 2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT / 2)
        )

        self.gru = nn.GRU(128, hidden_dim, num_layers=9, batch_first=True,
                          dropout=DROPOUT, bidirectional=True)

        # Layer normalization for better training stability
        self.norm = nn.LayerNorm(hidden_dim * 2)

        # Attention mechanism - important for focusing on key movement patterns
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Global feature processing with consistent dimensions for the residual connection
        # Fix the dimension mismatch by ensuring both layers have the same output dim
        self.global_fc1 = nn.Linear(global_dim, 128)  # Changed to output 128
        self.global_fc2 = nn.Linear(128, 128)  # Takes 128, outputs 128
        self.global_norm = nn.LayerNorm(128)
        self.global_dropout = nn.Dropout(DROPOUT)

        # Final classifier with key feature emphasis
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(DROPOUT / 2),

            nn.Linear(64, 2)
        )

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name and 'norm' not in name and 'batch' not in name:
                if len(param.shape) > 1:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x, g, lengths):
        # Process sequences of different lengths
        valid_seq = []
        for i, length in enumerate(lengths):
            valid_seq.append(x[i, :length])

        # Apply CNN feature extraction (requires channel-first format)
        conv_out = []
        for seq in valid_seq:
            # Transpose to [features, time]
            seq = seq.transpose(0, 1).unsqueeze(0)
            # Apply convolutions
            c_out = self.conv_layers(seq)
            # Transpose back to [time, features]
            c_out = c_out.transpose(1, 2)
            conv_out.append(c_out.squeeze(0))

        # Re-pad the sequences for GRU
        padded = nn.utils.rnn.pad_sequence(conv_out, batch_first=True)

        # Make sure padded is contiguous
        padded = padded.contiguous()

        # Handle GRU differently on CPU vs CUDA to avoid cuDNN issues
        if DEVICE.type == 'cuda':
            # For CUDA, we'll avoid the packed sequence approach which can cause issues
            # Create a mask for valid sequence positions
            mask = torch.zeros(padded.size(0), padded.size(1), dtype=torch.bool, device=padded.device)
            for i, length in enumerate(lengths):
                if length > 0:  # Ensure length is valid
                    mask[i, :min(length, padded.size(1))] = True  # Use min to avoid index errors

            # Process with GRU
            gru_out, _ = self.gru(padded)

            # Zero out invalid positions
            gru_out = gru_out * mask.unsqueeze(-1)
        else:
            # On CPU, we can use packed sequence safely
            # Need to handle case where all lengths might be 0
            if all(length <= 0 for length in lengths):
                return torch.zeros((x.size(0), 2), device=x.device)  # Return dummy output

            # Ensure all lengths are > 0 for pack_padded_sequence
            valid_lengths = torch.clamp(lengths, min=1)
            packed = pack_padded_sequence(padded, valid_lengths.cpu(), batch_first=True, enforce_sorted=True)
            gru_out, _ = self.gru(packed)
            gru_out, _ = pad_packed_sequence(gru_out, batch_first=True)

        # Apply layer normalization
        normalized = self.norm(gru_out)

        # Attention mechanism
        attn_weights = torch.softmax(self.attention(normalized), dim=1)
        sequence_encoding = torch.sum(attn_weights * normalized, dim=1)

        # Process global features with residual connection - fixed dimensions
        g1 = self.global_fc1(g)  # [batch_size, 128]
        g2 = self.global_fc2(g1)  # [batch_size, 128]
        global_out = self.global_norm(g1 + g2)  # Now both tensors are [batch_size, 128]
        global_out = self.global_dropout(global_out)

        # Combine sequence and global features
        combined = torch.cat((sequence_encoding, global_out), dim=1)

        return self.classifier(combined)


def count_parameters(model):
    """Count and print the number of trainable parameters in the model"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[yellow]Total trainable parameters: {total_params:,}")

    # Count parameters by layer type
    layer_counts = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_type = name.split('.')[0]
            if layer_type not in layer_counts:
                layer_counts[layer_type] = 0
            layer_counts[layer_type] += param.numel()

    print("[yellow]Parameters by layer type:")
    for layer_type, count in layer_counts.items():
        print(f"  - {layer_type}: {count:,} parameters ({count / total_params * 100:.1f}%)")

    return total_params


def compute_stats(dataset):
    """Compute mean and std for global features"""
    all_g = np.stack([sample[1].numpy() for sample in dataset])
    return all_g.mean(axis=0), all_g.std(axis=0)


def train_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss, correct = 0, 0
    all_preds, all_labels = [], []

    for x, g, y, lengths in tqdm(loader, desc="Training", leave=False):
        x, g, y, lengths = x.to(DEVICE), g.to(DEVICE), y.to(DEVICE), lengths.to(DEVICE)
        optimizer.zero_grad()
        out = model(x, g, lengths)
        loss = criterion(out, y)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

        pred = out.argmax(1)
        correct += torch.sum(pred == y).item()

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    # Calculate precision, recall and F1 for training set
    p, r, f, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    return total_loss / len(loader), correct / len(loader.dataset), p, r, f


def evaluate(model, loader, criterion):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    all_probs = []

    with torch.no_grad():
        for x, g, y, lengths in tqdm(loader, desc="Evaluating", leave=False):
            x, g, y, lengths = x.to(DEVICE), g.to(DEVICE), y.to(DEVICE), lengths.to(DEVICE)
            out = model(x, g, lengths)
            loss = criterion(out, y)
            total_loss += loss.item()

            # Get class probabilities
            probs = torch.softmax(out, dim=1)
            all_probs.extend(probs.cpu().numpy())

            pred = out.argmax(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Debug class distribution in predictions
    pred_counts = Counter(all_preds)
    print(f"Prediction distribution: {dict(pred_counts)}")

    # Handle case where all predictions are the same class
    if len(set(all_preds)) <= 1:
        print("[red]WARNING: Model is predicting only one class!")

    p, r, f, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)

    # Calculate balanced accuracy
    from sklearn.metrics import balanced_accuracy_score
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    print(f"Balanced accuracy: {balanced_acc:.4f}")

    return total_loss / len(loader), acc, p, r, f, all_labels, all_preds, np.array(all_probs)


def debug_predictions(model, loader):
    """Debug function to analyze model predictions"""
    model.eval()
    preds_count = {0: 0, 1: 0}
    true_count = {0: 0, 1: 0}
    correct_count = {0: 0, 1: 0}
    total = 0

    with torch.no_grad():
        for x, g, y, lengths in loader:
            x, g, lengths = x.to(DEVICE), g.to(DEVICE), lengths.to(DEVICE)
            out = model(x, g, lengths)
            preds = out.argmax(1).cpu().numpy()
            labels = y.cpu().numpy()

            for i, (pred, label) in enumerate(zip(preds, labels)):
                preds_count[pred] += 1
                true_count[label] += 1
                if pred == label:
                    correct_count[label] += 1
                total += 1

                # Log high confidence misclassifications
                prob = torch.softmax(out[i], dim=0)
                if pred != label and prob[pred] > 0.9:
                    print(f"[red] High confidence misclassification: true={label}, pred={pred}, prob={prob[pred]:.4f}")

    print(f"\nPrediction distribution: {preds_count}")
    print(f"True label distribution: {true_count}")
    print(f"Percentages: {preds_count[0] / total * 100:.1f}% predicted as class 0,"
          f" {preds_count[1] / total * 100:.1f}% predicted as class 1")

    # Per-class accuracy
    for label in [0, 1]:
        if true_count[label] > 0:
            print(f"Class {label} accuracy: {correct_count[label] / true_count[label] * 100:.1f}%")
        else:
            print(f"Class {label} accuracy: N/A (no samples)")

    # Calculate decision boundary statistics
    confidences = []
    with torch.no_grad():
        for x, g, y, lengths in loader:
            x, g, lengths = x.to(DEVICE), g.to(DEVICE), lengths.to(DEVICE)
            out = model(x, g, lengths)
            probs = torch.softmax(out, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            confidences.extend(max_probs.cpu().numpy())

    print(f"\nAverage prediction confidence: {np.mean(confidences):.4f}")
    print(f"Confidence percentiles: 10%={np.percentile(confidences, 10):.4f},"
          f" 50%={np.percentile(confidences, 50):.4f},"
          f" 90%={np.percentile(confidences, 90):.4f}")


def calculate_model_score(val_loss, f1):
    """Calculate a combined score for model selection.
    
    The score is a weighted combination of validation loss, F1 score, precision, and recall.
    - Lower is better for val_loss, so we take the negative.
    - Higher is better for F1, precision, and recall.
    """
    # Normalize val_loss (convert to a score where higher is better)
    # Combined score: higher is better
    return ((1 - val_loss) + f1) / 2


def train_for_folder(folder_name, use_smote=True):
    print(f"[magenta] \n{'=' * 40}\nTraining on {folder_name}\n{'=' * 40}")
    main_log_dir = os.path.join("logs_GRU_optimized_SMOTE_v3", folder_name)
    os.makedirs(main_log_dir, exist_ok=True)

    print(f"[yellow]Using batch size: {BATCH_SIZE}")

    # Create and analyze dataset
    dataset = MouseMovementDataset(folder_name)
    mean, std = compute_stats(dataset)
    dataset = MouseMovementDataset(folder_name, mean, std)

    # Calculate class weights for dealing with imbalance
    labels = [dataset.samples[i][1] for i in range(len(dataset))]
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    print(f"Class weights: {class_weights}")

    # Use k-fold cross-validation for more robust evaluation
    kfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    fold_results = []

    print(f"[yellow]Starting {NUM_FOLDS}-fold cross-validation")

    # Run k-fold cross-validation
    for fold, (train_idx, test_idx) in enumerate(kfold.split(np.zeros(len(labels)), labels)):
        print(f"[magenta]\n{'=' * 40}\nFold {fold + 1}/{NUM_FOLDS}\n{'=' * 40}")

        # Create a subdirectory for this fold
        fold_log_dir = os.path.join(main_log_dir, f"fold_{fold + 1}")
        os.makedirs(fold_log_dir, exist_ok=True)

        # Further split the training data into train and validation
        train_labels = [labels[i] for i in train_idx]
        train_idx_final, val_idx = train_test_split(
            train_idx,
            test_size=0.2,
            stratify=train_labels,
            random_state=42 + fold  # Different seed for each fold
        )

        # Create subsets
        train_set = Subset(dataset, train_idx_final)
        val_set = Subset(dataset, val_idx)
        test_set = Subset(dataset, test_idx)

        # Verify split distributions
        train_labels = [labels[i] for i in train_idx_final]
        val_labels = [labels[i] for i in val_idx]
        test_labels = [labels[i] for i in test_idx]

        print(f"Train set: {len(train_set)} samples, distribution: {Counter(train_labels)}")
        print(f"Val set: {len(val_set)} samples, distribution: {Counter(val_labels)}")
        print(f"Test set: {len(test_set)} samples, distribution: {Counter(test_labels)}")

        # Apply SMOTE for better class balance if enabled
        if use_smote and SMOTE_AVAILABLE:
            # Prepare feature matrices for SMOTE
            print("[yellow]Preparing data for SMOTE...")
            x_train_combined = []
            y_train = []

            # We'll use global features + trajectory summary for SMOTE
            for idx in train_idx:
                traj, g, label, length = dataset[idx]

                # Extract summary statistics from trajectory
                if length > 0:
                    traj_np = traj.numpy()[:length]
                    # Calculate trajectory statistics
                    traj_mean = np.mean(traj_np, axis=0)
                    traj_std = np.std(traj_np, axis=0)
                    traj_max = np.max(traj_np, axis=0)
                    traj_min = np.min(traj_np, axis=0)
                    traj_features = np.concatenate([traj_mean, traj_std, traj_max, traj_min])
                else:
                    # Handle empty trajectories
                    traj_features = np.zeros(INPUT_DIM * 4)

                # Combine with global features
                combined_features = np.concatenate([traj_features, g.numpy()])
                x_train_combined.append(combined_features)
                y_train.append(label.item())

            x_train_combined = np.array(x_train_combined)
            y_train = np.array(y_train)

            # Apply SMOTE to the combined features
            x_train_resampled, y_train_resampled = apply_smote(x_train_combined, y_train)

            # Create new indices for resampled data
            print("[green]Creating balanced training set...")
            resampled_train_idx = []

            # First, add all original training indices
            resampled_train_idx.extend(train_idx)

            # Track which synthetic samples we've created
            synthetic_count = 0

            # Then create synthetic samples for the minority class
            for i, (_, label) in enumerate(zip(x_train_resampled, y_train_resampled)):
                if i >= len(train_idx):  # This is a synthetic sample
                    # Find the nearest neighbors in the original dataset
                    minority_indices = [idx for idx, lbl in enumerate(train_labels) if lbl == label]
                    if minority_indices:
                        # Use a random sample from the minority class as a base
                        base_idx = random.choice(minority_indices)
                        resampled_train_idx.append(train_idx[base_idx])
                        synthetic_count += 1

            print(f"[green]Added {synthetic_count} synthetic samples through resampling")

            # Update training set with resampled data
            train_idx = resampled_train_idx
            train_labels = [dataset.samples[i % len(dataset.samples)][1] for i in train_idx]
            train_set = Subset(dataset, [i % len(dataset) for i in train_idx])

            # Recalculate class weights after SMOTE resampling
            class_distribution = Counter(train_labels)
            print(f"[green]Class distribution after SMOTE: {class_distribution}")

            # Recalculate balanced weights (if SMOTE didn't create perfect balance)
            class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
            print(f"[green]Updated class weights after SMOTE: {class_weights}")

        # Create weighted sampler for balanced batches
        class_sample_count = np.bincount(train_labels)
        weight = 1. / class_sample_count
        samples_weight = torch.tensor([weight[t] for t in train_labels], dtype=torch.float32)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        # Create data loaders
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler,
                                  collate_fn=collate_fn, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=collate_fn, num_workers=NUM_WORKERS)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                                 collate_fn=collate_fn, num_workers=NUM_WORKERS)

        # Initialize model
        try:
            model = DeceptionNet().to(DEVICE)

            # Count and log model parameters
            total_params = count_parameters(model)
            with open(os.path.join(fold_log_dir, "model_info.txt"), "w") as f:
                f.write(f"Total trainable parameters: {total_params:,}\n")
                f.write(f"Model architecture:\n{model}\n")

            # AdamW optimizer with weight decay for better regularization
            optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)

            # Use cosine annealing scheduler for better convergence
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,  # Restart every 10 epochs
                T_mult=1,
                eta_min=1e-6
            )

            # Use class weights and focal loss for imbalanced data
            criterion = FocalLoss(alpha=2.0, gamma=2.0, class_weights=class_weights.to(DEVICE))

            # Initialize tracking variables for best model
            best_val_f1 = 0
            best_val_precision = 0
            best_val_recall = 0
            best_val_loss = float('inf')
            delta = 0.001
            best_model_path = os.path.join(fold_log_dir, "best_model.pt")
            stopper = EarlyStopping(patience=PATIENCE)

            # Training history for plotting
            history = {
                'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_precision': [], 'train_recall': [],
                'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_precision': [], 'val_recall': []
            }

            # Training loop
            for epoch in range(1, EPOCHS + 1):
                print(f"\nEpoch {epoch}/{EPOCHS}")
                start_time = time.time()

                # Train
                t_loss, t_acc, t_p, t_r, t_f = train_epoch(model, train_loader, optimizer, None, criterion)

                # Step scheduler once per epoch 
                scheduler.step()

                # Validate
                v_loss, v_acc, v_p, v_r, v_f, _, _, _ = evaluate(model, val_loader, criterion)

                # Record history
                history['train_loss'].append(t_loss)
                history['train_acc'].append(t_acc)
                history['train_f1'].append(t_f)
                history['train_precision'].append(t_p)
                history['train_recall'].append(t_r)

                history['val_loss'].append(v_loss)
                history['val_acc'].append(v_acc)
                history['val_f1'].append(v_f)
                history['val_precision'].append(v_p)
                history['val_recall'].append(v_r)

                duration = time.time() - start_time

                print(
                    f"\nEpoch {epoch:03}: Train Loss={t_loss:.4f} Acc={t_acc:.2%} F1={t_f:.4f} P={t_p:.4f} R={t_r:.4f}"
                    f" | Val Loss={v_loss:.4f} Acc={v_acc:.2%} F1={v_f:.4f} P={v_p:.4f} R={v_r:.4f} |"
                    f" Time={duration:.2f}s")

                # Save if improved
                improved = False
                if v_loss < best_val_loss - delta and v_f > best_val_f1 + delta:
                    print(
                        f"[green]Model Checkpoint: \nF1 improved: {best_val_f1:.4f} -> {v_f:.4f}."
                        f"\nVal Loss improved: {best_val_loss:.4f} -> {v_loss:.4f}\nSaving model.")
                    best_val_f1 = v_f
                    best_val_loss = v_loss
                    torch.save(model.state_dict(), best_model_path)
                    improved = True

                # Early stopping check
                if stopper(v_loss, v_f):
                    if improved:
                        print("[green]Model improved!")
                elif stopper.early_stop:
                    print("[red]Early stopping triggered.")
                    break

                # Debug predictions every 10 epochs
                if epoch % 10 == 0 or epoch == 1:
                    print("\nDebug validation predictions:")
                    debug_predictions(model, val_loader)

            # Plot training history
            plt.figure(figsize=(15, 10))
            # Plot loss
            plt.subplot(2, 3, 1)
            plt.plot(history['train_loss'], label='Train Loss')
            plt.plot(history['val_loss'], label='Val Loss')
            plt.title('Loss History')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            # Plot accuracy
            plt.subplot(2, 3, 2)
            plt.plot(history['train_acc'], label='Train Accuracy')
            plt.plot(history['val_acc'], label='Val Accuracy')
            plt.title('Accuracy History')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

            # Plot F1 score
            plt.subplot(2, 3, 3)
            plt.plot(history['train_f1'], label='Train F1')
            plt.plot(history['val_f1'], label='Val F1')
            plt.title('F1 Score History')
            plt.xlabel('Epoch')
            plt.ylabel('F1 Score')
            plt.legend()

            # Plot precision
            plt.subplot(2, 3, 4)
            plt.plot(history['train_precision'], label='Train Precision')
            plt.plot(history['val_precision'], label='Val Precision')
            plt.title('Precision History')
            plt.xlabel('Epoch')
            plt.ylabel('Precision')
            plt.legend()

            # Plot recall
            plt.subplot(2, 3, 5)
            plt.plot(history['train_recall'], label='Train Recall')
            plt.plot(history['val_recall'], label='Val Recall')
            plt.title('Recall History')
            plt.xlabel('Epoch')
            plt.ylabel('Recall')
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(fold_log_dir, "training_history.png"))
            plt.close()

            # Evaluate on test set
            print("\nLoading best model for final evaluation...")
            model.load_state_dict(torch.load(best_model_path))
            test_loss, test_acc, test_p, test_r, test_f, y_true, y_pred, y_probs = evaluate(model, test_loader,
                                                                                            criterion)

            # Save fold results
            fold_result = {'fold': fold + 1, 'test_acc': test_acc, 'test_precision': test_p, 'test_recall': test_r,
                           'test_f1': test_f, 'val_loss': best_val_loss, 'val_f1': best_val_f1,
                           'val_precision': best_val_precision, 'val_recall': best_val_recall,
                           'model_path': best_model_path, 'log_dir': fold_log_dir, 'y_true': y_true, 'y_pred': y_pred,
                           'y_probs': y_probs, 'combined_score': calculate_model_score(
                            test_loss,
                            test_f
                            )}

            fold_results.append(fold_result)

            print(f"\nFold {fold + 1} Test: Acc={test_acc:.4f} Loss={test_loss:.4f}"
                  f" Precision={test_p:.4f} Recall={test_r:.4f} F1={test_f:.4f}")

            # Generate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(10, 8))

            # Regular confusion matrix
            plt.subplot(1, 2, 1)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Truthful', 'Deceitful'],
                        yticklabels=['Truthful', 'Deceitful'])
            plt.title(f'Fold {fold + 1} - Test Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')

            # Normalized confusion matrix
            plt.subplot(1, 2, 2)
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                        xticklabels=['Truthful', 'Deceitful'],
                        yticklabels=['Truthful', 'Deceitful'])
            plt.title(f'Fold {fold + 1} - Normalized Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')

            plt.tight_layout()
            plt.savefig(os.path.join(fold_log_dir, "confusion_matrix.png"))
            plt.close()

            # Generate ROC curve
            from sklearn.metrics import roc_curve, roc_auc_score
            fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
            auc = roc_auc_score(y_true, y_probs[:, 1])

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Fold {fold + 1} - ROC Curve')
            plt.legend()
            plt.savefig(os.path.join(fold_log_dir, "roc_curve.png"))
            plt.close()

            # Save classification report
            report = classification_report(y_true, y_pred, output_dict=True)
            with open(os.path.join(fold_log_dir, "classification_report.txt"), "w") as f:
                f.write(classification_report(y_true, y_pred))
                f.write("\n\nPrecision, Recall, F1 by class:\n")
                for cls in [0, 1]:
                    f.write(f"Class {cls}: Precision={report[str(cls)]['precision']:.4f}, ")
                    f.write(f"Recall={report[str(cls)]['recall']:.4f}, ")
                    f.write(f"F1={report[str(cls)]['f1-score']:.4f}, ")
                    f.write(f"Support={report[str(cls)]['support']}\n")

        except Exception as exc:
            print(f"[red]Error in fold {fold + 1}: {exc}")
            import traceback
            traceback.print_exc()

    # After all folds, select the best model
    if fold_results:
        # Sort folds by combined score (higher is better)
        sorted_folds = sorted(fold_results, key=lambda x: x['combined_score'], reverse=True)
        best_fold = sorted_folds[0]

        print(f"\n[green]Best fold: Fold {best_fold['fold']}")
        print(f"Test Accuracy: {best_fold['test_acc']:.4f}")
        print(f"Test F1 Score: {best_fold['test_f1']:.4f}")
        print(f"Test Precision: {best_fold['test_precision']:.4f}")
        print(f"Test Recall: {best_fold['test_recall']:.4f}")

        # Copy the best model files to the main log directory
        print("[green]Copying best model files to main directory...")
        best_fold_dir = best_fold['log_dir']

        # Copy model file
        source_model_path = best_fold['model_path']
        final_model_path = os.path.join(main_log_dir, "final_model.pt")
        shutil.copy(source_model_path, final_model_path)

        # Create CPU version
        model = DeceptionNet().to('cpu')
        model.load_state_dict(torch.load(source_model_path, map_location='cpu'))
        torch.save(model.state_dict(), os.path.join(main_log_dir, "final_model_cpu.pt"))

        # Copy other result files
        for file in ['confusion_matrix.png', 'roc_curve.png', 'classification_report.txt', 'training_history.png']:
            source_file = os.path.join(best_fold_dir, file)
            if os.path.exists(source_file):
                target_file = os.path.join(main_log_dir, file)
                shutil.copy(source_file, target_file)

        # Calculate average metrics across all folds
        avg_acc = np.mean([fold['test_acc'] for fold in fold_results])
        avg_f1 = np.mean([fold['test_f1'] for fold in fold_results])
        avg_precision = np.mean([fold['test_precision'] for fold in fold_results])
        avg_recall = np.mean([fold['test_recall'] for fold in fold_results])

        # Save a summary of all folds
        with open(os.path.join(main_log_dir, "all_folds_summary.txt"), "w") as f:
            f.write(f"Summary of {NUM_FOLDS}-fold cross-validation:\n\n")
            f.write("Average metrics across all folds:\n")
            f.write(f"  Accuracy: {avg_acc:.4f}\n")
            f.write(f"  F1 Score: {avg_f1:.4f}\n")
            f.write(f"  Precision: {avg_precision:.4f}\n")
            f.write(f"  Recall: {avg_recall:.4f}\n\n")

            f.write(f"Best fold (Fold {best_fold['fold']}):\n")
            f.write(f"  Test Accuracy: {best_fold['test_acc']:.4f}\n")
            f.write(f"  Test F1 Score: {best_fold['test_f1']:.4f}\n")
            f.write(f"  Test Precision: {best_fold['test_precision']:.4f}\n")
            f.write(f"  Test Recall: {best_fold['test_recall']:.4f}\n\n")

            f.write("Detailed results by fold:\n")
            for fold in sorted_folds:
                f.write(f"Fold {fold['fold']}:\n")
                f.write(f"  Test Accuracy: {fold['test_acc']:.4f}\n")
                f.write(f"  Test F1 Score: {fold['test_f1']:.4f}\n")
                f.write(f"  Test Precision: {fold['test_precision']:.4f}\n")
                f.write(f"  Test Recall: {fold['test_recall']:.4f}\n")
                f.write(f"  Val Loss: {fold['val_loss']:.4f}\n")
                f.write(f"  Combined Score: {fold['combined_score']:.4f}\n")
                f.write("\n")

        # Plot comparison of folds
        plt.figure(figsize=(15, 10))

        # Plot test metrics
        metrics = ['test_acc', 'test_f1', 'test_precision', 'test_recall']
        metric_labels = ['Accuracy', 'F1 Score', 'Precision', 'Recall']

        for i, metric in enumerate(metrics):
            plt.subplot(2, 2, i + 1)
            values = [fold[metric] for fold in fold_results]
            folds = [f"Fold {fold['fold']}" for fold in fold_results]

            # Plot bars
            bars = plt.bar(folds, values)

            # Highlight best fold
            best_idx = [i for i, fold in enumerate(fold_results) if fold['fold'] == best_fold['fold']][0]
            bars[best_idx].set_color('red')

            # Add average line
            avg_value = np.mean(values)
            plt.axhline(y=avg_value, color='green', linestyle='--', label=f'Avg: {avg_value:.4f}')

            plt.ylabel(metric_labels[i])
            plt.title(f'Test {metric_labels[i]} by Fold')
            plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(main_log_dir, "folds_comparison.png"))
        plt.close()

        # Rename the main log directory with performance metrics from the best fold
        best_acc = best_fold['test_acc']
        new_folder_name = f"{folder_name} ({best_acc * 100:.2f}% Test Acc)"
        new_log_dir = os.path.join("logs_GRU_optimized_SMOTE_v3", new_folder_name)
        try:
            if os.path.exists(main_log_dir):
                for i in range(0, 5):
                    fold_log_dir = os.path.join(main_log_dir, f"fold_{i + 1}")
                    os.remove(fold_log_dir)
        except Exception as exc:
            print(f"[red]Could not clean directory: {exc}")
        try:
            if os.path.exists(main_log_dir) and not os.path.exists(new_log_dir):
                os.rename(main_log_dir, new_log_dir)
                print(f"[green]Results directory renamed to: {new_log_dir}")
        except Exception as exc:
            print(f"[red]Could not rename directory: {exc}")


if __name__ == '__main__':
    folders = ["data_new_truncated_final", "data_new_truncated", "data_new"]

    for folder in folders:
        try:
            # Clear GPU memory before each run
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Run garbage collection to free memory
            import gc

            gc.collect()

            # Train on the folder
            train_for_folder(folder, USE_SMOTE)

        except Exception as e:
            print(f"[red]Error training on {folder}: {e}")
            import traceback

            traceback.print_exc()
