import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm
from rich import print

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

# Hyperparameters - adjusted for better performance and reduced memory usage
BATCH_SIZE = 8  # Reduced from 64
EPOCHS = 100
NUM_WORKERS = 4  # Reduced from 4 to minimize resource usage
PATIENCE = 15
DROPOUT = 0.3  # Increased from 0.3 for better regularization
HIDDEN_DIM = 256  # Reduced from 128 for memory efficiency
INPUT_DIM = 9
GLOBAL_DIM = 10


class FocalLoss(nn.Module):
    def __init__(self, alpha=2.0, gamma=2.0, reduction='mean', class_weights=None):
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


class MouseMovementDataset(Dataset):
    def __init__(self, data_folder, global_means=None, global_stds=None):
        self.samples = []
        self.class_counts = {0: 0, 1: 0}  # Track class distribution

        for root, _, files in os.walk(data_folder):
            label = 1 if "deceitful" in root.lower() else 0
            self.class_counts[label] += len([f for f in files if f.endswith(".json")])

            for file in files:
                if file.endswith(".json"):
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
        with open(path) as f:
            data = json.load(f)

        traj = data.get("trajectory", [])
        g = np.array([
            data.get("trajectory_metrics", {}).get(k, 0)
            for k in [
                'final_decision_path_efficiency', 'decision_path_efficiency', 'total_time', 'hesitation_time',
                'time_to_first_movement', 'hesitation_count', 'direction_changes',
                'total_pause_time', 'pause_count', 'answer_changes']
        ], dtype=np.float32)

        # Get screen dimensions for normalization (can be adjusted based on your data)
        max_x, max_y = 1920, 1080  # Default screen resolution
        max_v, max_a = 1, 1
        if traj:
            max_x = max(p.get('x', 0) for p in traj) or max_x
            max_y = max(p.get('y', 0) for p in traj) or max_y
            max_v = max(p.get('velocity', 0) for p in traj) or max_v
            max_a = max(p.get('acceleration', 0) for p in traj) or max_a

        features = []
        for p in traj:
            x = p.get('x', 0) / max_x  # Normalize x to [0,1]
            y = p.get('y', 0) / max_y  # Normalize y to [0,1]
            dx = p.get('dx', 0) / max_x  # Normalize dx relative to screen width
            dy = p.get('dy', 0) / max_y  # Normalize dy relative to screen height
            
            # Other features
            velocity = p.get('velocity', 0) / max_v  # Normalize velocity to [0,1]
            acceleration = p.get('acceleration', 0) / max_a  # Normalize acceleration to [0,1]
            curvature = p.get('curvature', 0)
            jerk = p.get('jerk', 0)
            timestamp = p.get('timestamp', 0)
            
            features.append([x, y, dx, dy, velocity, acceleration, curvature, jerk, timestamp])

        # Handle empty trajectory case
        if not features:
            features = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

        features = np.array(features, dtype=np.float32)
        
        # Replace timestamp with normalized time (0-1)
        # Rather than keeping both, which would be redundant
        if features.shape[0] > 1:
            # Replace the timestamp column with evenly spaced time values
            features[:, -1] = np.linspace(0, 1, len(features))
        else:
            features[:, -1] = 0.5  # For single-point trajectories
        
        # Normalize global features
        if self.global_means is not None:
            g = (g - self.global_means) / (self.global_stds + 1e-8)

        return torch.tensor(features, dtype=torch.float32), torch.tensor(g), torch.tensor(label), len(features)


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    seqs, globals_, labels, lengths = zip(*batch)
    padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    return padded, torch.stack(globals_), torch.tensor(labels), torch.tensor(lengths)


class DeceptionNet(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, global_dim=GLOBAL_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()

        # Simplified 1D CNN feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 16, kernel_size=3, padding=1),  # Increased to 64
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),  # Increased to 128
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # Simplified bidirectional LSTM with 7 layers
        self.lstm = nn.LSTM(128, hidden_dim, num_layers=9, batch_first=True,  # Increased to 7 layers
                            dropout=DROPOUT, bidirectional=True)

        # Layer normalization for better training stability
        self.norm = nn.LayerNorm(hidden_dim * 2)

        # Simplified attention mechanism (single-head instead of multi-head)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Global feature processing with residual connection
        self.global_fc1 = nn.Linear(global_dim, 128)  # Increased to 64
        self.global_fc2 = nn.Linear(128, 128)         # Increased to 64
        self.global_norm = nn.LayerNorm(128)
        self.global_dropout = nn.Dropout(DROPOUT)

        # Combined classifier with simplified structure
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 128, 128),  # Reduced input size
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

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

        # Re-pad the sequences for LSTM
        padded = nn.utils.rnn.pad_sequence(conv_out, batch_first=True)

        # Make sure padded is contiguous
        padded = padded.contiguous()
        
        # Handle LSTM differently on CPU vs CUDA to avoid cuDNN issues
        if DEVICE.type == 'cuda':
            # For CUDA, we'll avoid the packed sequence approach which can cause issues with cuDNN
            # Create a mask for valid sequence positions
            mask = torch.zeros(padded.size(0), padded.size(1), dtype=torch.bool, device=padded.device)
            for i, length in enumerate(lengths):
                mask[i, :length] = True
                
            # Process with LSTM
            lstm_out, _ = self.lstm(padded)
            
            # Zero out invalid positions
            lstm_out = lstm_out * mask.unsqueeze(-1)
        else:
            # On CPU, we can use packed sequence safely
            packed = pack_padded_sequence(padded, lengths.cpu(), batch_first=True, enforce_sorted=True)
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        # Apply layer normalization (lstm_out is already unpacked if we're on CUDA)
        normalized = self.norm(lstm_out)

        # Single-head attention (simplified from multi-head)
        attn_weights = torch.softmax(self.attention(normalized), dim=1)
        sequence_encoding = torch.sum(attn_weights * normalized, dim=1)

        # Process global features with residual connection
        g1 = self.global_fc1(g)
        g2 = self.global_fc2(g1)
        global_out = self.global_norm(g1 + g2)
        global_out = self.global_dropout(global_out)

        # Combine sequence and global features
        combined = torch.cat((sequence_encoding, global_out), dim=1)

        return self.classifier(combined)


def compute_stats(dataset):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Reduced from 5.0

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

    with torch.no_grad():
        for x, g, y, lengths in tqdm(loader, desc="Evaluating", leave=False):
            x, g, y, lengths = x.to(DEVICE), g.to(DEVICE), y.to(DEVICE), lengths.to(DEVICE)
            out = model(x, g, lengths)
            loss = criterion(out, y)
            total_loss += loss.item()

            pred = out.argmax(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Debug class distribution in predictions
    pred_counts = {0: all_preds.count(0), 1: all_preds.count(1)}
    print(f"Prediction distribution: {pred_counts}")

    # Handle case where all predictions are the same class
    if len(set(all_preds)) <= 1:
        print("[red]WARNING: Model is predicting only one class!")

    p, r, f, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)

    # Calculate balanced accuracy
    from sklearn.metrics import balanced_accuracy_score
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    print(f"Balanced accuracy: {balanced_acc:.4f}")

    return total_loss / len(loader), acc, p, r, f, all_labels, all_preds


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


def create_reduced_model():
    """Create a smaller model for memory-intensive datasets"""
    # Further reduce dimensions and complexity
    global HIDDEN_DIM
    orig_hidden_dim = HIDDEN_DIM
    HIDDEN_DIM = 64  # Reduce hidden dimension
    
    model = DeceptionNet(hidden_dim=HIDDEN_DIM)
    
    # Restore original value
    HIDDEN_DIM = orig_hidden_dim
    return model


def train_for_folder(folder_name):
    print(f"[magenta] \n{'=' * 40}\nTraining on {folder_name}\n{'=' * 40}")
    log_dir = os.path.join("logs_LSTM", folder_name)
    os.makedirs(log_dir, exist_ok=True)
    actual_batch_size = BATCH_SIZE

    print(f"[yellow]Using batch size: {actual_batch_size}")

    # Create and analyze dataset
    dataset = MouseMovementDataset(folder_name)
    mean, std = compute_stats(dataset)
    dataset = MouseMovementDataset(folder_name, mean, std)

    # Calculate class weights for dealing with imbalance
    labels = [dataset.samples[i][1] for i in range(len(dataset))]
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    print(f"Class weights: {class_weights}")

    # Use stratified sampling for dataset splits
    indices = list(range(len(dataset)))

    # Get stratified splits
    train_idx, temp_idx = train_test_split(
        indices, 
        test_size=0.5,
        stratify=labels,
        random_state=42
    )

    # Get labels for temp indices
    temp_labels = [labels[i] for i in temp_idx]

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=2/5,  # 2/5 of the temp set (20% of total)
        stratify=temp_labels,
        random_state=42
    )

    # Create subsets
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    # Verify split distributions
    train_labels = [labels[i] for i in train_idx]
    val_labels = [labels[i] for i in val_idx]
    test_labels = [labels[i] for i in test_idx]

    print(f"Train set: {len(train_set)} samples, distribution: {np.bincount(train_labels)}")
    print(f"Val set: {len(val_set)} samples, distribution: {np.bincount(val_labels)}")
    print(f"Test set: {len(test_set)} samples, distribution: {np.bincount(test_labels)}")

    # Create weighted sampler for balanced batches
    class_sample_count = np.bincount(train_labels)
    weight = 1. / class_sample_count
    samples_weight = torch.tensor([weight[t] for t in train_labels])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    # Create data loaders - use sampler for training
    train_loader = DataLoader(train_set, batch_size=actual_batch_size, sampler=sampler, 
                              collate_fn=collate_fn, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=actual_batch_size, shuffle=False, 
                            collate_fn=collate_fn, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_set, batch_size=actual_batch_size, shuffle=False, 
                             collate_fn=collate_fn, num_workers=NUM_WORKERS)

    # Try-except block to catch CUDA errors and reduce resources if needed
    try:
        # Initialize model - use smaller model for larger datasets
        if folder_name == "data_reconstructed" or folder_name == "data_combined" and DEVICE.type == 'cuda':
            print("[yellow]Using reduced model for large dataset")
            model = create_reduced_model().to(DEVICE)
        else:
            model = DeceptionNet().to(DEVICE)
        
        # AdamW optimizer with weight decay for better regularization
        optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
        
        # Use cosine annealing scheduler for better convergence
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,  # Restart every 10 epochs
            T_mult=1,
            eta_min=1e-6
        )
    
        # Use class weights and focal loss for imbalanced data
        criterion = FocalLoss(alpha=2.0, gamma=2.0, class_weights=class_weights.to(DEVICE))
    
        best_f1 = 0
        best_v_loss = np.inf
        delta = 0.001
        best_model_path = os.path.join(log_dir, "best_model.pt")
        stopper = EarlyStopping(patience=PATIENCE)
    
        # Training history for plotting
        history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': []
        }
    
        for epoch in range(1, EPOCHS + 1):
            print(f"\nEpoch {epoch}/{EPOCHS}")
            start_time = time.time()
    
            # Train
            t_loss, t_acc, _, _, t_f = train_epoch(model, train_loader, optimizer, None, criterion)
            
            # Step scheduler once per epoch 
            scheduler.step()
    
            # Validate
            v_loss, v_acc, _, _, v_f, _, _ = evaluate(model, val_loader, criterion)
    
            # Record history
            history['train_loss'].append(t_loss)
            history['train_acc'].append(t_acc)
            history['train_f1'].append(t_f)
            history['val_loss'].append(v_loss)
            history['val_acc'].append(v_acc)
            history['val_f1'].append(v_f)
    
            duration = time.time() - start_time
    
            print(f"\nEpoch {epoch:03}: Train Loss={t_loss:.4f} Acc={t_acc:.2%} F1={t_f:.4f} |"
                  f" Val Loss={v_loss:.4f} Acc={v_acc:.2%} F1={v_f:.4f} |"
                  f" Time={duration:.2f}s |"
                  f" LR={optimizer.param_groups[0]['lr']:.6f}")
    
            # Save if improved
            improved = False
            if v_loss < best_v_loss - delta and v_f > best_f1 + delta:
                print(f"[green]Model Checkpoint: \nF1 improved: {best_f1:.4f} -> {v_f:.4f}."
                      f"\nVal Loss improved: {best_v_loss:.4f} -> {v_loss:.4f}\nSaving model.")
                best_f1 = v_f
                best_v_loss = v_loss
                torch.save(model.state_dict(), best_model_path)
                improved = True
    
            # Early stopping check
            if stopper(v_loss, v_f):
                if improved:
                    print("[green]Model improved!")
            elif stopper.early_stop:
                print("[red]Early stopping triggered.")
                break
    
            # Debug predictions every 5 epochs
            if epoch % 5 == 0 or epoch == 1:
                print("\nDebug validation predictions:")
                debug_predictions(model, val_loader)
    
        # Plot training history
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
    
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Accuracy')
        plt.plot(history['val_acc'], label='Val Accuracy')
        plt.plot(history['train_f1'], label='Train F1')
        plt.plot(history['val_f1'], label='Val F1')
        plt.title('Metrics History')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
    
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "training_history.png"))
        plt.close()
    
        # Final evaluation on test set
        print("\nLoading best model for final evaluation...")
        model.load_state_dict(torch.load(best_model_path))
        _, acc, p, r, f, y_true, y_pred = evaluate(model, test_loader, criterion)
        print(f"Test: Acc={acc:.4f} Precision={p:.4f} Recall={r:.4f} F1={f:.4f}")
    
        # Detailed debug of test set
        print("\nTest set prediction analysis:")
        debug_predictions(model, test_loader)
    
        # Save classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        with open(os.path.join(log_dir, "classification_report.txt"), "w") as f:
            f.write(classification_report(y_true, y_pred))
            f.write("\n\nPrecision, Recall, F1 by class:\n")
            for cls in [0, 1]:
                f.write(f"Class {cls}: Precision={report[str(cls)]['precision']:.4f}, ")
                f.write(f"Recall={report[str(cls)]['recall']:.4f}, ")
                f.write(f"F1={report[str(cls)]['f1-score']:.4f}, ")
                f.write(f"Support={report[str(cls)]['support']}\n")
    
        # Confusion matrix with normalization
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        class_names = ['Truthful', 'Deceitful']
    
        # Plot regular confusion matrix
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
    
        # Plot normalized confusion matrix
        plt.subplot(1, 2, 2)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title("Normalized Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "confusion_matrix.png"))
        plt.close()

        # Save ROC curve
        from sklearn.metrics import roc_curve, roc_auc_score
        model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for x, g, y, lengths in test_loader:
                x, g, lengths = x.to(DEVICE), g.to(DEVICE), lengths.to(DEVICE)
                out = model(x, g, lengths)
                probs = torch.softmax(out, dim=1)[:, 1]  # Probability of class 1
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(y.numpy())

        # ROC curve
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        auc = roc_auc_score(all_labels, all_probs)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig(os.path.join(log_dir, "roc_curve.png"))
        plt.close()

        # Save model as CPU (for prediction)
        try:
            # CPU model saving for compatibility
            model.to("cpu")
            torch.save(model.state_dict(), os.path.join(log_dir, "deception_model_cpu.pt"))
            print("[green]Model saved successfully for CPU inference.")
            new_folder_name = f"{folder_name} ({acc*100:.2f}% Test Acc)"
            log_dir_score = os.path.join("logs_LSTM", new_folder_name)
            os.rename(log_dir, log_dir_score)
        except Exception as exp:
            print(f"[red]Error saving model: {exp}")
            
    except RuntimeError as run_err:
        if "CUDA out of memory" in str(run_err):
            print(f"[red]⚠️ CUDA out of memory in {folder}. Try reducing batch size or model size.")
            raise run_err
        elif "CUDNN_STATUS_NOT_SUPPORTED" in str(run_err):
            print(f"[red]⚠️ cuDNN error in {folder}. This is likely due to non-contiguous tensors.")
            raise run_err
        else:
            raise run_err


if __name__ == '__main__':
    for folder in ["data_new_truncated", "data_new"]:  # "data_combined", "data_reconstructed"]:
        try:
            # Set memory limit for PyTorch on CUDA to avoid OOM errors
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory
                fraction = 0.8  # Use 80% for others
                
                torch.cuda.set_per_process_memory_fraction(fraction)
                print(f"[yellow]Setting memory fraction to {fraction:.1f} ({fraction*total_memory/1e9:.1f}GB)")
                
                # Empty cache before each run
                torch.cuda.empty_cache()
            
            # Clear any remaining references
            import gc
            gc.collect()
            
            train_for_folder(folder)
            
        except Exception as e:
            print(f"[red]Error training on {folder}: {e}")
            import traceback
            traceback.print_exc()
            
            # If we get a CUDA error, try again with CPU
            if "CUDA" in str(e):
                print(f"[yellow]\nRetrying {folder} on CPU...")
                orig_device = DEVICE
                globals()["DEVICE"] = torch.device("cpu")
                try:
                    train_for_folder(folder)
                except Exception as e2:
                    print(f"[red]CPU training also failed: {e2}")
                    traceback.print_exc()
                finally:
                    # Restore device
                    globals()["DEVICE"] = orig_device
