import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from scipy import stats
from tqdm import tqdm


# Function to recursively find all JSON files in a directory
def find_json_files(directory):
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.json'):
                files.append(os.path.join(root, filename))
    return files


def extract_features_from_trajectory(traj_data, metrics_data=None):
    """Extract features from trajectory data"""
    if not traj_data:
        return None

    try:
        # Basic statistics from the trajectory
        x_coords = [p.get('x', 0) for p in traj_data]
        y_coords = [p.get('y', 0) for p in traj_data]
        velocities = [p.get('velocity', 0) for p in traj_data]
        accelerations = [p.get('acceleration', 0) for p in traj_data]
        curvatures = [p.get('curvature', 0) for p in traj_data]
        jerks = [p.get('jerk', 0) for p in traj_data]
        clicks = [p.get('click', 0) for p in traj_data]
        timestamps = [p.get('timestamp', 0) for p in traj_data]

        # Calculate time taken for the entire trajectory
        if len(traj_data) >= 2:
            time_taken = (timestamps[-1] - timestamps[0]) / 1000.0  # in seconds
        else:
            time_taken = 0

        # Calculate directional changes
        dx_values = [p.get('dx', 0) for p in traj_data]
        dy_values = [p.get('dy', 0) for p in traj_data]

        # Count direction changes (sign changes in dx and dy)
        dir_changes_x = sum(1 for i in range(1, len(dx_values)) if dx_values[i] * dx_values[i - 1] < 0)
        dir_changes_y = sum(1 for i in range(1, len(dy_values)) if dy_values[i] * dy_values[i - 1] < 0)

        # Detect pauses (where velocity is close to 0)
        pauses = sum(1 for v in velocities if abs(v) < 0.01)

        # Calculate pause durations - a pause is defined as consecutive points with very low velocity
        pause_durations = []
        current_pause = 0
        pause_start = 0  # Initialize pause_start to fix the reference error

        for i in range(1, len(velocities)):
            if abs(velocities[i]) < 0.01:
                if current_pause == 0 and i > 0:  # Start of a new pause
                    pause_start = timestamps[i - 1]
                current_pause += 1
            elif current_pause > 0:  # End of a pause
                pause_end = timestamps[i - 1]
                pause_durations.append((pause_end - pause_start) / 1000.0)  # in seconds
                current_pause = 0

        # If we end in a pause, add it
        if current_pause > 0 and len(timestamps) > 0:
            pause_durations.append((timestamps[-1] - pause_start) / 1000.0)

        # Detect hesitations (velocity drops significantly then increases again)
        hesitations = 0
        hesitation_durations = []
        in_hesitation = False
        hesitation_start = 0

        for i in range(1, len(velocities) - 1):
            # Start of hesitation: velocity drops significantly
            if not in_hesitation and velocities[i] < velocities[i - 1] * 0.3:
                in_hesitation = True
                hesitation_start = timestamps[i]
            # End of hesitation: velocity increases significantly
            elif in_hesitation and velocities[i + 1] > velocities[i] * 2:
                hesitations += 1
                hesitation_durations.append((timestamps[i] - hesitation_start) / 1000.0)
                in_hesitation = False

        # Calculate path efficiency (ratio of direct distance to actual distance traveled)
        if len(x_coords) >= 2:
            direct_distance = np.sqrt((x_coords[-1] - x_coords[0]) ** 2 + (y_coords[-1] - y_coords[0]) ** 2)
            path_distances = [np.sqrt((x_coords[i] - x_coords[i - 1]) ** 2 + (y_coords[i] - y_coords[i - 1]) ** 2)
                              for i in range(1, len(x_coords))]
            total_distance = sum(path_distances)
            path_efficiency = direct_distance / total_distance if total_distance > 0 else 0
        else:
            path_efficiency = 0
            direct_distance = 0
            total_distance = 0

        # Calculate temporal features
        if len(timestamps) >= 2:
            # Time between consecutive points
            time_diffs = [(timestamps[i] - timestamps[i - 1]) / 1000.0 for i in range(1, len(timestamps))]
            avg_time_between_points = np.mean(time_diffs) if time_diffs else 0
            std_time_between_points = np.std(time_diffs) if time_diffs else 0
        else:
            avg_time_between_points = 0
            std_time_between_points = 0

        # Calculate click-related features
        first_click_time = 0
        if any(clicks) and len(timestamps) > 0:
            first_click_idx = clicks.index(1)
            first_click_time = (timestamps[first_click_idx] - timestamps[0]) / 1000.0

        # Calculate velocity and acceleration patterns
        velocity_changes = [abs(velocities[i] - velocities[i - 1]) for i in range(1, len(velocities))]
        accel_changes = [abs(accelerations[i] - accelerations[i - 1]) for i in range(1, len(accelerations))]

        avg_velocity_change = np.mean(velocity_changes) if velocity_changes else 0
        avg_accel_change = np.mean(accel_changes) if accel_changes else 0

        # Calculate variability in movement
        velocity_std = np.std(velocities) if velocities else 0
        acceleration_std = np.std(accelerations) if accelerations else 0

        # Calculate advanced movement features
        jerk_mean = np.mean(jerks) if jerks else 0
        jerk_std = np.std(jerks) if jerks else 0
        curvature_mean = np.mean(curvatures) if curvatures else 0
        curvature_std = np.std(curvatures) if curvatures else 0

        # Compute movement smoothness using normalized jerk
        if len(jerks) > 2 and time_taken > 0:
            normalized_jerk = np.sum(np.array(jerks) ** 2) * (time_taken ** 3)
            movement_smoothness = -np.log(normalized_jerk) if normalized_jerk > 0 else 0
        else:
            movement_smoothness = 0
            normalized_jerk = 0

        # Calculate spatial features
        if len(x_coords) > 0 and len(y_coords) > 0:
            x_std = np.std(x_coords)
            y_std = np.std(y_coords)
            xy_covariance = np.cov(x_coords, y_coords)[0, 1] if len(x_coords) > 1 else 0
        else:
            x_std = 0
            y_std = 0
            xy_covariance = 0

        # Extract metrics directly from metrics_data if available
        extracted_metrics = {}
        if metrics_data:
            for key, value in metrics_data.items():
                # Convert to float to ensure numeric values
                try:
                    extracted_metrics[f"metric_{key}"] = float(value)
                except (ValueError, TypeError):
                    # Skip non-numeric values
                    pass

        # Combine all features
        features = {
            'trajectory_length': len(traj_data),
            'time_taken': time_taken,
            'mean_velocity': np.mean(velocities) if velocities else 0,
            'max_velocity': np.max(velocities) if velocities else 0,
            'mean_acceleration': np.mean(accelerations) if accelerations else 0,
            'max_acceleration': np.max(accelerations) if velocities else 0,
            'velocity_std': velocity_std,
            'acceleration_std': acceleration_std,
            'direction_changes_x': dir_changes_x,
            'direction_changes_y': dir_changes_y,
            'total_direction_changes': dir_changes_x + dir_changes_y,
            'pauses_count': pauses,
            'avg_pause_duration': np.mean(pause_durations) if pause_durations else 0,
            'total_pause_duration': sum(pause_durations) if pause_durations else 0,
            'hesitations_count': hesitations,
            'avg_hesitation_duration': np.mean(hesitation_durations) if hesitation_durations else 0,
            'total_hesitation_duration': sum(hesitation_durations) if hesitation_durations else 0,
            'path_efficiency': path_efficiency,
            'direct_distance': direct_distance,
            'total_distance': total_distance,
            'jerk_mean': jerk_mean,
            'jerk_std': jerk_std,
            'curvature_mean': curvature_mean,
            'curvature_std': curvature_std,
            'movement_smoothness': movement_smoothness,
            'normalized_jerk': normalized_jerk,
            'x_range': max(x_coords) - min(x_coords) if x_coords else 0,
            'y_range': max(y_coords) - min(y_coords) if y_coords else 0,
            'x_std': x_std,
            'y_std': y_std,
            'xy_covariance': xy_covariance,
            'avg_time_between_points': avg_time_between_points,
            'std_time_between_points': std_time_between_points,
            'first_click_time': first_click_time,
            'click_count': sum(clicks),
            'avg_velocity_change': avg_velocity_change,
            'avg_accel_change': avg_accel_change,
            'velocity_accel_ratio': np.mean(velocities) / np.mean(accelerations) if np.mean(accelerations) != 0 else 0,
        }

        # Add metrics from the metrics_data
        features.update(extracted_metrics)

        return features

    except Exception as e:
        print(f"Error extracting features: {e}")
        return None


def load_and_process_data(data_dir):
    """Load and process all JSON files from the data directory"""
    print(f"Searching for JSON files in {data_dir}")
    all_files = find_json_files(data_dir)
    print(f"Found {len(all_files)} JSON files")

    data = []
    labels = []
    label_counts = {"truthful": 0, "deceitful": 0}

    for file_path in tqdm(all_files, desc="Loading files"):
        try:
            with open(file_path, 'r') as f:
                json_data = json.load(f)

            # Get label from the file or folder name
            file_label = json_data.get('label', '').lower()

            # Simplified label determination logic
            if ('deceptive' in file_label or 'deceitful' in file_label
                    or any(term in file_path.lower() for term in ['deceitful', 'deceptive'])):
                label = 1
                label_counts["deceitful"] += 1
            else:
                label = 0
                label_counts["truthful"] += 1

            # Extract trajectory data - handle both array and object formats
            trajectory = json_data.get('trajectory', [])

            # Make sure trajectory is a list of dictionaries
            if isinstance(trajectory, dict):
                # Convert from object format to array if needed
                trajectory = [trajectory]

            # Get metrics data
            metrics = json_data.get('trajectory_metrics', {})

            # Extract features
            features = extract_features_from_trajectory(trajectory, metrics)

            if features:
                data.append(features)
                labels.append(label)
            else:
                print(f"Skipping {file_path}: Could not extract features")
        except json.JSONDecodeError:
            print(f"Error: {file_path} is not a valid JSON file")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"Processed {len(data)} files successfully")
    print(f"Label distribution: {label_counts}")

    return data, labels


def analyze_feature_importance(x, y):
    """Analyze feature importance using multiple methods"""
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

    # Feature importance from Random Forest with additional hyperparameters
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        min_samples_leaf=5,  # Added hyperparameter
        max_features='sqrt'   # Added hyperparameter
    )
    rf.fit(x_train, y_train)

    # Evaluate the model
    y_pred = rf.predict(x_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    print(f"Random Forest Performance - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    print(classification_report(y_test, y_pred))

    # Feature importance scores
    feature_names = x.columns
    rf_importance = pd.DataFrame({
        'Feature': feature_names,
        'RF_Importance': rf.feature_importances_
    }).sort_values('RF_Importance', ascending=False)

    # Mutual information (measures dependency between variables)
    mi_scores = mutual_info_classif(x, y, random_state=42)
    mi_importance = pd.DataFrame({
        'Feature': feature_names,
        'MI_Importance': mi_scores
    }).sort_values('MI_Importance', ascending=False)

    # Statistical tests for each feature
    stat_tests = {}
    for feature in feature_names:
        truthful_values = x[feature][np.array(y) == 0]
        deceitful_values = x[feature][np.array(y) == 1]

        # T-test
        t_stat, p_value = stats.ttest_ind(truthful_values, deceitful_values, equal_var=False, nan_policy='omit')
        stat_tests[feature] = {
            'p_value': p_value,
            't_statistic': t_stat,
            'truthful_mean': np.mean(truthful_values),
            'deceitful_mean': np.mean(deceitful_values),
            'difference': np.mean(deceitful_values) - np.mean(truthful_values),
            'effect_size': abs(np.mean(deceitful_values) - np.mean(truthful_values)) / np.sqrt(
                (np.std(truthful_values) ** 2 + np.std(deceitful_values) ** 2) / 2) if len(truthful_values) > 0 and len(
                deceitful_values) > 0 else 0
        }

    # Convert to DataFrame and sort by effect size
    stat_df = pd.DataFrame.from_dict(stat_tests, orient='index')
    stat_df['Feature'] = stat_df.index
    stat_df = stat_df.sort_values('effect_size', ascending=False)

    # Combine all importance scores
    combined_importance = pd.merge(rf_importance, mi_importance, on='Feature', how="inner", validate="many_to_many")
    combined_importance = pd.merge(combined_importance, stat_df, on='Feature', how="inner", validate="many_to_many")

    # Normalize importance scores
    for col in ['RF_Importance', 'MI_Importance', 'effect_size']:
        if combined_importance[col].max() > 0:
            combined_importance[f'{col}_norm'] = combined_importance[col] / combined_importance[col].max()
        else:
            combined_importance[f'{col}_norm'] = 0

    # Initialize combined score column with zeros
    combined_importance['combined_score'] = 0

    # Calculate combined score (safely handle missing columns)
    norm_cols = [col for col in ['RF_Importance_norm', 'MI_Importance_norm', 'effect_size_norm']
                 if col in combined_importance.columns]

    if norm_cols:
        combined_importance['combined_score'] = combined_importance[norm_cols].mean(axis=1)

    combined_importance = combined_importance.sort_values('combined_score', ascending=False)

    return combined_importance, rf, (x_test, y_test)


def visualize_feature_distributions(x, y, top_features, output_dir='feature_analysis'):
    """Visualize the distribution of top features between truthful and deceitful samples"""
    os.makedirs(output_dir, exist_ok=True)

    # Create a DataFrame with features and label
    df = x.copy()
    df['label'] = y
    df['label'] = df['label'].map({0: 'Truthful', 1: 'Deceitful'})

    # Plot histograms of top features
    for feature in top_features:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=feature, hue='label', kde=True, element='step', common_norm=False, bins=30)
        plt.title(f'Distribution of {feature} by Label')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{feature}_hist.png'))
        plt.close()

        # Removed boxplot creation as requested

    # Create a correlation matrix of top features (numeric only)
    top_features_df = df[top_features + ['label']]
    plt.figure(figsize=(12, 10))
    # Only compute correlation on numeric features, not the 'label' column
    correlation = top_features_df[top_features].corr()
    mask = np.triu(correlation)
    sns.heatmap(correlation, annot=True, cmap='coolwarm', mask=mask, vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Top Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()


def main():
    # Directory containing data
    data_dir = 'data_new_truncated'  # Replace with the path to your data directory
    output_dir = 'feature_analysis'  # Changed output directory name as requested

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load and process the data
    print("Loading and processing data...")
    features_list, labels = load_and_process_data(data_dir)

    # Convert to DataFrame for easier manipulation
    if features_list:
        df = pd.DataFrame(features_list)

        # Handle missing values
        df.fillna(0, inplace=True)

        # Create a copy of the original DataFrame for visualization
        df_viz = df.copy()

        # Normalize features
        scaler = StandardScaler()
        x_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

        # Analyze feature importance
        print("\nAnalyzing feature importance...")
        importance_df, model, test_data = analyze_feature_importance(x_scaled, labels)

        # Print top 20 most important features
        print("\nTop 20 most discriminative features:")
        top_features = importance_df.head(20)
        print(top_features[['Feature', 'combined_score', 'p_value', 'difference', 'truthful_mean', 'deceitful_mean']])

        # Visualize the distributions of the top 10 features
        print("\nCreating visualizations...")
        top_10_features = top_features['Feature'].head(10).tolist()
        visualize_feature_distributions(df_viz, labels, top_10_features, output_dir)

        # Save importance data to CSV in the output directory
        importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)

        # Evaluate model with only top features
        x_test, y_test = test_data
        top_n_features = top_features['Feature'].head(10).tolist()

        y_pred = model.predict(x_test)
        print("\nModel performance with all features:")
        print(classification_report(y_test, y_pred))

        # Train a model with only top features
        x_top = x_scaled[top_n_features]
        x_train_top, x_test_top, y_train_top, y_test_top = train_test_split(
            x_top, labels, test_size=0.3, random_state=42, stratify=labels
        )

        # Added hyperparameters to RandomForestClassifier
        model_top = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            min_samples_leaf=5,  # Added hyperparameter
            max_features='sqrt'   # Added hyperparameter
        )
        model_top.fit(x_train_top, y_train_top)

        y_pred_top = model_top.predict(x_test_top)
        print("\nModel performance with only top 10 features:")
        print(classification_report(y_test_top, y_pred_top))

        # Create confusion matrix
        cm = confusion_matrix(y_test_top, y_pred_top)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Truthful', 'Deceitful'],
                    yticklabels=['Truthful', 'Deceitful'])
        plt.title('Confusion Matrix (Top Features)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()

        # Feature importance plot for the top features
        plt.figure(figsize=(12, 8))
        top_20 = importance_df.head(20)
        sns.barplot(x='combined_score', y='Feature', data=top_20)
        plt.title('Top 20 Features by Combined Importance Score')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_features_importance.png'))

        print("\nAnalysis complete. Results saved to feature_analysis directory.")
    else:
        print("No valid data found. Please check your data directory.")


if __name__ == "__main__":
    main()
