import os
import csv
import json
import math
import re
from datetime import datetime


def extract_timestamp_from_filename(filename):
    """Extract timestamp from filename like 'tracking_20250319_112111_yes.csv'"""
    match = re.search(r'tracking_(\d{8})_(\d{6})_', filename)
    if match:
        date_str = match.group(1)
        time_str = match.group(2)
        datetime_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
        try:
            return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S").timestamp()
        except ValueError:
            return 0
    return 0


def calculate_pause_metrics(trajectory_data):
    """Calculate pause-related metrics from trajectory data_new"""
    pause_threshold = 10  # ms, same as in the original script
    pause_count = 0
    total_pause_time = 0
    last_movement_time = None

    for i in range(1, len(trajectory_data)):
        current_point = trajectory_data[i]
        prev_point = trajectory_data[i - 1]

        # Extract timestamp and position
        timestamp = current_point["timestamp"]
        prev_timestamp = prev_point["timestamp"]

        # If there's no movement between consecutive data_new points
        if (current_point["dx"] == 0 and current_point["dy"] == 0 and
                (current_point["x"] == prev_point["x"] and current_point["y"] == prev_point["y"])):

            # If this is the beginning of a pause
            if last_movement_time is not None:
                pause_duration = timestamp - prev_timestamp
                if pause_duration >= pause_threshold:
                    pause_count += 1
                    total_pause_time += pause_duration
                    last_movement_time = None
        else:
            # There was movement
            last_movement_time = timestamp

    return pause_count, total_pause_time


def calculate_hover_metrics(trajectory_data):
    """Calculate hover metrics based on click locations

    This function estimates hover time and count around areas where clicks occur.
    It creates a bounding box around each click point to simulate button areas,
    then measures time spent within these areas before the click occurs.
    """
    # Define bounds for button size based on CSS for answer-btn
    button_width = 200  # pixels (max-width from CSS)
    button_height = 48  # pixels (approximated from padding)

    # Extract click points from the trajectory
    click_points = []
    for i, point in enumerate(trajectory_data):
        if point.get("click", 0) == 1:
            click_points.append((i, point))

    # If no clicks, use default method
    if not click_points:
        return 0, 0

    # Initialize metrics
    hover_count = 0
    hover_time = 0

    # Create button areas around each click point
    button_areas = []
    for _, click_point in click_points:
        # Create a bounding box centered on the click
        x_center = click_point["x"]
        y_center = click_point["y"]

        # Calculate bounds
        x_min = x_center - button_width/2
        x_max = x_center + button_width/2
        y_min = y_center - button_height/2
        y_max = y_center + button_height/2

        button_areas.append({
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "timestamp": click_point["timestamp"]
        })

    # Track state for each button area
    hovering_states = [False] * len(button_areas)
    hover_starts = [None] * len(button_areas)

    # Process trajectory to find hover events
    for point in trajectory_data:
        point_time = point["timestamp"]
        point_x = point["x"]
        point_y = point["y"]

        # Check each button area
        for i, area in enumerate(button_areas):
            # Only consider hovering before the click
            if point_time >= area["timestamp"]:
                continue

            in_button_area = (
                    area["x_min"] <= point_x <= area["x_max"] and
                    area["y_min"] <= point_y <= area["y_max"]
            )

            if in_button_area and not hovering_states[i]:
                # Start of hover
                hovering_states[i] = True
                hover_starts[i] = point_time
                hover_count += 1
            elif not in_button_area and hovering_states[i]:
                # End of hover
                hovering_states[i] = False
                if hover_starts[i] is not None:
                    hover_time += point_time - hover_starts[i]

    # Add any ongoing hovers at the end of trajectory
    for i, hovering in enumerate(hovering_states):
        if hovering and hover_starts[i] is not None and i < len(button_areas):
            # Use click timestamp as the end time
            hover_time += button_areas[i]["timestamp"] - hover_starts[i]

    return hover_count, hover_time


def calculate_direction_changes(trajectory_data):
    """Calculate the number of direction changes in the trajectory"""
    if len(trajectory_data) < 3:
        return 0

    direction_changes = 0
    last_direction = None

    for i in range(1, len(trajectory_data)):
        dx = trajectory_data[i]["dx"]
        dy = trajectory_data[i]["dy"]

        if dx == 0 and dy == 0:
            continue

        # Calculate movement direction (in 8 directions)
        angle = math.atan2(dy, dx) * 180 / math.pi
        direction = round(angle / 45) * 45

        # Check if direction changed significantly
        if last_direction is not None and direction != last_direction:
            direction_changes += 1

        last_direction = direction

    return direction_changes


def calculate_answer_changes(trajectory_data):
    """Estimate the number of times the answer was changed
    based on clicks and movement patterns"""
    # Look for clicks in the trajectory data_new
    clicks = [point for point in trajectory_data if point["click"] == 1]

    if len(clicks) <= 1:
        return 0

    # We'll assume each click after the first one represents a change
    return len(clicks) - 1


def calculate_time_to_first_movement(trajectory_data):
    """Calculate time from start to first significant movement"""
    if len(trajectory_data) < 2:
        return 0

    start_time = trajectory_data[0]["timestamp"]

    for point in trajectory_data[1:]:
        if point["dx"] != 0 or point["dy"] != 0:
            return (point["timestamp"] - start_time) / 1000  # convert to seconds

    return 0


def calculate_decision_path_efficiency(trajectory_data):
    """Calculate path efficiency up to first click decision"""
    if len(trajectory_data) < 2:
        return 1.0

    # Find the first click
    first_click_index = None
    for i, point in enumerate(trajectory_data):
        if point["click"] == 1:
            first_click_index = i
            break

    if first_click_index is None:
        # If no click, use the entire trajectory
        first_click_index = len(trajectory_data) - 1

    # Get the subset of the trajectory up to the first click
    decision_path = trajectory_data[:first_click_index + 1]

    # Calculate efficiency
    start = decision_path[0]
    end = decision_path[-1]

    # Direct distance (straight line)
    direct_distance = math.sqrt(
        (end["x"] - start["x"]) ** 2 + (end["y"] - start["y"]) ** 2
    )

    # Actual path length
    path_length = 0
    for i in range(1, len(decision_path)):
        segment_length = math.sqrt(
            (decision_path[i]["x"] - decision_path[i - 1]["x"]) ** 2 +
            (decision_path[i]["y"] - decision_path[i - 1]["y"]) ** 2
        )
        path_length += segment_length

    # Avoid division by zero
    if path_length == 0:
        return 1.0

    return direct_distance / path_length


def calculate_final_decision_path_efficiency(trajectory_data):
    """Calculate path efficiency up to the last click decision"""
    if len(trajectory_data) < 2:
        return 1.0

    # Find the last click
    last_click_index = None
    for i in range(len(trajectory_data) - 1, -1, -1):
        if trajectory_data[i]["click"] == 1:
            last_click_index = i
            break

    if last_click_index is None:
        # If no click, use the entire trajectory
        last_click_index = len(trajectory_data) - 1

    # Get the subset of the trajectory up to the last click
    decision_path = trajectory_data[:last_click_index + 1]

    # Calculate efficiency the same way as for the first decision
    start = decision_path[0]
    end = decision_path[-1]

    direct_distance = math.sqrt(
        (end["x"] - start["x"]) ** 2 + (end["y"] - start["y"]) ** 2
    )

    path_length = 0
    for i in range(1, len(decision_path)):
        segment_length = math.sqrt(
            (decision_path[i]["x"] - decision_path[i - 1]["x"]) ** 2 +
            (decision_path[i]["y"] - decision_path[i - 1]["y"]) ** 2
        )
        path_length += segment_length

    if path_length == 0:
        return 1.0

    return direct_distance / path_length


def calculate_acceleration_jerk(trajectory_data):
    """Calculate acceleration and jerk for each point in the trajectory"""
    # Set up time window for smoothing
    window_size = 5  # Number of samples to use for smoothing

    # Initialize velocity buffer for acceleration calculation
    velocity_buffer = []
    # Store previous accelerations for jerk calculation
    acceleration_buffer = []

    # Sampling interval in seconds (100Hz = 0.01s)
    sampling_interval = 0.01

    # Process trajectory to add acceleration and jerk
    for i, point in enumerate(trajectory_data):
        # Calculate velocity if not present
        if "velocity" not in point or point["velocity"] == 0:
            # If this is not the first point, calculate velocity from displacement
            if i > 0:
                prev_point = trajectory_data[i-1]
                dx = point["x"] - prev_point["x"]
                dy = point["y"] - prev_point["y"]
                displacement = math.sqrt(dx**2 + dy**2)
                time_diff = (point["timestamp"] - prev_point["timestamp"]) / 1000  # convert to seconds
                if time_diff > 0:
                    point["velocity"] = displacement / time_diff
                else:
                    point["velocity"] = 0
            else:
                point["velocity"] = 0

        # Update velocity buffer
        velocity_buffer.append(point["velocity"])
        if len(velocity_buffer) > window_size:
            velocity_buffer.pop(0)

        # Calculate acceleration
        if len(velocity_buffer) >= 3:
            # Use linear regression to find the slope of velocity over time
            x = list(range(len(velocity_buffer)))
            y = velocity_buffer

            # Calculate means
            x_mean = sum(x) / len(x)
            y_mean = sum(y) / len(y)

            # Calculate slope (acceleration)
            numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
            denominator = sum((xi - x_mean) ** 2 for xi in x)

            # Convert to acceleration in pixels/sec²
            if denominator != 0:
                acceleration = numerator / denominator / sampling_interval
            else:
                acceleration = 0
        elif i > 0 and "velocity" in trajectory_data[i-1]:
            # Fallback to simple calculation when not enough samples
            velocity_diff = point["velocity"] - trajectory_data[i-1]["velocity"]
            time_diff = sampling_interval  # assume constant sampling rate
            acceleration = velocity_diff / time_diff
        else:
            acceleration = 0

        # Store the calculated acceleration
        point["acceleration"] = acceleration

        # Update acceleration buffer
        acceleration_buffer.append(acceleration)
        if len(acceleration_buffer) > window_size:
            acceleration_buffer.pop(0)

        # Calculate jerk (rate of change of acceleration)
        if len(acceleration_buffer) >= 3:
            # Similar to acceleration calculation, use linear regression for jerk
            x = list(range(len(acceleration_buffer)))
            y = acceleration_buffer

            # Calculate means
            x_mean = sum(x) / len(x)
            y_mean = sum(y) / len(y)

            # Calculate slope (jerk)
            numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
            denominator = sum((xi - x_mean) ** 2 for xi in x)

            # Convert to jerk in pixels/sec³
            if denominator != 0:
                jerk = numerator / denominator / sampling_interval
            else:
                jerk = 0
        elif i > 0 and "acceleration" in trajectory_data[i-1]:
            # Fallback to simple calculation
            acceleration_diff = acceleration - trajectory_data[i-1]["acceleration"]
            time_diff = sampling_interval  # assume constant sampling rate
            jerk = acceleration_diff / time_diff
        else:
            jerk = 0

        # Store the calculated jerk
        point["jerk"] = jerk

    return trajectory_data


def calculate_curvature(trajectory_data):
    """Calculate curvature for each point in the trajectory"""
    for i in range(1, len(trajectory_data) - 1):
        p1 = trajectory_data[i-1]
        p2 = trajectory_data[i]
        p3 = trajectory_data[i+1]

        # Convert to vectors
        v1 = {"x": p2["x"] - p1["x"], "y": p2["y"] - p1["y"]}
        v2 = {"x": p3["x"] - p2["x"], "y": p3["y"] - p2["y"]}

        # Calculate the cross product magnitude
        cross_product = v1["x"] * v2["y"] - v1["y"] * v2["x"]

        # Calculate magnitudes
        v1_mag = math.sqrt(v1["x"] * v1["x"] + v1["y"] * v1["y"])
        v2_mag = math.sqrt(v2["x"] * v2["x"] + v2["y"] * v2["y"])

        # Avoid division by zero
        if v1_mag * v2_mag > 0:
            # Calculate curvature (K = |v1 × v2| / (|v1| * |v2|))
            trajectory_data[i]["curvature"] = abs(cross_product) / (v1_mag * v2_mag)
        else:
            trajectory_data[i]["curvature"] = 0

    # Set curvature for first and last points
    if len(trajectory_data) > 0:
        trajectory_data[0]["curvature"] = 0
    if len(trajectory_data) > 1:
        trajectory_data[-1]["curvature"] = 0

    return trajectory_data


def parse_csv_to_json(csv_filepath):
    """Convert a tracking CSV file to the new JSON format"""
    with open(csv_filepath, 'r', newline='', encoding='utf-8') as csvfile:
        # Try to sniff the dialect to determine delimiter
        sample = csvfile.read(1024)
        csvfile.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample)
            reader = csv.DictReader(csvfile, dialect=dialect)
        except Exception as e:
            # Fallback to basic comma delimiter
            print(f"Warning: Could not detect CSV dialect: {e}. Falling back to comma delimiter.")
            reader = csv.DictReader(csvfile)

        rows = list(reader)

        if not rows:
            print(f"Warning: No data_new found in {csv_filepath}")
            return None

        # Extract label (yes/no) from filename
        label_match = re.search(r'_(yes|no|deceitful)\.', csv_filepath)
        label = label_match.group(1) if label_match else "unknown"

        # Convert trajectory data_new
        trajectory = []
        for row in rows:
            # Make sure all required fields exist
            required_fields = ['timestamp', 'x', 'y', 'dx', 'dy', 'velocity']
            if not all(field in row for field in required_fields):
                print(f"Warning: Missing required fields in {csv_filepath}")
                continue

            # Convert numerical fields
            try:
                entry = {
                    "timestamp": int(float(row.get('timestamp', 0))),
                    "x": int(float(row.get('x', 0))),
                    "y": int(float(row.get('y', 0))),
                    "dx": int(float(row.get('dx', 0))),
                    "dy": int(float(row.get('dy', 0))),
                    "velocity": float(row.get('velocity', 0)),
                    "acceleration": float(row.get('acceleration', 0)) if 'acceleration' in row else 0,
                    "curvature": float(row.get('curvature', 0)) if 'curvature' in row else 0,
                    "jerk": float(row.get('jerk', 0)) if 'jerk' in row else 0,
                    "click": int(float(row.get('click', 0))) if 'click' in row else 0
                }
                trajectory.append(entry)
            except ValueError as e:
                print(f"Warning: Error converting values in {csv_filepath}: {e}")
                continue

        # Check for empty trajectory
        if not trajectory:
            print(f"Warning: No valid trajectory data_new in {csv_filepath}")
            return None

        # Calculate acceleration and jerk if not already present or need recalculation
        recalculate_motion = any(
            point["acceleration"] == 0 and point["jerk"] == 0
            for point in trajectory if point["velocity"] > 0
        )

        if recalculate_motion:
            trajectory = calculate_acceleration_jerk(trajectory)
            trajectory = calculate_curvature(trajectory)

        # Calculate metrics
        pause_count, total_pause_time = calculate_pause_metrics(trajectory)
        hover_count, hover_time = calculate_hover_metrics(trajectory)
        direction_changes = calculate_direction_changes(trajectory)
        time_to_first_movement = calculate_time_to_first_movement(trajectory)

        # Extract sequence ID from the first timestamp
        sequence_id = trajectory[0]["timestamp"] if trajectory else 0

        # Calculate total time in seconds
        if len(trajectory) < 2:
            total_time = 0
        else:
            total_time = (trajectory[-1]["timestamp"] - trajectory[0]["timestamp"]) / 1000

        if rows and len(rows) > 0:
            # Use existing metrics if available
            if 'decision_path_efficiency' in rows[0]:
                decision_path_efficiency_val = float(rows[0].get('decision_path_efficiency', 0))
            else:
                decision_path_efficiency_val = calculate_decision_path_efficiency(trajectory)

            if 'final_decision_path_efficiency' in rows[0]:
                final_decision_path_efficiency_val = float(rows[0].get('final_decision_path_efficiency', 0))
            else:
                final_decision_path_efficiency_val = calculate_final_decision_path_efficiency(trajectory)

            if 'changes_of_mind' in rows[0]:
                answer_changes_val = int(float(rows[0].get('changes_of_mind', 0)))
            else:
                answer_changes_val = calculate_answer_changes(trajectory)
        else:
            # Calculate if not available
            decision_path_efficiency_val = calculate_decision_path_efficiency(trajectory)
            final_decision_path_efficiency_val = calculate_final_decision_path_efficiency(trajectory)
            answer_changes_val = calculate_answer_changes(trajectory)

        # Create the JSON structure
        json_data = {
            "sequence_id": sequence_id,
            "label": label,
            "trajectory_metrics": {
                "decision_path_efficiency": decision_path_efficiency_val,
                "final_decision_path_efficiency": final_decision_path_efficiency_val,
                "total_time": total_time,
                "hesitation_time": hover_time / 1000,  # convert to seconds
                "time_to_first_movement": time_to_first_movement,
                "hesitation_count": hover_count,
                "direction_changes": direction_changes,
                "hover_time": hover_time / 1000,  # convert to seconds
                "hover_count": hover_count,
                "total_pause_time": total_pause_time / 1000,  # convert to seconds
                "pause_count": pause_count,
                "answer_changes": answer_changes_val
            },
            "trajectory": trajectory
        }

        return json_data


def process_directory():
    """Process CSV files from data_csv directory and convert to JSON"""
    backup_base_dir = "data_csv"
    output_base_dir = "data_reconstructed"

    # Ensure output directories exist
    truthful_output_dir = os.path.join(output_base_dir, "truthful")
    deceptive_output_dir = os.path.join(output_base_dir, "deceitful")
    os.makedirs(truthful_output_dir, exist_ok=True)
    os.makedirs(deceptive_output_dir, exist_ok=True)

    # Collect files from truthful and deceitful folders
    truthful_dir = os.path.join(backup_base_dir, "truthful")
    deceptive_dir = os.path.join(backup_base_dir, "deceitful")

    # Check if directories exist
    truthful_files = []
    if not os.path.exists(truthful_dir):
        print(f"Warning: Directory {truthful_dir} does not exist.")
    else:
        truthful_files = [f for f in os.listdir(truthful_dir) if f.endswith('.csv') and f.startswith('tracking_')]

    deceptive_files = []
    if not os.path.exists(deceptive_dir):
        print(f"Warning: Directory {deceptive_dir} does not exist.")
    else:
        deceptive_files = [f for f in os.listdir(deceptive_dir) if f.endswith('.csv') and f.startswith('tracking_')]

    # Sort by timestamp in filename
    truthful_files.sort(key=extract_timestamp_from_filename)
    deceptive_files.sort(key=extract_timestamp_from_filename)

    # Cap at 400 files each (you have 500 for truthful in your code, keeping that)
    truthful_files = truthful_files[:500]
    deceptive_files = deceptive_files[:400]

    # Only apply exclusion logic to truthful files
    files_to_exclude = []
    if truthful_files:
        # Group files in sections of 10
        file_groups = [truthful_files[i:i + 10] for i in range(0, len(truthful_files), 10)]

        # For each group, find the two files with the most recent timestamps to exclude
        for group in file_groups:
            if len(group) > 2:  # Only exclude if there are at least 3 files in the group
                group_with_timestamps = [(f, extract_timestamp_from_filename(f)) for f in group]
                sorted_by_time = sorted(group_with_timestamps, key=lambda x: x[1], reverse=True)
                # Add the two most recent files to exclusion list
                files_to_exclude.extend([f for f, _ in sorted_by_time[:2]])

    print(f"Found {len(truthful_files)} truthful files and {len(deceptive_files)} deceitful files.")
    print(f"Excluding {len(files_to_exclude)} recent truthful files.")

    # Process truthful files
    processed_truthful = 0
    for current_file in truthful_files:
        if current_file in files_to_exclude:
            print(f"Skipping excluded truthful file: {current_file}")
            continue

        csv_path = os.path.join(truthful_dir, current_file)

        print(f"Processing truthful file: {current_file}...")

        # Convert to JSON
        json_data = parse_csv_to_json(csv_path)

        if json_data:
            # Create output filename
            json_filename = os.path.splitext(current_file)[0] + '.json'
            json_path = os.path.join(truthful_output_dir, json_filename)

            # Write JSON file
            with open(json_path, 'w', encoding='utf-8') as json_file:
                json.dump(json_data, json_file, indent=2)

            print(f"  → Converted to {json_filename}")
            processed_truthful += 1
        else:
            print(f"  → Failed to convert {current_file}")

    # Process deceitful files
    processed_deceptive = 0
    for current_file in deceptive_files:
        csv_path = os.path.join(deceptive_dir, current_file)

        print(f"Processing deceitful file: {current_file}...")

        # Convert to JSON
        json_data = parse_csv_to_json(csv_path)

        if json_data:
            # Create output filename
            json_filename = os.path.splitext(current_file)[0] + '.json'
            json_path = os.path.join(deceptive_output_dir, json_filename)

            # Write JSON file
            with open(json_path, 'w', encoding='utf-8') as json_file:
                json.dump(json_data, json_file, indent=2)

            print(f"  → Converted to {json_filename}")
            processed_deceptive += 1
        else:
            print(f"  → Failed to convert {current_file}")

    print(f"Conversion complete! Processed {processed_truthful} truthful and {processed_deceptive} deceitful files.")


if __name__ == "__main__":
    process_directory()
    print("Conversion complete!")
