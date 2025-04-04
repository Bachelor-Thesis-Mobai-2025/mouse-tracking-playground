import os
import json

def truncate_trajectory_after_last_click(input_folder="data_new", output_folder="data_new_truncated"):
    """
    Recursively walks through `input_folder` and its subfolders,
    finds all JSON files, trims the trajectory data after the last click=1,
    and also removes initial points where both x and y are 0,
    then saves results under `output_folder`, maintaining subfolder structure.
    """

    for root, dirs, files in os.walk(input_folder):
        # Figure out sub-path relative to input_folder
        relative_path = os.path.relpath(root, input_folder)
        # Construct the equivalent output path
        output_path = os.path.join(output_folder, relative_path)
        # Ensure the output subfolder exists
        os.makedirs(output_path, exist_ok=True)

        for filename in files:
            if filename.lower().endswith(".json"):
                input_file = os.path.join(root, filename)
                output_file = os.path.join(
                    output_path,
                    os.path.splitext(filename)[0] + "_trunc.json"
                )

                with open(input_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Sanity-check that "trajectory" is present and is a list
                if "trajectory" in data and isinstance(data["trajectory"], list):
                    last_click_index = None
                    # Find the last occurrence of click=1
                    for i, entry in enumerate(data["trajectory"]):
                        if entry.get("click") == 1:
                            last_click_index = i

                    if last_click_index is not None:
                        # Keep everything up to and including that index
                        data["trajectory"] = data["trajectory"][:last_click_index + 1]
                    
                    # Truncate initial points where both x and y are 0
                    first_non_zero_index = None
                    for i, entry in enumerate(data["trajectory"]):
                        if entry.get("x", 0) != 0 or entry.get("y", 0) != 0:
                            first_non_zero_index = i
                            break
                    
                    if first_non_zero_index is not None:
                        # Only keep points from the first non-zero position onwards
                        data["trajectory"] = data["trajectory"][first_non_zero_index:]

                # Write out the truncated file
                with open(output_file, "w", encoding="utf-8") as out_f:
                    json.dump(data, out_f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    folder_name = "data_new_doubled"
    truncate_trajectory_after_last_click(folder_name, f"{folder_name}_truncated_final")