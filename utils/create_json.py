import os
import shutil
import json

# Define source and destination directories
source_root = "./data/brats-met/images/test/abnormal"
destination_folder = "test/abnormal"
os.makedirs(destination_folder, exist_ok=True)

# Prepare JSON data
json_data = []

# Iterate over subfolders in source_root
for subfolder in sorted(os.listdir(source_root)):
    subfolder_path = os.path.join(source_root, subfolder)
    if os.path.isdir(subfolder_path):
        files = sorted(os.listdir(subfolder_path))
        if files:
            first_file = files[0]  # Get the first file in sorted order
            source_file = os.path.join(subfolder_path, first_file)
            destination_file = os.path.join(destination_folder, first_file)
            
            # Copy the file to the new location
            shutil.copy2(source_file, destination_file)
            
            # Append file details to JSON list
            json_data.append({
                "filename": destination_file,
                "label": 1,
                "label_name": "abnormal",
                "clsname": "abnormal"
            })

# Write JSON data to a file
json_path = os.path.join(destination_folder, "metadata.json")
with open(json_path, "w") as json_file:
    for entry in json_data:
        json_file.write(json.dumps(entry) + "\n")

print(f"Processed {len(json_data)} files. JSON metadata saved at {json_path}.")
