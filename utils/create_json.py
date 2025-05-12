import os
import json

# Set the path to the folder containing the folders
base_path = "../data/brats-met/Testing"
output_file = "../data/brats-met/Testing/test.json"

# Get list of directories in base_path
entries = [
    {"filename": os.path.join("Testing", folder), 
     "label": 1, 
     "label_name": "abnormal", 
     "clsname": "abnormal"}
    for folder in os.listdir(base_path)
    if os.path.isdir(os.path.join(base_path, folder))
]

# Write to file as newline-delimited JSON
with open(output_file, "w") as f:
    for entry in entries:
        json.dump(entry, f)
        f.write("\n")
