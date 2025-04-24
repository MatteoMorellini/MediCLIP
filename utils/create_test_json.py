import os
import json

# useful when you all the folders in Testing but missing the meta.json

def create_label_file_from_folders(root='.', output_file='meta.json'):
    with open(output_file, 'w') as f:
        for folder in os.listdir(root):
            folder_path = os.path.join(root, folder)
            if os.path.isdir(folder_path):
                # Customize this path and logic if needed
                entry = {
                    "filename": f"Testing/{folder}",
                    "label": 1,
                    "label_name": "abnormal",
                    "clsname": "abnormal"
                }
                f.write(json.dumps(entry) + '\n')
    print(f"Labels written to {output_file}")

create_label_file_from_folders()