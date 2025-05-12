from pathlib import Path
import json
import os
import shutil

meta_path = Path("data/brats-met/meta.json")

with open(meta_path) as f:
    data = json.load(f)

training_patients = set()
training_data = data['train']['brain']

for sample in training_data:
    patient = sample['img_path'].split('/')[6]
    if patient not in training_patients:
        training_patients.add(patient)

testing_patients = set()
testing_data = data['test']['brain']

for sample in testing_data:
    patient = sample['img_path'].split('/')[6]
    if patient not in testing_data:
        testing_patients.add(patient)

path = Path('data/brats-met')
source_dir = path / "Training"
destination_dir = path / "Testing"

os.makedirs(destination_dir, exist_ok = True)

for folder in testing_patients:
    source_path = source_dir / folder 
    destination_path = destination_dir / folder 

    if os.path.isdir(source_path):
        shutil.move(source_path, destination_path)