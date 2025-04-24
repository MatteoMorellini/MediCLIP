import os
from pathlib import Path
import random
import numpy as np
import nibabel as nib
from PIL import Image
import json

# useful when you have the initial dataset and you want to divide in train and test

mod = "t2w"
seg = "seg"

dataset_path = Path("../data/brats-met/vanilla-dataset").resolve()
train_path = Path("../data/brats-met/images/train/abnormal").resolve()
test_path = Path("test/abnormal")
test_json_path = Path('../data/brats-met/samples/test.json')


def normalize(data):
    if data.min() == data.max():
        return data
    return (data - data.min()) / (data.max() - data.min())


def save_slices(patient_dir, image_data, mod):
    dest_dir = patient_dir / mod
    dest_dir.mkdir(parents=True, exist_ok=True)
    for i in range(image_data.shape[2]):  # Assuming the slices are along the z-axis
        slice_data = normalize(image_data[:, :, i]) * 255
        slice_img = Image.fromarray(slice_data)
        slice_img = slice_img.convert("L")  # Convert to grayscale
        slice_img.save(dest_dir / f"{i:03d}.jpeg")


patients = list(dataset_path.glob("*BraTS*"))

random.shuffle(patients)

# Split 70/30
split_idx = int(len(patients) * 0.7)
train_patients = patients[:split_idx]

with open(test_json_path, 'w') as f:
    pass

for patient in patients:
    if patient in train_patients:
        dest_path = train_path 
        dest_patient_dir = dest_path / patient.name
    else:
        dest_path = test_path
        dest_patient_dir = dest_path / patient.name
        data = {'filename': str(dest_patient_dir), "label": 1, "label_name": "abnormal", "clsname": "abnormal"}
        with open(test_json_path, 'a') as f:
            f.write(json.dumps(data) + '\n')
            
    dest_patient_dir.mkdir(parents=True, exist_ok=True)

    # img
    img = nib.load(patient / f"{patient.name}-{mod}.nii.gz")
    img_data = img.get_fdata()
    img_data = np.rot90(img_data, k=1)
    save_slices(dest_patient_dir, img_data, mod=mod)

    # mask
    img = nib.load(patient / f"{patient.name}-{seg}.nii.gz")
    img_data = img.get_fdata()
    img_data = np.rot90(img_data, k=1) # before was 3
    save_slices(dest_patient_dir, np.int32(img_data > 0), mod=seg)
    