import os
from pathlib import Path

import numpy as np
import nibabel as nib
from PIL import Image

mod = "t2w"
seg = "seg"

dataset_path = Path("../data/brats-met/fmri-reg").resolve()
dest_path = Path("../data/brats-met/images/train/abnormal").resolve()


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


patients = dataset_path.glob("*BraTS*")
for patient in patients:
    dest_patient_dir = dest_path / patient.name
    dest_patient_dir.mkdir(parents=True, exist_ok=True)

    # img
    img = nib.load(patient / f"{patient.name}-{mod}.nii.gz")
    img_data = img.get_fdata()
    img_data = np.rot90(img_data, k=3)
    save_slices(dest_patient_dir, img_data, mod=mod)

    # mask
    img = nib.load(patient / f"{patient.name}-{seg}.nii.gz")
    img_data = img.get_fdata()
    img_data = np.rot90(img_data, k=3)
    save_slices(dest_patient_dir, np.int32(img_data > 0), mod=seg)
