import random
import shutil
from pathlib import Path

# useful when all the folders of the patients are in Training

# === CONFIG ===
train_dir = Path("..data/brats-met/Training")
test_dir = Path("..data/brats-met/Testing")
split_ratio = 0.3
seed = 42

# === Ensure test directory exists
test_dir.mkdir(parents=True, exist_ok=True)

# === Get list of patient folders
patient_dirs = [p for p in train_dir.iterdir() if p.is_dir()]

# === Shuffle and split
random.seed(seed)
random.shuffle(patient_dirs)

split_idx = int(len(patient_dirs) * split_ratio)
test_patients = patient_dirs[:split_idx]

# === Move test patients to test folder
for patient in test_patients:
    shutil.move(str(patient), test_dir / patient.name)

print(f"✅ Moved {len(test_patients)} folders to {test_dir}")
print(f"Remaining in {train_dir}: {len(list(train_dir.iterdir()))}")
