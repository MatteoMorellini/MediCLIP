from pathlib import Path
import json
from PIL import Image
from collections import defaultdict

# Generates a JSON metadata file (meta.json) for all patients in the 
# Training directory of the BraTS-MET dataset.

meta = {"train": {"brain": {}}, "test": {"brain": {}}}

meta_train = defaultdict(dict)

brats_train_dir = Path("../data/brats-met/Training/")

for patient in brats_train_dir.iterdir():

    t2w = patient / "t2w"
    seg = patient / "seg"

    if not (patient.is_dir() and seg.is_dir() and t2w.is_dir()):
        continue
    for slice, mask in zip(t2w.iterdir(), seg.iterdir()):
        mask_image = Image.open(mask).convert("L")
        anomaly = 0 if all(pixel == 0 for pixel in mask_image.getdata()) else 1
        image = {
            "img_path": str(slice.resolve()),
            "mask_path": str(mask.resolve()),
            "cls_name": "brain",
            "specie_name": None,
            "anomaly": anomaly,
        }
        meta_train[patient.name][mask.name.split('.')[0]] = image

meta["train"]["brain"] = meta_train

with open("../data/brats-met/Training/meta.json", "w") as f:
    json.dump(meta, f, indent=4)
