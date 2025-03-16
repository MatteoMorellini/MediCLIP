from pathlib import Path
import nibabel as nib
import numpy as np
import shutil

nii_dir = Path('../data/brats-met/images/test/abnormal').resolve()
mask_dir = Path('../data/brats-met/fmri').resolve()

if not mask_dir.exists():
    mask_dir.mkdir()

for patient in nii_dir.glob('*.nii.gz'):
    stem =  mask_dir / str(patient.stem).split('.')[0]
    stem.mkdir(exist_ok=True)

    file = stem / f"{patient.stem}.gz"
    shutil.move(str(patient), str(file))

    mri = nib.load(file.resolve())
    z_axis = mri.shape[2]
    mask = np.random.randint(0,2, size=(224, 224, z_axis), dtype=np.uint8)
    nii_image = nib.Nifti1Image(mask, mri.affine)
    nib.save(nii_image, stem / 'mask.nii.gz')
    #file.touch()
    