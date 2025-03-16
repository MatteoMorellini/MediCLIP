from pathlib import Path
import nibabel as nib
import numpy as np
import shutil

target_dir = Path('../data/brats-met/images/train/abnormal').resolve()
source_dir = Path('../data/brats-met/fmri-reg').resolve()

mod = 't2w'
seg = 'seg'

if not target_dir.exists():
    target_dir.mkdir()

"""
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
    """

for patient in source_dir.glob('*BraTS*'):
    patient_stem = patient.stem
    target_patient_dir = target_dir / patient_stem

    target_patient_dir.mkdir(parents=True, exist_ok=True)
 
    # img
    img = nib.load(patient / f"{patient.name}-{mod}.nii.gz")
    img_data = img.get_fdata()
    img_data = np.rot90(img_data, k=3)
    nii_image = nib.Nifti1Image(img_data, img.affine)
    nib.save(nii_image, target_patient_dir / f'{patient_stem}-{mod}.nii.gz')
 
    # mask
    img = nib.load(patient / f"{patient.name}-{seg}.nii.gz")
    img_data = img.get_fdata()
    img_data = np.rot90(img_data, k=3)
    nii_image = nib.Nifti1Image(img_data, img.affine)
    nib.save(nii_image, target_patient_dir / f'{patient_stem}-{seg}.nii.gz')