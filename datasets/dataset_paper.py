import os
import random
from enum import Enum
import PIL
import torch
from torchvision import transforms
import json
from PIL import Image
import numpy as np
import nibabel as nib
from pathlib import Path
import math

from medsyn.tasks import (
    CutPastePatchBlender,
    SmoothIntensityChangeTask,
    GaussIntensityChangeTask,
    SinkDeformationTask,
    SourceDeformationTask,
    IdentityTask,
)

# todo: merge in a unique file
# todo: fix folders for the others test set


class TrainDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
        source,
        preprocess,
        k_shot=-1,
        **kwargs,
    ):
        super().__init__()
        self.args = args
        self.source = source
        self.k_shot = k_shot
        self.transform_img = preprocess
        self.data_to_iterate = self.get_image_data()
        self.augs, self.augs_pro = self.load_anomaly_syn()
        assert sum(self.augs_pro) == 1.0

    def __getitem__(self, idx):
        info = self.data_to_iterate[idx]
        image_path = os.path.join(self.source, "images", info["filename"])
        image = self.read_image(image_path)
        choice_aug = np.random.choice(
            a=[aug for aug in self.augs],
            p=[pro for pro in self.augs_pro],
            size=(1,),
            replace=False,
        )[0]
        image, mask = choice_aug(image)
        image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        image = self.transform_img(image)
        mask = torch.from_numpy(mask)
        return {
            "image": image,
            "mask": mask,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def read_image(self, path):
        image = (
            PIL.Image.open(path)
            .resize(
                (self.args.image_size, self.args.image_size),
                PIL.Image.Resampling.BILINEAR,
            )
            .convert("L")
        )
        image = np.array(image).astype(np.uint8)
        return image

    def get_image_data(self):
        data_to_iterate = []
        with open(os.path.join(self.source, "samples", "train.json"), "r") as f_r:
            for line in f_r:
                meta = json.loads(line)
                data_to_iterate.append(meta)
        if self.k_shot != -1:
            data_to_iterate = random.sample(data_to_iterate, self.k_shot)
        return data_to_iterate

    def load_anomaly_syn(self):
        tasks = []
        task_probability = []
        for task_name in self.args.anomaly_tasks.keys():
            if task_name == "CutpasteTask":
                support_images = [
                    self.read_image(
                        os.path.join(self.source, "images", data["filename"])
                    )
                    for data in self.data_to_iterate
                ]
                task = CutPastePatchBlender(support_images)
            elif task_name == "SmoothIntensityTask":
                task = SmoothIntensityChangeTask(30.0)
            elif task_name == "GaussIntensityChangeTask":
                task = GaussIntensityChangeTask()
            elif task_name == "SinkTask":
                task = SinkDeformationTask()
            elif task_name == "SourceTask":
                task = SourceDeformationTask()
            elif task_name == "IdentityTask":
                task = IdentityTask()
            else:
                raise NotImplementedError(
                    "task must in [CutpasteTask, "
                    "SmoothIntensityTask, "
                    "GaussIntensityChangeTask,"
                    "SinkTask, SourceTask, IdentityTask]"
                )

            tasks.append(task)
            task_probability.append(self.args.anomaly_tasks[task_name])
        return tasks, task_probability


class ChexpertTestDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
        source,
        preprocess,
        **kwargs,
    ):
        super().__init__()
        self.args = args
        self.source = source

        self.transform_img = preprocess
        self.data_to_iterate = self.get_image_data()

    def __getitem__(self, idx):
        info = self.data_to_iterate[
            idx
        ]  # Accesss the metadata information for the image at index idx
        image_path = os.path.join(self.source, "images", info["filename"])
        image = (
            PIL.Image.open(image_path)
            .convert("RGB")
            .resize(
                (self.args.image_size, self.args.image_size),
                PIL.Image.Resampling.BILINEAR,
            )
        )
        mask = np.zeros((self.args.image_size, self.args.image_size), dtype=float)
        image = self.transform_img(image)
        mask = torch.from_numpy(mask)

        return {
            "image": image,
            "mask": mask,
            "classname": info["clsname"],
            "is_anomaly": info["label"],
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        data_to_iterate = []
        with open(os.path.join(self.source, "samples", "test.json"), "r") as f_r:
            for line in f_r:
                meta = json.loads(line)
                data_to_iterate.append(meta)
        return data_to_iterate


class BrainMRITestDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
        source,
        preprocess,
        **kwargs,
    ):
        super().__init__()
        self.args = args
        self.source = source
        self.transform_img = preprocess
        self.data_to_iterate = self.get_image_data()

    def __getitem__(self, idx):
        info = self.data_to_iterate[idx]
        image_path = os.path.join(self.source, "images", info["filename"])

        image = (
            PIL.Image.open(image_path)
            .convert("RGB")
            .resize(
                (self.args.image_size, self.args.image_size),
                PIL.Image.Resampling.BILINEAR,
            )
        )
        
        mask = np.zeros((self.args.image_size, self.args.image_size), dtype=float)
        image = self.transform_img(image)
        mask = torch.from_numpy(mask)

        return {
            "image": image,
            "mask": mask,
            "classname": info["clsname"],
            "is_anomaly": info["label"],
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        data_to_iterate = []
        with open(os.path.join(self.source, "samples", "test.json"), "r") as f_r:
            for line in f_r:
                meta = json.loads(line)
                data_to_iterate.append(meta)
        return data_to_iterate


class BusiTestDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
        source,
        preprocess,
        **kwargs,
    ):
        super().__init__()
        self.args = args
        self.source = source
        self.transform_img = preprocess
        self.data_to_iterate = self.get_image_data()

    def __getitem__(self, idx):
        info = self.data_to_iterate[idx]
        image_path = os.path.join(self.source, "images", info["filename"])
        image = (
            PIL.Image.open(image_path)
            .convert("RGB")
            .resize(
                (self.args.image_size, self.args.image_size),
                PIL.Image.Resampling.BILINEAR,
            )
        )

        if info.get("mask", None):
            mask = os.path.join(self.source, "images", info["mask"])
            mask = (
                PIL.Image.open(mask)
                .convert("L")
                .resize(
                    (self.args.image_size, self.args.image_size),
                    PIL.Image.Resampling.NEAREST,
                )
            )
            mask = np.array(mask).astype(float) / 255.0
            mask[mask != 0.0] = 1.0
        else:
            mask = np.zeros((self.args.image_size, self.args.image_size), dtype=float)

        image = self.transform_img(image)
        mask = torch.from_numpy(mask)

        return {
            "image": image,
            "mask": mask,
            "classname": info["clsname"],
            "is_anomaly": info["label"],
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        data_to_iterate = []
        with open(os.path.join(self.source, "samples", "test.json"), "r") as f_r:
            for line in f_r:
                meta = json.loads(line)
                data_to_iterate.append(meta)
        return data_to_iterate


class BratsMetTestDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
        source,
        preprocess,
        slice_idx,
        **kwargs,
    ):
        super().__init__()
        self.args = args
        self.source = source
        self.transform_img = preprocess
        self.slice_idx = slice_idx
        self.data_to_iterate = self.get_image_data()

    def __getitem__(self, idx):
        info = self.data_to_iterate[idx] 
        # eg: {'filename': 'Testing/BraTS-MET-00415-000', 'label': 1, 'label_name': 'abnormal', 'clsname': 'abnormal', 'slice': 150}

        if self.slice_idx == -1:
            image_path = os.path.join(self.source, info["filename"], 't2w', str(info['slice']).zfill(3)+'.jpeg')  
            mask_path = os.path.join(self.source, info["filename"], 'seg', str(info['slice']).zfill(3)+'.jpeg')
        else:
            image_path = os.path.join(self.source, info["filename"], 't2w', str(self.slice_idx).zfill(3)+'.jpeg')    
            mask_path = os.path.join(self.source, info["filename"], 'seg', str(self.slice_idx).zfill(3)+'.jpeg')

        image = (
            PIL.Image.open(image_path)
            .convert("RGB")
            .resize(
                (self.args.image_size, self.args.image_size),
                PIL.Image.Resampling.BILINEAR,
            )
        )
        if os.path.exists(mask_path):
            mask = (
                PIL.Image.open(mask_path)
                .convert("L")
                .resize(
                    (self.args.image_size, self.args.image_size),
                    PIL.Image.Resampling.NEAREST,
                )
            )
            mask = np.array(mask).astype(float) / 255.0
            mask[mask != 0.0] = 1.0
        else:
            mask = np.zeros((self.args.image_size, self.args.image_size), dtype=float)

        image = self.transform_img(image)
        mask = torch.from_numpy(mask)
        num_ones = (mask == 1).sum().item()
        return {
            "image": image,
            "mask": mask,
            "classname": 'abnormal' if num_ones > 10 else 'normal',
            "is_anomaly": 1 if num_ones > 10 else 0,
            "image_path": str(image_path),
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        data_to_iterate = []

        with open(os.path.join(self.source, 'Testing', "test.json"), 'r') as f:
            data = [json.loads(line) for line in f]

        for row in data:
            if self.slice_idx == -1:
                for slice in range(120, 155):
                #for slice in range(0, 155, int(self.args.distance_per_slice)*6):
                    row['slice'] = slice
                    data_to_iterate.append(row.copy())
            else:
                data_to_iterate.append(row)
        return data_to_iterate
        