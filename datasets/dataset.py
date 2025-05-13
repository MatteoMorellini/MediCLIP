import torch.utils.data as data
import json
import random
from PIL import Image
import numpy as np
import torch
import os
from pathlib import Path
from collections import defaultdict
from medsyn.tasks import (
    CutPastePatchBlender,
    SmoothIntensityChangeTask,
    GaussIntensityChangeTask,
    SinkDeformationTask,
    SourceDeformationTask,
    IdentityTask,
)

"""def filter_data(meta_info):  # Filter out anomaly slices for few-shot learning
    filtered_meta_info = {
        split: {
            category: [item for item in items if item.get("anomaly") == 0]
            for category, items in categories.items()
        }
        for split, categories in meta_info.items()
    }
    return filtered_meta_info

def filter_data(data_to_iterate):
    data_to_iterate = [slice for slice in data_to_iterate if slice.get('anomaly')==0]"""

def sample_per_slice(data_to_iterate, k_shot):
    all_samples = []
    for key in data_to_iterate.keys():
        values = data_to_iterate[key]
        if len(values) < k_shot:
            raise ValueError(f"Not enough elements to sample {k_shot} from key {key}")
        #print(random.sample(values, k_shot))
        all_samples.extend(random.sample(values, k_shot))
    return all_samples

class TrainDataset(data.Dataset):
    def __init__(
        self,
        args,
        root,
        transform,
        target_transform,
        mode="test",
        save_dir="fewshot",
        k_shot=0,
    ):
        self.args = args
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.__image_path_key = (
            "img_path" if args.train_dataset == "brats-met" else "filename"
        )

        meta_info = self._get_meta_info(mode, k_shot)

        self.data_all = self._load_slices(meta_info, mode, save_dir, k_shot)

        self.augs, self.augs_pro = self.load_anomaly_syn()
        assert sum(self.augs_pro) == 1.0
        self.length = len(self.data_all)


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_path = self.data_all[
            index
        ]  # data_all is a list of dictionaries for each sample
        if self.args.train_dataset != 'brats-met':
            img_path = os.path.join(self.root, "images", img_path)
        image = self.read_image(img_path)

        # During traning we apply the augmentation to the image, hence we don't care about the original mask
        # Augmentation applied EACH time the item is retrieved
        choice_aug = np.random.choice(
            a=[aug for aug in self.augs],
            p=[pro for pro in self.augs_pro],
            size=(1,),
            replace=False,
        )
        choice_aug = choice_aug[0]
        image, mask = choice_aug(image)
        
        image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        image = self.transform(image)

        # ? normalization
        
        mask = torch.from_numpy(mask)

        return {"image": image, "mask": mask}

    def _get_meta_info(self, mode, k_shot):
        if self.args.train_dataset == "brats-met":
            # ? how to intend random sample for patients:
            # option 1 - shuffle the dictionary and take the first K healthy slice
            # option 2 - take all ith healthy slices and select K out of them
            # I prefer option 2 since it's easier 
            meta_info = json.load(open(f"{self.root}/Training/meta.json", "r"))
            data_to_iterate = defaultdict(list)
            # iterate through patients
            for key, value in meta_info['train']['brain'].items():
                # iterate through slices of a patient
                for id, slice in value.items():
                    if int(id) % self.args.distance_per_slice == 0 and slice.get('anomaly')==0:
                        data_to_iterate[id].append(slice)
                #data_to_iterate.append(value[str(slice_idx).zfill(3)])
            if k_shot != -1: 
                data_to_iterate = sample_per_slice(data_to_iterate, k_shot)
            # Filter out anomaly slices for few-shot learning
            #meta_info = filter_data(data_to_iterate)  
        else:
            data_to_iterate = []
            with open(os.path.join(self.root, "samples", "train.json"), "r") as f_r:
                for line in f_r:
                    meta = json.loads(line)
                    data_to_iterate.append(meta)
            if k_shot != -1:
                data_to_iterate = random.sample(data_to_iterate, k_shot)

        return data_to_iterate

    def _get_cls_names(self, meta_info, mode, save_dir):
        if mode == "train":
            cls_names = ["brain"]
            Path(save_dir).mkdir(exist_ok=True)
            save_dir = os.path.join(save_dir, "k_shot.txt")
        else:
            cls_names = list(
                meta_info.keys()
            )  # During testing all available classes from the metadata are used
            # This allows the model to be evaluated across all anatomical regions in the dataset
        return cls_names, save_dir

    def _load_slices(self, meta_info, mode, save_dir, k_shot):
        self.cls_names, save_dir = self._get_cls_names(meta_info, mode, save_dir)
        data_all = []
        for cls_name in self.cls_names:
            if mode == "train":
                # Clean the file before writing
                with open(save_dir, "w"): pass
                for image in meta_info:
                    # image_path = {self.__image_path_key: image[self.__image_path_key]}
                    data_all.append(image[self.__image_path_key])
                    # Write the image path of the selected samples to a file
                    with open(save_dir, "a") as f:  
                        f.write(image[self.__image_path_key] + "\n")
                    # This creates a file with the paths of the selected samples, useful for reproducibility
            else:
                # TODO: data_all is now a list of img_paths and not dictionaries
                data_all.extend(
                    meta_info[cls_name]
                )  # for testing, all samples are used
        return data_all

    def read_image(self, path):
        image = (
            Image.open(path)
            .resize(
                (self.args.image_size, self.args.image_size), Image.Resampling.BILINEAR
            )
            .convert("L")
        )
        image = np.array(image).astype(np.uint8)
        return image

    def load_anomaly_syn(self):
        tasks = []
        task_probability = []
        for task_name in self.args.anomaly_tasks.keys():
            if task_name == "CutpasteTask":
                support_images = [
                    self.read_image(os.path.join(self.root, "images", data))
                    for data in self.data_all
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


def filter_healthy_patients(meta_info, k_shot, max_anomaly_ratio=0.1):
    """
    Filters meta_info to include only k_shot patients with a low proportion of anomaly slices.

    A patient is accepted if their anomaly slices are below `max_anomaly_ratio`.
    """
    patient_slices = {}  # Store slices per patient

    # Step 1: Collect all patient slices
    for item in meta_info["brain"]:
        patient_id = Path(item["img_path"]).parent
        if patient_id not in patient_slices:
            patient_slices[patient_id] = []
        patient_slices[patient_id].append(item)

    # Step 2: Filter patients based on anomaly ratio
    selected_patients = []
    for patient, slices in patient_slices.items():
        total_slices = len(slices)
        anomaly_slices = sum(s["anomaly"] for s in slices)
        anomaly_ratio = anomaly_slices / total_slices

        if anomaly_ratio <= max_anomaly_ratio:
            selected_patients.append(patient)

    # Step 3: Ensure we have enough patients
    if len(selected_patients) < k_shot:
        raise ValueError(
            f"Only {len(selected_patients)} patients meet the criteria, but k_shot={k_shot} was requested."
        )

    # Step 4: Select k_shot patients & filter meta_info
    selected_patients = set(selected_patients[:k_shot])
    filtered_meta = {
        "brain": [s for p in selected_patients for s in patient_slices[p]]
    }  # List of metadata of slices, each belongs to a healthy patient

    return filtered_meta


class TrainDatasetFewShot(
    TrainDataset
):  # Few-shot on patients and not single slices as in TrainDataset
    def __init__(
        self,
        args,
        root,
        transform,
        target_transform,
        mode="test",
        save_dir="fewshot",
        k_shot=0,
    ):
        super().__init__(
            args, root, transform, target_transform, mode, save_dir, k_shot
        )

    def _load_slices(self, meta_info, mode, save_dir):
        self.cls_names, save_dir = self._get_cls_names(meta_info, mode, save_dir)
        data_all = []
        for cls_name in self.cls_names:
            if mode == "train":
                data_tmp = meta_info[cls_name]
                # Random selection already done in filter_healthy_patients
                # Take all slices from selected patients and put them in the same list
                for item in data_tmp:
                    data_all.append(item["img_path"])
                    with open(save_dir, "a") as f:
                        f.write(item["img_path"] + "\n")
            else:
                data_all.extend(meta_info[cls_name])
        return data_all

    def _get_meta_info(self, mode, k_shot):
        meta_info = json.load(open(f"{self.root}/meta.json", "r"))
        meta_info = meta_info[mode]
        if k_shot > 0:  # Filter out anomaly patients for few-shot learning
            meta_info = filter_healthy_patients(meta_info, k_shot)

        return meta_info
