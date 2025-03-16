import torch.utils.data as data
import json
import random
from PIL import Image
import numpy as np
import torch
import os
from pathlib import Path
from medsyn.tasks import CutPastePatchBlender,\
                        SmoothIntensityChangeTask,\
                        GaussIntensityChangeTask,\
                        SinkDeformationTask,\
                        SourceDeformationTask,\
                        IdentityTask
from torchvision.transforms import ToPILImage, ToTensor
 
def generate_class_info(dataset_name):
    class_name_map_class_id = {}
    if dataset_name == 'mvtec':
        obj_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                    'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
    elif dataset_name == 'visa':
        obj_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                    'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    elif dataset_name == 'mpdd':
        obj_list = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']
    elif dataset_name == 'btad':
        obj_list = ['01', '02', '03']
    elif dataset_name == 'DAGM_KaggleUpload':
        obj_list = ['Class1','Class2','Class3','Class4','Class5','Class6','Class7','Class8','Class9','Class10']
    elif dataset_name == 'SDD':
        obj_list = ['electrical commutators']
    elif dataset_name == 'DTD':
        obj_list = ['Woven_001', 'Woven_127', 'Woven_104', 'Stratified_154', 'Blotchy_099', 'Woven_068', 'Woven_125', 'Marbled_078', 'Perforated_037', 'Mesh_114', 'Fibrous_183', 'Matted_069']
    elif dataset_name == 'colon':
        obj_list = ['colon']
    elif dataset_name == 'ISBI':
        obj_list = ['skin']
    elif dataset_name == 'Chest':
        obj_list = ['chest']
    elif dataset_name == 'thyroid':
        obj_list = ['thyroid']
    elif dataset_name == 'brats':
        obj_list = ['brain']
    for k, index in zip(obj_list, range(len(obj_list))):
        class_name_map_class_id[k] = index

    return obj_list, class_name_map_class_id

def filter_data(meta_info): # Filter out anomaly slices for few-shot learning
    filtered_meta_info = {
        split: {
            category: [item for item in items if item.get("anomaly") == 0]
            for category, items in categories.items()
        }
        for split, categories in meta_info.items()
    }
    return filtered_meta_info

class TrainDataset(data.Dataset):
    def __init__(self, args, root, transform, target_transform, dataset_name, mode='test', save_dir='fewshot', k_shot=0):
        self.args = args
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        
        meta_info = self._get_meta_info(mode, k_shot)

        self.data_all = self._load_slices(meta_info, mode, save_dir, k_shot)
        
        self.augs, self.augs_pro = self.load_anomaly_syn()
        assert sum(self.augs_pro)==1.0
        self.length = len(self.data_all)
        self.obj_list, self.class_name_map_class_id = generate_class_info(dataset_name)
 
    def __len__(self):
        return self.length
 
    def __getitem__(self, index):
        data = self.data_all[index] # data_all is a list of dictionaries for each sample
        img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
                                                              data['specie_name'], data['anomaly']

        image = self.read_image(img_path)
        

        # * During traning we apply the augmentation to the image, hence we don't care about the original mask
        
        choice_aug = np.random.choice(a=[aug for aug in self.augs],
                                         p = [pro for pro in self.augs_pro],
                                         size=(1,), replace=False)
        choice_aug = choice_aug[0]
        image, mask = choice_aug(image)
        
        image = Image.fromarray(image.astype(np.uint8)).convert('RGB')
        image = self.transform(image)
       
        mask = torch.from_numpy(mask)
        
        return {'image': image, 'mask': mask, 'cls_name': cls_name, 'anomaly': anomaly,
                'img_path': img_path, "cls_id": self.class_name_map_class_id[cls_name]}  
    
    def _get_meta_info(self, mode, k_shot):
        meta_info = json.load(open(f'{self.root}/meta.json', 'r'))

        if k_shot > 0: # Filter out anomaly slices for few-shot learning
            meta_info = filter_data(meta_info) # Ensure that the model focuses on learning the characteristics of healthy slices
        
        return meta_info[mode]

    
    def _get_cls_names(self, meta_info, mode, save_dir):
        if mode == 'train':
            cls_names = ['brain']
            Path(save_dir).mkdir(exist_ok=True)
            save_dir = os.path.join(save_dir, 'k_shot.txt')
        else:
            cls_names = list(meta_info.keys()) # During testing all available classes from the metadata are used
            # This allows the model to be evaluated across all anatomical regions in the dataset
        return cls_names, save_dir
    
    def _load_slices(self, meta_info, mode, save_dir, k_shot): 
        self.cls_names, save_dir = self._get_cls_names(meta_info, mode, save_dir)
        data_all = []
        for cls_name in self.cls_names:
            if mode == 'train':
                data_tmp = meta_info[cls_name] #  Retrieve all data for the current class
                indices = torch.randint(0, len(data_tmp), (k_shot,)) # Randomly select k_shot samples
                for i in range(len(indices)): 
                    data_all.append(data_tmp[indices[i]])
                    with open(save_dir, "a") as f: # Write the image path of the selected samples to a file
                        f.write(data_tmp[indices[i]]['img_path'] + '\n') 
                        # This creates a file with the paths of the selected samples, useful for reproducibility
            else:
                data_all.extend(meta_info[cls_name]) # for testing, all samples are used
        return data_all
    
    def read_image(self,path):
        image = Image.open(path).resize((self.args.image_size,self.args.image_size),
                                            Image.Resampling.BILINEAR).convert("L")
        image = np.array(image).astype(np.uint8)
        return image
    
    def load_anomaly_syn(self):
        tasks = []
        task_probability = []
        for task_name in self.args.anomaly_tasks.keys():
            if task_name =='CutpasteTask':
                support_images = [self.read_image(data['img_path']) for data in self.data_all]
                task = CutPastePatchBlender(support_images)
            elif task_name == 'SmoothIntensityTask':
                task = SmoothIntensityChangeTask(30.0)
            elif task_name == 'GaussIntensityChangeTask':
                task = GaussIntensityChangeTask()
            elif task_name == 'SinkTask':
                task = SinkDeformationTask()
            elif task_name == 'SourceTask':
                task = SourceDeformationTask()
            elif task_name == 'IdentityTask':
                task = IdentityTask()
            else:
                raise NotImplementedError("task must in [CutpasteTask, "
                                          "SmoothIntensityTask, "
                                          "GaussIntensityChangeTask,"
                                          "SinkTask, SourceTask, IdentityTask]")

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
    for item in meta_info['brain']:
        patient_id = Path(item['img_path']).parent
        if patient_id not in patient_slices:
            patient_slices[patient_id] = []
        patient_slices[patient_id].append(item)
 
    # Step 2: Filter patients based on anomaly ratio
    selected_patients = []
    for patient, slices in patient_slices.items():
        total_slices = len(slices)
        anomaly_slices = sum(s['anomaly'] for s in slices)
        anomaly_ratio = anomaly_slices / total_slices
 
        if anomaly_ratio <= max_anomaly_ratio:
            selected_patients.append(patient)
 
    # Step 3: Ensure we have enough patients
    if len(selected_patients) < k_shot:
        raise ValueError(f"Only {len(selected_patients)} patients meet the criteria, but k_shot={k_shot} was requested.")
 
    # Step 4: Select k_shot patients & filter meta_info
    selected_patients = set(selected_patients[:k_shot])
    filtered_meta = {'brain': [s for p in selected_patients for s in patient_slices[p]]} # List of metadata of slices, each belongs to a healthy patient
 
    return filtered_meta
 
class TrainDatasetFewShot(TrainDataset): # Few-shot on patients and not single slices as in TrainDataset
    def __init__(self, args, root, transform, target_transform, dataset_name, mode='test', save_dir='fewshot', k_shot=0):

        super().__init__(args, root, transform, target_transform, dataset_name, mode, save_dir, k_shot)

    def _load_slices(self, meta_info, mode, save_dir, k_shot): 
        self.cls_names, save_dir = self._get_cls_names(meta_info, mode, save_dir)
        data_all = []
        for cls_name in self.cls_names:
            if mode == 'train':
                data_tmp = meta_info[cls_name] 
                # Random selection already done in filter_healthy_patients
                # Take all slices from selected patients and put them in the same list
                for item in data_tmp:
                    data_all.append(item)
                    with open(save_dir, "a") as f:
                        f.write(item['img_path'] + '\n')
            else:
                data_all.extend(meta_info[cls_name])
        return data_all
    
    def _get_meta_info(self, mode, k_shot):
        meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
        meta_info = meta_info[mode]
        if k_shot > 0: # Filter out anomaly patients for few-shot learning
            meta_info = filter_healthy_patients(meta_info, k_shot) 
        
        return meta_info
