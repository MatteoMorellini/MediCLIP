import numpy as np
import open_clip
import torch
import yaml
from easydict import EasyDict
from models.Necker import Necker
from models.Adapter import Adapter
import math
import argparse
import warnings
from utils.misc_helper import *
from datasets.dataset_paper import (
    ChexpertTestDataset,
    BusiTestDataset,
    BrainMRITestDataset,
    BratsMetTestDataset,
)
from torch.utils.data import DataLoader
from models.MapMaker import MapMaker
import pprint
import torch.nn.functional as F
from pytorch_grad_cam.utils.image import show_cam_on_image
from tqdm import tqdm
from PIL import Image
import cv2
import nibabel as nib
import PIL
import matplotlib.pyplot as plt
from pathlib import Path


warnings.filterwarnings("ignore")


def normalization(segmentations, image_size, avgpool_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segmentations = torch.tensor(segmentations[:, None, ...]).to(
        device
    )  # N x 1 x H x W
    segmentations = F.interpolate(
        segmentations, (image_size, image_size), mode="bilinear", align_corners=True
    )

    segmentations_ = (
        F.avg_pool2d(segmentations, (avgpool_size, avgpool_size), stride=1)
        .cpu()
        .numpy()
    )

    min_scores = segmentations_.reshape(-1).min(axis=-1).reshape(1)

    max_scores = segmentations_.reshape(-1).max(axis=-1).reshape(1)

    segmentations = segmentations.squeeze(1).cpu().numpy()
    segmentations = (segmentations - min_scores) / (max_scores - min_scores)
    segmentations = np.clip(segmentations, a_min=0, a_max=1)

    segmentations = cv2.GaussianBlur(segmentations, (5, 5), 0)
    return segmentations


@torch.no_grad()
def make_vision_takens_info(model, model_cfg, layers_out):
    img = torch.ones(
        (
            1,
            3,
            model_cfg["vision_cfg"]["image_size"],
            model_cfg["vision_cfg"]["image_size"],
        )
    ).to(model.device)

    img_feature, tokens = model.encode_image(img, layers_out)

    # img_feature - represents the global image embedding, usually as [CLS] token so shape [Batch, Feature Dim] = [1, 768]
    # to match text embedding dim, perhaps through a linear layer. Used for image-text similarity
    # tokens - list of tensors, each tensor represents the tokenized image features at different layers
    # tokens[0].shape - [1, 257, 1024]
    # tokens[1].shape - [1, 257, 1024]
    # tokens[2].shape - [1, 257, 1024]

    if len(tokens[0].shape) == 3:
        model.token_size = [
            int(math.sqrt(token.shape[1] - 1)) for token in tokens
        ]  # how many tokens in each dimension in each layer
        model.token_c = [
            token.shape[-1] for token in tokens
        ]  # number of channels in each token
    else:
        model.token_size = [token.shape[2] for token in tokens]
        model.token_c = [token.shape[1] for token in tokens]

    model.embed_dim = model_cfg["embed_dim"]
    print(
        "model token size is {}".format(model.token_size),
        " model token dim is {}".format(model.token_c),
    )


@torch.no_grad()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config_path) as f:
        args.config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    set_seed(seed=args.config.random_seed)

    model, preprocess, model_cfg = open_clip.create_model_and_transforms(
        args.config.model_name, args.config.image_size, device=device
    )

    for param in model.parameters():
        param.requires_grad_(False)

    args.config.model_cfg = model_cfg

    make_vision_takens_info(model, args.config.model_cfg, args.config.layers_out)

    necker = Necker(clip_model=model).to(model.device)
    adapter = Adapter(clip_model=model, target=args.config.model_cfg["embed_dim"]).to(
        model.device
    )

    if args.config.prompt_maker == "coop":
        from models.CoOp import PromptMaker
    else:
        raise NotImplementedError("type of prompt must in ['coop']")

    prompt_maker = PromptMaker(
        prompts=args.config.prompts,
        clip_model=model,
        n_ctx=args.config.n_learnable_token,
        CSC=args.config.CSC,
        class_token_position=args.config.class_token_positions,
    ).to(model.device)

    map_maker = MapMaker(image_size=args.config.image_size).to(model.device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ? how is zero-shot feasible without training the adapters?

    checkpoints = torch.load(
        args.checkpoint_path, map_location=str(device)
    )  # Pass device as string
    adapter.load_state_dict(checkpoints["adapter_state_dict"])
    prompt_maker.prompt_learner.load_state_dict(checkpoints["prompt_state_dict"])
    prompt_maker.prompt_learner.eval()
    adapter.eval()

    for test_dataset_name in args.config.test_datasets:
        if test_dataset_name == "chexpert":
            test_dataset = ChexpertTestDataset(
                args=args.config,
                source=os.path.join(args.config.data_root, test_dataset_name),
                preprocess=preprocess,
            )

        elif test_dataset_name == "brainmri":
            test_dataset = BrainMRITestDataset(
                args=args.config,
                source=os.path.join(args.config.data_root, test_dataset_name),
                preprocess=preprocess,
            )
        elif test_dataset_name == "busi":
            test_dataset = BusiTestDataset(
                args=args.config,
                source=os.path.join(args.config.data_root, test_dataset_name),
                preprocess=preprocess,
            )
        elif test_dataset_name == "brats-met":
            test_dataset = BratsMetTestDataset(
                args=args.config,
                source=os.path.join(args.config.data_root, test_dataset_name),
                preprocess=preprocess,
                slice_idx = args.slice_idx,
            )

        else:
            raise NotImplementedError(
                "dataset must in ['chexpert','busi','brainmri', 'brats-met]"
            )
        
        test_dataloader = DataLoader(
            test_dataset, batch_size=args.config.batch_size, num_workers=1
        )
        # iterates over batches, which are made by the value returned by getitem
        # e.g. getitem returns image,label -> dataloader returns batches of images and labels
        # since here it returns a dict, dataloader returns a dictionary where each
        # entry is a batch of the corresponding key

        results = validate(
            args,
            test_dataset_name,
            test_dataloader,
            model,
            necker,
            adapter,
            prompt_maker,
            map_maker,
        )

        if test_dataset_name not in {'busi', 'brats-met'}:
            print(
                "{}, image auroc: {:.4f}".format(
                    test_dataset_name, results["image-auroc"]
                )
            )
        else:
            print(
                "{}, image auroc: {:.4f}, pixel_auroc: {:.4f}".format(
                    test_dataset_name, results["image-auroc"], results["pixel-auroc"]
                )
            )


def validate(
    args,
    dataset_name,
    test_dataloader,
    clip_model,
    necker,
    adapter,
    prompt_maker,
    map_maker,
):
    image_preds = []
    image_gts = []  # ground-truth labels for whole image

    pixel_preds = []
    pixel_gts = []

    image_paths = []

    # iterate over batches of test images to predict
    for i, input in enumerate(test_dataloader):  
        images = input["image"].to(clip_model.device)
        # images.shape = [8, 3, 224, 224]
        image_paths.extend(input["image_path"])
        _, image_tokens = clip_model.encode_image(
            images, out_layers=args.config.layers_out
        )

        image_features = necker(image_tokens)
        vision_adapter_features = adapter(image_features)
        prompt_adapter_features = prompt_maker(vision_adapter_features)
        # compare vision and prompt features to get anomaly heatmaps
        anomaly_map = map_maker(vision_adapter_features, prompt_adapter_features)

        B, _, H, W = anomaly_map.shape  # _ is the number of channels: Sn and Sa
        anomaly_map = anomaly_map[:, 1, :, :]  # keep only the anomaly map

        pixel_preds.append(anomaly_map)  # append since we have a whole matrix
        anomaly_score, _ = torch.max(anomaly_map.view((B, H * W)), dim=-1)
        image_preds.extend(
            anomaly_score.cpu().numpy().tolist()
        )  # extend since it is a list of scalar
        image_gts.extend(input["is_anomaly"].cpu().numpy().tolist())

        if dataset_name == "busi" or dataset_name == "brats-met":
            pixel_gts.append(input["mask"].cpu().numpy())
    # pixel_preds = [Tensor[B, H, W], Tensor[B, H, W], ...] -> Tensor[N, H, W]

    pixel_preds_np = [pixel_pred.cpu().numpy() for pixel_pred in pixel_preds]

    pixel_preds = normalization(
            torch.cat(pixel_preds, dim=0), args.config.image_size
        )

    if dataset_name == "busi" or dataset_name == "brats-met":
        pixel_gts = np.concatenate(pixel_gts, axis=0)

    save_images_root = os.path.join(args.vis_save_root, "{}".format(dataset_name))
    os.makedirs(save_images_root, exist_ok=True)

    if dataset_name == "busi" or dataset_name == "brats-met":
        iter = tqdm(
            zip(image_paths, image_gts, pixel_preds, pixel_gts),
            total=len(image_paths),
            desc="Generating Segmentation Images...",
            leave=False,
        )
    else:
        iter = tqdm(
            zip(image_paths, image_gts, pixel_preds),
            total=len(image_paths),
            desc="Generating Segmentation Images...",
            leave=False,
        )

    # generate and save visualization images, one image per iteration
    for i, data in enumerate(iter):
        if dataset_name == "busi" or dataset_name == "brats-met":
            image_path, image_gt, pixel_pred, pixel_gt = data
        else:
            image_path, image_gt, pixel_pred = data

        _, image_name = os.path.split(image_path)
        image_name = image_name.split(".")[0]

        """if dataset_name == "brats-met":
            fmri = nib.load(image_path)

            n_slices = fmri.shape[-1]

            for _ in range(n_slices - 1):
                slice = fmri.get_fdata()[:, :, _]

                slice_norm = (slice - np.min(slice)) / (
                    np.max(slice) - np.min(slice) + 1e-8
                )  # Avoid division by zero
                slice = (slice_norm * 255).astype(np.uint8)

                image = (
                    PIL.Image.fromarray(slice)
                    .convert("RGB")
                    .resize(
                        (args.config.image_size, args.config.image_size),
                        PIL.Image.Resampling.BILINEAR,
                    )
                )

                heat = show_cam_on_image(
                    np.array(image) / 255, pixel_pred[_], use_rgb=True
                )

                label_ = "normal" if image_gt == 0 else "abnormal"

                merge = [image, heat]

                Image.fromarray(np.concatenate(merge, axis=1).astype(np.uint8)).save(
                    os.path.join(save_images_root, f"{image_name}_{_}.png")
                )

            data_array = np.stack(pixel_pred, axis=-1) * 1000
            nii_image = nib.Nifti1Image(
                data_array, affine=np.eye(4)
            )  # Identity affine matrix
            nib.save(nii_image, os.path.join(save_images_root, f"{image_name}.nii.gz"))
        else:"""
        image = Image.open(image_path).convert("RGB")
        image = image.resize((args.config.image_size, args.config.image_size))
        image = np.array(image).astype(np.uint8)

        heat = show_cam_on_image(image / 255, pixel_pred, use_rgb=True)

        label_ = "normal" if image_gt == 0 else "abnormal"

        merge = [image, heat]

        if dataset_name == "busi":
            pixel_gt = np.repeat(np.expand_dims(pixel_gt, axis=-1), 3, axis=-1)
            merge.append(pixel_gt * 255)

        img = Image.fromarray(np.concatenate(merge, axis=1).astype(np.uint8))

        img.save(
            os.path.join(
                save_images_root, "{}_{}_{}.jpg".format(i, label_, image_name)
            )
        )
    
    metric = compute_imagewise_metrics(image_preds, image_gts)
    if (dataset_name == "busi" or dataset_name == "brats-met"):  
        # for these datasets, since we have a ground truth mask, we can compute pixel-wise metrics
        metric.update(compute_pixelwise_metrics(pixel_preds_np, pixel_gts))
    return metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MediCLIP")
    parser.add_argument("--config_path", type=str, help="model configs")
    parser.add_argument("--checkpoint_path", type=str, help="the checkpoint path")
    parser.add_argument("--vis_save_root", type=str, default="vis_results")
    parser.add_argument("--slice_idx", type=int, default=100)
    args = parser.parse_args()
    torch.multiprocessing.set_start_method("spawn")
    main(args)
