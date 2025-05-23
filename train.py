import open_clip
import os
import torch
import yaml
from easydict import EasyDict
from models.Necker import Necker
from models.Adapter import Adapter
import math
import argparse
import warnings
from utils.misc_helper import (
    AverageMeter,
    compute_imagewise_metrics,
    compute_pixelwise_metrics,
    get_current_time,
    create_logger,
    set_seed
)
from torch.utils.data import DataLoader
from models.MapMaker import MapMaker
from utils.losses import FocalLoss, BinaryDiceLoss
from datasets.dataset_paper import (
    ChexpertTestDataset,
    BusiTestDataset,
    BrainMRITestDataset,
    BratsMetTestDataset,
)
from datasets.dataset import TrainDataset, TrainDatasetFewShot
import pprint
from tqdm import tqdm
import multiprocessing
import numpy as np
import random
from pathlib import Path
import json

warnings.filterwarnings("ignore")

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

    if len(tokens[0].shape) == 3: # (B, N, C)
        # number of tokens along one side of the square grid
        model.token_size = [int(math.sqrt(token.shape[1] - 1)) for token in tokens]
        # number of channels - feature dimensionality of each token embedding
        model.token_c = [token.shape[-1] for token in tokens]
    else: # (B, C, H, W)
        model.token_size = [token.shape[2] for token in tokens]
        model.token_c = [token.shape[1] for token in tokens]

    model.embed_dim = model_cfg["embed_dim"]
    print(
        "model token size is {}".format(model.token_size),
        "model token dim is {}".format(model.token_c),
    )


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.config_path) as f:
        args.config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    set_seed(seed=args.config.random_seed)

    model, preprocess, model_cfg = open_clip.create_model_and_transforms(
        args.config.model_name, args.config.image_size, device=device
    )

    if args.pmc:
        model_path = '/home/aldo_marzullo/.cache/huggingface/hub/models--ryanyip7777--pmc_vit_l_14/snapshots/a1db5a40d0a07855a435d9fabde846fdf7028be4/open_clip_pytorch_model.bin'
        model.load_state_dict(torch.load(model_path, weights_only=True))

    for param in model.parameters():
        param.requires_grad_(False)

    args.config.model_cfg = model_cfg

    make_vision_takens_info(model, args.config.model_cfg, args.config.layers_out)

    current_time = get_current_time()
    args.config.save_root = os.path.join(args.config.save_root, current_time)

    if not os.path.exists(args.config.save_root):
        os.makedirs(args.config.save_root)

    logger = create_logger("logger", os.path.join(args.config.save_root, "logger.log"))
    #logger.info("config: {}".format(pprint.pformat(args)))

    necker = Necker(clip_model=model).to(model.device)
    adapter = Adapter(clip_model=model, target=args.config.model_cfg["embed_dim"]).to(
        model.device
    )

    if args.config.prompt_maker == "coop":
        from models.CoOp import PromptMaker

        logger.info("load CoOp")
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

    if args.checkpoint_path:
        checkpoints = torch.load(
            args.checkpoint_path, map_location=str(device)
        )  # Pass device as string
        adapter.load_state_dict(checkpoints["adapter_state_dict"])
        prompt_maker.prompt_learner.load_state_dict(checkpoints["prompt_state_dict"])

    optimizer = torch.optim.Adam(
        [
            {"params": prompt_maker.prompt_learner.parameters(), "lr": 0.001},
            {"params": adapter.parameters(), "lr": 0.001},
        ],
        lr=0.001,
        betas=(0.5, 0.999),
    )

    source = os.path.join(args.config.data_root, args.config.train_dataset)
    if args.patients:
        assert args.config.train_dataset == "brats-met", (
            "patients training supported only for brats-met dataset"
        )
        train_dataset = TrainDatasetFewShot(
            args=args.config,
            root=source,
            mode="train",
            target_transform=None,
            transform=preprocess,
            k_shot=args.k_shot,
        )
    else:
        train_dataset = TrainDataset(
            args=args.config,
            root=source,
            mode="train",
            target_transform=None,
            transform=preprocess,
            k_shot=args.k_shot,
        )

    num_workers = max(1, multiprocessing.cpu_count() - 1)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.config.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_dataloaders = {}
    best_record = {}

    for test_dataset_name in args.config.test_datasets:
        if test_dataset_name == "chexpert":
            test_dataset = ChexpertTestDataset(
                args=args.config,
                source=os.path.join(
                    args.config.data_root, test_dataset_name
                ),  # args.config.data_root = 'data' always
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
                slice_idx=-1
            )
        else:
            raise NotImplementedError(
                "dataset must in ['chexpert','busi','brainmri', 'brats-met'] "
            )

        test_dataloader = DataLoader(
            test_dataset, batch_size=args.config.batch_size, num_workers=2
        )

        test_dataloaders[test_dataset_name] = test_dataloader

        best_record[test_dataset_name] = None

    logger.info(
        "train data ({}) len {}".format(args.config.train_dataset, len(train_dataset))
    )

    for test_dataset_name in test_dataloaders:
        logger.info(
            "test data ({}) len {}".format(
                test_dataset_name, len(test_dataloaders[test_dataset_name].dataset)
            )
        )

    for task_name in args.config.anomaly_tasks:
        logger.info(
            "anomaly syn task is {}, sampling probability is {}".format(
                task_name, args.config.anomaly_tasks[task_name]
            )
        )

    for epoch in range(0, args.config.epoch):
        last_iter = epoch * len(train_dataloader)
        
        train_one_epoch(
            args,
            train_dataloader,
            optimizer,
            epoch,
            last_iter,
            logger,
            model,
            necker,
            adapter,
            prompt_maker,
            map_maker,
        )
        
        if (epoch+1) % args.config.val_freq_epoch == 0:
            results = validate(
                args,
                test_dataloaders,
                epoch,
                model,
                necker,
                adapter,
                prompt_maker,
                map_maker,
            )
            save_flag = False

            for test_dataset_name in results:
                if best_record[test_dataset_name] is None:
                    if test_dataset_name == "busi" or test_dataset_name == "brats-met":
                        best_record[test_dataset_name] = [
                            results[test_dataset_name]["image-auroc"],
                            results[test_dataset_name]["pixel-auroc"],
                        ]
                    else:
                        best_record[test_dataset_name] = [
                            results[test_dataset_name]["image-auroc"]
                        ]

                    save_flag = True
                else:
                    if np.mean(
                        [
                            results[test_dataset_name][key]
                            for key in results[test_dataset_name]
                        ]
                    ) > np.mean(best_record[test_dataset_name]):
                        if (
                            test_dataset_name == "busi"
                            or test_dataset_name == "brats-met"
                        ):
                            best_record[test_dataset_name] = [
                                results[test_dataset_name]["image-auroc"],
                                results[test_dataset_name]["pixel-auroc"],
                            ]
                        else:
                            best_record[test_dataset_name] = [
                                results[test_dataset_name]["image-auroc"]
                            ]
                        save_flag = True

                if test_dataset_name == "busi" or test_dataset_name == "brats-met":
                    logger.info(
                        "({}): Epoch: {}, image auroc: {:.4f}, pixel_auroc: {:.4f},".format(
                            test_dataset_name,
                            epoch + 1,
                            results[test_dataset_name]["image-auroc"],
                            results[test_dataset_name]["pixel-auroc"],
                        )
                    )
                else:
                    logger.info(
                        "({}): Epoch: {}, image auroc: {:.4f},".format(
                            test_dataset_name,
                            epoch + 1,
                            results[test_dataset_name]["image-auroc"],
                        )
                    )

            for test_dataset_name in results:
                if test_dataset_name == "busi" or test_dataset_name == "brats-met":
                    logger.info(
                        "({} best): image auroc: {:.4f}, pixel auroc: {:.4f},".format(
                            test_dataset_name,
                            best_record[test_dataset_name][0],
                            best_record[test_dataset_name][1],
                        )
                    )
                else:
                    logger.info(
                        "({} best): image auroc: {:.4f},".format(
                            test_dataset_name,
                            best_record[test_dataset_name][0],
                        )
                    )

            if save_flag:
                logger.info("save checkpoints in epoch: {}".format(epoch + 1))
                torch.save(
                    {
                        "adapter_state_dict": adapter.state_dict(),
                        "prompt_state_dict": prompt_maker.prompt_learner.state_dict(),
                    },
                    os.path.join(
                        args.config.save_root, "checkpoints_{}.pkl".format(epoch + 1)
                    ),
                )


def train_one_epoch(
    args,
    train_dataloader,
    optimizer,
    epoch,
    start_iter,
    logger,
    clip_model,
    necker,
    adapter,
    prompt_maker,
    map_maker,
):
    loss_meter = AverageMeter(args.config.print_freq_step)

    focal_criterion = FocalLoss()
    dice_criterion = BinaryDiceLoss()

    adapter.train()
    prompt_maker.train()

    for i, input in enumerate(train_dataloader):
        curr_step = start_iter + i
        images = input["image"].to(clip_model.device)
        gt_mask = input["mask"].to(clip_model.device)
        
        with torch.no_grad():
            _, image_tokens = clip_model.encode_image(
                images, out_layers=args.config.layers_out
            )
            # align shape
            image_features = necker(image_tokens)

        # adapter outside 'torch.no_grad' since its parameters are updated 
        # align number of channels
        vision_adapter_features = adapter(image_features)
        prompt_adapter_features = prompt_maker(vision_adapter_features)
        anomaly_map = map_maker(vision_adapter_features, prompt_adapter_features)
        loss = []
        # S_n and S_a have shapes [B, H, W]
        # then anomaly_map has shape [B, 2, H, W] where [0] is S_n and [1] is S_a
    
        """function applies a softmax across the class dimension (dim=1) to turn 
        the [B, 2, H, W] into probabilities per class per pixel.
        Then, for each pixel:
            It extracts the probability for the true class (using gt_mask)
        """
        # at the beginning the weights are around 0, hence the model almost always predicts 0
        # since a mask is almost all 0s, the error is already low
        loss.append(focal_criterion(anomaly_map, gt_mask))
        #print(f"focal loss: {loss[-1]}")
        """Unlike Focal Loss or CrossEntropyLoss, which operate on probabilities 
        (and therefore penalize over- or under-confidence), Dice Loss works on 
        soft masks and is focused on overlap between prediction and ground truth.
        It doesn't care that much about the confidence:
            - In CrossEntropy/Focal Loss, confidence hugely change the loss depending 
            on how close they are to the target (e.g., log(0.95) vs log(0.55) is very different).
            - In Dice Loss, it just gets multiplied by the ground truth and added up
        """
        # ? investigate changes to this loss function
        loss.append(dice_criterion(anomaly_map[:, 1, :, :], gt_mask))
        #print(f"dice loss: {loss[-1]}")
        loss = torch.sum(torch.stack(loss))
        loss_meter.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (curr_step + 1) % args.config.print_freq_step == 0:
            logger.info(
                "Epoch: [{0}/{1}]\t"
                "Iter: [{2}/{3}]\t"
                "Loss {loss.val:.5f} ({loss.avg:.5f})\t".format(
                    epoch + 1,
                    args.config.epoch,
                    curr_step + 1,
                    len(train_dataloader) * args.config.epoch,
                    loss=loss_meter,
                )
            )


def validate(
    args, test_dataloaders, epoch, clip_model, necker, adapter, prompt_maker, map_maker
):
    adapter.eval()
    prompt_maker.eval()
    results = {}
    for test_dataset_name in test_dataloaders:
        test_dataloader = test_dataloaders[test_dataset_name]
        anomaly_maps = []
        anomaly_gts = []

        image_scores = []
        image_labels = []

        with torch.no_grad():
            for i, input in enumerate(tqdm(test_dataloader, desc=test_dataset_name)):
                images = (input["image"].to(clip_model.device))  
                _, image_tokens = clip_model.encode_image(
                    images, out_layers=args.config.layers_out
                )
                image_features = necker(image_tokens)
                vision_adapter_features = adapter(image_features)
                propmt_adapter_features = prompt_maker(vision_adapter_features)
                anomaly_map = map_maker(
                    vision_adapter_features, propmt_adapter_features
                )

                B, _, H, W = anomaly_map.shape

                anomaly_map = anomaly_map[:, 1, :, :]
                
                anomaly_gt = input["mask"] 

                anomaly_maps.append(anomaly_map.cpu().numpy())
                anomaly_gts.append(anomaly_gt.cpu().numpy())

                anomaly_scores, _ = torch.max(anomaly_map.view((B, H * W)), dim=-1)

                image_scores.extend(anomaly_scores.cpu().numpy().tolist())

                image_labels.extend(input["is_anomaly"].cpu().numpy().tolist())

        metric = compute_imagewise_metrics(image_scores, image_labels)

        if test_dataset_name == "busi" or test_dataset_name == "brats-met":
            metric.update(compute_pixelwise_metrics(anomaly_maps, anomaly_gts))
        results[test_dataset_name] = metric
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MediCLIP")
    parser.add_argument(
        "--config_path", type=str, default="config/brainmri.yaml", help="model configs"
    )
    parser.add_argument("--k_shot", type=int, default=16, help="normal image number")
    parser.add_argument(
        "--patients",
        type=bool,
        default=False,
        help="whether to k-shot refers to patients",
    )
    parser.add_argument("--checkpoint_path", type=str, help="the checkpoint path", default=None)
    parser.add_argument('--pmc', type=bool, help = 'use pmc as backbone', default=False) 
    args = parser.parse_args()
    torch.multiprocessing.set_start_method("spawn")
    main(args)
