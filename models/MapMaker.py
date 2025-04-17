import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class MapMaker(nn.Module):
    def __init__(self, image_size):
        super(MapMaker, self).__init__()
        self.image_size = image_size

    def forward(self, vision_adapter_features, prompt_adapter_features):
        # vision_adapter_features is a list of 3 tensor, each is a feature
        # each feature has shape [8, 16, 16, 768] ie [B, H, W, C]
        
        # prompt_adapter_features is a tensor of shape [768, 2]
        anomaly_maps = []
        for i, vision_adapter_feature in enumerate(vision_adapter_features):
            B, H, W, C = vision_adapter_feature.shape
            anomaly_map = (
                (vision_adapter_feature.view((B, H * W, C)) @ prompt_adapter_features)
                .contiguous()
                .view((B, H, W, -1))
                .permute(0, 3, 1, 2) # [8, 2, 16, 16]
            )
            anomaly_maps.append(anomaly_map)
        # anomaly_maps[0] = [Sn1, Sn2, Sn3] simplifying with batch_size=1
        # anomaly_maps[1] = [Sa1, Sa2, Sa3]
        # combine into tensor of shape [3, 8, 2, 16, 16], compute mean along dim 0
        anomaly_map = torch.stack(anomaly_maps, dim=0).mean(dim=0)
        anomaly_map = F.interpolate(
            anomaly_map,
            (self.image_size, self.image_size),
            mode="bilinear",
            align_corners=True,
        )
        return torch.softmax(anomaly_map, dim=1)
