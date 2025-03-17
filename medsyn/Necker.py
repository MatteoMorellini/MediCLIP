import torch
import torch.nn as nn
import math


# processes token embeddings from different transformer layers and aligns them spatially using bilinear upsampling
class Necker(nn.Module):
    def __init__(self, clip_model):
        super(Necker, self).__init__()
        self.clip_model = clip_model
        target = max(self.clip_model.token_size)  # determine the largest token size
        for i, size in enumerate(
            self.clip_model.token_size
        ):  # for each layer's token output size
            self.add_module(
                "{}_upsample".format(i),  # create an upsampling layer to match target
                nn.UpsamplingBilinear2d(scale_factor=target / size),
            )

    @torch.no_grad()
    def forward(self, tokens):
        align_features = []
        for i, token in enumerate(tokens):
            if len(token.shape) == 3:  # [Batch, Num_tokens, Channels]
                B, N, C = token.shape
                token = token[
                    :, 1:, :
                ]  # remove the [CLS] token, else reshaping is impossible
                token = token.view(
                    (B, int(math.sqrt(N - 1)), int(math.sqrt(N - 1)), C)
                ).permute(0, 3, 1, 2)  # reshape to [Batch, Channels, Height, Width]
            align_features.append(
                getattr(self, "{}_upsample".format(i))(  # retrieve upsampling layer,
                    token
                )
            )  # upsample the token
        return align_features
