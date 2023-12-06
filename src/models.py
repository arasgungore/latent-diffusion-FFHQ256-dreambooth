import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL, UNet2DModel

class FaceAutoencoder(AutoencoderKL):
    def __init__(self):
        super(FaceAutoencoder, self).__init__(input_shape=(256, 256, 3), latent_dims=(32, 32))

class LatentDiffusionModel(UNet2DModel):
    def __init__(self):
        super(LatentDiffusionModel, self).__init__(in_channels=32, out_channels=32)
