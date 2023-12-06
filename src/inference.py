import torch
from models import LatentDiffusionModel, FaceAutoencoder
from utils.inference_utils import generate_samples

# Load the trained diffusion model
diffusion_model = LatentDiffusionModel()
diffusion_model.load_state_dict(torch.load('checkpoints/latent_diffusion.pth'))
diffusion_model.eval()

# Generate samples using DDIM sampler
generated_samples = generate_samples(diffusion_model, num_samples=5)

# Load the finetuned Dreambooth model
dreambooth_model = FaceAutoencoder()
dreambooth_model.load_state_dict(torch.load('checkpoints/finetune.pth'))
dreambooth_model.eval()

# Generate samples of the subject using DDIM sampler
subject_samples = generate_samples(dreambooth_model, num_samples=5)

# Display or save the generated samples
