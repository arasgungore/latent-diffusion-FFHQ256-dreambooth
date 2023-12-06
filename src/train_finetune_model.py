import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from models import FaceAutoencoder, LatentDiffusionModel
from data_loading import prepare_data_loader
from utils.dreambooth_utils import dreambooth_finetune

# Define paths
data_path = 'data/subject'
log_path = 'logs/finetune'
checkpoint_path = 'checkpoints/finetune.pth'

# Hyperparameters
epochs = 10
lr = 0.0001
batch_size = 8

# Prepare data loader
data_loader = prepare_data_loader(data_path, batch_size=batch_size)

# Load pre-trained diffusion model from Task 1
diffusion_model = LatentDiffusionModel()
diffusion_model.load_state_dict(torch.load('checkpoints/latent_diffusion.pth'))
diffusion_model.eval()

# Initialize Dreambooth model
dreambooth_model = FaceAutoencoder()

# Initialize optimizer for Dreambooth model
optimizer = optim.Adam(dreambooth_model.parameters(), lr=lr)

# Fine-tune using Dreambooth method
dreambooth_finetune(dreambooth_model, diffusion_model, data_loader, optimizer, epochs)

# Save the finetuned model
torch.save(dreambooth_model.state_dict(), checkpoint_path)
