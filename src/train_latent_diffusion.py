import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from models import FaceAutoencoder, LatentDiffusionModel
from data_loading import prepare_data_loader

# Define paths
data_path = 'data/ffhq256'
log_path = 'logs/latent_diffusion'
checkpoint_path = 'checkpoints/latent_diffusion.pth'

# Hyperparameters
epochs = 20
lr = 0.001
batch_size = 32

# Prepare data loader
data_loader = prepare_data_loader(data_path, batch_size=batch_size)

# Initialize models and optimizer
autoencoder = FaceAutoencoder()
diffusion_model = LatentDiffusionModel()
optimizer = optim.Adam(diffusion_model.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    for batch in data_loader:
        # Forward pass through autoencoder
        latent_representation = autoencoder(batch)

        # Forward pass through diffusion model
        output = diffusion_model(latent_representation)

        # Compute loss and backpropagate
        loss = F.mse_loss(output, latent_representation)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Log loss to TensorBoard
    writer = SummaryWriter(log_path)
    writer.add_scalar('Loss', loss.item(), epoch)
    writer.close()

# Save the trained model
torch.save(diffusion_model.state_dict(), checkpoint_path)
