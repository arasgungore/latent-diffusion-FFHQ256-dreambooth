import torch
import torch.nn.functional as F

def dreambooth_finetune(dreambooth_model, diffusion_model, data_loader, optimizer, epochs):
    for epoch in range(epochs):
        for batch in data_loader:
            # Forward pass through diffusion model
            latent_representation = diffusion_model(batch)

            # Forward pass through Dreambooth model
            output = dreambooth_model(latent_representation)

            # Compute loss and backpropagate
            loss = F.mse_loss(output, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
