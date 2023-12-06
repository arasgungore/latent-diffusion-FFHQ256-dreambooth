import torch
import matplotlib.pyplot as plt
import numpy as np

def generate_samples(model, num_samples=5):
    with torch.no_grad():
        # Generate random noise for sampling
        noise = torch.randn(num_samples, 32, 32)

        # Generate samples using the model
        generated_samples = model.generate(noise)

    return generated_samples

def display_or_save_images(images, filename=None):
    # Rescale images from [-1, 1] to [0, 1]
    images = (images + 1) / 2.0

    # Convert PyTorch tensor to NumPy array
    images = images.permute(0, 2, 3, 1).cpu().numpy()

    # Create a grid of images
    rows = int(np.sqrt(images.shape[0]))
    cols = int(np.ceil(images.shape[0] / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

    for i in range(rows):
        for j in range(cols):
            index = i * cols + j
            if index < images.shape[0]:
                axes[i, j].imshow(images[index])
                axes[i, j].axis('off')

    if filename:
        plt.savefig(filename)
    else:
        plt.show()
