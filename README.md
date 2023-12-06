# latent-diffusion-FFHQ256-dreambooth

An unconditional generative model trained on FFHQ face data set in 256×256 resolution and then fine-tuned using Dreambooth method. The Diffusers library from HuggingFace is utilized for implementing the latent diffusion model.



## Project Structure

```plaintext
FaceGenerativeModel/
|-- data/
|   |-- ffhq/   # FFHQ dataset (downloaded separately)
|   |-- subject/   # Small face dataset of a single subject
|-- diffusers/   # Cloned Diffusers library from HuggingFace
|-- src/
|   |-- data_loading.py   # Script for loading and preparing data
|   |-- models.py   # Implementation of FaceAutoencoder and LatentDiffusionModel
|   |-- train_latent_diffusion.py   # Script for Task 1 - Training Latent Diffusion Model
|   |-- train_finetune_model.py   # Script for Task 2 - Finetuning the Model
|   |-- inference.py   # Inference script for generating samples
|-- utils/
|   |-- dreambooth_utils.py   # Utilities for Dreambooth method
|   |-- inference_utils.py   # Utilities for inference script
|-- README.md   # Detailed README with instructions and project overview
|-- report.pdf   # 2-3 pages report with sampled images and loss curves
```



## Prerequisites

- Python 3.7+
- GPU (recommended for faster training)
- Google Colab or Google Cloud (optional, for free GPU usage)



## Data Preparation

1. Download the FFHQ dataset from [here](https://www.kaggle.com/datasets/denislukovnikov/ffhq256-images-only) and save it in the `data/ffhq/` folder.
2. Prepare a small face dataset of a single subject in the `data/subject/` folder using 10-15 photos.



## Training Latent Diffusion Model (Task 1)

1. Run the following command to train the latent diffusion model on the FFHQ dataset:
   ```bash
   python src/train_latent_diffusion.py --data_path data/ffhq
   ```
2. After training, generate new face samples using:
   ```bash
   python src/inference.py --model_path checkpoints/latent_diffusion.pth
   ```



## Fine-tuning the Model (Task 2)

1. Run the following command to finetune the model on the subject dataset:
   ```bash
   python src/train_finetune_model.py --data_path data/subject --pretrained_model_path checkpoints/latent_diffusion.pth
   ```
2. After finetuning, generate new face samples of the subject using:
   ```bash
   python src/inference.py --model_path checkpoints/finetune.pth
   ```



## Author

👤 **Aras Güngöre**

- LinkedIn: [@arasgungore](https://www.linkedin.com/in/arasgungore)
- GitHub: [@arasgungore](https://github.com/arasgungore)
