import argparse
import math
import os
import shutil
import time
from pathlib import Path
import logging

import accelerate
import numpy as np
import PIL
import PIL.Image
import timm
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from datasets import load_dataset
from marigold.discriminator import Discriminator
from huggingface_hub import create_repo
from packaging import version
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import transforms
from tqdm import tqdm

from diffusers import VQModel, AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_wandb_available
from dataset import flow_blur_dataset

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    logging.warning("CUDA is not available. Running on CPU will be slow.")
logging.info(f"device = {device}")

model_path = '/workspace/Marigold/checkpoint/stable-diffusion-2/vae'
val_txt_file = 'dataset/dog/dog_val.txt'

model = AutoencoderKL.from_pretrained(model_path)
model = model.to(device)
model.eval()

### validation images ###
validation_images = []
original_images = []
with open (val_txt_file, 'r') as f:
    paths = f.readlines()
    for path in paths:
        validation_images.append(path)

validation_transform = transforms.Compose(
    [
        transforms.Resize(768, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ]
)

for image_path in validation_images:
    image = PIL.Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image = validation_transform(image).to(device)
    original_images.append(image[None])

with torch.no_grad():
    images = []
    for original_image in original_images:
        image = model(original_image).sample
        images.append(image)

original_images = torch.cat(original_images, dim=0)
images = torch.cat(images, dim=0)

# Convert to PIL images
images = torch.clamp(images, 0.0, 1.0)
original_images = torch.clamp(original_images, 0.0, 1.0)
images *= 255.0
original_images *= 255.0
images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
original_images = original_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
images = np.concatenate([original_images, images], axis=2)
images = [Image.fromarray(image) for image in images]
print('Done')