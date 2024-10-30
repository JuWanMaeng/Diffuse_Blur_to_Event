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
from ptlflow.utils import flow_utils
from tqdm import tqdm


def normalize_flow_to_tensor(flow):
    # Calculate the magnitude of the flow vectors
    u, v = flow[:,:,0], flow[:,:,1]
    magnitude = np.sqrt(u**2 + v**2)
    
    # Avoid division by zero by setting small magnitudes to a minimal positive value
    magnitude[magnitude == 0] = 1e-8
    
    # Normalize u and v components to get unit vectors for x and y
    x = u / magnitude
    y = v / magnitude
            
    # Normalize the magnitude to [0, 1] range for the z component
    z = magnitude / 147
    z = np.clip(z, 0, 1) 


    # Stack x, y, and z to create the 3D tensor C with shape (H, W, 3)
    C = np.stack((x, y, z), axis=-1)

    return C

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
logging.warning("CUDA is not available. Running on CPU will be slow.")
logging.info(f"device = {device}")

model_path = '/workspace/Marigold/checkpoint/stable-diffusion-2/vae'
# model_path = '/workspace/data/AE-output-KL-pretrained/checkpoint-6000/aemodel'

model = AutoencoderKL.from_pretrained(model_path)
model = model.to(device)
model.eval()

### validation images ###
original_images = []

with open ('dataset/test_into_future.txt', 'r') as f:
    paths = f.readlines()
    for path in paths[:2]:
        ### flow ###
        blur_path = path.strip()
        flow_path = blur_path.replace('blur', 'flow/flows')
        # flow_path = flow_path.replace('png', 'pfm')
        if flow_path.split('/')[3] == 'Monkaa_my':
            flow_path = flow_path.replace('png', 'pfm')
        else:
            flow_path = flow_path.replace('png', 'flo')

        max_flow = 10000
        flow = flow_utils.flow_read(flow_path)

        nan_mask = np.isnan(flow)
        flow[nan_mask] = max_flow + 1
        flow[nan_mask] = 0
        flow = np.clip(flow, -max_flow, max_flow)
        n_flow = normalize_flow_to_tensor(flow)  # H,W,3
        n_flow = torch.from_numpy(n_flow.transpose(2,0,1)[None]).to(device)
        original_images.append(n_flow)


total_MSE = 0
with torch.no_grad():
    images = []
    for original_image in tqdm(original_images, ncols=80):
        image = model(original_image).sample
        mse = torch.mean((original_image - image) ** 2)
        total_MSE += mse
print("MSE:", total_MSE.item()/len(original_images))
