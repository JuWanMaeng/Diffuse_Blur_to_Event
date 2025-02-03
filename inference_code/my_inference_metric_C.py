# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------


import argparse
import logging
import os
from glob import glob

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from marigold.b2f_pipeline_C import B2FPipeline_C
import cv2
from ptlflow.utils import flow_utils
from ptlflow.utils.flow_utils import flow_to_rgb
import torch.nn.functional as F

EXTENSION_LIST = [".jpg", ".jpeg", ".png"]

def normalize_flow(flow):

    # Calculate the magnitude of the flow vectors
    u, v = flow[:,:,0], flow[:,:,1]
    magnitude = np.sqrt(u**2 + v**2)
    
    # Avoid division by zero by setting small magnitudes to a minimal positive value
    magnitude[magnitude == 0] = 1e-8
    
    # Normalize u and v components to get unit vectors for x and y
    x = u / magnitude
    y = v / magnitude

    # Normalize the magnitude to [0, 1] range for the z component
    z = magnitude / 100  # 100
    z = np.clip(z, 0, 1)
    z = z * 2 - 1


    # Stack x, y, and z to create the 3D tensor C with shape (H, W, 3)
    C = np.stack((x, y, z), axis=-1)

    return C


if "__main__" == __name__:
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using Marigold."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint/my",
        help="Checkpoint path or hub name.",
    )

    parser.add_argument(
        "--input_rgb_dir",
        type=str,
        default='/workspace/Marigold/dataset/train_part4.txt',
        help="Path to the input image folder.",
    )

    # inference setting
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=50,
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed. For the original (DDIM) version, it's recommended to use 10-50 steps, while for LCM 1-4 steps.",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=1,
        help="Number of predictions to be ensembled, more inference gives better results but runs slower.",
    )
    parser.add_argument(
        "--half_precision",
        "--fp16",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )

    # resolution setting
    parser.add_argument(
        "--processing_res",
        type=int,
        default=None,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 768.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, out put depth at resized operating resolution. Default: False.",
    )
    parser.add_argument(
        "--resample_method",
        choices=["bilinear", "bicubic", "nearest"],
        default="bilinear",
        help="Resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic` or `nearest`. Default: `bilinear`",
    )

    # depth map colormap
    parser.add_argument(
        "--color_map",
        type=str,
        default="Spectral",
        help="Colormap used to render depth predictions.",
    )

    # other settings
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Reproducibility seed. Set to `None` for unseeded inference.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Inference batch size. Default: 0 (will be set automatically).",
    )
    parser.add_argument(
        "--apple_silicon",
        action="store_true",
        help="Flag of running on Apple Silicon.",
    )

    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    input_rgb_dir = args.input_rgb_dir

    denoise_steps = args.denoise_steps
    ensemble_size = args.ensemble_size
    if ensemble_size > 15:
        logging.warning("Running with large ensemble size will be slow.")
    half_precision = args.half_precision

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res
    if 0 == processing_res and match_input_res is False:
        logging.warning(
            "Processing at native resolution without resizing output might NOT lead to exactly the same resolution, due to the padding and pooling properties of conv layers."
        )
    resample_method = args.resample_method

    color_map = args.color_map
    seed = args.seed
    batch_size = args.batch_size


    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Data --------------------
    # rgb_filename_list = glob(os.path.join(input_rgb_dir, "*"))
    input_path_txt = input_rgb_dir
    rgb_filename_list = []
    with open(input_path_txt, 'r') as file:
        for line in file:
            rgb_filename_list.append(line.strip())

    # rgb_filename_list = sorted(rgb_filename_list)
    n_images = len(rgb_filename_list)
    if n_images > 0:
        logging.info(f"Found {n_images} images")
    else:
        logging.error(f"No image found in '{input_rgb_dir}'")
        exit(1)


    # -------------------- Model --------------------
    if half_precision:
        dtype = torch.float16
        variant = "fp16"
        logging.info(
            f"Running with half precision ({dtype}), might lead to suboptimal result."
        )
    else:
        dtype = torch.float32
        variant = None

    pipe: B2FPipeline_C = B2FPipeline_C.from_pretrained(
        checkpoint_path, variant=variant, torch_dtype=dtype
    )

    # try:
    #     pipe.enable_xformers_memory_efficient_attention()
    # except ImportError:
    #     pass  # run without xformers

    pipe = pipe.to(device)
    logging.info(
        f"scale_invariant: {pipe.scale_invariant}, shift_invariant: {pipe.shift_invariant}"
    )

    # Print out config
    logging.info(
        f"Inference settings: checkpoint = `{checkpoint_path}`, "
        f"with denoise_steps = {denoise_steps or pipe.default_denoising_steps}, "
        f"ensemble_size = {ensemble_size}, "
        f"processing resolution = {processing_res or pipe.default_processing_resolution}, "
        f"seed = {seed}; "
        f"color_map = {color_map}."
    )

    # -------------------- Inference and saving --------------------
    max_flow = 10000
    total_mse = 0
    save_path = '/workspace/data/results/C/Gopro/train'

    pass_flow = 0
    with torch.no_grad():
        for idx,blur_path in enumerate(tqdm(rgb_filename_list, desc="Estimating depth", leave=True)):
            # Read input image
            phase = blur_path.split('/')[4]

            if phase == 'test':
                next_scene = 1111
            else:
                next_scene = 2103

            input_image = Image.open(blur_path)
            img_name = blur_path.split('/')[-1]
            img_num = img_name[:-4]

            # Optical flow array
            flow_path = blur_path.replace('blur', 'flow/flows')
            flow_path = flow_path.replace('png', 'flo')

            # Random number generator
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)

            # Predict depth
            pipe_outs = pipe(
                input_image,
                denoising_steps=denoise_steps,
                ensemble_size=ensemble_size,
                processing_res=processing_res,
                match_input_res=match_input_res,
                batch_size=batch_size,
                color_map=color_map,
                show_progress_bar=True,
                resample_method=resample_method,
                generator=generator,
            )  # 720,1280,3
            
            # l-flow
            max_flow = 10000
            flow = flow_utils.flow_read(flow_path)
            nan_mask = np.isnan(flow)
            flow[nan_mask] = max_flow + 1
            flow[nan_mask] = 0
            flow = np.clip(flow, -max_flow, max_flow)
            l_flow = normalize_flow(flow)
            l_flow = (l_flow + 1) /2

            # r-flow
            scene = flow_path.split('/')[5]
            new_scene = int(scene) + next_scene
            new_scene = str(new_scene).zfill(6)
            flow_path = flow_path.replace(scene,new_scene)
            flow = flow_utils.flow_read(flow_path)
            nan_mask = np.isnan(flow)
            flow[nan_mask] = max_flow + 1
            flow[nan_mask] = 0
            flow = np.clip(flow, -max_flow, max_flow)
            r_flow = normalize_flow(flow)
            r_flow = (r_flow + 1) / 2


            mse = 100
            os.makedirs(f'{save_path}/{img_num}', exist_ok=True)
            for idx,pipe_out in enumerate(pipe_outs):
                
                
                pipe_out = (pipe_out + 1) / 2

                l_mse = np.mean((pipe_out - l_flow) ** 2)
                r_mse = np.mean((pipe_out - r_flow) ** 2)

                target_mse = min(l_mse,r_mse)
                if target_mse < 0.01:
                    np.savez_compressed(f'{save_path}/{img_num}/{idx+1}.npz',pipe_out)
                else:
                    pass_flow += 1

                mse = min(r_mse, l_mse, mse)

            total_mse += mse



    print(total_mse / len(rgb_filename_list))
    print(f'pass flow:{pass_flow}')