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

from marigold.b2f_pipeline import B2FPipeline
import cv2
from ptlflow.utils import flow_utils
from ptlflow.utils.flow_utils import flow_read, flow_to_rgb

from color import get_dominant_color, compare_colors

EXTENSION_LIST = [".jpg", ".jpeg", ".png"]


if "__main__" == __name__:
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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
        default='dataset/train_part4.txt',
        help="Path to the input image folder.",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default='Gopro_with_gt',
        help="Path to the input image folder.",
    )

    parser.add_argument(
        "--output_dir", type=str, default='results/', help="Output directory."
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
        default=10,
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
        default=0,
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
    output_dir = os.path.join(args.output_dir, args.dataset_name)

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


    logging.info(f"output dir = {output_dir}")

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

    pipe: B2FPipeline = B2FPipeline.from_pretrained(
        checkpoint_path, variant=variant, torch_dtype=dtype
    )

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        pass  # run without xformers

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
    with torch.no_grad():
        os.makedirs(output_dir, exist_ok=True)

        for i,rgb_path in enumerate(tqdm(rgb_filename_list, desc="Estimating depth", leave=True)):

            # Read input image
            input_image = Image.open(rgb_path)
            img_name = rgb_path.split('/')[-1]
            img_num = img_name[:-4]

            # Random number generator
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)

            # make save path
            os.makedirs(os.path.join(output_dir,img_num),exist_ok=True)
            ensemble_candidate_path = os.path.join(output_dir,img_num)

            p_img_num = int(img_num) + 2103
            p_img_num = f'{p_img_num:06}'
            
            # psudo-GT flow imgs
            future_gt = f'/workspace/data/Gopro_my/train/{img_num}/flow/flows/{img_num}.flo'
            past_gt = f'/workspace/data/Gopro_my/train/{p_img_num}/flow/flows/{p_img_num}.flo'

            future_gt_flow = flow_read(future_gt)
            future_gt_img = flow_to_rgb(future_gt_flow, max_radius=150)
            past_gt_flow = flow_read(past_gt)
            past_gt_img = flow_to_rgb(past_gt_flow, max_radius=150)

            # find dominat color
            future_dominant_color = get_dominant_color(future_gt_img)
            past_dominant_color = get_dominant_color(past_gt_img)

            # save pusdo-GT flow
            cv2.imwrite(os.path.join(ensemble_candidate_path,'0.png'),future_gt_img)
            cv2.imwrite(os.path.join(ensemble_candidate_path,'1.png'),past_gt_img)

            idx = 2
            counter = {'future':1, 'past':1}

            while True:
                print(img_num, counter)
                if counter['future'] == 10 and counter['past'] == 10:
                    break
                
                # Predict depth
                pipe_out = pipe(
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
                )


                for out_img in pipe_out:
                    gen_dominant_color = get_dominant_color(out_img)

                    distant_to_future = compare_colors(gen_dominant_color,future_dominant_color)
                    distant_to_past = compare_colors(gen_dominant_color, past_dominant_color)

                    if distant_to_future > distant_to_past:
                        if counter['future'] < 10:
                            counter['future'] += 1
                            cv2.imwrite(os.path.join(ensemble_candidate_path,f'{idx}.png'),out_img)
                            idx += 1
                        else:
                            continue
                    else:
                        if counter['past'] < 10:
                            counter['past'] +=1
                            cv2.imwrite(os.path.join(ensemble_candidate_path,f'{idx}.png'),out_img)
                            idx += 1
                        else:
                            continue



