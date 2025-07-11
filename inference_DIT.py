import argparse
import logging
import os
from glob import glob

import numpy as np
import torch
import torchvision as v
from PIL import Image
from tqdm.auto import tqdm
from dataset.h5_image_dataset import H5ImageDataset, concatenate_h5_datasets

from marigold.b2e_pipeline_dit import B2EPipeline_DIT
from torch.utils.data import DataLoader
from event_metric import metric_and_output

EXTENSION_LIST = [".jpg", ".jpeg", ".png"]


if "__main__" == __name__:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using Marigold."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint/my_DIT",
        help="Checkpoint path or hub name.",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default='Gopro_Event_Train',
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

    opt = {'crop_size': None,
           'use_flip' : False,
           'folder_path' : '/workspace/data/GOPRO/train',
           }
    
    dataset = concatenate_h5_datasets(H5ImageDataset, opt)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
    )

    # -------------------- Model --------------------
    dtype = torch.float32
    variant = None

    pipe: B2EPipeline_DIT = B2EPipeline_DIT.from_pretrained(
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
    total_rmse = 0

    with open('results/Gopro_Event_Train.txt','a', buffering=1) as f:
        with torch.no_grad():
            # os.makedirs(output_dir, exist_ok=True)

            for idx,data in enumerate(tqdm(dataloader, desc="Estimating depth", leave=True)):
                # Read input image
                input_image = data['frame']
                img_path = data['path']

                # Random number generator
                if seed is None:
                    generator = None
                else:
                    generator = torch.Generator(device=device)
                    generator.manual_seed(seed)

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

                # save output folder
                os.makedirs(os.path.join(output_dir, img_path[0]),exist_ok=True)


                gt_event = data['voxel'][0]  # [6,H,W]
                gt_event = np.array(gt_event)

        

                min_navie_rmse, min_reversed_rmse, once_rmse, avg_rmse, best_pred = metric_and_output(pipe_out,gt_event)
                best_rmse = min(min_navie_rmse, min_reversed_rmse)
                total_rmse += best_rmse

                f.write(f'{img_path[0]}: {min_navie_rmse:.3f} {min_reversed_rmse:.3f} {once_rmse:.3f} {avg_rmse:.3f}\n')
                np.save(os.path.join(output_dir, img_path[0], 'out'), best_pred)


            print(total_rmse/len(dataloader))



                

