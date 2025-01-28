import argparse
import logging
import os
import shutil
from datetime import datetime, timedelta
from dataset.h5_image_dataset import H5ImageDataset, concatenate_h5_datasets

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm


from marigold.b2e_pipeline_dit import B2EPipeline_DIT
from src.trainer import get_trainer_cls
from src.util.config_util import (
    find_value_in_omegaconf,
    recursive_load_config,
)
from src.util.depth_transform import (
    DepthNormalizerBase,
    get_depth_normalizer,
)
from src.util.logging_util import (
    config_logging,
    init_wandb,
    load_wandb_job_id,
    log_slurm_job_id,
    save_wandb_job_id,
    tb_logger,
)
from src.util.slurm_util import get_local_scratch_dir, is_on_slurm

if "__main__" == __name__:
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    t_start = datetime.now()
    print(f"start at {t_start}")

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(description="Train your cute model!")
    parser.add_argument(
        "--config",
        type=str,
        default="/workspace/Marigold/config/train_dit.yaml",
        help="Path to config file.",
    )
    parser.add_argument(
        "--resume_run",
        action="store",
        default=None,
        help="Path of checkpoint to be resumed. If given, will ignore --config, and checkpoint in the config",
    )
    parser.add_argument(
        "--output_dir", type=str, default='output/DIT', help="directory to save checkpoints"
    )
    parser.add_argument("--no_cuda", action="store_true", help="Do not use cuda.")
    parser.add_argument(
        "--exit_after",
        type=int,
        default=-1,
        help="Save checkpoint and exit after X minutes.",
    )
    parser.add_argument("--no_wandb", action="store_true", help="run without wandb")

    parser.add_argument(
        "--base_ckpt_dir",
        type=str,
        default='/workspace/Marigold/checkpoint/',
        help="directory of pretrained checkpoint",
    )
    parser.add_argument(
        "--add_datetime_prefix",
        action="store_true",
        help="Add datetime to the output folder name",
    )

    args = parser.parse_args()
    resume_run = args.resume_run
    output_dir = args.output_dir
    base_ckpt_dir = (
        args.base_ckpt_dir
        if args.base_ckpt_dir is not None
        else os.environ["BASE_CKPT_DIR"]
    )

    # -------------------- Initialization --------------------
    # Resume previous run
    if resume_run is not None:
        print(f"Resume run: {resume_run}")
        out_dir_run = os.path.dirname(os.path.dirname(resume_run))
        job_name = os.path.basename(out_dir_run)
        # Resume config file
        cfg = OmegaConf.load(os.path.join(out_dir_run, "config.yaml"))
    else:
        # Run from start
        cfg = recursive_load_config(args.config)
        # Full job name
        pure_job_name = os.path.basename(args.config).split(".")[0]
        # Add time prefix
        if args.add_datetime_prefix:
            job_name = f"{t_start.strftime('%y_%m_%d-%H_%M_%S')}-{pure_job_name}"
        else:
            job_name = pure_job_name

        # Output dir
        if output_dir is not None:
            # out_dir_run = os.path.join(output_dir, job_name)
            out_dir_run = os.path.join(output_dir)
        else:
            out_dir_run = os.path.join("./output", job_name)

        os.makedirs(out_dir_run, exist_ok=True)
        # os.makedirs(out_dir_run, exist_ok=False)

    cfg_data = cfg.dataset

    # Other directories
    out_dir_ckpt = os.path.join(out_dir_run, "checkpoint")
    if not os.path.exists(out_dir_ckpt):
        os.makedirs(out_dir_ckpt)
    out_dir_tb = os.path.join(out_dir_run, "tensorboard")
    if not os.path.exists(out_dir_tb):
        os.makedirs(out_dir_tb)
    out_dir_eval = os.path.join(out_dir_run, "evaluation")
    if not os.path.exists(out_dir_eval):
        os.makedirs(out_dir_eval)
    out_dir_vis = os.path.join(out_dir_run, "visualization")
    if not os.path.exists(out_dir_vis):
        os.makedirs(out_dir_vis)

    # -------------------- Logging settings --------------------
    config_logging(cfg.logging, out_dir=out_dir_run)
    logging.debug(f"config: {cfg}")

    # Initialize wandb
    if not args.no_wandb:
        if resume_run is not None:
            wandb_id = load_wandb_job_id(out_dir_run)
            wandb_cfg_dic = {
                "id": wandb_id,
                "resume": "must",
                **cfg.wandb,
            }
        else:
            wandb_cfg_dic = {
                "config": dict(cfg),
                "name": job_name,
                "mode": "online",
                **cfg.wandb,
            }
        wandb_cfg_dic.update({"dir": out_dir_run})
        wandb_run = init_wandb(enable=True, **wandb_cfg_dic)
        save_wandb_job_id(wandb_run, out_dir_run)
    else:
        init_wandb(enable=False)

    # Tensorboard (should be initialized after wandb)
    tb_logger.set_dir(out_dir_tb)

    log_slurm_job_id(step=0)

    # -------------------- Device --------------------
    cuda_avail = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if cuda_avail else "cpu")
    logging.info(f"device = {device}")

    # -------------------- Snapshot of code and config --------------------
    if resume_run is None:
        _output_path = os.path.join(out_dir_run, "config.yaml")
        with open(_output_path, "w+") as f:
            OmegaConf.save(config=cfg, f=f)
        logging.info(f"Config saved to {_output_path}")
        # Copy and tar code on the first run
        _temp_code_dir = os.path.join(out_dir_run, "code_tar")
        _code_snapshot_path = os.path.join(out_dir_run, "code_snapshot.tar")
        os.system(
            f"rsync --relative -arhvz --quiet --filter=':- .gitignore' --exclude '.git' . '{_temp_code_dir}'"
        )
        os.system(f"tar -cf {_code_snapshot_path} {_temp_code_dir}")
        os.system(f"rm -rf {_temp_code_dir}")
        logging.info(f"Code snapshot saved to: {_code_snapshot_path}")


    # -------------------- Gradient accumulation steps --------------------
    eff_bs = cfg.dataloader.effective_batch_size
    accumulation_steps = eff_bs / cfg.dataloader.max_train_batch_size
    assert int(accumulation_steps) == accumulation_steps
    accumulation_steps = int(accumulation_steps)

    logging.info(
        f"Effective batch size: {eff_bs}, accumulation steps: {accumulation_steps}"
    )

    # -------------------- Data --------------------
    loader_seed = cfg.dataloader.seed
    if loader_seed is None:
        loader_generator = None
    else:
        loader_generator = torch.Generator().manual_seed(loader_seed)

    # Training dataset
    # train_dataset = Flow_Blur_dataset_C(
    #     blur_txt_path='/workspace/Marigold/dataset/train/train_vimeo+monkaa.txt',
    #     seed=loader_seed  # Seed를 설정하여 무작위성 고정
    # )
    opt = {'crop_size':(544,960),
           'use_flip': True,
           'folder_path' : '/workspace/data/GOPRO/train'
           }
    
    train_dataset = concatenate_h5_datasets(H5ImageDataset, opt)
    # train_dataset = H5ImageDataset(opt, data_path,)


    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.dataloader.max_train_batch_size,
        num_workers=cfg.dataloader.num_workers,
        shuffle=True,
        generator=loader_generator,
    )

    # -------------------- Model --------------------
    _pipeline_kwargs = cfg.pipeline.kwargs if cfg.pipeline.kwargs is not None else {}
    model = B2EPipeline_DIT.from_pretrained(
        os.path.join(base_ckpt_dir, cfg.model.pretrained_path), **_pipeline_kwargs
    )

    ### If you want to train Unet from scratch ###
    # def reset_unet_weights(unet):
    #     for layer in unet.parameters():
    #         if layer.requires_grad:
    #             torch.nn.init.normal_(layer, mean=0.0, std=0.02)

    # reset_unet_weights(model.unet)


    # -------------------- Trainer --------------------
    # Exit time
    if args.exit_after > 0:
        t_end = t_start + timedelta(minutes=args.exit_after)
        logging.info(f"Will exit at {t_end}")
    else:
        t_end = None

    trainer_cls = get_trainer_cls(cfg.trainer.name)
    logging.debug(f"Trainer: {trainer_cls}")
    trainer = trainer_cls(
        cfg=cfg,
        model=model,
        train_dataloader=train_loader,
        device=device,
        base_ckpt_dir=base_ckpt_dir,
        out_dir_ckpt=out_dir_ckpt,
        out_dir_eval=out_dir_eval,
        out_dir_vis=out_dir_vis,
        accumulation_steps=accumulation_steps,
        val_dataloaders=None,
        vis_dataloaders=None,
    )

    # -------------------- Checkpoint --------------------
    if resume_run is not None:
        trainer.load_checkpoint(
            resume_run, load_trainer_state=True, resume_lr_scheduler=True
        )

    # -------------------- Training & Evaluation Loop --------------------
    try:
        trainer.train(t_end=t_end)
    except Exception as e:
        logging.exception(e)
