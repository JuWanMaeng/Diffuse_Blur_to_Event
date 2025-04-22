import time
import os
import math
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from accelerate import Accelerator
from dataset.h5_image_dataset import H5ImageDataset, concatenate_h5_datasets
from src.model.NAFNet_Recon import NAFNetRecon
from src.model.NAFNet_recon_KL import NAFNetRecon_VAE
import matplotlib.pyplot as plt
import numpy as np


def encode_rgb(model, rgb_in):
    """
    Encode RGB image into latent.

    Args:
        rgb_in (`torch.Tensor`):
            Input RGB image to be encoded.

    Returns:
        `torch.Tensor`: Image latent.
    """
    # encode
    h = model.encoder(rgb_in)
    moments = model.quant_conv(h)
    mean, logvar = torch.chunk(moments, 2, dim=1)
    # scale latent
    rgb_latent = mean * 0.18215
    return rgb_latent


def main():
    # Accelerate 초기화
    accelerator = Accelerator()
    device = accelerator.device

    # 사전 학습된 VAE 모델 로드 및 평가 모드 전환
    # model =  NAFNetRecon(img_channel=6, width=64, middle_blk_num=28, enc_blk_nums=[1,1,1], dec_blk_nums=[1,1,1],latent_dim=128)
    VAE = AutoencoderKL.from_pretrained('/workspace/Marigold/checkpoint/stable-diffusion-2/vae')
    VAE.to(device)
    VAE.eval()

    NAFVAE =  NAFNetRecon(img_channel=6, width=64, middle_blk_num=28, enc_blk_nums=[1,1,1], dec_blk_nums=[1,1,1],latent_dim=8)
    weight = '/workspace/Marigold/checkpoint/NAF_VAE_8.pth'
    checkpoint = torch.load(weight)

    NAFVAE.load_state_dict(checkpoint['params'])
    NAFVAE.to(device)
    NAFVAE.eval()

    # 테스트 데이터셋 로드 (GOPRO test 폴더 사용)
    opt = {
        'crop_size': None,
        'use_flip': False,
        'folder_path': '/workspace/data/GOPRO/train'
    }
    test_dataset = concatenate_h5_datasets(H5ImageDataset, opt)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=1,
    )

    os.makedirs("plots/NAFVAE", exist_ok=True)
    os.makedirs("plots/VAE", exist_ok=True)
    os.makedirs("plots/NAFVAE_norm", exist_ok=True)

    for step, batch in enumerate(test_dataloader):
        # 1) 데이터 준비
        event = batch["voxel"].to(device, non_blocking=True)
        with torch.no_grad():
            NAFVAE_fmap = NAFVAE(event)[0]
            VAE_fmap_1 = encode_rgb(VAE, event[:, :3, :, :])
            VAE_fmap_2 = encode_rgb(VAE, event[:, 3:, :, :])
        VAE_fmap = torch.cat([VAE_fmap_1, VAE_fmap_2], dim=0)

        # 2) NumPy로 변환
        NAFVAE_np = NAFVAE_fmap.cpu().numpy().ravel()
        VAE_np    = VAE_fmap.cpu().numpy().ravel()

        # 3) 원시 통계값 계산·출력
        naf_mu, naf_sigma = -0.000058,0.043284
        vae_mu, vae_sigma = VAE_np.mean(),    VAE_np.std()
        print(f"Step {step:04d} NAFVAE raw → mean: {naf_mu:.4f}, std: {naf_sigma:.4f}")
        print(f"Step {step:04d} VAE    raw → mean: {vae_mu:.4f}, std: {vae_sigma:.4f}")

        # 4) 정규화 및 통계 출력
        NAFVAE_norm = (NAFVAE_np - naf_mu) / naf_sigma
        norm_mu, norm_sigma = NAFVAE_norm.mean(), NAFVAE_norm.std()
        print(f"Step {step:04d} NAFVAE norm → mean: {norm_mu:.4f}, std: {norm_sigma:.4f}")

        # 5) 히스토그램 그리기·저장
        # 5-1) NAFVAE raw
        plt.figure(figsize=(6,4))
        plt.hist(NAFVAE_np, bins=100, alpha=0.7, range=(-5,5))
        plt.title(f"Step {step:04d} | NAFVAE Raw\nμ={naf_mu:.4f}, var={naf_sigma**2:.6f}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.xlim(-5,5)
        plt.tight_layout()
        plt.savefig(f"plots/NAFVAE/NAFVAE_hist_step_{step:04d}.png", dpi=150)
        plt.close()

        # VAE raw 히스토그램
        plt.figure(figsize=(6,4))
        plt.hist(VAE_np, bins=100, alpha=0.7, range=(-5,5))
        plt.title(f"Step {step:04d} | VAE Raw\nμ={vae_mu:.4f}, var={vae_sigma**2:.6f}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.xlim(-5,5)
        plt.tight_layout()
        plt.savefig(f"plots/VAE/VAE_hist_step_{step:04d}.png", dpi=150)
        plt.close()

        # NAFVAE normalized 히스토그램
        plt.figure(figsize=(6,4))
        plt.hist(NAFVAE_norm, bins=100, alpha=0.7, range=(-5,5))
        plt.title(f"Step {step:04d} | NAFVAE Norm\nμ={norm_mu:.4f}, var={norm_sigma**2:.6f}")
        plt.xlabel("Value ((x−μ)/σ)")
        plt.ylabel("Frequency")
        plt.xlim(-5,5)
        plt.tight_layout()
        plt.savefig(f"plots/NAFVAE_norm/NAFVAE_norm_hist_step_{step:04d}.png", dpi=150)
        plt.close()

        # print(f"Saved: {out_naf} | {out_vae} | {out_norm}\n")



if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    main()