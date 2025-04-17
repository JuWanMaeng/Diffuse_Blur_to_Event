import time
import os
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from accelerate import Accelerator
from dataset.h5_image_dataset import H5ImageDataset, concatenate_h5_datasets
from src.model.NAFNet_Recon import NAFNetRecon

def main():
    # Accelerate 초기화
    accelerator = Accelerator()
    device = accelerator.device

    # 사전 학습된 NAFVAE(=NAFNetRecon) 모델 로드
    model = NAFNetRecon(
        img_channel=6,
        width=64,
        middle_blk_num=28,
        enc_blk_nums=[1,1,1],
        dec_blk_nums=[1,1,1],
        latent_dim=128
    )
    checkpoint = torch.load('/workspace/Marigold/checkpoint/NAF_VAE_128.pth')
    model.load_state_dict(checkpoint['params'])
    model.to(device)
    model.eval()

    # 테스트 데이터셋 로드
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

    # 실험할 noise 비율(a) 리스트
    noise_ratio_list = [i/100 for i in range(0, 21)]  # 0.00 ~ 0.20
    avg_latent_rmse_list = []

    for a in noise_ratio_list:
        print(f"\n[Noise Ratio (a): {a:.2f}]")
        latent_rmse_list = []

        for step, batch in enumerate(test_dataloader):
            event = batch["voxel"].to(device, non_blocking=True)  # clean input

            # 노이즈 추가
            noise = torch.randn_like(event)
            event_noisy = (1 - a) * event + a * noise
            event_noisy = torch.clamp(event_noisy, -1.0, 1.0)

            with torch.no_grad():
                # **latent 추출**
                latent_clean = model.encode(event)         # [1, latent_dim, H', W']
                latent_noisy = model.encode(event_noisy)   # same shape

            # Get min/max values to determine range
            min_clean = latent_clean.min()
            max_clean = latent_clean.max()
            min_noisy = latent_noisy.min() 
            max_noisy = latent_noisy.max()
            
            # Normalize latents to [0,1] range
            latent_clean = (latent_clean - min_clean) / (max_clean - min_clean)
            latent_noisy = (latent_noisy - min_noisy) / (max_noisy - min_noisy)
            # **latent 공간 RMSE 계산**
            mse_latent = F.mse_loss(latent_noisy, latent_clean, reduction="mean")
            rmse_latent = torch.sqrt(mse_latent)
            latent_rmse_list.append(rmse_latent.item())

            if step == 500:
                break

        # 노이즈 비율별 평균 latent-RMSE
        avg_latent = sum(latent_rmse_list) / len(latent_rmse_list)
        avg_latent_rmse_list.append(avg_latent)
        print(f">>> [a={a:.2f}] Avg latent-RMSE: {avg_latent:.6f}")

    # 결과 플롯
    plt.figure(figsize=(8,5))
    plt.plot(noise_ratio_list, avg_latent_rmse_list, marker='o')
    plt.title("Noise Ratio vs Average Latent RMSE")
    plt.xlabel("Noise Ratio (a)")
    plt.ylabel("Average Latent RMSE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("latent_rmse_vs_noise.png")
    print(">>> Plot saved as 'latent_rmse_vs_noise.png'")

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    main()
