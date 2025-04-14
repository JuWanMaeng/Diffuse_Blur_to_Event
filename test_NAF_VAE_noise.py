import time
import os
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from diffusers import AutoencoderKL
from accelerate import Accelerator
from dataset.h5_image_dataset import H5ImageDataset, concatenate_h5_datasets
from src.model.NAFNet_Recon import NAFNetRecon
import matplotlib.ticker as ticker

def main():
    # Accelerate 초기화
    accelerator = Accelerator()
    device = accelerator.device

    # 사전 학습된 VAE 모델 로드
    model = NAFNetRecon(img_channel=6, width=64, middle_blk_num=28, enc_blk_nums=[1,1,1], dec_blk_nums=[1,1,1], latent_dim=128)
    weight = '/workspace/Marigold/checkpoint/NAF_VAE_128.pth'
    checkpoint = torch.load(weight)
    model.load_state_dict(checkpoint['params'])
    model.to(device)
    model.eval()

    # 테스트 데이터셋 로드
    opt = {
        'crop_size': None,
        'use_flip': False,
        'folder_path': '/workspace/data/GOPRO/test'
    }
    test_dataset = concatenate_h5_datasets(H5ImageDataset, opt)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=1,
    )

    # 실험할 noise 비율(a) 리스트
    noise_ratio_list = [i/100 for i in range(0,21)]
    print(noise_ratio_list)
    avg_rmse_list = []

    # 각 노이즈 비율마다 RMSE 계산
    for a in noise_ratio_list:
        print(f"\n[Noise Ratio (a): {a}]")
        rmse_list = []
        for step, batch in enumerate(test_dataloader):
            event = batch["voxel"].to(device, non_blocking=True)

            # (1-a)*event + a*noise 로 노이즈 추가
            noise = torch.randn_like(event)
            event_noisy = (1 - a) * event + a * noise
            event_noisy = torch.clamp(event_noisy, -1.0, 1.0)

            with torch.no_grad():
                fmap = model(event_noisy)

            fmap = (fmap + 1) / 2
            event_clean = (event + 1) / 2

            mse = F.mse_loss(fmap, event_clean, reduction="mean")
            rmse = torch.sqrt(mse)
            rmse_list.append(rmse.item())

            print(f"Step: {step} | RMSE: {rmse.item():.6f}")


        avg_rmse = sum(rmse_list) / len(rmse_list) if rmse_list else float('nan')
        avg_rmse_list.append(avg_rmse)
        print(f">>> [Noise Ratio (a): {a}] Average RMSE: {avg_rmse:.6f}")


    # 결과 플롯 저장
    plt.figure(figsize=(8, 5))
    plt.plot(noise_ratio_list, avg_rmse_list, marker='o')
    plt.title("Noise Ratio vs Average RMSE")
    plt.xlabel("Noise Ratio (a)")
    plt.ylabel("Average RMSE")
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("rmse_vs_noise.png")
    print(">>> Plot saved as 'rmse_vs_noise.png'")

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
