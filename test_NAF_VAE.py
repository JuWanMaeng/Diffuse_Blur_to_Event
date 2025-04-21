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
from tqdm import tqdm
import numpy as np

def main():
    # Accelerate 초기화
    accelerator = Accelerator()
    device = accelerator.device

    # 사전 학습된 VAE 모델 로드 및 평가 모드 전환
    # model =  NAFNetRecon(img_channel=6, width=64, middle_blk_num=28, enc_blk_nums=[1,1,1], dec_blk_nums=[1,1,1],latent_dim=128)
    model =  NAFNetRecon_VAE(img_channel=6, width=64, middle_blk_num=28, enc_blk_nums=[1,1,1], dec_blk_nums=[1,1,1],latent_dim=128)
    weight = 'checkpoint/NAF_VAE_1e-5.pth'
    checkpoint = torch.load(weight)

    model.load_state_dict(checkpoint['params'])
    model.to(device)
    model.eval()

    # 테스트 데이터셋 로드 (GOPRO test 폴더 사용)
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

    # 테스트 루프: 각 배치마다 재구성 결과와 원본 간의 RMSE 계산
    rmse_list = []
    for step, batch in enumerate(test_dataloader):
        # "voxel" 키에 해당하는 데이터를 가져옵니다.
        event = batch["voxel"].to(device, non_blocking=True)

        # 모델 추론: fmap은 재구성된 결과
        with torch.no_grad():
            fmap = model(event)[0]

        # RMSE 계산: sqrt(mean((원본 - 재구성)**2))
        fmap = (fmap+1) / 2
        event = (event+1) / 2

        mse = F.mse_loss(fmap, event, reduction="mean")
        rmse = torch.sqrt(mse)
        rmse_list.append(rmse.item())

        print(f"Step: {step} | RMSE: {rmse.item():.6f}")

    # 전체 테스트 데이터에 대한 평균 RMSE 출력
    avg_rmse = sum(rmse_list) / len(rmse_list) if rmse_list else float('nan')
    print(f"Average RMSE over test dataset: {avg_rmse:.6f}")


def cal_mean_std():
    # 1) Accelerate 초기화
    accelerator = Accelerator()
    device = accelerator.device


    NAFVAE = NAFNetRecon(
        img_channel=6, width=64,
        middle_blk_num=28, enc_blk_nums=[1,1,1], dec_blk_nums=[1,1,1],
        latent_dim=128
    )
    checkpoint = torch.load('/workspace/Marigold/checkpoint/NAF_VAE_128.pth')
    NAFVAE.load_state_dict(checkpoint['params'])
    NAFVAE.to(device).eval()

    # 3) 데이터로더 준비
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

    # 4) 전체 데이터셋에 대한 통계 누적 변수
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    # 5) 배치별 NAFVAE 출력 누적
    for step, batch in enumerate(tqdm(test_dataloader, ncols=80)):
        event = batch["voxel"].to(device, non_blocking=True)

        with torch.no_grad():
            fmap = NAFVAE(event)[0]      # (C, H, W)

        arr = fmap.cpu().numpy().ravel()
        total_sum += arr.sum()
        total_sq_sum += np.sum(arr * arr)
        total_count += arr.size

    # 6) 최종 평균·표준편차 계산 및 출력
    mean = total_sum / total_count
    var = total_sq_sum / total_count - mean**2
    std = np.sqrt(var)

    print(f"Dataset-wide NAFVAE output → mean: {mean:.6f}, std: {std:.6f}")



if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    cal_mean_std()
