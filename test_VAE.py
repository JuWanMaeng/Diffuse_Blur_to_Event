import time
import os
import math
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from accelerate import Accelerator
from dataset.h5_image_dataset import H5ImageDataset, concatenate_h5_datasets
import matplotlib.pyplot as plt

def main():
    # Accelerate 초기화
    accelerator = Accelerator()
    device = accelerator.device

    # 사전 학습된 VAE 모델 로드 및 평가 모드 전환
    # model = AutoencoderKL.from_pretrained('/workspace/Marigold/checkpoint/stable-diffusion-2/vae')
    model = AutoencoderKL.from_pretrained('/workspace/data/AE-output-KL-pretrained/checkpoint-3000/aemodel')
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
            fmap, _ = model(event, return_dict=False)


        if torch.max(event) > 1:
            max_val = torch.max(torch.abs(event))
            event_01 = event/max_val

            max_val = torch.max(torch.abs(fmap))
            fmap_01 = fmap / max_val
        # RMSE 계산: sqrt(mean((원본 - 재구성)**2))
        else:
            fmap_01 = (fmap+1) / 2
            event_01 = (event+1) / 2

        mse = F.mse_loss(fmap_01, event_01, reduction="mean")
        rmse = torch.sqrt(mse)
        rmse_list.append(rmse.item())

########################
        # fmap_np = event.detach().cpu().numpy()[0]  # shape: (3, H, W)


        # # 채널별로 플롯
        # fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        # for i in range(3):
        #     axs[i].imshow(fmap_np[i], cmap='seismic', vmin=-1, vmax=1)
        #     axs[i].set_title(f"Channel {i}")
        #     axs[i].axis('off')

        # plt.tight_layout()
        # plt.savefig("fmap_3channels.png", bbox_inches='tight')
        # plt.close(fig)
###########################
        print(f"Step: {step} | RMSE: {rmse.item():.6f}")
        # break

    # 전체 테스트 데이터에 대한 평균 RMSE 출력
    avg_rmse = sum(rmse_list) / len(rmse_list) if rmse_list else float('nan')
    print(f"Average RMSE over test dataset: {avg_rmse:.6f}")

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    main()
