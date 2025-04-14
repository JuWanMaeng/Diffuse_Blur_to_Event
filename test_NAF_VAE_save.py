import numpy as np
import os
import torch
from accelerate import Accelerator
from dataset.h5_image_dataset import H5ImageDataset, concatenate_h5_datasets
from src.model.NAFNet_Recon import NAFNetRecon
from src.model.NAFNet_recon_KL import NAFNetRecon_VAE

def main():
    # Accelerate 초기화
    accelerator = Accelerator()
    device = accelerator.device

    # 사전 학습된 VAE 모델 로드
    model = NAFNetRecon(img_channel=6, width=64, middle_blk_num=28, enc_blk_nums=[1,1,1], dec_blk_nums=[1,1,1], latent_dim=128)
    # model = NAFNetRecon_VAE(img_channel=6, width=64, middle_blk_num=28, enc_blk_nums=[1,1,1], dec_blk_nums=[1,1,1], latent_dim=128)
    weight = 'checkpoint/NAF_VAE_128.pth'
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
    noise_ratio_list = [0.05]
    print(noise_ratio_list)
    avg_rmse_list = []

    
    # 각 노이즈 비율마다 RMSE 계산
    for a in noise_ratio_list:
        save_folder = f'results/debug/{a}'
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

            fmap = fmap[0]
            event = event[0]    

            save_path = os.path.join(save_folder,batch['path'][0])

            fmap = np.array(fmap.detach().cpu())
            event = np.array(event.cpu())
            
            os.makedirs(save_path,exist_ok=True)
            np.save(os.path.join(save_path,'out.npy'), fmap)
            np.save(os.path.join(save_path,'gt.npy'), event)




if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    main()
