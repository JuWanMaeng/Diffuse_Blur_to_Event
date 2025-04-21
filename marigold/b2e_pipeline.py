import logging
from typing import Dict, Optional, Union

import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
    UNet2DConditionModel,
)
from src.model.NAFNet_recon_KL import NAFNetRecon_VAE
from src.model.NAFNet_Recon import NAFNetRecon
from diffusers.utils import BaseOutput
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from .util.batchsize import find_batch_size
from .util.ensemble import ensemble_depth
from .util.image_util import (
    chw2hwc,
    colorize_depth_maps,
    get_tv_resample_method,
    resize_max_res,
)

import cv2, os
import matplotlib.pyplot as plt



class B2EPipeline(DiffusionPipeline):

    rgb_latent_scale_factor = 0.18215
    event_latent_scale_factor = 0.18215

    def __init__(
        self,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, LCMScheduler],
        text_encoder: CLIPTextModel,
        vae : AutoencoderKL,
        tokenizer: CLIPTokenizer,
        scale_invariant: Optional[bool] = True,
        shift_invariant: Optional[bool] = True,
        default_denoising_steps: Optional[int] = None,
        default_processing_resolution: Optional[int] = None,
    ):
        super().__init__()
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,

        )
        self.register_to_config(
            scale_invariant=scale_invariant,
            shift_invariant=shift_invariant,
            default_denoising_steps=default_denoising_steps,
            default_processing_resolution=default_processing_resolution,
        )

        self.scale_invariant = scale_invariant
        self.shift_invariant = shift_invariant
        self.default_denoising_steps = default_denoising_steps
        self.default_processing_resolution = default_processing_resolution
        self.empty_text_embed = None

        self.event_vae =  NAFNetRecon(img_channel=6, width=64, middle_blk_num=28, 
                                      enc_blk_nums=[1,1,1], dec_blk_nums=[1,1,1],latent_dim=8)
        weight = 'checkpoint/NAF_VAE_8.pth'
        checkpoint = torch.load(weight)
        self.event_vae.load_state_dict(checkpoint['params'])
        self.event_vae = self.event_vae.cuda()
        self.event_vae.eval()


    @torch.no_grad()
    def __call__(
        self,
        input_image: Union[Image.Image, torch.Tensor],
        denoising_steps: Optional[int] = None,
        ensemble_size: int = 5,
        processing_res: Optional[int] = None,
        match_input_res: bool = True,
        resample_method: str = "bilinear",
        batch_size: int = 0,
        generator: Union[torch.Generator, None] = None,
        color_map: str = "Spectral",
        show_progress_bar: bool = True,
        ensemble_kwargs: Dict = None,
    ):

        # Model-specific optimal default values leading to fast and reasonable results.
        if denoising_steps is None:
            denoising_steps = self.default_denoising_steps
        if processing_res is None:
            processing_res = self.default_processing_resolution

        assert processing_res >= 0
        assert ensemble_size >= 1

        # Check if denoising step is reasonable
        self._check_inference_step(denoising_steps)

        resample_method: InterpolationMode = get_tv_resample_method(resample_method)

        # ----------------- Image Preprocess -----------------
        # Convert to torch tensor
        if isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
            # convert to torch tensor [H, W, rgb] -> [rgb, H, W]
            rgb = pil_to_tensor(input_image)
            rgb = rgb.unsqueeze(0)  # [1, rgb, H, W]
        elif isinstance(input_image, torch.Tensor):
            rgb = input_image
        else:
            raise TypeError(f"Unknown input type: {type(input_image) = }")
        input_size = rgb.shape
        assert (
            4 == rgb.dim() and 3 == input_size[-3]
        ), f"Wrong input shape {input_size}, expected [1, rgb, H, W]"

        # Resize image
        if processing_res > 0:
            rgb = resize_max_res(
                rgb,
                max_edge_resolution=processing_res,
                resample_method=resample_method,
            )

        # Normalize rgb values
        if torch.min(rgb) < 0:
            rgb_norm = rgb
        else:
            rgb_norm: torch.Tensor = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]

        rgb_norm = rgb_norm.to(self.dtype)
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        # ----------------- Predicting event -----------------
        # Batch repeated input image
        duplicated_rgb = rgb_norm.expand(ensemble_size, -1, -1, -1)
        single_rgb_dataset = TensorDataset(duplicated_rgb)
        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = find_batch_size(
                ensemble_size=ensemble_size,
                input_res=max(rgb_norm.shape[1:]),
                dtype=self.dtype,
            )

        single_rgb_loader = DataLoader(
            single_rgb_dataset, batch_size=_bs, shuffle=False
        )

        # Predict flow maps (batched)
        depth_pred_ls = []
        if show_progress_bar:
            iterable = tqdm(
                single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False, ncols=80
            )
        else:
            iterable = single_rgb_loader
        for batch in iterable:
            (batched_img,) = batch
            depth_pred_raw = self.single_infer(
                rgb_in=batched_img,
                num_inference_steps=denoising_steps,
                show_pbar=show_progress_bar,
                generator=generator,
            )
            depth_pred_ls.append(depth_pred_raw.detach())

            # debug 시각화
            save_dir = 'debug/infer_debug'
            gen_event = depth_pred_raw[0].cpu().numpy()
            maxv = np.max(np.abs(gen_event)); gen_event /= maxv if maxv>0 else 1
            fig, axs = plt.subplots(2, 3, figsize=(20, 10))
            for ch in range(6):
                axs.ravel()[ch].imshow(gen_event[ch],cmap='seismic',vmin=-1,vmax=1)
                axs.ravel()[ch].axis('off')
            fig.suptitle(f"Step={denoising_steps}")
            plt.tight_layout(rect=[0,0,1,0.96])
            os.makedirs(save_dir,exist_ok=True)
            plt.savefig(os.path.join(save_dir,f"event_debug_{denoising_steps:04d}.png"))
            plt.close(fig)

        depth_preds = torch.concat(depth_pred_ls, dim=0)
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        depth_preds = depth_preds
        pred_uncert = None

        num_samples = depth_preds.shape[0]
        res = []
        for i in range(num_samples):
            depth_pred = depth_preds[i:i+1,:,:,:]

            # Resize back to original resolution
            if match_input_res:
                depth_pred = resize(
                    depth_pred,
                    input_size[-2:],
                    interpolation=resample_method,
                    antialias=True,
                )

            # Convert to numpy
            depth_pred = depth_pred.squeeze()
            depth_pred = depth_pred.cpu().numpy()
            if pred_uncert is not None:
                pred_uncert = pred_uncert.squeeze().cpu().numpy()

            # Clip output range
            depth_pred = depth_pred.transpose(1,2,0)
            res.append(depth_pred)
            

        return res

    def _check_inference_step(self, n_step: int) -> None:
        """
        Check if denoising step is reasonable
        Args:
            n_step (`int`): denoising steps
        """
        assert n_step >= 1

        if isinstance(self.scheduler, DDIMScheduler):
            if n_step < 10:
                logging.warning(
                    f"Too few denoising steps: {n_step}. Recommended to use the LCM checkpoint for few-step inference."
                )
        elif isinstance(self.scheduler, LCMScheduler):
            if not 1 <= n_step <= 4:
                logging.warning(
                    f"Non-optimal setting of denoising steps: {n_step}. Recommended setting is 1-4 steps."
                )
        else:
            raise RuntimeError(f"Unsupported scheduler type: {type(self.scheduler)}")

    def encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)

    @torch.no_grad()
    def single_infer(
        self,
        rgb_in: torch.Tensor,
        num_inference_steps: int,
        generator: Union[torch.Generator, None],
        show_pbar: bool,
    ) -> torch.Tensor:

        device = self.device
        rgb_in = rgb_in.to(device)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]

        # Encode image
        rgb_latent = self.encode_image(rgb_in)
        B,C,H,W = rgb_latent.shape

        # Initial depth map (noise)
        event_latent = torch.randn(
            (B, 128, H, W),  # 8 channels for event latent
            device=device,
            dtype=self.dtype,
            generator=generator,
        )  # [B, 4, h, w]

        # Batched empty text embedding
        if self.empty_text_embed is None:
            self.encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (rgb_latent.shape[0], 1, 1)
        ).to(device)  # [B, 2, 1024]

        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
                ncols=80
            )
        else:
            iterable = enumerate(timesteps)

        for i, t in iterable:
            unet_input = torch.cat(
                [rgb_latent, event_latent], dim=1
            )  # this order is important

            # predict the noise residual
            noise_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_empty_text_embed
            ).sample  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            event_latent = self.scheduler.step(
                noise_pred, t, event_latent, generator=generator
            ).prev_sample

        event = self.decode_event(event_latent)

        # clip prediction
        # event = torch.clip(event, -1.0, 1.0)

        return event

    def encode_image(self, image_in: torch.Tensor) -> torch.Tensor:

        # encode
        h = self.vae.encoder(image_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        image_latent = mean * self.rgb_latent_scale_factor
        
        return image_latent
    
    # def encode_event(self, event_in: torch.Tensor, key=None) -> torch.Tensor:
        
    #     with torch.no_grad():
    #         h,_ = self.event_vae.encode(event_in)
    #     # scale latent
    #     event_latent = h * self.rgb_latent_scale_factor
    #     return event_latent

    # def decode_event(self, event_latent: torch.Tensor) -> torch.Tensor:

    #     # scale latent
    #     # event_latent = event_latent / self.event_latent_scale_factor

    #     with torch.no_grad():
    #         z = self.event_vae.decode(event_latent)

    #     return z



############
# KL 안쓴 버전임
    def encode_event(self, event_in: torch.Tensor, key=None) -> torch.Tensor:
        
        with torch.no_grad():
            mean  = self.event_vae.encode(event_in)

        # scale latent
        event_latent = mean * self.event_latent_scale_factor
        return event_latent

    def decode_event(self, event_latent: torch.Tensor) -> torch.Tensor:

        # scale latent
        event_latent = event_latent / self.event_latent_scale_factor

        with torch.no_grad():
            z = self.event_vae.decode(event_latent)

        return z
    

    @torch.no_grad()
    def infer_and_save_debug(
        self,
        rgb_in: torch.Tensor,
        num_inference_steps: int,
        generator: Union[torch.Generator, None] = None,
        save_dir: str = "debug/infer_debug"
    ):
        """
        single_infer를 통해 생성된 event를 6채널 subplot 이미지로 저장.
        Args:
            rgb_in:       [B,3,H,W] 입력 RGB 텐서 (batch size B>=1)
            num_steps:    디노이징 단계 수
            generator:    랜덤 시드용 torch.Generator (선택)
            save_dir:     결과 이미지를 저장할 폴더
        """
        os.makedirs(save_dir, exist_ok=True)
        self.model.to(self.device)

        # 1) single_infer 호출
        event = self.single_infer(
            rgb_in=rgb_in,
            num_inference_steps=num_inference_steps,
            generator=generator,
            show_pbar=False
        )  # [B, C, H, W]
        
        # 2) 첫 번째 배치만 사용
        gen_event = event[0].cpu().numpy()  # (C, H, W)
        
        # 3) 값 정규화 (극성 파악용)
        maxv = np.max(np.abs(gen_event))
        if maxv > 0:
            gen_event = gen_event / maxv
        
        # 4) 6채널을 2x3 subplot으로 그려서 저장
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        axs = axs.ravel()
        for ch in range(gen_event.shape[0]):
            axs[ch].imshow(gen_event[ch], cmap="seismic", vmin=-1, vmax=1)
            axs[ch].set_title(f"channel {ch}", fontsize=12)
            axs[ch].axis("off")
        fig.suptitle(f"Inference Debug (steps={num_inference_steps})", fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # 5) 파일명에 4자리 0패딩 적용
        fname = f"event_debug_{num_inference_steps:04d}.png"
        save_path = os.path.join(save_dir, fname)
        plt.savefig(save_path)
        plt.close(fig)

        print(f">>> Saved debug image: {save_path}")
        return event