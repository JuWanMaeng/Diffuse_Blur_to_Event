import logging
from typing import Dict, Optional, Union, List

import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    LCMScheduler,
    UNet2DConditionModel,
    DiTPipeline,
    DiTTransformer2DModel
)

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
    resize_max_res_with_padding
)

from torch.nn import Conv2d
from torch.nn.parameter import Parameter

class B2EPipeline_DIT(DiTPipeline):

    rgb_latent_scale_factor = 0.18215
    event_latent_scale_factor = 0.18215

    def __init__(
        self,
        transformer: DiTTransformer2DModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, LCMScheduler],
        default_processing_resolution = 960,
        scale_invariant: Optional[bool] = True,
        shift_invariant: Optional[bool] = True,
        default_denoising_steps: Optional[int] = None,

    ):
        super().__init__(
            vae = vae,
            scheduler = scheduler,
            transformer=transformer)
        
        self.register_modules(
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            # text_encoder=text_encoder,
            # tokenizer=tokenizer,
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

        # assert processing_res >= 0
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
        rgb, p_top, p_bottom, p_left, p_right = resize_max_res_with_padding(
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

        depth_preds = torch.concat(depth_pred_ls, dim=0)
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        depth_preds = depth_preds
        pred_uncert = None

        num_samples = depth_preds.shape[0]
        res = []
        for i in range(num_samples):
            depth_pred = depth_preds[i:i+1,:,:,:]
            pred_h, pred_w = depth_pred.shape[2], depth_pred.shape[3]

            depth_pred = depth_pred[:,:,p_top:pred_h - p_bottom, p_left:pred_w -p_right]
            

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
        rgb_latent = self.encode(rgb_in)
        B,C,H,W = rgb_latent.shape

        # Initial depth map (noise)
        event_latent = torch.randn(
            (B, 8, H, W),  # 8 channels for event latent
            device=device,
            dtype=self.dtype,
            generator=generator,
        )  # [B, 4, h, w]

        batch_size = rgb_in.shape[0]
        class_labels = torch.zeros(batch_size, dtype=torch.long, device=device)

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
            transformer_input = torch.cat(
                [rgb_latent, event_latent], dim=1
            )  # this order is important

            time_step = torch.tensor([t], device=device)
            # predict the noise residual
            noise_pred = self.transformer(
                transformer_input, time_step, class_labels
            ).sample  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            
            event_latent = self.scheduler.step(
                noise_pred, t, event_latent, generator=generator
            ).prev_sample

        event = self.decode_event(event_latent)

        # clip prediction
        event = torch.clip(event, -1.0, 1.0)

        return event

    def encode(self, flow_in: torch.Tensor) -> torch.Tensor:

        # encode
        h = self.vae.encoder(flow_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        flow_latent = mean * self.rgb_latent_scale_factor
        return flow_latent
    

    def decode_event(self, event_latent: torch.Tensor) -> torch.Tensor:

        # scale latent
        event_latent = event_latent / self.event_latent_scale_factor

        # split event_latent 
        event_latent_1, event_latent_2 = torch.chunk(event_latent, 2, dim=1)
        # decode
        z_1 = self.vae.post_quant_conv(event_latent_1)
        z_2 = self.vae.post_quant_conv(event_latent_2)

        z_1 = self.vae.decoder(z_1)
        z_2 = self.vae.decoder(z_2)

        stacked = torch.cat([z_1, z_2], dim=1)

        return stacked


    def replace_transformer_proj(self):  # setting when inference
        # 기존 프로젝션 레이어를 가져옴
        _weight = self.transformer.pos_embed.proj.weight.clone()  # [embed_dim, 4, patch_size, patch_size]
        _bias = self.transformer.pos_embed.proj.bias.clone()  # [embed_dim]

        # 기존 가중치를 repeat하여 12개의 채널로 확장
        _weight = _weight.repeat(1, 3, 1, 1)  # 4채널에서 12채널로 확장
        _weight *= 1 / 3.0  # 확장된 채널 크기를 보정

        # 기존 출력 채널(embed_dim)을 유지한 새로운 Conv2d 레이어 생성
        _n_proj_out_channels = self.transformer.pos_embed.proj.out_channels  # embed_dim
        _patch_size = self.transformer.pos_embed.proj.kernel_size[0]  # 패치 크기 가져오기

        _new_proj = Conv2d(
            in_channels=12,  # 새 입력 채널 크기
            out_channels=_n_proj_out_channels,  # 기존 출력 채널 크기
            kernel_size=(_patch_size, _patch_size),  # 기존 패치 크기 유지
            stride=_patch_size,  # 기존 stride 유지
            bias=True,
        )

        # 새로운 레이어에 가중치와 바이어스를 할당
        _new_proj.weight = Parameter(_weight)
        _new_proj.bias = Parameter(_bias)

        # 모델의 프로젝션 레이어 교체
        self.transformer.pos_embed.proj = _new_proj
        logging.info("transformer proj layer is replaced with 12 channels")
        logging.info("transformer config is updated")

        return