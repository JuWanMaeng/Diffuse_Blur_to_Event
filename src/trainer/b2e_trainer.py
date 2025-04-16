import logging
import os
import shutil
from datetime import datetime
from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F

from diffusers import DDPMScheduler
from omegaconf import OmegaConf
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image


from marigold.b2e_pipeline import B2EPipeline

from src.util import metric
from src.util.data_loader import skip_first_batches
from src.util.logging_util import tb_logger, eval_dic_to_text
from src.util.loss import get_loss
from src.util.lr_scheduler import IterExponential
from src.util.metric import MetricTracker
from src.util.multi_res_noise import multi_res_noise_like
from src.util.alignment import align_depth_least_square
from src.util.seeding import generate_seed_sequence

import wandb
import matplotlib.pyplot as plt


class B2ETrainer:
    def __init__(
        self,
        cfg: OmegaConf,
        model: B2EPipeline,
        train_dataloader: DataLoader,
        device,
        base_ckpt_dir,
        out_dir_ckpt,
        out_dir_eval,
        out_dir_vis,
        accumulation_steps: int,
        val_dataloaders: List[DataLoader] = None,
        vis_dataloaders: List[DataLoader] = None,
    ):
        self.cfg: OmegaConf = cfg
        self.model: B2EPipeline = model
        self.device = device
        self.seed: Union[int, None] = (
            self.cfg.trainer.init_seed
        )  # used to generate seed sequence, set to `None` to train w/o seeding
        self.out_dir_ckpt = out_dir_ckpt
        self.out_dir_eval = out_dir_eval
        self.out_dir_vis = out_dir_vis
        self.train_loader: DataLoader = train_dataloader
        self.val_loaders: List[DataLoader] = val_dataloaders
        self.vis_loaders: List[DataLoader] = vis_dataloaders
        self.accumulation_steps: int = accumulation_steps


        self.use_gan = False


        self.use_consistency_loss = False

        # Adapt input layers
        if 12 != self.model.unet.config["in_channels"]:
            self._replace_unet_conv_in()

        if 8 != self.model.unet.config["out_channels"]:
            self._replace_unet_conv_out()

        # Encode empty text prompt
        self.model.encode_empty_text()
        self.empty_text_embed = self.model.empty_text_embed.detach().clone().to(device)

        self.model.unet.enable_xformers_memory_efficient_attention()

        # Trainability
        self.model.vae.requires_grad_(False)
        self.model.text_encoder.requires_grad_(False)
        # self.model.img_vae.reguires_grad(False)
        self.model.unet.requires_grad_(True)

        # Optimizer !should be defined after input layer is adapted
        lr = self.cfg.lr
        self.optimizer = Adam(self.model.unet.parameters(), lr=lr)
        

        # LR scheduler
        lr_func = IterExponential(
            total_iter_length=self.cfg.lr_scheduler.kwargs.total_iter,
            final_ratio=self.cfg.lr_scheduler.kwargs.final_ratio,
            warmup_steps=self.cfg.lr_scheduler.kwargs.warmup_steps,
        )
        self.lr_scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=lr_func)

        if self.use_gan:
            self.discriminator_scheduler = LambdaLR(
                optimizer=self.discriminator_optimizer,
                lr_lambda=lr_func
    )
        # Loss
        self.loss = get_loss(loss_name=self.cfg.loss.name, **self.cfg.loss.kwargs)

        # Training noise scheduler
        self.training_noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(
            os.path.join(
                base_ckpt_dir,
                cfg.trainer.training_noise_scheduler.pretrained_path,
                "scheduler",
            )
        )
        self.prediction_type = self.training_noise_scheduler.config.prediction_type
        assert (
            self.prediction_type == self.model.scheduler.config.prediction_type
        ), "Different prediction types"
        self.scheduler_timesteps = (
            self.training_noise_scheduler.config.num_train_timesteps
        )

        # Eval metrics
        self.metric_funcs = [getattr(metric, _met) for _met in cfg.eval.eval_metrics]
        self.train_metrics = MetricTracker(*["loss","diff_loss", "g_loss","d_loss", "real_loss", "fake_loss","cons_loss"])
        self.val_metrics = MetricTracker(*[m.__name__ for m in self.metric_funcs])
        # main metric for best checkpoint saving
        self.main_val_metric = cfg.validation.main_val_metric
        self.main_val_metric_goal = cfg.validation.main_val_metric_goal
        assert (
            self.main_val_metric in cfg.eval.eval_metrics
        ), f"Main eval metric `{self.main_val_metric}` not found in evaluation metrics."
        self.best_metric = 1e8 if "minimize" == self.main_val_metric_goal else -1e8

        # Settings
        self.max_epoch = self.cfg.max_epoch
        self.max_iter = self.cfg.max_iter
        self.gradient_accumulation_steps = accumulation_steps
        self.gt_depth_type = self.cfg.gt_depth_type
        self.gt_mask_type = self.cfg.gt_mask_type
        self.save_period = self.cfg.trainer.save_period
        self.backup_period = self.cfg.trainer.backup_period
        self.val_period = self.cfg.trainer.validation_period
        self.vis_period = self.cfg.trainer.visualization_period

        # Multi-resolution noise
        self.apply_multi_res_noise = self.cfg.multi_res_noise is not None
        if self.apply_multi_res_noise:
            self.mr_noise_strength = self.cfg.multi_res_noise.strength
            self.annealed_mr_noise = self.cfg.multi_res_noise.annealed
            self.mr_noise_downscale_strategy = (
                self.cfg.multi_res_noise.downscale_strategy
            )

        # Internal variables
        self.epoch = 1
        self.n_batch_in_epoch = 0  # batch index in the epoch, used when resume training
        self.effective_iter = 0  # how many times optimizer.step() is called
        self.in_evaluation = False
        self.global_seed_sequence: List = []  # consistent global seed sequence, used to seed random generator, to ensure consistency when resuming

    def _replace_unet_conv_in(self):
        # replace the first layer to accept 8 in_channels
        _weight = self.model.unet.conv_in.weight.clone()  # [320, 4, 3, 3]
        _bias = self.model.unet.conv_in.bias.clone()  # [320]
        _weight = _weight.repeat((1, 3, 1, 1))  # Keep selected channel(s)

        # half the activation magnitude
        _weight *= 1 / 3.0  # 3배 확장된 것을 보정
        # new conv_in channel
        _n_convin_out_channel = self.model.unet.conv_in.out_channels
        _new_conv_in = Conv2d(
            12, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        _new_conv_in.weight = Parameter(_weight)
        _new_conv_in.bias = Parameter(_bias)
        self.model.unet.conv_in = _new_conv_in
        logging.info("Unet conv_in layer is replaced with 12 channels")
        # replace config
        self.model.unet.config["in_channels"] = 12
        logging.info("Unet config is updated")
        return

    def _replace_unet_conv_out(self):
        """
        Replace the last conv_out layer of U-Net to output 12 channels.
        """
        # Clone existing weight and bias
        _weight = self.model.unet.conv_out.weight.clone()  # 기존 가중치
        _bias = self.model.unet.conv_out.bias.clone()      # 기존 bias

        # 기존 가중치 채널 확장 (output 채널을 늘리기 위해)
        _weight = _weight.repeat((2, 1, 1, 1))  #

        # 출력 값의 스케일 유지 (가중치 값 조정)
        _weight *= 0.5

        # 새로운 conv_out 레이어 생성
        _n_convin_in_channel = self.model.unet.conv_out.in_channels
        _new_conv_out = Conv2d(in_channels=_n_convin_in_channel, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))


        _new_conv_out.weight = Parameter(_weight)
        _new_conv_out.bias = Parameter(_bias.repeat(2))  # bias도 3배로 복제

        # U-Net의 마지막 conv_out 레이어 교체
        self.model.unet.conv_out = _new_conv_out
        logging.info("Unet conv_out layer is replaced with 8 channels")

        # U-Net config 업데이트
        self.model.unet.config["out_channels"] = 8
        logging.info("Unet config is updated")
        return
    

    def compute_generator_gan_loss_v_prediction(
        self,
        discriminator,
        x_t,  # [B, C, H, W] noisy latent at time t
        v_pred,  # [B, C, H, W] model output (v)
        x0_gt,   # [B, C, H, W] ground-truth x0 (encoded event latent)
        timesteps,  # [B] each sample's t
        alphas_cumprod,  # tensor of shape [num_timesteps], holds \bar{alpha}_t
    ):
        """
        - discriminator: 판별자
        - x_t: 현재 시점 t의 noisy latent
        - v_pred: UNet이 예측한 v (v-prediction)
        - x0_gt: ground-truth x0 (VAE 인코딩된 event 이미지)
        - timesteps: [B], 배치별로 서로 다른 t값 가능
        - alphas_cumprod: scheduler_timesteps

        반환: generator 측 gan loss (scalar) -> 목표값 : pred_fake -> -1 그러므로 gan_loss -> 1
        """

        device = x_t.device

        # 1) alpha_bar_t 가져오기
        alpha_bar_t = alphas_cumprod[timesteps]  # shape [B]
        alpha_bar_t = alpha_bar_t.reshape(-1, 1, 1, 1)  # [B,1,1,1]

        # sqrt, 1 - alpha_bar
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)  # [B,1,1,1]
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)  # [B,1,1,1]

        #  x0 추정
        #    x0 = sqrt(alpha_bar_t)*x_t - sqrt(1 - alpha_bar_t)*v_pred
        x0_fake = sqrt_alpha_bar_t * x_t - sqrt_one_minus_alpha_bar_t * v_pred

        #### latent -> pixel ####
        with torch.no_grad():
            x0_fake = self.model.decode_event(x0_fake)

        # 3) 가짜 vs. 진짜
        #    여기서 fake_data = x0_fake, real_data = x0_gt
        # pred_fake = discriminator(x0_fake)  # [B]
        # # Hinge Loss (Generator 입장) 예시
        # gan_loss_g = - pred_fake.mean()

        pred_fake = torch.tanh(discriminator(x0_fake))  # [-1, 1]
        gan_loss_g = torch.mean((pred_fake + 1.0) ** 2)  # 목표: pred_fake -> -1

        return x0_fake, gan_loss_g
    
    def compute_discriminator_loss_hinge(self, discriminator, real_data, fake_data):
        """
        Hinge Loss 기반 Discriminator 손실을 계산합니다.
        
        Args:
            discriminator (nn.Module): Discriminator 네트워크
            real_data (torch.Tensor): [B, C, H, W], 실제 데이터 입력
            fake_data (torch.Tensor): [B, C, H, W], 가짜(생성된) 데이터 입력
        
        Returns:
            torch.Tensor: Discriminator hinge loss (스칼라)
        """
        # 1) Discriminator로부터 예측 점수 받기
        # pred_real = discriminator(real_data)  # [B]
        # pred_fake = discriminator(fake_data)  # [B]
        
        # # 2) Hinge Loss 구성
        # # real_data는 1 이상으로, fake_data는 -1 이하로 분류되도록 유도
        # real_loss = torch.mean(F.relu(1.0 - pred_real))  # D(real) >= 1
        # fake_loss = torch.mean(F.relu(1.0 + pred_fake))  # D(fake) <= -1

        pred_real = torch.tanh(discriminator(real_data))  # [-1, 1]
        pred_fake = torch.tanh(discriminator(fake_data))  # [-1, 1]

        real_loss = torch.mean(F.relu(0.5 - pred_real))  # Real data: pred >= 0.5
        fake_loss = torch.mean(F.relu(0.5 + pred_fake))  # Fake data: pred <= -0.5

        d_loss = real_loss + fake_loss
        return d_loss, real_loss, fake_loss

    def compute_channel_consistency_regularization(self, x0):
        """
        SCER 형식 데이터의 채널 간 연속성을 강제하는 Consistency Regularization.

        Args:
            x0 (torch.Tensor): [B, 6, H, W] 형태의 VAE Decoder 출력 (pixel space)

        Returns:
            torch.Tensor: Consistency Regularization Loss (scalar)
        """
        # 6채널 데이터를 3채널씩 분리
        event_1 = x0[:, 0:3, :, :]  # [B, 3, H, W]
        event_2 = x0[:, 3:6, :, :]  # [B, 3, H, W]
        event_2 = torch.flip(event_2, dims=[1])

        # event_1 내 채널 간 consistency (2번 채널 값은 1번 채널에 포함, 1번 채널은 0번 채널에 포함)
        loss_event_1_01 = torch.mean(torch.relu(event_1[:, 1, :, :] - event_1[:, 0, :, :]))  # event_1[:, 1] <= event_1[:, 0]
        loss_event_1_12 = torch.mean(torch.relu(event_1[:, 2, :, :] - event_1[:, 1, :, :]))  # event_1[:, 2] <= event_1[:, 1]

        # event_2 내 채널 간 consistency
        loss_event_2_01 = torch.mean(torch.relu(event_2[:, 1, :, :] - event_2[:, 0, :, :]))  # event_2[:, 1] <= event_2[:, 0]
        loss_event_2_12 = torch.mean(torch.relu(event_2[:, 2, :, :] - event_2[:, 1, :, :]))  # event_2[:, 2] <= event_2[:, 1]

        # 최종 Consistency Loss (두 더미의 모든 관계 강제)
        consistency_loss = loss_event_1_01 + loss_event_1_12 + loss_event_2_01 + loss_event_2_12

        return consistency_loss

    def train(self, t_end=None):
        logging.info("Start training")

        device = self.device
        self.model.to(device)
        
        if self.use_gan:
            self.discriminator.to(device)

        self.train_metrics.reset()
        accumulated_step = 0  # Gradient Accumulation 카운트
    

        for epoch in range(self.epoch, self.max_epoch + 1):
            self.epoch = epoch
            logging.debug(f"epoch: {self.epoch}")

            # Skip previous batches when resume
            for batch in skip_first_batches(self.train_loader, self.n_batch_in_epoch):
                ########################################
                # 1) 준비 작업 (모델 train 모드, rng 등)
                ########################################
                self.model.unet.train()
                if self.use_gan and self.effective_iter > self.gan_start_iter:
                    self.discriminator.train()

                if self.seed is not None:
                    local_seed = self._get_next_seed()
                    rand_num_generator = torch.Generator(device=device)
                    rand_num_generator.manual_seed(local_seed)
                else:
                    rand_num_generator = None

                ########################################
                # 2) 배치 데이터 준비 및 인코딩
                ########################################
                rgb = batch['frame'].to(device)                  # [B, 3, H, W]
                event_gt_for_latent = batch['voxel'].to(device)  # [B, 6, H, W], 
                event_1 = event_gt_for_latent[:, 0:3, :, :]
                event_2 = event_gt_for_latent[:, 3:, :, :]
                gt_event = torch.cat([event_1, event_2], dim=1)

                batch_size = rgb.shape[0]

                with torch.no_grad():
                    # Encode
                    event_latent_1 = self.model.encode(event_1)  # [B, 4, h, w] 
                    event_latent_2 = self.model.encode(event_2)
                    rgb_latent = self.model.encode(rgb)

                    # Ground Truth x0 in latent space
                    event_latent = torch.cat([event_latent_1, event_latent_2], dim=1)
                    # shape 예: [B, 8, h, w]

                ########################################
                # 3) Diffusion Forward Process (노이즈 추가)
                ########################################
                timesteps = torch.randint(
                    0, self.scheduler_timesteps, (batch_size,),
                    device=device, generator=rand_num_generator
                ).long()  # [B]

                # noise (multi_res_noise_like or randn)
                if self.apply_multi_res_noise:
                    strength = self.mr_noise_strength
                    if self.annealed_mr_noise:
                        strength = strength * (timesteps / self.scheduler_timesteps)
                    noise = multi_res_noise_like(
                        event_latent,
                        strength=strength,
                        downscale_strategy=self.mr_noise_downscale_strategy,
                        generator=rand_num_generator,
                        device=device,
                    )
                else:
                    noise = torch.randn(
                        event_latent.shape,
                        device=device,
                        generator=rand_num_generator,
                    )  # [B, C, h, w]

                noisy_latents = self.training_noise_scheduler.add_noise(
                    event_latent, noise, timesteps
                )  # [B, C, h, w]

                ########################################
                # 4) U-Net 전방연산 (노이즈 or v 예측)
                ########################################
                text_embed = self.empty_text_embed.to(device).repeat((batch_size, 1, 1))
                cat_latents = torch.cat([rgb_latent, noisy_latents], dim=1).float()

                model_pred = self.model.unet(cat_latents, timesteps, text_embed).sample
                if torch.isnan(model_pred).any():
                    logging.warning("model_pred contains NaN.")

                # Diffusion 기본 로스 계산
                if self.prediction_type == "sample":
                    target = event_latent
                elif self.prediction_type == "epsilon":
                    target = noise
                elif self.prediction_type == "v_prediction":
                    target = self.training_noise_scheduler.get_velocity(
                        event_latent, noise, timesteps
                    )
                else:
                    raise ValueError(f"Unknown prediction type {self.prediction_type}")

                ########## Debug ###########
                x0 = self.predict_x0_from_v_single(x_t=noisy_latents[0], v=model_pred[0], timestep=timesteps[0], alphas_cumprod=self.training_noise_scheduler.alphas_cumprod)  #
                x0 = x0.unsqueeze(0)
                with torch.no_grad():
                    debug_out = self.model.decode_event(x0)
                tmp_out = debug_out[0]
                tmp_out = tmp_out.detach().cpu().numpy()
                np.save(f'out.npy', tmp_out)

                latent_loss = self.loss(model_pred.float(), target.float())
                diff_loss = latent_loss.mean()

                ########################################
                # 5) Generator(U-Net) & Discriminator 업데이트
                ########################################
                if self.use_gan and self.effective_iter > self.gan_start_iter :
                    # ========== Generator(U-Net) Loss ==========
                    # 5a) GAN Loss(Generator 측)
                    if self.prediction_type == "v_prediction":
                        # VAE output을 discriminator에 넣을까? latent를 넣을까?
                        x0_fake, gan_loss_g = self.compute_generator_gan_loss_v_prediction(
                            discriminator=self.discriminator,
                            x_t=noisy_latents,
                            v_pred=model_pred,
                            x0_gt=event_latent,
                            timesteps=timesteps,
                            alphas_cumprod=self.training_noise_scheduler.alphas_cumprod,
                        )
                    elif self.prediction_type == "epsilon":
                        raise NotImplementedError("GAN loss for epsilon mode is not implemented yet.")
                    else:
                        raise ValueError(f"Unknown prediction type {self.prediction_type}")

                    # ========== Consistency Regularization 추가 ==========
                    # x0_fake에 대해 채널 간 Consistency Loss 계산
                    if self.use_consistency_loss:
                        consistency_loss = self.compute_channel_consistency_regularization(x0_fake)
                        g_loss = diff_loss + self.gan_weight * gan_loss_g + self.consistency_weight * consistency_loss
                        self.train_metrics.update("cons_loss", consistency_loss.item())

                    else:
                        g_loss = diff_loss + self.gan_weight * gan_loss_g

                    # 역전파 (Generator)
                    g_loss_for_bp = g_loss / self.gradient_accumulation_steps
                    g_loss_for_bp.backward() 

                    # ========== Discriminator Loss ==========
                    real_data = gt_event
                    fake_data = x0_fake.detach()  # gradient 분리
                    d_loss, real_loss, fake_loss = self.compute_discriminator_loss_hinge(   # d_loss = real_loss + fake_loss
                        self.discriminator, real_data=real_data, fake_data=fake_data
                    )
                    d_loss_for_bp = d_loss / self.gradient_accumulation_steps
                    d_loss_for_bp.backward()

                    # 로깅
                    self.train_metrics.update("diff_loss", diff_loss.item())
                    self.train_metrics.update("g_loss", gan_loss_g.item())
                    self.train_metrics.update("d_loss", d_loss.item())
                    self.train_metrics.update("real_loss", real_loss.item())
                    self.train_metrics.update("fake_loss", fake_loss.item())

                else:
                    # GAN 미사용 시 → Diffusion Loss만 역전파
                    g_loss = diff_loss
                    g_loss_for_bp = g_loss / self.gradient_accumulation_steps
                    g_loss_for_bp.backward()

                    # 로깅
                    self.train_metrics.update("diff_loss", diff_loss.item())
                    d_loss = None  # GAN 미사용


                self.train_metrics.update("loss", g_loss.item())

                # Gradient Accumulation 카운트
                accumulated_step += 1
                self.n_batch_in_epoch += 1

                ########################################
                # 6) Accumulation 완료 시 Optimizer step
                ########################################
                if accumulated_step >= self.gradient_accumulation_steps:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    if self.use_gan and self.effective_iter > self.gan_start_iter:
                        self.discriminator_optimizer.step()
                        self.discriminator.zero_grad()

                    self.lr_scheduler.step()
                    if self.use_gan and self.effective_iter > self.gan_start_iter:
                        self.discriminator_scheduler.step()

                    accumulated_step = 0
                    self.effective_iter += 1

                    # 로깅
                    accumulated_loss = self.train_metrics.result()["loss"]
                    logging.info(
                        f"iter {self.effective_iter:5d} (epoch {epoch:2d}): loss={accumulated_loss:.5f}"
                    )
                    wandb.log({'iter': self.effective_iter, 'loss': float(f"{accumulated_loss:.5f}")})

                    if self.use_gan and self.effective_iter > self.gan_start_iter:
                        diffusion_loss = self.train_metrics.result()["diff_loss"]
                        discriminator_loss = self.train_metrics.result()["d_loss"]
                        d_real_loss = self.train_metrics.result()['real_loss']
                        d_fake_loss = self.train_metrics.result()['fake_loss']
                        generator_loss = self.train_metrics.result()["g_loss"]
     
                        wandb.log({'iter': self.effective_iter, 'diffusion_loss': float(f"{diffusion_loss:.5f}")})
                        wandb.log({'iter': self.effective_iter, 'd_loss': float(f"{discriminator_loss:.5f}")})
                        wandb.log({'iter': self.effective_iter, 'g_loss': float(f"{generator_loss:.5f}")})
                        wandb.log({'iter': self.effective_iter, 'd_real_loss': float(f"{d_real_loss:.5f}")})
                        wandb.log({'iter': self.effective_iter, 'd_fake_loss': float(f"{d_fake_loss:.5f}")})

                        if self.use_consistency_loss:
                            cons_loss = self.train_metrics.result()["cons_loss"]
                            wandb.log({'iter': self.effective_iter, 'cons_loss': float(f"{cons_loss:.5f}")})

                    self.train_metrics.reset()

                    # Per-step callback
                    self._train_step_callback()

                    # End of training
                    if self.max_iter > 0 and self.effective_iter >= self.max_iter:
                        self.save_checkpoint(
                            ckpt_name=self._get_backup_ckpt_name(),
                            save_train_state=False,
                        )
                        logging.info("Training ended.")
                        return
                    # Time's up
                    elif t_end is not None and datetime.now() >= t_end:
                        self.save_checkpoint(ckpt_name="latest", save_train_state=True)
                        logging.info("Time is up, training paused.")
                        return

                    torch.cuda.empty_cache()
                    # <<< Effective batch end <<<

            # Epoch end
            self.n_batch_in_epoch = 0



    def _train_step_callback(self):
        """Executed after every iteration"""
        # Save backup (with a larger interval, without training states)
        if self.backup_period > 0 and 0 == self.effective_iter % self.backup_period:
            self.save_checkpoint(
                ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
            )

        _is_latest_saved = False
        # Validation
        if self.val_period > 0 and 0 == self.effective_iter % self.val_period:
            self.in_evaluation = True  # flag to do evaluation in resume run if validation is not finished
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)
            _is_latest_saved = True
            self.validate()
            self.in_evaluation = False
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)

        # Save training checkpoint (can be resumed)
        if (
            self.save_period > 0
            and 0 == self.effective_iter % self.save_period
            and not _is_latest_saved
        ):
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)

        # Visualization
        if self.vis_period > 0 and 0 == self.effective_iter % self.vis_period:
            self.visualize()

    def validate(self):
        for i, val_loader in enumerate(self.val_loaders):
            val_dataset_name = val_loader.dataset.disp_name
            val_metric_dic = self.validate_single_dataset(
                data_loader=val_loader, metric_tracker=self.val_metrics
            )
            logging.info(
                f"Iter {self.effective_iter}. Validation metrics on `{val_dataset_name}`: {val_metric_dic}"
            )
            tb_logger.log_dic(
                {f"val/{val_dataset_name}/{k}": v for k, v in val_metric_dic.items()},
                global_step=self.effective_iter,
            )
            # save to file
            eval_text = eval_dic_to_text(
                val_metrics=val_metric_dic,
                dataset_name=val_dataset_name,
                sample_list_path=val_loader.dataset.filename_ls_path,
            )
            _save_to = os.path.join(
                self.out_dir_eval,
                f"eval-{val_dataset_name}-iter{self.effective_iter:06d}.txt",
            )
            with open(_save_to, "w+") as f:
                f.write(eval_text)

            # Update main eval metric
            if 0 == i:
                main_eval_metric = val_metric_dic[self.main_val_metric]
                if (
                    "minimize" == self.main_val_metric_goal
                    and main_eval_metric < self.best_metric
                    or "maximize" == self.main_val_metric_goal
                    and main_eval_metric > self.best_metric
                ):
                    self.best_metric = main_eval_metric
                    logging.info(
                        f"Best metric: {self.main_val_metric} = {self.best_metric} at iteration {self.effective_iter}"
                    )
                    # Save a checkpoint
                    self.save_checkpoint(
                        ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
                    )

    def visualize(self):
        for val_loader in self.vis_loaders:
            vis_dataset_name = val_loader.dataset.disp_name
            vis_out_dir = os.path.join(
                self.out_dir_vis, self._get_backup_ckpt_name(), vis_dataset_name
            )
            os.makedirs(vis_out_dir, exist_ok=True)
            _ = self.validate_single_dataset(
                data_loader=val_loader,
                metric_tracker=self.val_metrics,
                save_to_dir=vis_out_dir,
            )


    def _get_next_seed(self):
        if 0 == len(self.global_seed_sequence):
            self.global_seed_sequence = generate_seed_sequence(
                initial_seed=self.seed,
                length=self.max_iter * self.gradient_accumulation_steps,
            )
            logging.info(
                f"Global seed sequence is generated, length={len(self.global_seed_sequence)}"
            )
        return self.global_seed_sequence.pop()

    def save_checkpoint(self, ckpt_name, save_train_state):
        ckpt_dir = os.path.join(self.out_dir_ckpt, ckpt_name)
        logging.info(f"Saving checkpoint to: {ckpt_dir}")
        # Backup previous checkpoint
        temp_ckpt_dir = None
        if os.path.exists(ckpt_dir) and os.path.isdir(ckpt_dir):
            temp_ckpt_dir = os.path.join(
                os.path.dirname(ckpt_dir), f"_old_{os.path.basename(ckpt_dir)}"
            )
            if os.path.exists(temp_ckpt_dir):
                shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            os.rename(ckpt_dir, temp_ckpt_dir)
            logging.debug(f"Old checkpoint is backed up at: {temp_ckpt_dir}")

        # Save UNet
        unet_path = os.path.join(ckpt_dir, "unet")
        self.model.unet.save_pretrained(unet_path, safe_serialization=False)
        logging.info(f"UNet is saved to: {unet_path}")

        # 추가: Discriminator 가중치 저장
        if self.use_gan and self.effective_iter > self.gan_start_iter:
            disc_path = os.path.join(ckpt_dir, "discriminator.ckpt")
            torch.save(self.discriminator.state_dict(), disc_path)
            logging.info(f"Discriminator weights are saved to: {disc_path}")


        if save_train_state:
            state = {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "config": self.cfg,
                "effective_iter": self.effective_iter,
                "epoch": self.epoch,
                "n_batch_in_epoch": self.n_batch_in_epoch,
                "best_metric": self.best_metric,
                "in_evaluation": self.in_evaluation,
                "global_seed_sequence": self.global_seed_sequence,
            }
            train_state_path = os.path.join(ckpt_dir, "trainer.ckpt")
            torch.save(state, train_state_path)
            # iteration indicator
            f = open(os.path.join(ckpt_dir, self._get_backup_ckpt_name()), "w")
            f.close()

            logging.info(f"Trainer state is saved to: {train_state_path}")

        # Remove temp ckpt
        if temp_ckpt_dir is not None and os.path.exists(temp_ckpt_dir):
            shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            logging.debug("Old checkpoint backup is removed.")

    def load_checkpoint(
        self, ckpt_path, load_trainer_state=True, resume_lr_scheduler=True
    ):
        logging.info(f"Loading checkpoint from: {ckpt_path}")
        # Load UNet
        _model_path = os.path.join(ckpt_path, "unet", "diffusion_pytorch_model.bin")
        self.model.unet.load_state_dict(
            torch.load(_model_path, map_location=self.device)
        )
        self.model.unet.to(self.device)
        logging.info(f"UNet parameters are loaded from {_model_path}")

        # Load training states
        if load_trainer_state:
            checkpoint = torch.load(os.path.join(ckpt_path, "trainer.ckpt"))
            self.effective_iter = checkpoint["effective_iter"]
            self.epoch = checkpoint["epoch"]
            self.n_batch_in_epoch = checkpoint["n_batch_in_epoch"]
            self.in_evaluation = checkpoint["in_evaluation"]
            self.global_seed_sequence = checkpoint["global_seed_sequence"]

            self.best_metric = checkpoint["best_metric"]

            self.optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(f"optimizer state is loaded from {ckpt_path}")

            if resume_lr_scheduler:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                logging.info(f"LR scheduler state is loaded from {ckpt_path}")

        logging.info(
            f"Checkpoint loaded from: {ckpt_path}. Resume from iteration {self.effective_iter} (epoch {self.epoch})"
        )
        return

    def _get_backup_ckpt_name(self):
        return f"iter_{self.effective_iter:06d}"


    def predict_x0_from_v_single(self,x_t, v, timestep, alphas_cumprod, verbose=False):
        """
        v-pred 기반의 x0 복원 수식 사용
        """
        device = x_t.device

        if not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep, dtype=torch.long, device=device)

        alpha_bar = alphas_cumprod[timestep].to(device)
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

        # 🔄 v-pred 공식 사용
        x0 = sqrt_alpha_bar * x_t - sqrt_one_minus_alpha_bar * v

        if verbose:
            print(f"[t={timestep.item()}] alpha_bar: {alpha_bar.item():.6f}, sqrt_alpha_bar: {sqrt_alpha_bar.item():.6f}, sqrt_1m_alpha_bar: {sqrt_one_minus_alpha_bar.item():.6f}")

        return x0
    

    @torch.no_grad()
    def debug_reconstruction_error_vs_timestep(
        self,
        save_dir="./debug",
        timesteps_list=None,
        max_batches=5
    ):
        """
        각 timestep에 대해 RMSE(x0_pred, z0) 측정 → 그래프 저장
        """
        os.makedirs(save_dir, exist_ok=True)
        device = self.device
        self.model.to(device)
        self.model.unet.eval()

        if timesteps_list is None:
            timesteps_list = list(range(990, -1, -10))

        rmse_by_timestep = []

        for t_val in tqdm(timesteps_list, desc="Evaluating RMSE vs Timestep"):
            rmse_list = []

            for i, batch in enumerate(self.train_loader):
                if i >= max_batches:
                    break

                rgb = batch['frame'].to(device)
                event_gt_for_latent = batch['voxel'].to(device)  # [B, 6, H, W], 
                event_1 = event_gt_for_latent[:, 0:3, :, :]
                event_2 = event_gt_for_latent[:, 3:, :, :]
                event_latent_1 = self.model.encode(event_1)  # [B, 4, h, w] 
                event_latent_2 = self.model.encode(event_2)

                B = event_gt_for_latent.shape[0]

                # latent 생성
                z0 = torch.cat([event_latent_1, event_latent_2], dim=1)
                rgb_latent = self.model.encode(rgb)
                t = torch.tensor([t_val] * B, dtype=torch.long, device=device)

                # noise 주입
                noise = torch.randn_like(z0)
                z_t = self.training_noise_scheduler.add_noise(z0, noise, t)

                # U-Net 예측
                text_embed = self.empty_text_embed.to(device).repeat((B, 1, 1))
                x_in = torch.cat([rgb_latent, z_t], dim=1).float()
                v_pred = self.model.unet(x_in, t, text_embed).sample

                # 복원
                x0_pred = torch.stack([
                    self.predict_x0_from_v_single(
                        x_t=z_t[j],
                        v=v_pred[j],
                        timestep=t[j],
                        alphas_cumprod=self.training_noise_scheduler.alphas_cumprod
                    )
                    for j in range(B)
                ], dim=0)

                # [-1,1] → [0,1] 스케일
                x0_pred = (x0_pred + 1) / 2
                z0_scaled = (z0 + 1) / 2

                # RMSE
                rmse = torch.sqrt(F.mse_loss(x0_pred, z0_scaled, reduction="mean"))
                rmse_list.append(rmse.item())

            avg_rmse = sum(rmse_list) / len(rmse_list)
            print(f"[t={t_val}] avg RMSE: {avg_rmse:.6f}")
            rmse_by_timestep.append(avg_rmse)

        # 그래프 저장
        plt.figure(figsize=(8, 5))
        plt.plot(timesteps_list, rmse_by_timestep, marker='o')
        plt.xlabel("Timestep (t)")
        plt.ylabel("Avg RMSE between z0 and x0_pred")
        plt.title("gt latent vs output latent rmse by Timestep")
        plt.grid(True)
        plt.tight_layout()
        out_path = os.path.join(save_dir, "rmse_vs_timestep_original.png")
        plt.savefig(out_path)
        print(f">>> RMSE plot saved to: {out_path}")