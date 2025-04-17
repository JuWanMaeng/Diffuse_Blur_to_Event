import logging
import os
import shutil
from datetime import datetime
from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision as v

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

        # Adapt input layers
        if 132 != self.model.unet.config["in_channels"]:
            self._replace_unet_conv_in()

        if 128 != self.model.unet.config["out_channels"]:
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
        # 기존 UNet의 첫 번째 conv layer weight와 bias 복사 (원래 shape: [320, 4, 3, 3])
        _weight = self.model.unet.conv_in.weight.clone()  # [320, 4, 3, 3]
        _bias = self.model.unet.conv_in.bias.clone()        # [320]
        
        # 4채널을 68채널로 확장 (4 * 17 = 68)
        _weight = _weight.repeat((1, 33, 1, 1))  # 새로운 weight shape: [320, 132, 3, 3]
        
        # 복제된 채널에 따른 활성화 값 보정
        _weight *= 1 / 33.0  # 33배 확장된 것을 보정
        
        # 새 conv_in layer 생성: in_channels를 68로 설정
        _n_conv_in_out_channel = self.model.unet.conv_in.out_channels
        _new_conv_in = Conv2d(
            132, _n_conv_in_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        _new_conv_in.weight = Parameter(_weight)
        _new_conv_in.bias = Parameter(_bias)
        self.model.unet.conv_in = _new_conv_in
        
        logging.info("Unet conv_in layer is replaced with 68 channels")
        # config 업데이트
        self.model.unet.config["in_channels"] = 132
        logging.info("Unet config is updated")
        return

    def _replace_unet_conv_out(self):
        """
        Replace the last conv_out layer of U-Net to output 64 channels.
        """
        # Clone existing weight and bias
        _weight = self.model.unet.conv_out.weight.clone()  # 기존 가중치, shape: [4, in_channels, 3, 3] (예시)
        _bias = self.model.unet.conv_out.bias.clone()        # 기존 bias, shape: [4]

        # 기존 가중치 채널 확장 (output 채널을 늘리기 위해)
        # 예를 들어, 기존 채널이 4개라면 64개로 만들기 위해 16배 반복합니다.
        _weight = _weight.repeat(32, 1, 1, 1)  # 새로운 weight shape: [4*16=64, in_channels, 3, 3]

        # 출력 값의 스케일 유지 (가중치 값 조정)
        _weight *= 1 / 32.0  # 16배 확장된 것을 보정

        # 새로운 conv_out 레이어 생성 (out_channels를 64로 설정)
        _n_convin_in_channel = self.model.unet.conv_out.in_channels
        _new_conv_out = Conv2d(
            in_channels=_n_convin_in_channel, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )

        _new_conv_out.weight = Parameter(_weight)
        _new_conv_out.bias = Parameter(_bias.repeat(32))  # bias도 동일하게 16배로 복제

        # U-Net의 마지막 conv_out 레이어 교체
        self.model.unet.conv_out = _new_conv_out
        logging.info("Unet conv_out layer is replaced with 64 channels")

        # U-Net config 업데이트
        self.model.unet.config["out_channels"] = 128
        logging.info("Unet config is updated")
        return

    
    

    def train(self, t_end=None):
        logging.info("Start training")

        device = self.device
        self.model.to(device)
        
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
                event = batch['voxel'].to(device)  # [B, 6, H, W], 

                batch_size = rgb.shape[0]

                with torch.no_grad():
                    # Encode
                    event_latent = self.model.encode_event(event)  # [B, 4, h, w] 
                    rgb_latent = self.model.encode_image(rgb)

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


                latent_loss = self.loss(model_pred.float(), target.float())
                diff_loss = latent_loss.mean()
                self.train_metrics.update("loss", diff_loss.item())

                diff_loss = diff_loss / self.gradient_accumulation_steps
                diff_loss.backward()


                # Gradient Accumulation 카운트
                accumulated_step += 1
                self.n_batch_in_epoch += 1

                ########################################
                # 6) Accumulation 완료 시 Optimizer step
                ########################################
                if accumulated_step >= self.gradient_accumulation_steps:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    self.lr_scheduler.step()

                    accumulated_step = 0
                    self.effective_iter += 1

                    # 로깅
                    accumulated_loss = self.train_metrics.result()["loss"]
                    logging.info(
                        f"iter {self.effective_iter:5d} (epoch {epoch:2d}): loss={accumulated_loss:.5f}"
                    )
                    wandb.log({'iter': self.effective_iter, 'loss': float(f"{accumulated_loss:.5f}")})

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
        max_batches=1
    ):
        """
        각 timestep에 대해 RMSE(x0_pred, z0) 측정 및 디코딩된 event 결과를 6채널 이미지로 저장하고,
        전체 RMSE vs timestep 그래프를 저장하는 함수.
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
                if i > max_batches:
                    break

                rgb = batch['frame'].to(device)
                event = batch['voxel'].to(device)
                B = event.shape[0]

                # latent 생성
                z0 = self.model.encode_event(event)  # [B, C, H, W]
                rgb_latent = self.model.encode_image(rgb)
                t = torch.tensor([t_val] * B, dtype=torch.long, device=device)

                # 노이즈 주입: VAE latent에 디퓨전 noise 추가
                noise = torch.randn_like(z0)
                z_t = self.training_noise_scheduler.add_noise(z0, noise, t)

                # U-Net 예측
                text_embed = self.empty_text_embed.to(device).repeat((B, 1, 1))
                x_in = torch.cat([rgb_latent, z_t], dim=1).float()
                v_pred = self.model.unet(x_in, t, text_embed).sample

                # 복원: 각 배치의 각 샘플에 대해 x0_pred 계산 (리스트 컴프리헨션 사용)
                x0_pred = torch.stack([
                    self.predict_x0_from_v_single(
                        x_t=z_t[j],
                        v=v_pred[j],
                        timestep=t[j],
                        alphas_cumprod=self.training_noise_scheduler.alphas_cumprod
                    )
                    for j in range(B)
                ], dim=0)

                # 디코딩: 복원된 latent를 event 이미지로 복원 (여기서 debug_out은 [B, C, H, W] 형태라고 가정)
                with torch.no_grad():
                    debug_out = self.model.decode_event(x0_pred)
                # 여기서는 첫 번째 sample만 선택
                gen_event = debug_out[0].detach().cpu().numpy()  # shape: (C, H, W)

                # 저장 전에 극성 비교를 위해 값 정규화: 최대 절대값으로 나누어 [-1,1] 유지
                max_val = np.max(np.abs(gen_event))
                if max_val != 0:
                    gen_event = gen_event / max_val

                # 6채널 이미지를 2x3 서브플롯으로 그리기
                fig, axs = plt.subplots(2, 3, figsize=(20, 10))
                axs = axs.ravel()
                for ch in range(6):
                    channel_data = gen_event[ch]  # (H, W)
                    im = axs[ch].imshow(channel_data, cmap='seismic', vmin=-1, vmax=1)
                    axs[ch].axis('off')
                plt.tight_layout()
                # 이미지 파일 저장 (예: gen_event_{t_val}.png)
                save_path = os.path.join(save_dir, f"gen_event_{t_val}.png")
                plt.savefig(save_path)
                plt.close(fig)
                
                # [-1,1] → [0,1] 스케일 for RMSE 계산
                x0_pred_scaled = (x0_pred + 1) / 2
                z0_scaled = (z0 + 1) / 2

                # RMSE 계산 (전체 배치에 대해)
                rmse = torch.sqrt(F.mse_loss(x0_pred_scaled, z0_scaled, reduction="mean"))
                rmse_list.append(rmse.item())

            avg_rmse = sum(rmse_list) / len(rmse_list)
            print(f"[t={t_val}] avg RMSE: {avg_rmse:.6f}")
            rmse_by_timestep.append(avg_rmse)

        # RMSE vs timestep 그래프 저장
        plt.figure(figsize=(8, 5))
        plt.plot(timesteps_list, rmse_by_timestep, marker='o')
        plt.xlabel("Timestep (t)")
        plt.ylabel("Avg RMSE between z0 and x0_pred")
        plt.title("gt latent vs output latent RMSE by Timestep")
        plt.grid(True)
        plt.tight_layout()
        out_path = os.path.join(save_dir, "rmse_vs_timestep_NAFVAE_test.png")
        plt.savefig(out_path)
        plt.close()
        print(f">>> RMSE plot saved to: {out_path}")


    @torch.no_grad()
    def debug_recon_vs_timestep(
        self,
        save_dir="./debug/NAFVAE",
        timesteps_list=None
    ):
        """
        고정된 배치에 대해 각 timestep에서 RMSE(x0_pred, z0) 측정 및 디코딩된 event 결과를 6채널 이미지로 저장하고,
        전체 RMSE vs timestep 그래프를 저장하는 함수.
        """
        os.makedirs(save_dir, exist_ok=True)
        device = self.device
        self.model.to(device)
        self.model.unet.eval()

        # 만약 별도로 지정하지 않으면 990부터 0까지 10씩 감소하는 timesteps 사용
        if timesteps_list is None:
            timesteps_list = list(range(990, -1, -10))

        rmse_by_timestep = []

        # dataloader에서 첫 번째 배치를 고정으로 가져오기
        fixed_batch = next(iter(self.train_loader))
        rgb = fixed_batch['frame'].to(device)
        event = fixed_batch['voxel'].to(device)
        B = event.shape[0]

        # latent 계산 (고정 배치)
        z0 = self.model.encode_event(event)  # [B, C, H, W]
        rgb_latent = self.model.encode_image(rgb)

        for t_val in tqdm(timesteps_list, desc="Evaluating RMSE vs Timestep"):
            # timestep 벡터 생성
            t = torch.tensor([t_val] * B, dtype=torch.long, device=device)

            # 노이즈 주입: noise 생성 후 디퓨전 스케줄러를 통해 z_t 생성
            noise = torch.randn_like(z0)
            z_t = self.training_noise_scheduler.add_noise(z0, noise, t)

            # U-Net 예측
            text_embed = self.empty_text_embed.to(device).repeat((B, 1, 1))
            x_in = torch.cat([rgb_latent, z_t], dim=1).float()
            v_pred = self.model.unet(x_in, t, text_embed).sample

            # 복원: 배치 내 각 샘플에 대해 x0_pred 계산
            x0_pred = torch.stack([
                self.predict_x0_from_v_single(
                    x_t=z_t[j],
                    v=v_pred[j],
                    timestep=t[j],
                    alphas_cumprod=self.training_noise_scheduler.alphas_cumprod
                )
                for j in range(B)
            ], dim=0)

            # 디코딩: 복원된 latent를 event 이미지로 복원 → 첫 번째 샘플 선택
            debug_out = self.model.decode_event(x0_pred)
            gen_event = debug_out[0].detach().cpu().numpy()  # shape: (C, H, W)

            # 저장 전에 극성 비교를 위해 최대 절대값으로 정규화하여 [-1, 1] 유지
            max_val = np.max(np.abs(gen_event))
            if max_val != 0:
                gen_event = gen_event / max_val

            # 6채널 이미지를 2x3 subplot으로 그리기
            fig, axs = plt.subplots(2, 3, figsize=(20, 10))
            axs = axs.ravel()
            for ch in range(6):
                channel_data = gen_event[ch]  # (H, W)
                axs[ch].imshow(channel_data, cmap='seismic', vmin=-1, vmax=1)
                axs[ch].axis('off')
            plt.tight_layout()

            # 각 timestep마다 결과 이미지 저장 (서브 폴더 생성)
            save_path = os.path.join(save_dir, f"gen_event_{t_val}.png")
            plt.savefig(save_path)
            plt.close(fig)