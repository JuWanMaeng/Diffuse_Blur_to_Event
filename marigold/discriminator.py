import torch.nn as nn
import torch.nn.functional as F

class SCERDiscriminator(nn.Module):
    """
    간단한 2D Conv 기반 Discriminator 예시.
    입력: [B, 6, H, W] 형태 (SCER)
    출력: [B, 1] (진짜/가짜 판별 로짓)
    """
    def __init__(self, in_channels=8, base_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            # Conv1
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Conv2
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Conv3
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Conv4
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 출력 채널 1개
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=0)
        )

    def forward(self, x):
        """
        x: [B, 6, H, W] (SCER)
        """
        out = self.net(x)  # [B, 1, H', W'] (보통 H',W'는 대략 1x1에 근접)
        return out.view(-1)  # [B], 진짜/가짜 점수(로짓)
