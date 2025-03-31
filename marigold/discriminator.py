import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Residual Block with optional downsampling.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 첫 번째 컨볼루션: stride가 2이면 다운샘플링
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        # 두 번째 컨볼루션
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 입력 차원과 출력 차원이 다르거나 stride가 1이 아니면 스킵 연결을 맞춰줌
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class Discriminator(nn.Module):
    """
    Residual Block을 도입한 Discriminator.
    입력: [B, 6, H, W] 형태 (SCER)
    출력: [B], 진짜/가짜 판별 로짓
    """
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        # 초기 Conv: 채널 수를 base_channels로 변경 및 다운샘플링
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Residual Blocks를 통해 특징 추출 및 다운샘플링 진행
        self.layer1 = ResidualBlock(base_channels, base_channels * 2, stride=2)
        self.layer2 = ResidualBlock(base_channels * 2, base_channels * 4, stride=2)
        self.layer3 = ResidualBlock(base_channels * 4, base_channels * 8, stride=2)
        # 최종 출력: Conv 레이어를 통해 1채널 출력 (패치 단위 판별)
        self.conv_out = nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=0)

    def forward(self, x):
        """
        x: [B, 6, H, W] (SCER)
        """
        out = self.initial(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.conv_out(out)  # [B, 1, H', W']
        return out.view(-1)  # [B] - 각 배치에 대한 판별 로짓

# 간단한 테스트 코드
if __name__ == "__main__":
    model = Discriminator()
    dummy_input = torch.randn(4, 6, 128, 128)  # 배치 크기 4, 128x128 이미지
    output = model(dummy_input)
    print(output.shape)  # 예상 출력: torch.Size([4])
