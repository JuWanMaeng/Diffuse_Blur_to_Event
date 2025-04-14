import torch
import torch.nn as nn
import torch.nn.functional as F

# --- LayerNormFunction, LayerNorm2d, SimpleGate, NAFBlock --- #
class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=(0,2,3)), grad_output.sum(dim=(0,2,3)), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1, bias=True)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, kernel_size=1, bias=True),
        )
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma
# --- End of NAFBlock --- #


# --- VAE 스타일의 NAFNetRecon --- #
class NAFNetRecon_VAE(nn.Module):
    def __init__(self, img_channel=6, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], latent_dim=128):
        """
        Args:
            img_channel: 입력 이미지 채널 수
            width: 초기 feature channel 수
            middle_blk_num: Middle block의 총 개수 (짝수여야 함)
            enc_blk_nums: 각 encoder stage에서 NAFBlock의 개수 리스트
            dec_blk_nums: 각 decoder stage에서 NAFBlock의 개수 리스트
            latent_dim: 잠재 공간의 차원 (여기서는 평균과 로그분산 각각 latent_dim를 가짐)
        """
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, kernel_size=3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, kernel_size=3, padding=1, bias=True)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, kernel_size=2, stride=2))
            chan *= 2

        assert middle_blk_num % 2 == 0, "middle_blk_num must be even."
        half_middle = middle_blk_num // 2
        self.middle_encoder = nn.Sequential(*[NAFBlock(chan) for _ in range(half_middle)])
        self.middle_decoder = nn.Sequential(*[NAFBlock(chan) for _ in range(half_middle)])
        
        # latent 변환: 출력 채널이 2*latent_dim 이어야 합니다.
        self.latent_to_8 = nn.Conv2d(chan, 2 * latent_dim, kernel_size=1, bias=True)
        self.latent_from_8 = nn.Conv2d(latent_dim, chan, kernel_size=1, bias=True)

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, kernel_size=1, bias=False),
                nn.PixelShuffle(2)
            ))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
        
        self.padder_size = 2 ** len(self.encoders)

    def encode(self, inp):
        x = self.intro(inp)
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            x = down(x)
        x = self.middle_encoder(x)
        latent_params = self.latent_to_8(x)
        # latent_params를 (mu, logvar)로 나누기 (채널 차원에서 절반씩)
        mu, logvar = torch.chunk(latent_params, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # 디코더 전에 잠재 벡터 z를 복원합니다.
        z = self.latent_from_8(z)
        x = self.middle_decoder(z)
        for decoder, up in zip(self.decoders, self.ups):
            x = up(x)
            x = decoder(x)
        x = self.ending(x)
        return x

    def forward(self, y):
        # 인코딩 단계에서 mu와 logvar 반환
        mu, logvar = self.encode(y)
        # 학습 시 reparameterize를 통해 z 샘플링
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        # 필요에 따라 (reconstructed image, mu, logvar)로 반환
        return recon, mu, logvar

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

# KL divergence loss 함수 정의
def kl_divergence_loss(mu, logvar):
    # 배치 평균을 사용하는 예시
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

# --- Example Training Step --- #
if __name__ == '__main__':
    dummy_input = torch.randn(2, 6, 256, 256)
    model = NAFNetRecon_VAE(img_channel=6, width=64, middle_blk_num=28, enc_blk_nums=[1, 1, 1], dec_blk_nums=[1, 1, 1], latent_dim=128)
    
    recon, mu, logvar = model(dummy_input)
    print("Input shape:", dummy_input.shape)
    print("Reconstruction shape:", recon.shape)
    print("mu shape:", mu.shape)
    print("logvar shape:", logvar.shape)

    # 예시 손실 계산
    recon_loss = F.mse_loss(recon, dummy_input)
    kl_loss = kl_divergence_loss(mu, logvar)
    beta = 1.0  # 필요에 따라 조절
    total_loss = recon_loss + beta * kl_loss
    print("Reconstruction loss:", recon_loss.item())
    print("KL loss:", kl_loss.item())
    print("Total loss:", total_loss.item())