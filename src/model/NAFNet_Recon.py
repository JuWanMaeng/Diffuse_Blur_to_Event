import torch
import torch.nn as nn
import torch.nn.functional as F


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
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

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
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

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

class NAFNetRecon(nn.Module):
    def __init__(self, img_channel=6, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[],latent_dim=128):
        """
        Args:
            img_channel: 입력 이미지 채널 수
            width: 초기 feature channel 수
            middle_blk_num: Middle block의 총 개수 (짝수여야 합니다)
            enc_blk_nums: 각 encoder stage에서 NAFBlock의 개수 리스트
            dec_blk_nums: 각 decoder stage에서 NAFBlock의 개수 리스트
        """
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, bias=True)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        # skip connection 없이 encoder 진행
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, kernel_size=2, stride=2)
            )
            chan *= 2

        # middle block를 반으로 분할 (짝수인지 확인)
        assert middle_blk_num % 2 == 0, "middle_blk_num must be even to split equally."
        half_middle = middle_blk_num // 2
        self.middle_encoder = nn.Sequential(*[NAFBlock(chan) for _ in range(half_middle)])
        self.middle_decoder = nn.Sequential(*[NAFBlock(chan) for _ in range(half_middle)])
        
        # latent 변환 모듈 추가: encoder에서 나온 latent를 64채널로 줄이고, decoder에 넣기 전에 복원
        self.latent_to_8 = nn.Conv2d(chan, latent_dim, kernel_size=1, stride=1, bias=True)
        self.latent_from_8 = nn.Conv2d(latent_dim, chan, kernel_size=1, stride=1, bias=True)

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, kernel_size=1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan //= 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )

        self.padder_size = 2 ** len(self.encoders)

    def encode(self, inp):
        """Encoder 부분: intro, encoder stages, downsampling, 그리고 middle_encoder 이후 latent 변환.
           Skip connection은 사용하지 않습니다."""
        B, C, H, W = inp.shape

        x = self.intro(inp)
        # encoder + downsampling 진행 (skip connection 없이)
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            x = down(x)

        latent = self.middle_encoder(x)
        # latent을 64채널로 변환
        latent = self.latent_to_8(latent)
        return latent

    def decode(self, latent):
        """Decoder 부분: latent 복원, middle_decoder, upsampling 및 최종 복원.
           입력 이미지(inp)는 사용하지 않습니다."""
        # latent를 다시 원래 채널 수로 복원하여 middle_decoder에 넣음
        latent = self.latent_from_8(latent)
        x = self.middle_decoder(latent)
        # upsampling과 decoder 진행 (skip connection 없이)
        for decoder, up in zip(self.decoders, self.ups):
            x = up(x)
            x = decoder(x)
        x = self.ending(x)

        return x

    def forward(self, y):
        latent = self.encode(y)
        return self.decode(latent)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

if __name__ == '__main__':
    # 디버깅을 위한 임의 입력 데이터 생성 (batch_size, 채널, height, width)
    dummy_input = torch.randn(1, 6, 256, 256)

    # 모델 생성: 예시로 encoder와 decoder stage에 각각 1개의 NAFBlock을 사용합니다.
    model = NAFNetRecon(img_channel=6, width=64, middle_blk_num=28, enc_blk_nums=[1,1,1], dec_blk_nums=[1,1,1])
    
    # 모델의 forward 경로 테스트
    output = model(dummy_input)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)

    # encode와 decode를 별도로 테스트
    latent, orig_size = model.encode(dummy_input)
    print("Latent shape after conversion to 64 channels:", latent.shape)
    recon = model.decode(latent, orig_size)
    print("Reconstructed output shape:", recon.shape)
