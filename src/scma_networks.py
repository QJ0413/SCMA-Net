"""
SCMA-Net: Structure-Consistent Mask-Aware Network
Based on svip10.tex

Key modules:
  - DAFD: Damage-Aware Feature Downsampling
  - MBDM: Multi-Branch Dependency Modeling
  - STCB: Structure-Texture Complementary Block
  - SCMANet: full encoder-decoder generator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange


# ---------------------------------------------------------------------------
# Shared utilities (reused from HINT)
# ---------------------------------------------------------------------------

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super().__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    """Gated FFN with depthwise conv, identical to HINT."""
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


# ---------------------------------------------------------------------------
# Spectral norm helper (for Discriminator)
# ---------------------------------------------------------------------------

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module


# ---------------------------------------------------------------------------
# DAFD: Damage-Aware Feature Downsampling
# ---------------------------------------------------------------------------

class PixelBasedDownsample(nn.Module):
    """PDB: PixelUnshuffle-based spatial reorganisation (no channel change)."""
    def __init__(self):
        super().__init__()
        self.pus = nn.PixelUnshuffle(2)  # (B,C,H,W) -> (B,4C,H/2,W/2)

    def forward(self, x):
        return self.pus(x)


class DAFD(nn.Module):
    """
    Damage-Aware Feature Downsampling (Section 2.2 of svip10.tex).

    Given F ∈ R^{H×W×C} and M ∈ {0,1}^{H×W×1}:
      1. F' = Conv3x3(F)
      2. F_hat = PDB(F'),  M_hat = PDB(M)
         -> F_hat ∈ R^{H/2 × W/2 × 4C},  M_hat ∈ R^{H/2 × W/2 × 4}
      3. Interleaved concat: each feature channel paired with its mask indicator
         -> F_C ∈ R^{H/2 × W/2 × 8C}
      4. Spatial attention: A = σ(Conv2(ReLU(Conv1(F_C))))
         -> F'_C = A ⊙ F_C
      5. Depthwise-separable conv -> F_out ∈ R^{H/2 × W/2 × 2C}
    """
    def __init__(self, in_channels):
        super().__init__()
        C = in_channels
        self.proj = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1, bias=False)
        self.pdb_feat = nn.PixelUnshuffle(2)   # C -> 4C
        self.pdb_mask = nn.PixelUnshuffle(2)   # 1 -> 4

        # After interleaved concat: 4C feat + 4C (mask broadcast) = 8C channels
        fused_ch = 8 * C
        # Spatial attention: two convolutions
        mid_ch = max(fused_ch // 4, 1)
        self.attn_conv1 = nn.Conv2d(fused_ch, mid_ch, kernel_size=1, bias=True)
        self.attn_conv2 = nn.Conv2d(mid_ch, fused_ch, kernel_size=1, bias=True)

        # Depthwise-separable conv: 8C -> 2C (standard output of downsampling)
        self.dw_conv = nn.Conv2d(fused_ch, fused_ch, kernel_size=3, stride=1,
                                 padding=1, groups=fused_ch, bias=False)
        self.pw_conv = nn.Conv2d(fused_ch, 2 * C, kernel_size=1, bias=False)

    def forward(self, F_in, M):
        # 1. project
        Fp = self.proj(F_in)                       # (B, C, H, W)
        # 2. PDB
        F_hat = self.pdb_feat(Fp)                  # (B, 4C, H/2, W/2)
        M_hat = self.pdb_mask(M)                   # (B, 4,  H/2, W/2)
        B, C4, Hh, Wh = F_hat.shape
        C = C4 // 4

        # 3. Interleaved concat: [f0, m0, f1, m1, ..., f_{4C-1}, m_{(i mod 4)}]
        # Build a (B, 8C, H/2, W/2) tensor by interleaving feature and mask channels
        parts = []
        for i in range(C4):
            parts.append(F_hat[:, i:i+1, :, :])         # feature channel i
            parts.append(M_hat[:, i % 4:i % 4 + 1, :, :])  # mask indicator
        F_C = torch.cat(parts, dim=1)              # (B, 8C, H/2, W/2)

        # 4. Spatial attention
        A = self.attn_conv2(F.relu(self.attn_conv1(F_C)))
        A = torch.sigmoid(A)
        F_C2 = A * F_C

        # 5. Depthwise-separable conv -> (B, 2C, H/2, W/2)
        out = self.dw_conv(F_C2)
        out = self.pw_conv(out)
        return out

    # small helper to avoid shadowing torch.nn.functional
    @staticmethod
    def relu(x):
        return F.relu(x, inplace=True)


# ---------------------------------------------------------------------------
# MBDM: Multi-Branch Dependency Modeling
# ---------------------------------------------------------------------------

class MBDM(nn.Module):
    """
    Multi-Branch Dependency Modeling (Section 2.3.2).

    Four branches on X1 (after FFN1 + LayerNorm):
      - Local: channel-transposed self-attention
      - Global: dilated conv (dilation=2)
      - Spatial: avg-pool → conv → upsample → spatial modulation mask
      - Channel: global avg-pool → conv → channel modulation vector

    Fusion:
      X_{G+L} = (X_L ⊙ X'_S ⊙ X'_C) + (X_G ⊙ X'_S)
      X2      = θ(X_{G+L})   [1×1 conv]
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = LayerNorm(dim)

        # --- Local branch: channel-transposed self-attention ---
        # DW conv to generate Q, K, V  (all C channels each)
        self.qkv_dw = nn.Conv2d(dim, dim * 3, kernel_size=3, stride=1,
                                 padding=1, groups=dim, bias=False)
        self.alpha = nn.Parameter(torch.ones(1))  # learnable scaling factor

        # --- Global branch: dilated conv ---
        self.global_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1,
                                     padding=2, dilation=2, bias=False)
        self.global_bn = nn.BatchNorm2d(dim)

        # --- Spatial branch ---
        self.spatial_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.spatial_upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                            align_corners=False)
        self.spatial_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

        # --- Channel branch ---
        self.channel_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # --- Aggregation ---
        self.theta = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, X1):
        B, C, H, W = X1.shape
        Xn = self.norm(X1)

        # Local branch
        qkv = self.qkv_dw(Xn)                      # (B, 3C, H, W)
        Q, K, V = qkv.chunk(3, dim=1)               # each (B, C, H, W)
        # reshape for channel-wise attention: Q (B,C,HW) @ K (B,HW,C) -> (B,C,C)
        Q_r = Q.reshape(B, C, H * W)               # (B, C, HW)
        K_r = K.reshape(B, C, H * W).transpose(1, 2)  # (B, HW, C)
        V_r = V.reshape(B, H * W, C)               # (B, HW, C)
        A = torch.softmax(Q_r @ K_r / self.alpha, dim=-1)  # (B, C, C)
        X_L = (V_r @ A.transpose(1, 2)).transpose(1, 2)    # (B, C, HW)
        X_L = X_L.reshape(B, C, H, W)

        # Global branch
        X_G = F.relu(self.global_bn(self.global_conv(Xn)), inplace=True)

        # Spatial branch: downsample -> conv -> upsample
        Xs = self.spatial_pool(Xn)
        Xs = self.spatial_conv(Xs)
        Xs = self.spatial_upsample(Xs)
        # ensure size matches (in case of odd dimensions)
        if Xs.shape[-2:] != (H, W):
            Xs = F.interpolate(Xs, size=(H, W), mode='bilinear', align_corners=False)
        X_S = torch.sigmoid(self.spatial_proj(Xs))  # (B, C, H, W)

        # Channel branch
        X_C = self.channel_fc(self.channel_pool(Xn))  # (B, C, 1, 1)

        # Fusion
        X_GL = (X_L * X_S * X_C) + (X_G * X_S)
        X2 = self.theta(X_GL)
        return X2


# ---------------------------------------------------------------------------
# STCB: Structure-Texture Complementary Block
# ---------------------------------------------------------------------------

class STCB(nn.Module):
    """
    Structure-Texture Complementary Block (Section 2.3).
    Sandwich: FFN1 -> MBDM -> FFN2, with residual around FFN2.

        X1      = FFN1(LayerNorm(X))       # feature conditioning
        X2      = MBDM(X1)                 # multi-branch dependency
        X_out   = X + FFN2(LayerNorm(X2))  # residual integration
    """
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.ffn1 = FeedForward(dim, ffn_expansion_factor, bias)

        self.mbdm = MBDM(dim)

        self.norm2 = LayerNorm(dim)
        self.ffn2 = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, X):
        X1 = X + self.ffn1(self.norm1(X))
        X2 = self.mbdm(X1)
        X_out = X + self.ffn2(self.norm2(X2))
        return X_out


# ---------------------------------------------------------------------------
# Upsample (identical role to HINT's Upsample)
# ---------------------------------------------------------------------------

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


# ---------------------------------------------------------------------------
# Gated embedding (identical to HINT's GatedEmb)
# ---------------------------------------------------------------------------

class GatedEmb(nn.Module):
    def __init__(self, in_c=4, embed_dim=48, bias=False):
        super().__init__()
        self.gproj1 = nn.Conv2d(in_c, embed_dim * 2, kernel_size=3,
                                stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.gproj1(x)
        x1, x2 = x.chunk(2, dim=1)
        return F.gelu(x1) * x2


# ---------------------------------------------------------------------------
# SCMANet: full 7-level encoder-decoder (Section 2.1)
# num_blocks = [4, 6, 6, 8, 6, 6, 4]  (paper Section 3.1)
# ---------------------------------------------------------------------------

class SCMANet(nn.Module):
    """
    7-stage hierarchical encoder-decoder.
    Encoder: E1 -> DAFD -> E2 -> DAFD -> E3 -> DAFD -> latent
    Decoder: latent -> Up -> D3 -> Up -> D2 -> Up -> D1 -> output
    Skip connections between encoder and decoder stages.
    """
    def __init__(self,
                 inp_channels=4,
                 out_channels=3,
                 dim=48,
                 num_blocks=None,
                 ffn_expansion_factor=2.66,
                 bias=False):
        super().__init__()
        if num_blocks is None:
            num_blocks = [4, 6, 6, 8, 6, 6, 4]

        self.patch_embed = GatedEmb(inp_channels, dim, bias)

        # Encoder
        self.encoder_level1 = nn.Sequential(*[
            STCB(dim, ffn_expansion_factor, bias) for _ in range(num_blocks[0])])

        self.down1_2 = DAFD(dim)  # dim -> 2*dim, H/2
        self.encoder_level2 = nn.Sequential(*[
            STCB(dim * 2, ffn_expansion_factor, bias) for _ in range(num_blocks[1])])

        self.down2_3 = DAFD(dim * 2)  # 2*dim -> 4*dim, H/4
        self.encoder_level3 = nn.Sequential(*[
            STCB(dim * 4, ffn_expansion_factor, bias) for _ in range(num_blocks[2])])

        self.down3_4 = DAFD(dim * 4)  # 4*dim -> 8*dim, H/8
        self.latent = nn.Sequential(*[
            STCB(dim * 8, ffn_expansion_factor, bias) for _ in range(num_blocks[3])])

        # Decoder
        self.up4_3 = Upsample(dim * 8)   # 8*dim -> 4*dim (via PixelShuffle)
        self.reduce3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            STCB(dim * 4, ffn_expansion_factor, bias) for _ in range(num_blocks[4])])

        self.up3_2 = Upsample(dim * 4)   # 4*dim -> 2*dim
        self.reduce2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            STCB(dim * 2, ffn_expansion_factor, bias) for _ in range(num_blocks[5])])

        self.up2_1 = Upsample(dim * 2)   # 2*dim -> dim
        # After skip: dim (up) + dim (enc1) = 2*dim
        self.decoder_level1 = nn.Sequential(*[
            STCB(dim * 2, ffn_expansion_factor, bias) for _ in range(num_blocks[6])])

        self.output = nn.Conv2d(dim * 2, out_channels, kernel_size=3,
                                stride=1, padding=1, bias=bias)

    def forward(self, inp_img, mask):
        """
        inp_img: (B, 3, H, W)  masked input image
        mask:    (B, 1, H, W)  binary damage mask (1=damaged)
        """
        # Pre-scales masks for each DAFD level
        mask_half    = F.interpolate(mask, scale_factor=0.5,  mode='nearest')
        mask_quarter = F.interpolate(mask, scale_factor=0.25, mode='nearest')

        # Embedding: concatenate image + mask channel
        x = self.patch_embed(torch.cat([inp_img, mask], dim=1))  # (B, dim, H, W)

        # Encoder
        enc1 = self.encoder_level1(x)                         # (B, dim,   H,   W)
        enc2 = self.encoder_level2(self.down1_2(enc1, mask))  # (B, 2dim, H/2, W/2)
        enc3 = self.encoder_level3(self.down2_3(enc2, mask_half))    # (B, 4dim, H/4, W/4)
        lat  = self.latent(self.down3_4(enc3, mask_quarter))         # (B, 8dim, H/8, W/8)

        # Decoder
        d3 = self.up4_3(lat)                              # (B, 4dim, H/4, W/4)
        d3 = self.reduce3(torch.cat([d3, enc3], dim=1))  # (B, 4dim, H/4, W/4)
        d3 = self.decoder_level3(d3)

        d2 = self.up3_2(d3)                               # (B, 2dim, H/2, W/2)
        d2 = self.reduce2(torch.cat([d2, enc2], dim=1))  # (B, 2dim, H/2, W/2)
        d2 = self.decoder_level2(d2)

        d1 = self.up2_1(d2)                               # (B, dim, H, W)
        d1 = torch.cat([d1, enc1], dim=1)                 # (B, 2dim, H, W)
        d1 = self.decoder_level1(d1)

        out = self.output(d1)
        out = (torch.tanh(out) + 1) / 2
        return out


# ---------------------------------------------------------------------------
# Discriminator (reused verbatim from HINT networks.py)
# ---------------------------------------------------------------------------

class BaseNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                          or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)
        self.apply(init_func)


class Discriminator(BaseNetwork):
    def __init__(self, in_channels=3, use_sigmoid=True,
                 use_spectral_norm=True, init_weights=True):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, 64, 4, 2, 1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(256, 512, 4, 1, 1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(512, 1, 4, 1, 1,
                                    bias=not use_spectral_norm), use_spectral_norm),
        )
        if init_weights:
            self.init_weights()

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        out = torch.sigmoid(c5) if self.use_sigmoid else c5
        return out, [c1, c2, c3, c4, c5]
