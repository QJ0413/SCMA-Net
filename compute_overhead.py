"""
计算 HINT / SCMA-Net / SyFormer 的计算开销（CPU 兼容版）：
  - 参数量 (Params, M)
  - FLOPs (G, 256x256 输入)
  - 推理时间 (ms/image, batch=1)
"""
import os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

# 把项目根目录和 src 加入路径
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.networks import HINT
from src.scma_networks import SCMANet
from Syformer import SyFormer
from thop import profile, clever_format

# Patch HINT's Downsample.forward to avoid hardcoded .cuda() (CPU compatibility)
def _downsample_forward_cpu(self, x, mask):
    out = self.body(x)
    out_mask = self.body2(mask)
    b, n, h, w = out.shape
    t = torch.zeros((b, 2 * n, h, w), dtype=out.dtype, device=out.device)
    for i in range(n):
        t[:, 2 * i, :, :] = out[:, i, :, :]
    for i in range(n):
        if i <= 3:
            t[:, 2 * i + 1, :, :] = out_mask[:, i, :, :]
        else:
            t[:, 2 * i + 1, :, :] = out_mask[:, i % 4, :, :]
    return self.proj(t)

from src.networks import Downsample as _HintDown
_HintDown.forward = _downsample_forward_cpu

device = 'cpu'
H, W = 256, 256


# ---------------------------------------------------------------------------
# 参数量
# ---------------------------------------------------------------------------
def count_params(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total / 1e6


# ---------------------------------------------------------------------------
# FLOPs（thop）
# ---------------------------------------------------------------------------
def count_flops(model, dummy_inputs):
    """dummy_inputs: tuple of tensors passed to model.forward"""
    flops, params = profile(model, inputs=dummy_inputs, verbose=False)
    return flops / 1e9   # GFLOPs


# ---------------------------------------------------------------------------
# 推理时间
# ---------------------------------------------------------------------------
def measure_time(model, dummy_inputs, warmup=5, iters=20):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(*dummy_inputs)
        t0 = time.time()
        for _ in range(iters):
            model(*dummy_inputs)
        return (time.time() - t0) * 1000 / iters   # ms


# ---------------------------------------------------------------------------
# HINT 包装（thop 要求单 forward call）
# ---------------------------------------------------------------------------
class HINTWrapper(nn.Module):
    def __init__(self, hint_net):
        super().__init__()
        self.net = hint_net

    def forward(self, inp_img, mask):
        mh = F.interpolate(mask, scale_factor=0.5,    mode='nearest')
        mq = F.interpolate(mask, scale_factor=0.25,   mode='nearest')
        mt = F.interpolate(mask, scale_factor=0.125,  mode='nearest')
        return self.net(inp_img, mask, mh, mq, mt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f'Device: {device}  |  Input: {H}x{W}  |  Batch: 1\n')

    # ── HINT ──────────────────────────────────────────────────────────────
    hint_raw = HINT(inp_channels=4, out_channels=3, dim=48,
                    num_blocks=[4, 6, 6, 8], heads=[1, 2, 4, 8]).eval()
    hint = HINTWrapper(hint_raw).eval()

    p_hint = count_params(hint)

    img  = torch.randn(1, 3, H, W)
    mask = torch.randint(0, 2, (1, 1, H, W)).float()

    f_hint = count_flops(hint, (img, mask))
    t_hint = measure_time(hint, (img, mask))

    # ── SCMA-Net ───────────────────────────────────────────────────────────
    scma = SCMANet(inp_channels=4, out_channels=3, dim=48,
                   num_blocks=[4, 6, 6, 8, 6, 6, 4]).eval()

    p_scma = count_params(scma)
    f_scma = count_flops(scma, (img, mask))
    t_scma = measure_time(scma, (img, mask))

    # ── SyFormer ───────────────────────────────────────────────────────────
    # SyFormer takes two images + mask; input size must be divisible by n_win=8
    syformer = SyFormer(num_classes=3, normal_init=False).eval()
    p_sy = count_params(syformer)
    f_sy = count_flops(syformer, (img, img, mask))
    t_sy = measure_time(syformer, (img, img, mask))

    # ── 打印结果 ──────────────────────────────────────────────────────────
    header = f'{"Metric":<22}{"HINT":>12}{"SCMA-Net":>12}{"SyFormer":>12}'
    sep    = '-' * 58
    print(header)
    print(sep)
    print(f'{"Params (M)":<22}{p_hint:>12.2f}{p_scma:>12.2f}{p_sy:>12.2f}')
    print(f'{"FLOPs (G)":<22}{f_hint:>12.2f}{f_scma:>12.2f}{f_sy:>12.2f}')
    print(f'{"Inference (ms)":<22}{t_hint:>12.2f}{t_scma:>12.2f}{t_sy:>12.2f}')
    print(sep)

    # ── 论文用小结 ────────────────────────────────────────────────────────
    print('\n[论文数值参考]')
    print(f'  HINT     : Params={p_hint:.2f}M  FLOPs={f_hint:.2f}G  Time={t_hint:.1f}ms')
    print(f'  SCMA-Net : Params={p_scma:.2f}M  FLOPs={f_scma:.2f}G  Time={t_scma:.1f}ms')
    print(f'  SyFormer : Params={p_sy:.2f}M  FLOPs={f_sy:.2f}G  Time={t_sy:.1f}ms')


if __name__ == '__main__':
    main()
