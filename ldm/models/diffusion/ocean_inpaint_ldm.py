# ldm/models/diffusion/ocean_inpaint_ldm.py
import torch
from typing import Any
from ldm.models.diffusion.ddpm import LatentDiffusion

class OceanInpaintLDM(LatentDiffusion):
    """
    使用 concat 条件：将 'cond'（masked_image+mask，其中我们只编码 masked_image 前C通道）编码到潜空间，
    与噪声潜变量在 UNet 输入通道拼接。
    """

    def __init__(self, *args, **kwargs):
        # 强制使用 concat 条件，并从 batch['cond'] 取条件
        kwargs.setdefault("conditioning_key", "concat")
        super().__init__(*args, **kwargs)
        # 告诉父类：条件来自 batch 的哪个 key
        self.cond_stage_key = "cond"

    @torch.no_grad()
    def get_learned_conditioning(self, c: Any):
        """
        c: (B, C+1, H, W) ，我们只取前 C 通道（masked_image）走 VAE 编码
        """
        if isinstance(c, dict) and "cond" in c:
            c = c["cond"]
        # 仅编码 masked_image 部分（去掉最后的 mask 通道）
        if c.dim() == 4 and c.size(1) >= 2:
            masked = c[:, :-1, ...]
        else:
            masked = c
        return self.encode_first_stage(masked)  # (B, zc, h, w)

    def apply_model(self, x_noisy, t, cond=None, **kwargs):
        """
        将 cond_z 通道拼接到 x_noisy，然后送入 UNet。
        DDIMSampler/训练都会调用到这里。
        """
        assert cond is not None, "concat 条件不能为空"
        x_in = torch.cat([x_noisy, cond], dim=1)
        return self.model(x_in, t, **kwargs)
