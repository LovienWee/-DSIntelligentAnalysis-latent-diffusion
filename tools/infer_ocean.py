# tools/infer_ocean.py
import os
import argparse
import torch
import xarray as xr
import numpy as np
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.data.ocean_inpaint import OceanInpaintDataset

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="configs/latent-diffusion/ocean_ldm_inpaint.yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="trained inpaint LDM ckpt")
    parser.add_argument("--out", type=str, required=True, help="output NetCDF path")
    parser.add_argument("--split_json", type=str, required=True)
    parser.add_argument("--nc_paths", nargs="+", required=True)
    parser.add_argument("--split_key", type=str, default="test")
    parser.add_argument("--crop_size", type=int, nargs=2, default=[128,128])
    parser.add_argument("--keep_ratio", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=50, help="DDIM steps")
    parser.add_argument("--eta", type=float, default=0.0)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    model = instantiate_from_config(cfg.model)
    state = torch.load(args.ckpt, map_location="cpu")
    if "state_dict" in state: state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model = model.to(args.device)
    model.eval()

    sampler = DDIMSampler(model)

    ds = OceanInpaintDataset(
        nc_paths=args.nc_paths,
        split_json=args.split_json,
        split_key=args.split_key,
        crop_size=tuple(args.crop_size),
        keep_ratio=args.keep_ratio,
        keep_ratio_jitter=0.0,
        random_flip=False
    )

    preds = []
    gts = []
    masks = []

    for i in range(len(ds)):
        sample = ds[i]
        image = sample["image"].unsqueeze(0).to(args.device)  # (1,C,H,W), [-1,1]
        cond  = sample["cond"].unsqueeze(0).to(args.device)   # (1,C+1,H,W)
        mask  = sample["mask"].numpy()                        # (1,H,W)

        # 条件潜向量（concat）
        cond_z = model.get_learned_conditioning(cond)         # (1, zc, h, w)
        zc = cond_z

        # 目标潜空间形状（与 VAE latent 一致）
        z = model.encode_first_stage(image)
        _, zc_dim, h, w = z.shape

        # 从高斯噪声开始用 DDIM 反演
        samples, _ = sampler.sample(
            S=args.steps,
            conditioning=zc,                # concat 模式下就是 cond_z
            batch_size=1,
            shape=(zc_dim, h, w),           # 目标 latent 形状
            verbose=False,
            eta=args.eta
        )
        rec = model.decode_first_stage(samples)               # (1,C,H,W) in [-1,1]

        preds.append(rec.squeeze(0).detach().cpu().numpy())
        gts.append(image.squeeze(0).detach().cpu().numpy())
        masks.append(mask)

    # [-1,1] -> [0,1]
    def inv(x):
        return np.clip((x + 1.0) * 0.5, 0.0, 1.0)

    pr = inv(np.stack(preds, axis=0))  # (T,C,H,W)
    gt = inv(np.stack(gts, axis=0))
    mk = np.stack(masks, axis=0)       # (T,1,H,W)

    T, C, H, W = pr.shape
    coords = {
        "time": np.arange(T),
        "channel": np.arange(C),
        "y": np.arange(H),
        "x": np.arange(W)
    }
    ds_out = xr.Dataset(
        {
            "pred": (("time","channel","y","x"), pr.astype(np.float32)),
            "gt":   (("time","channel","y","x"), gt.astype(np.float32)),
            "mask": (("time","y","x"), mk.astype(np.float32).squeeze(1))
        },
        coords=coords
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ds_out.to_netcdf(args.out)
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
