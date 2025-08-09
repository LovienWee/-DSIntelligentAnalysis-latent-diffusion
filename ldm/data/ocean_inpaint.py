# ldm/data/ocean_inpaint.py
import os
import json
import random
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import warnings

# ---------------- 可按需调整的设置 ----------------
DEFAULT_USE_SST = True
DEFAULT_USE_SOS = True
DEFAULT_USE_THETAO = True
DEFAULT_USE_SO = True
DEFAULT_THETAO_DEPTH_IDX_LIST = [0, 2, 4]
DEFAULT_SO_DEPTH_IDX_LIST = [0, 2, 4]
# -------------------------------------------------

def _to_float32(x):
    return x.astype(np.float32) if x.dtype != np.float32 else x

def _nan_to_num(x, fill_value=0.0):
    mask = ~np.isnan(x)
    x = np.nan_to_num(x, nan=fill_value)
    return x, mask

def _minmax_norm(x, min_val=None, max_val=None):
    if min_val is None:
        min_val = float(np.min(x))
    if max_val is None:
        max_val = float(np.max(x))
    if max_val <= min_val:
        return np.zeros_like(x, dtype=np.float32), (min_val, max_val)
    x01 = (x - min_val) / (max_val - min_val)
    return x01.astype(np.float32), (min_val, max_val)

def _to_minus1_1(x01):
    return (x01 * 2.0 - 1.0).astype(np.float32)

def _center_crop(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    H, W = arr.shape
    if H == out_h and W == out_w:
        return arr
    top = max(0, (H - out_h) // 2)
    left = max(0, (W - out_w) // 2)
    return arr[top:top + out_h, left:left + out_w]

def _rand_mask(shape: Tuple[int, int], keep_ratio: float, min_block: int = 4, max_block: int = 16) -> np.ndarray:
    """块状稀疏掩膜：1=观测保留，0=缺测"""
    H, W = shape
    mask = np.zeros((H, W), dtype=np.float32)
    target = int(H * W * keep_ratio)
    filled = 0
    rng = np.random.default_rng()
    while filled < target:
        bh = int(rng.integers(min_block, max_block + 1))
        bw = int(rng.integers(min_block, max_block + 1))
        r = int(rng.integers(0, max(1, H - bh + 1)))
        c = int(rng.integers(0, max(1, W - bw + 1)))
        mask[r:r + bh, c:c + bw] = 1.0
        filled = int(mask.sum())
    return mask

class OceanInpaintDataset(Dataset):
    """
    输出：
      image: (C,H,W) 完整目标 [-1,1]
      cond : (C+1,H,W) = masked_image (前C通道) + mask (最后1通道)
      mask : (1,H,W)
    """
    def __init__(
        self,
        nc_paths: List[str],
        split_json: str,
        split_key: str = "train",
        crop_size: Tuple[int, int] = (128, 128),
        use_sst: bool = DEFAULT_USE_SST,
        use_sos: bool = DEFAULT_USE_SOS,
        use_thetao: bool = DEFAULT_USE_THETAO,
        use_so: bool = DEFAULT_USE_SO,
        thetao_depth_idx_list: List[int] = DEFAULT_THETAO_DEPTH_IDX_LIST,
        so_depth_idx_list: List[int] = DEFAULT_SO_DEPTH_IDX_LIST,
        keep_ratio: float = 0.1,
        keep_ratio_jitter: float = 0.05,
        external_mask_nc: Optional[str] = None,  # 可选：提供预制 mask 的 nc（变量名 'mask'）
        random_flip: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.nc_paths = nc_paths
        self.split_json = split_json
        self.split_key = split_key
        self.crop_h, self.crop_w = crop_size
        self.use_sst = use_sst
        self.use_sos = use_sos
        self.use_thetao = use_thetao
        self.use_so = use_so
        self.thetao_depth_idx_list = thetao_depth_idx_list
        self.so_depth_idx_list = so_depth_idx_list
        self.keep_ratio = keep_ratio
        self.keep_ratio_jitter = keep_ratio_jitter
        self.random_flip = random_flip
        random.seed(seed)
        np.random.seed(seed)

        self.datasets = [xr.open_dataset(p) for p in nc_paths]
        self.time_lens = [ds.sizes["time"] for ds in self.datasets]
        self.cum_time = np.cumsum([0] + self.time_lens)

        with open(split_json, "r", encoding="utf-8") as f:
            split = json.load(f)
        if self.split_key not in split:
            raise ValueError(f"{split_json} 缺少键 {self.split_key}")
        self.indices = split[self.split_key]

        sample_ds = self.datasets[0]
        self.lat_size = sample_ds.sizes["latitude"]
        self.lon_size = sample_ds.sizes["longitude"]

        self.external_mask = None
        if external_mask_nc is not None and os.path.exists(external_mask_nc):
            dsm = xr.open_dataset(external_mask_nc)
            if "mask" in dsm.variables:
                self.external_mask = dsm["mask"]  # 期望与 (time,lat,lon) 对齐
            else:
                warnings.warn(f"{external_mask_nc} 无变量 'mask'，忽略。")

        self.norm_stats = {}  # varname -> (min,max)，供需要时反归一

    def __len__(self):
        return len(self.indices)

    def _map_global_t(self, g_t: int) -> Tuple[int, int]:
        idx_ds = int(np.searchsorted(self.cum_time, g_t, side="right") - 1)
        local_t = g_t - self.cum_time[idx_ds]
        return idx_ds, local_t

    def _get_var_2d(self, ds: xr.Dataset, varname: str, t: int) -> np.ndarray:
        if varname not in ds.variables:
            warnings.warn(f"缺变量 {varname} ，填0通道")
            arr = np.zeros((self.lat_size, self.lon_size), dtype=np.float32)
            self.norm_stats.setdefault(varname, (0.0, 1.0))
            return _to_minus1_1(arr)
        arr = ds[varname].isel(time=t).values
        arr = _to_float32(arr); arr, _ = _nan_to_num(arr, 0.0)
        x01, (mn, mx) = _minmax_norm(arr)
        self.norm_stats.setdefault(varname, (mn, mx))
        return _to_minus1_1(x01)

    def _get_var_3d_depthpick(self, ds: xr.Dataset, varname: str, t: int, depth_idx_list: List[int]) -> List[np.ndarray]:
        if varname not in ds.variables:
            warnings.warn(f"缺变量 {varname} ，跳过")
            return []
        out = []
        raw = ds[varname].isel(time=t)  # (depth, lat, lon)
        for k in depth_idx_list:
            sub = raw.isel(depth=k).values
            sub = _to_float32(sub); sub, _ = _nan_to_num(sub, 0.0)
            x01, (mn, mx) = _minmax_norm(sub)
            self.norm_stats.setdefault(f"{varname}_depth{k}", (mn, mx))
            out.append(_to_minus1_1(x01))
        return out

    def _build_channels(self, ds, t):
        channel_list = []
        names = []
        if self.use_sst:
            channel_list.append(self._get_var_2d(ds, "analysed_sst", t)); names.append("analysed_sst")
        if self.use_sos:
            channel_list.append(self._get_var_2d(ds, "sos", t)); names.append("sos")
        if self.use_thetao:
            arrs = self._get_var_3d_depthpick(ds, "thetao_glor", t, self.thetao_depth_idx_list)
            channel_list.extend(arrs)
            names.extend([f"thetao_glor_d{k}" for k in self.thetao_depth_idx_list])
        if self.use_so:
            arrs = self._get_var_3d_depthpick(ds, "so_glor", t, self.so_depth_idx_list)
            channel_list.extend(arrs)
            names.extend([f"so_glor_d{k}" for k in self.so_depth_idx_list])
        if len(channel_list) == 0:
            channel_list.append(np.zeros((self.lat_size, self.lon_size), dtype=np.float32)); names.append("zeros")
        img = np.stack(channel_list, axis=0)  # (C,H,W)
        return img, names

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        g_t = int(self.indices[idx])
        ds_id, local_t = self._map_global_t(g_t)
        ds = self.datasets[ds_id]

        img, _ = self._build_channels(ds, local_t)  # (C,H,W)

        # 中心裁剪
        C, H, W = img.shape
        out = np.zeros((C, self.crop_h, self.crop_w), dtype=np.float32)
        for c in range(C):
            out[c] = _center_crop(img[c], self.crop_h, self.crop_w)

        # 随机增强（翻转）
        if self.random_flip and self.split_key == "train":
            if random.random() < 0.5:
                out = out[:, :, ::-1].copy()
            if random.random() < 0.5:
                out = out[:, ::-1, :].copy()

        # mask
        if self.external_mask is not None:
            m = self.external_mask.isel(time=local_t).values.astype(np.float32)
            m = _center_crop(m, self.crop_h, self.crop_w)
        else:
            kr = np.clip(np.random.normal(self.keep_ratio, self.keep_ratio_jitter), 0.02, 0.8)
            m = _rand_mask((self.crop_h, self.crop_w), keep_ratio=float(kr))

        masked = out * m[None, :, :]
        cond = np.concatenate([masked, m[None, :, :]], axis=0)  # (C+1,H,W)

        return {
            "image": torch.from_numpy(out),
            "cond": torch.from_numpy(cond),
            "mask": torch.from_numpy(m[None, :, :])
        }

    def close(self):
        for ds in self.datasets:
            ds.close()

def make_inpaint_dataloader(
    nc_paths: List[str],
    split_json: str,
    split_key: str,
    batch_size: int,
    num_workers: int,
    **kwargs,
):
    dataset = OceanInpaintDataset(
        nc_paths=nc_paths,
        split_json=split_json,
        split_key=split_key,
        **kwargs,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split_key == "train"),
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )
    return loader
