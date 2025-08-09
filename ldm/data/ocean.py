import os
import json
import random
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import warnings

# ------------- 通道选择（先跑通可用的默认） -----------------
DEFAULT_USE_SST = True
DEFAULT_USE_SOS = True
DEFAULT_USE_THETAO = True
DEFAULT_USE_SO = True
DEFAULT_THETAO_DEPTH_IDX_LIST = [0]  # 只用表层
DEFAULT_SO_DEPTH_IDX_LIST = [0]
# ------------------------------------------------------------

def _to_float32(x):
    return x.astype(np.float32) if x.dtype != np.float32 else x

def _nan_to_num(x, fill_value=0.0):
    mask = ~np.isnan(x)
    x = np.nan_to_num(x, nan=fill_value)
    return x, mask

def _minmax_norm(x, min_val=None, max_val=None):
    if min_val is None:
        min_val = np.min(x)
    if max_val is None:
        max_val = np.max(x)
    if max_val <= min_val:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - min_val) / (max_val - min_val)).astype(np.float32)

def _to_minus1_1(x01):
    return (x01 * 2.0 - 1.0).astype(np.float32)

def _center_crop(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    H, W = arr.shape
    if H == out_h and W == out_w:
        return arr
    top = max(0, (H - out_h) // 2)
    left = max(0, (W - out_w) // 2)
    return arr[top:top + out_h, left:left + out_w]

class OceanDataset(Dataset):
    """
    从 .nc + JSON split 读取数据，输出 (C,H,W) 到 'image' 键，范围 [-1,1]
    """
    def __init__(
        self,
        nc_paths: List[str],
        split_json: str,
        split_key: str = "train",     # "train" | "val" | "test"
        crop_size: Tuple[int, int] = (128, 128),
        use_sst: bool = DEFAULT_USE_SST,
        use_sos: bool = DEFAULT_USE_SOS,
        use_thetao: bool = DEFAULT_USE_THETAO,
        use_so: bool = DEFAULT_USE_SO,
        thetao_depth_idx_list: List[int] = DEFAULT_THETAO_DEPTH_IDX_LIST,
        so_depth_idx_list: List[int] = DEFAULT_SO_DEPTH_IDX_LIST,
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

    def __len__(self):
        return len(self.indices)

    def _map_global_t(self, g_t: int) -> Tuple[int, int]:
        idx_ds = int(np.searchsorted(self.cum_time, g_t, side="right") - 1)
        local_t = g_t - self.cum_time[idx_ds]
        return idx_ds, local_t

    def _get_var_2d(self, ds: xr.Dataset, varname: str, t: int) -> np.ndarray:
        if varname not in ds.variables:
            warnings.warn(f"缺变量 {varname} ，填0通道")
            return np.zeros((self.lat_size, self.lon_size), dtype=np.float32)
        arr = ds[varname].isel(time=t).values
        arr = _to_float32(arr); arr, _ = _nan_to_num(arr, 0.0)
        return _to_minus1_1(_minmax_norm(arr))

    def _get_var_3d_depthpick(self, ds: xr.Dataset, varname: str, t: int, depth_idx_list: List[int]) -> List[np.ndarray]:
        if varname not in ds.variables:
            warnings.warn(f"缺变量 {varname} ，跳过")
            return []
        out = []
        for k in depth_idx_list:
            sub = ds[varname].isel(time=t, depth=k).values
            sub = _to_float32(sub); sub, _ = _nan_to_num(sub, 0.0)
            out.append(_to_minus1_1(_minmax_norm(sub)))
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        g_t = int(self.indices[idx])
        ds_id, local_t = self._map_global_t(g_t)
        ds = self.datasets[ds_id]

        channel_list = []
        if self.use_sst:
            channel_list.append(self._get_var_2d(ds, "analysed_sst", local_t))
        if self.use_sos:
            channel_list.append(self._get_var_2d(ds, "sos", local_t))
        if self.use_thetao:
            channel_list.extend(self._get_var_3d_depthpick(ds, "thetao_glor", local_t, self.thetao_depth_idx_list))
        if self.use_so:
            channel_list.extend(self._get_var_3d_depthpick(ds, "so_glor", local_t, self.so_depth_idx_list))

        if len(channel_list) == 0:
            channel_list.append(np.zeros((self.lat_size, self.lon_size), dtype=np.float32))

        img = np.stack(channel_list, axis=0)  # (C, H, W)

        # 中心裁剪
        C, H, W = img.shape
        out = np.zeros((C, self.crop_h, self.crop_w), dtype=np.float32)
        for c in range(C):
            out[c] = _center_crop(img[c], self.crop_h, self.crop_w)

        # 轻度增强
        if self.random_flip:
            if random.random() < 0.5:
                out = out[:, :, ::-1].copy()
            if random.random() < 0.5:
                out = out[:, ::-1, :].copy()

        return {"image": torch.from_numpy(out)}

    def close(self):
        for ds in self.datasets:
            ds.close()

def make_dataloader(
    nc_paths: List[str],
    split_json: str,
    split_key: str,
    batch_size: int,
    num_workers: int,
    **kwargs,
):
    dataset = OceanDataset(
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
