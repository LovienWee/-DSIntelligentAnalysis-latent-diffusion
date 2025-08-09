# tools/eval_ocean.py
import argparse
import xarray as xr
import numpy as np

def psnr(pred, gt, eps=1e-8):
    mse = np.mean((pred - gt) ** 2)
    if mse < eps: return 99.0
    return 10.0 * np.log10(1.0 / mse)  # [0,1] 区间

def ssim_simple(pred, gt, C1=0.01**2, C2=0.03**2):
    mu_x = pred.mean(); mu_y = gt.mean()
    var_x = pred.var();  var_y = gt.var()
    cov   = ((pred - mu_x) * (gt - mu_y)).mean()
    return ((2*mu_x*mu_y + C1)*(2*cov + C2)) / ((mu_x**2 + mu_y**2 + C1)*(var_x + var_y + C2) + 1e-8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc", type=str, required=True, help="infer output .nc")
    args = parser.parse_args()

    ds = xr.open_dataset(args.nc)
    pr = ds["pred"].values  # (T,C,H,W) in [0,1]
    gt = ds["gt"].values

    T, C, H, W = pr.shape
    rmse_c = []; mae_c = []; psnr_c = []; ssim_c = []
    for c in range(C):
        p = pr[:, c]; g = gt[:, c]
        rmse_c.append(np.sqrt(np.mean((p-g)**2)))
        mae_c.append(np.mean(np.abs(p-g)))
        psnr_c.append(psnr(p, g))
        ssim_c.append(ssim_simple(p, g))

    print("=== Metrics (averaged over time & space) ===")
    for name, arr in [("RMSE", rmse_c), ("MAE", mae_c), ("PSNR", psnr_c), ("SSIM", ssim_c)]:
        print(f"{name} per-channel:", [float(x) for x in arr])
        print(f"{name} mean:", float(np.mean(arr)))

if __name__ == "__main__":
    main()
