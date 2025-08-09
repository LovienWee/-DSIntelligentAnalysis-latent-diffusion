
---

# 🐬 Ocean Latent Diffusion 重构项目运行指南

本项目基于 **Latent Diffusion Model (LDM)**，结合自编码器（VAE）实现 **海洋三维温盐场重构**。流程包括两阶段训练（Autoencoder → Latent Diffusion）、条件补全（inpainting）以及推理评估。

---

## 📂 数据准备

1. 将原始 `.nc` 数据放入：

   ```
   oceandata/
     ├── IndianOcean/*.nc
     ├── SouthChinaSea/*.nc
     └── Split/*.json
   ```

2. 确保 `Split/*.json` 包含训练、验证、测试的索引划分。

---

## 🛠 环境配置

1. 安装 Miniconda（如已安装可跳过）：

   ```bash
   wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
   bash miniconda.sh -b -p "/root/miniconda3"
   eval "$(/root/miniconda3/bin/conda shell.bash hook)"
   ```

2. 创建并激活环境：

   ```bash
   conda env create -f environment.yaml -n ldm
   conda activate ldm
   ```

3. 安装缺失依赖：

   ```bash
   pip install xarray netCDF4
   pip install --no-cache-dir --force-reinstall "torchmetrics==0.6.0" "pytorch-lightning==1.4.2"
   ```

---

## 🚀 运行流程

### 1️⃣ 训练 Autoencoder（VAE）

```bash
python main.py --base configs/autoencoder/ocean_kl.yaml -t --gpus 0,
```

* 输出示例：`logs/autoencoder/version_x/checkpoints/last.ckpt`

---

### 2️⃣ 训练 Latent Diffusion Model（无条件）

1. 将上一步的 VAE `last.ckpt` 路径写入：

   ```yaml
   first_stage_config:
     ckpt_path: "logs/autoencoder/version_x/checkpoints/last.ckpt"
   ```

   （对应 `configs/latent-diffusion/ocean_ldm.yaml`）

2. 运行训练：

   ```bash
   python main.py --base configs/latent-diffusion/ocean_ldm.yaml -t --gpus 0,
   ```

---

### 3️⃣ 训练 Inpainting 条件 LDM

1. 将 `ocean_ldm_inpaint.yaml` 中的 VAE ckpt 替换为你的路径：

   ```yaml
   first_stage_config:
     ckpt_path: "logs/autoencoder/version_x/checkpoints/last.ckpt"
   ```

2. 运行训练：

   ```bash
   python main.py --base configs/latent-diffusion/ocean_ldm_inpaint.yaml -t --gpus 0,
   ```

---

### 4️⃣ 推理（重构稀疏观测）

```bash
python scripts/infer_ocean.py \
  --config configs/latent-diffusion/ocean_ldm_inpaint.yaml \
  --ckpt logs/inpaint/version_x/checkpoints/last.ckpt \
  --input_dir path/to/sparse_obs/ \
  --output_dir outputs/reconstructed_nc/
```

---

### 5️⃣ 评估

```bash
python scripts/eval_ocean.py \
  --pred_dir outputs/reconstructed_nc/ \
  --gt_dir path/to/ground_truth_nc/ \
  --metrics RMSE MAE SSIM
```

* 输出：误差指标、可视化剖面图、空间分布误差图。

---

## 📊 输出结果

* **训练日志**：`logs/`
* **模型权重**：`logs/*/checkpoints/*.ckpt`
* **重构 NetCDF 文件**：`outputs/reconstructed_nc/`
* **评估图表**：`outputs/eval_plots/`

---

## 📌 注意事项

* 确保 `.nc` 文件的变量名与 `ocean.py` 中的 `var_list` 一致；
* 在推理与评估阶段，数据会自动反归一化回物理单位（°C、PSU）；
* 稀疏观测的掩膜生成策略可在 `ocean_inpaint.py` 中调整。

---
