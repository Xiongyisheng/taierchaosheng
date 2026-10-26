# 胎儿超声 10 分类实验复现包

本目录于 2026-09-26 从 AutoDL“医院”实例（ID：`cq2dpc1jga-f378cb37`）的 `/root/autodl-tmp` 保存本课题的 baseline 与主体实验文件。服务器原始脚本和模型文件按相同的相对路径保存；`source_sha256.txt` 记录服务器原件的 SHA-256，可运行 `python verify_archive.py` 检查本地副本是否逐字节一致。

## 归档范围

- `train_dinov3.py`：DINOv3 全参数微调 baseline 训练脚本，从本地官方预训练权重加载骨干，接 10 类分类头，并继续训练骨干与分类头。
- `train_dinov3_mcph.py`：创新点一 MCPH（K=2）训练脚本。
- `train_dinov3_mcph_ppem.py`：最终 MCPH+PPEM 训练脚本。它从 DINOv3 预训练权重独立训练，不加载上一阶段的 MCPH checkpoint。
- `split/`：原实验固定的患者级划分，含 `patient_split.csv` 和 train/val/test 三表（分别 4,906 / 611 / 620 张；三个集合的患者无交叉）。
- `models/DINOv3-ViT-B16/`：实验使用的本地 DINOv3 预训练模型及原始许可文件。
- `experiments/DINOv3_ViTB16_FullFT_Baseline_v1/`、`experiments/DINOv3_ViTB16_MCPH_v1/`、`experiments/DINOv3_ViTB16_MCPH_PPEM_v1/`：各自的最佳 checkpoint、配置、环境记录及汇总指标。没有保存其他对比模型或 `last_model.pt`。
- `dataset/fetal_ultrasound.tar`：服务器当前 10 类图像目录的完整快照；已在服务器逐文件核对与在线目录内容一致。解包后的图像应位于 `fetal_ultrasound/`。
- `requirements_freeze.txt`：服务器原始完整环境记录；`requirements_repro_linux.txt`：从其中抽取的这三份训练脚本直接依赖，供相同 Linux/Python/CUDA 组合重建环境。

数据集文件名和划分表含患者标识，请限制访问，不要公开上传整个复现包。

在本地可运行 `python extract_dataset.py` 安全解包并逐文件校验图像；已经解包时可直接运行 `python check_dataset.py` 检查其与固定划分的对应关系。

## 使用方法

原始脚本固定使用 `/root/autodl-tmp` 作为工作目录。复现时先把**本地复现包复制到一台新的 Linux 实验机**的该目录，保持上述相对结构，并将 `dataset/fetal_ultrasound.tar` 解包到 `/root/autodl-tmp/fetal_ultrasound/`。压缩包根目录直接是 10 个类别文件夹，可用：

```bash
mkdir -p /root/autodl-tmp/fetal_ultrasound
tar -xf /root/autodl-tmp/dataset/fetal_ultrasound.tar -C /root/autodl-tmp/fetal_ultrasound
```

不要直接在此备份目录运行训练，因为脚本会在 `experiments/` 中写入新的 checkpoint 和指标。

历史环境为 Python 3.12.3、PyTorch 2.8.0+cu128、CUDA 12.8、Transformers 5.16.1；实验记录的 GPU 是 NVIDIA RTX 4090。`requirements_freeze.txt` 内含 Anaconda 构建机的 `file:///croot/...` 路径，不能直接作为可移植的 `pip install -r` 输入。相同 Linux x86_64、Python 3.12、CUDA 12.8 环境可从 `requirements_repro_linux.txt` 安装这三份脚本的依赖；该派生清单尚未在全新环境中实装验证。

恢复路径和环境后，可分别执行：

```bash
cd /root/autodl-tmp
python train_dinov3.py
python train_dinov3_mcph.py
python train_dinov3_mcph_ppem.py
```

运行前先确认数据集已完整解包，且 `split/` 中固定划分对应的 6,137 张图像均能找到。脚本的随机种子固定为 42；不同 GPU、驱动和库实现仍可能造成数值差异。若仅需保留或加载已有模型，请使用各实验目录中的 `best_model.pt`，无需重新训练。

本地检查可在复现包目录运行 `python verify_archive.py`（核对服务器 SHA-256）和 `python check_dataset.py`（核对解包图像与固定划分）。

划分 CSV 保留实验时的旧 Windows 路径 `D:\胎儿超声\胎儿各切面汇总`；原脚本读取时只取类别文件夹名和图像文件名，再映射到 `DATA_ROOT`，因此不要为了新机器位置而改写这些固定划分记录。

## 2026-09-26 本地核验

- 服务器当前图像目录与 `fetal_ultrasound.tar` 内的 6,137 个图像逐文件 SHA-256 一致。
- 本地归档的 34 个服务器源文件与 `source_sha256.txt` 全部匹配（0 项错误）。
- 本地解包并逐文件校验 6,137 张图像；固定划分 4,906 / 611 / 620 张，无缺失图像、重复路径或跨集合患者。
- 三份训练脚本和三个本地校验工具通过 Python 语法解析。这里没有在本机 GPU 上重新训练，因此不宣称跨硬件重跑指标完全相同。
