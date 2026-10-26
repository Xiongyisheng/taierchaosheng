#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DINOv3 ViT-B/16 + MCPH + PPEM — 胎儿超声标准切面 10 分类
==========================================================
正式创新点二：Patch-to-Prototype Evidence Matching（PPEM，Patch-原型证据匹配模块）。

最终完整模型 E2 = MCPH（创新点一，保持不变 K=2）+ PPEM（创新点二）。

结构：
    DINOv3 Final Hidden State
       ↙              ↘
    CLS Token      Patch Tokens (196)
       ↓                ↓
      MCPH             PPEM
       ↓                ↓
   Global Logits    Local Logits (Top-8 patch evidence mean)
       ↘              ↙
         Logit Addition
       final = global + 0.20 * local
              ↓
          10 Classes

关键：不是 feature fusion，MCPH 与 PPEM 都产生 class-level logits，最后固定权重相加。
PPEM 复用 MCPH 的 prototypes（不新增 prototype 参数），仅新增一个 Patch LayerNorm。

E2 独立从官方 facebook/dinov3-vitb16-pretrain-lvd1689m 开始训练，不加载 MCPH checkpoint。
"""

import argparse
import json
import math
import os
import random
import time
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from PIL import Image

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, roc_auc_score,
)

from transformers import AutoImageProcessor, AutoModel

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ====================================================================
# 常量配置（与 baseline/MCPH 完全一致）
# ====================================================================
SEED = 42
EPOCHS = 50
BATCH_SIZE = 32
IMG_SIZE = 224
BACKBONE_LR = 1e-5
HEAD_LR = 1e-4
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 5
EARLY_STOP_PATIENCE = 10
GRAD_CLIP_MAX_NORM = 1.0
NUM_CLASSES = 10
NUM_WORKERS = 16
NUM_PROTOTYPES = 2
MCPH_SCALE = 20.0
TOP_K_PATCHES = 8
LOCAL_WEIGHT = 0.20

CLASS_NAMES = ["丘脑横切面", "侧脑室横切面", "小脑横切面", "双眼横切面", "鼻唇冠状切面",
               "颜面正中矢状面", "脊柱纵切面", "双肾矢状切面", "双肾横切面", "腹围横切面"]

DATA_ROOT = "/root/autodl-tmp/fetal_ultrasound"
SPLIT_DIR = "/root/autodl-tmp/split"
MODEL_DIR = "/root/autodl-tmp/models/DINOv3-ViT-B16"
EXP_NAME = "DINOv3_ViTB16_MCPH_PPEM_v1"
EXP_DIR = f"/root/autodl-tmp/experiments/{EXP_NAME}"

# 已有消融结果（用于对比）
BASELINE = {
    "test_accuracy": 0.985484, "macro_precision": 0.985137, "macro_recall": 0.983619,
    "macro_f1": 0.983957, "weighted_f1": 0.985441, "macro_roc_auc": 0.999721,
    "class0_f1": 0.9307, "class1_f1": 0.9412,
    "total_errors": 9, "thalamus_to_ventricle": 6, "ventricle_to_thalamus": 1,
}
MCPH = {
    "test_accuracy": 0.987097, "macro_precision": 0.986072, "macro_recall": 0.985651,
    "macro_f1": 0.985835, "weighted_f1": 0.987095, "macro_roc_auc": 0.999701,
    "class0_f1": 0.943396, "class1_f1": 0.947368,
    "total_errors": 8, "thalamus_to_ventricle": 3, "ventricle_to_thalamus": 3,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_server_path(win_path, data_root):
    p = win_path.replace("\\", "/")
    parts = p.split("/")
    folder = parts[-2]
    filename = parts[-1]
    return os.path.join(data_root, folder, filename)


def load_csv(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        import csv as _csv
        reader = _csv.DictReader(f)
        for r in reader:
            r["class_id"] = int(r["class_id"])
            rows.append(r)
    return rows


# ====================================================================
# 数据一致性检查（完全一致）
# ====================================================================
def data_consistency_check():
    print("=" * 60)
    print("数据一致性检查")
    print("=" * 60)
    train = load_csv(os.path.join(SPLIT_DIR, "train.csv"))
    val = load_csv(os.path.join(SPLIT_DIR, "val.csv"))
    test = load_csv(os.path.join(SPLIT_DIR, "test.csv"))

    print(f"train.csv = {len(train)} images (期望 4906)")
    print(f"val.csv   = {len(val)} images (期望 611)")
    print(f"test.csv  = {len(test)} images (期望 620)")
    assert len(train) == 4906 and len(val) == 611 and len(test) == 620
    assert len(train) + len(val) + len(test) == 6137

    for split_name, split in [("train", train), ("val", val), ("test", test)]:
        cids = {r["class_id"] for r in split}
        assert cids == set(range(10)), f"{split_name} class_id 异常"
        for r in split:
            assert 0 <= r["class_id"] <= 9

    train_p = {r["patient_id"] for r in train}
    val_p = {r["patient_id"] for r in val}
    test_p = {r["patient_id"] for r in test}
    print(f"Train ∩ Val  = {len(train_p & val_p)}")
    print(f"Train ∩ Test = {len(train_p & test_p)}")
    print(f"Val ∩ Test   = {len(val_p & test_p)}")
    assert train_p.isdisjoint(val_p) and train_p.isdisjoint(test_p) and val_p.isdisjoint(test_p)

    all_rows = train + val + test
    missing = []
    print(f"检查 {len(all_rows)} 个文件存在性 ...")
    for r in all_rows:
        spath = to_server_path(r["filepath"], DATA_ROOT)
        if not os.path.exists(spath):
            missing.append(spath)
    print(f"missing files = {len(missing)}")
    assert len(missing) == 0, "存在缺失文件，终止训练！"

    print("DATA LEAKAGE CHECK = PASS")
    print("=" * 60)
    return train, val, test


# ====================================================================
# Dataset（完全一致）
# ====================================================================
class FetalUltrasoundDataset(Dataset):
    def __init__(self, rows, data_root, transform=None):
        self.rows = rows
        self.data_root = data_root
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        spath = to_server_path(r["filepath"], self.data_root)
        image = Image.open(spath).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, r["class_id"], r["patient_id"], spath


# ====================================================================
# MCPH（创新点一，保持不变）
# ====================================================================
class MultiCenterPrototypeHead(nn.Module):
    def __init__(self, dim=768, num_classes=10, num_prototypes=2, scale=20.0):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.scale = scale
        self.norm = nn.LayerNorm(dim)
        self.prototypes = nn.Parameter(torch.randn(num_classes, num_prototypes, dim))
        nn.init.trunc_normal_(self.prototypes, std=0.02)

    def forward(self, x):
        x = self.norm(x)
        x = F.normalize(x, dim=-1)
        p = F.normalize(self.prototypes, dim=-1)
        similarity = torch.einsum('bd,ckd->bck', x, p)
        scaled = self.scale * similarity
        logits = torch.logsumexp(scaled, dim=-1) - math.log(self.num_prototypes)
        return logits, similarity


# ====================================================================
# PPEM（创新点二）
# ====================================================================
class PPEM(nn.Module):
    """Patch-to-Prototype Evidence Matching。

    复用 MCPH 的 prototypes，对每个 patch 计算与所有类别原型的 cosine 相似度，
    LogSumExp 聚合同类两个原型得到 patch_class_score [B,N,C]，
    对每个类别取 Top-K 个最高 patch 得分的均值作为 local_logits。
    """

    def __init__(self, dim=768, num_classes=10, num_prototypes=2, scale=20.0, top_k=8):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.scale = scale
        self.top_k = top_k
        self.patch_norm = nn.LayerNorm(dim)

    def forward(self, patch_tokens, prototypes):
        # patch_tokens: [B, N, D]
        # prototypes: [C, K, D]（复用 MCPH，此处 L2 归一化）
        x = self.patch_norm(patch_tokens)          # [B, N, D]
        x = F.normalize(x, dim=-1)                 # [B, N, D]
        p = F.normalize(prototypes, dim=-1)        # [C, K, D]
        patch_sim = torch.einsum('bnd,ckd->bnck', x, p)   # [B, N, C, K]
        scaled = self.scale * patch_sim
        patch_class_score = torch.logsumexp(scaled, dim=-1) - math.log(self.num_prototypes)  # [B, N, C]
        pcs = patch_class_score.transpose(1, 2)    # [B, C, N]
        topk_vals, topk_idx = torch.topk(pcs, self.top_k, dim=-1)  # [B, C, top_k]
        local_logits = topk_vals.mean(dim=-1)      # [B, C]
        return local_logits, patch_class_score, topk_idx, x


# ====================================================================
# 模型：DINOv3 + MCPH + PPEM
# ====================================================================
class DINOv3MCPHPPEMClassifier(nn.Module):
    def __init__(self, backbone, hidden_size, num_classes,
                 num_prototypes=2, scale=20.0, top_k=8, local_weight=0.20,
                 num_register_tokens=4, num_patches=196):
        super().__init__()
        self.backbone = backbone
        self.num_register_tokens = num_register_tokens
        self.num_patches = num_patches
        self.local_weight = local_weight
        self.mcph = MultiCenterPrototypeHead(hidden_size, num_classes, num_prototypes, scale)
        self.ppem = PPEM(hidden_size, num_classes, num_prototypes, scale, top_k)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        hidden = outputs.last_hidden_state                 # [B, 1+4+196, D]
        cls_token = hidden[:, 0]                           # [B, D]
        patch_start = 1 + self.num_register_tokens
        patch_tokens = hidden[:, patch_start:patch_start + self.num_patches]  # [B, 196, D]

        global_logits, _ = self.mcph(cls_token)            # [B, C]
        local_logits, patch_class_score, topk_idx, _ = self.ppem(patch_tokens, self.mcph.prototypes)

        final_logits = global_logits + self.local_weight * local_logits
        return final_logits, global_logits, local_logits, patch_class_score, topk_idx


# ====================================================================
# 指标
# ====================================================================
def compute_metrics(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": acc,
        "macro_precision": macro_p, "macro_recall": macro_r, "macro_f1": macro_f1,
        "weighted_precision": weighted_p, "weighted_recall": weighted_r, "weighted_f1": weighted_f1,
    }


def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / max(EPOCHS - WARMUP_EPOCHS, 1)
    return 0.5 * (1 + math.cos(math.pi * progress))


# ====================================================================
# 评估（可选收集 PPEM 辅助信息）
# ====================================================================
@torch.no_grad()
def evaluate(model, loader, criterion, device, amp_dtype, collect_aux=False):
    model.eval()
    total_loss = 0.0
    y_true, y_pred, y_prob = [], [], []
    global_preds, patch_scores, topk_all, paths_all = [], [], [], []
    for images, labels, pids, paths in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            final_logits, global_logits, local_logits, patch_class_score, topk_idx = model(images)
            loss = criterion(final_logits, labels)
        total_loss += loss.item() * images.size(0)
        probs = torch.softmax(final_logits, dim=1).float()
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(probs.argmax(1).cpu().tolist())
        y_prob.append(probs.cpu().numpy())
        if collect_aux:
            gprobs = torch.softmax(global_logits, dim=1).float()
            global_preds.extend(gprobs.argmax(1).cpu().tolist())
            patch_scores.append(patch_class_score.float().cpu().numpy())   # [B, N, C]
            topk_all.append(topk_idx.cpu().numpy())                        # [B, C, K]
            paths_all.extend(paths)
    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)
    y_prob = np.concatenate(y_prob, axis=0).astype(np.float32)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics["loss"] = total_loss / len(loader.dataset)
    if collect_aux:
        global_preds = np.array(global_preds, dtype=np.int64)
        patch_scores = np.concatenate(patch_scores, axis=0).astype(np.float32)  # [N, 196, 10]
        topk_all = np.concatenate(topk_all, axis=0)  # [N, 10, 8]
        return metrics, y_true, y_pred, y_prob, global_preds, patch_scores, topk_all, paths_all
    return metrics, y_true, y_pred, y_prob


# ====================================================================
# 训练一个 epoch
# ====================================================================
def train_one_epoch(model, loader, optimizer, criterion, device, amp_dtype, scaler):
    model.train()
    total_loss = 0.0
    y_true, y_pred = [], []
    for images, labels, pids, paths in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            final_logits, _, _, _, _ = model(images)
            loss = criterion(final_logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(final_logits.argmax(1).cpu().tolist())
    acc = accuracy_score(y_true, y_pred)
    return {"loss": total_loss / len(loader.dataset), "accuracy": acc}


# ====================================================================
# 保存 checkpoint
# ====================================================================
def save_checkpoint(path, model, optimizer, epoch, best_val_f1, config):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_macro_f1": best_val_f1,
        "class_to_idx": {i: CLASS_NAMES[i] for i in range(NUM_CLASSES)},
        "model_name": "DINOv3-ViT-B16-MCPH-PPEM",
        "input_size": IMG_SIZE,
        "seed": SEED,
        "training_config": config,
    }, path)


# ====================================================================
# Smoke Test
# ====================================================================
def smoke_test(model, train_loader, val_loader, optimizer, criterion, device, amp_dtype, scaler):
    print("=" * 60)
    print("SMOKE TEST (MCPH + PPEM)")
    print("=" * 60)
    model.train()
    try:
        batch = next(iter(train_loader))
        images, labels, pids, paths = batch
        print(f"1. 数据读取: OK (batch {tuple(images.shape)})")
        images = images.to(device)
        labels = labels.to(device)

        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            outputs = model.backbone(pixel_values=images)
            hidden = outputs.last_hidden_state
            cls_token = hidden[:, 0]
            patch_start = 1 + model.num_register_tokens
            patch_tokens = hidden[:, patch_start:patch_start + model.num_patches]

            # MCPH
            global_logits, global_sim = model.mcph(cls_token)
            proto_norm = F.normalize(model.mcph.prototypes, dim=-1)
            # PPEM
            pn = model.ppem.patch_norm(patch_tokens)
            pn = F.normalize(pn, dim=-1)
            patch_sim = torch.einsum('bnd,ckd->bnck', pn, proto_norm)
            patch_class_score = torch.logsumexp(MCPH_SCALE * patch_sim, dim=-1) - math.log(NUM_PROTOTYPES)
            pcs = patch_class_score.transpose(1, 2)
            topk_vals, topk_idx = torch.topk(pcs, TOP_K_PATCHES, dim=-1)
            local_logits = topk_vals.mean(dim=-1)
            final_logits = global_logits + LOCAL_WEIGHT * local_logits
            loss = criterion(final_logits, labels)

        print(f"2. CLS shape = {tuple(cls_token.shape)}")
        print(f"3. Patch shape = {tuple(patch_tokens.shape)}")
        print(f"4. Prototype shape = {tuple(proto_norm.shape)}")
        print(f"5. Global similarity shape = {tuple(global_sim.shape)}")
        print(f"6. Patch similarity shape = {tuple(patch_sim.shape)}")
        print(f"7. Patch class score shape = {tuple(patch_class_score.shape)}")
        print(f"8. Local logits shape = {tuple(local_logits.shape)}")
        print(f"9. Global logits shape = {tuple(global_logits.shape)}")
        print(f"10. Final logits shape = {tuple(final_logits.shape)}")
        print(f"11. loss = {loss.item():.4f} (finite={math.isfinite(loss.item())})")

        assert tuple(cls_token.shape) == (images.shape[0], 768)
        assert tuple(patch_tokens.shape) == (images.shape[0], 196, 768)
        assert tuple(proto_norm.shape) == (NUM_CLASSES, NUM_PROTOTYPES, 768)
        assert tuple(patch_sim.shape) == (images.shape[0], 196, NUM_CLASSES, NUM_PROTOTYPES)
        assert tuple(patch_class_score.shape) == (images.shape[0], 196, NUM_CLASSES)
        assert tuple(local_logits.shape) == (images.shape[0], NUM_CLASSES)
        assert tuple(global_logits.shape) == (images.shape[0], NUM_CLASSES)
        assert tuple(final_logits.shape) == (images.shape[0], NUM_CLASSES)
        assert math.isfinite(loss.item())

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
        scaler.step(optimizer)
        scaler.update()
        print("12. backward + optimizer.step: OK")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[FAIL] 前向/反向失败: {e}")
        return False

    mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"13. GPU 显存峰值(allocated): {mem:.0f} MB")

    try:
        save_checkpoint(os.path.join(EXP_DIR, "best_model.pt"), model, optimizer, 0, 0.0, {})
        print("14. checkpoint 保存: OK")
    except Exception as e:
        print(f"[FAIL] checkpoint 保存失败: {e}")
        return False

    try:
        metrics, _, _, _ = evaluate(model, val_loader, criterion, device, amp_dtype)
        print(f"15. validation metrics: OK (val acc {metrics['accuracy']:.4f}, val macro_f1 {metrics['macro_f1']:.4f})")
    except Exception as e:
        print(f"[FAIL] validation 失败: {e}")
        return False

    print("MCPH + PPEM SMOKE TEST STATUS = PASS")
    print("=" * 60)
    return True


# ====================================================================
# PPEM 证据热力图
# ====================================================================
def save_ppem_examples(selected, out_dir, patch_grid=14):
    os.makedirs(out_dir, exist_ok=True)
    for tag, path, tid, pid, patch_score, topk_idx in selected:
        try:
            img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        except Exception as e:
            print(f"[WARN] 无法读取 {path}: {e}")
            continue
        # patch_score: [196] 某一类别的局部证据
        score_2d = patch_score.reshape(patch_grid, patch_grid)
        smax, smin = score_2d.max(), score_2d.min()
        score_norm = (score_2d - smin) / (smax - smin + 1e-8)

        # Top-K 掩码
        topk_flat = np.array(topk_idx).flatten().astype(int)  # [K]
        mask_2d = np.zeros(patch_grid * patch_grid, dtype=bool)
        for t in topk_flat:
            if 0 <= t < patch_grid * patch_grid:
                mask_2d[t] = True
        mask_2d = mask_2d.reshape(patch_grid, patch_grid)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(img); axes[0].set_title("Original"); axes[0].axis("off")
        im1 = axes[1].imshow(score_2d, cmap="jet")
        axes[1].set_title("Local Evidence (class logits)"); axes[1].axis("off"); plt.colorbar(im1, ax=axes[1], fraction=0.046)
        axes[2].imshow(mask_2d, cmap="Reds", vmin=0, vmax=1)
        axes[2].set_title(f"Top-{TOP_K_PATCHES} selected patches"); axes[2].axis("off")
        axes[3].imshow(img); axes[3].imshow(mask_2d, cmap="Reds", alpha=0.5)
        axes[3].set_title("Overlay"); axes[3].axis("off")
        fig.suptitle(f"{tag}\ntrue={CLASS_NAMES[tid]}  pred={CLASS_NAMES[pid]}")
        fig.savefig(os.path.join(out_dir, f"{tag}.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)


# ====================================================================
# 主流程
# ====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="仅运行 smoke test")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    global EXP_DIR
    os.makedirs(EXP_DIR, exist_ok=True)
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("环境信息")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"PyTorch: {torch.__version__}")
    import transformers
    print(f"transformers: {transformers.__version__}")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"AMP dtype: {amp_dtype}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)

    env_info = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "python": os.sys.version,
        "amp_dtype": str(amp_dtype),
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(EXP_DIR, "environment.txt"), "w", encoding="utf-8") as f:
        for k, v in env_info.items():
            f.write(f"{k}: {v}\n")

    train_rows, val_rows, test_rows = data_consistency_check()

    # 图像预处理（完全一致）
    processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
    mean = list(getattr(processor, "image_mean", [0.485, 0.456, 0.406]))
    std = list(getattr(processor, "image_std", [0.229, 0.224, 0.225]))
    print(f"Normalization mean={mean}, std={std}")

    train_transform = v2.Compose([
        v2.RandomResizedCrop(size=IMG_SIZE, scale=(0.8, 1.0)),
        v2.RandomRotation(degrees=10),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        v2.ToTensor(),
        v2.Normalize(mean=mean, std=std),
    ])
    eval_transform = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(IMG_SIZE),
        v2.ToTensor(),
        v2.Normalize(mean=mean, std=std),
    ])

    train_ds = FetalUltrasoundDataset(train_rows, DATA_ROOT, train_transform)
    val_ds = FetalUltrasoundDataset(val_rows, DATA_ROOT, eval_transform)
    test_ds = FetalUltrasoundDataset(test_rows, DATA_ROOT, eval_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
                            persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
                             persistent_workers=True)

    # 模型（重新从官方 pretrained 开始）
    backbone = AutoModel.from_pretrained(MODEL_DIR)
    hidden_size = backbone.config.hidden_size
    num_register_tokens = getattr(backbone.config, "num_register_tokens", 0)
    patch_size = getattr(backbone.config, "patch_size", 16)
    num_patches = (IMG_SIZE // patch_size) ** 2
    print(f"hidden_size={hidden_size}, num_register_tokens={num_register_tokens}, num_patches={num_patches}")

    model = DINOv3MCPHPPEMClassifier(
        backbone, hidden_size, NUM_CLASSES,
        num_prototypes=NUM_PROTOTYPES, scale=MCPH_SCALE,
        top_k=TOP_K_PATCHES, local_weight=LOCAL_WEIGHT,
        num_register_tokens=num_register_tokens, num_patches=num_patches,
    ).to(device)

    # 参数量统计
    backbone_params = sum(p.numel() for p in backbone.parameters())
    mcph_params = sum(p.numel() for p in model.mcph.parameters())
    ppem_params = sum(p.numel() for p in model.ppem.parameters())
    linear_head_params = sum(p.numel() for p in nn.Linear(hidden_size, NUM_CLASSES).parameters())
    baseline_total = backbone_params + linear_head_params
    mcph_total = backbone_params + mcph_params
    mcph_ppem_total = backbone_params + mcph_params + ppem_params
    n_params = mcph_ppem_total
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 60)
    print("参数量统计")
    print("=" * 60)
    print(f"Baseline parameters     = {baseline_total:,}")
    print(f"MCPH parameters         = {mcph_total:,}")
    print(f"MCPH + PPEM parameters  = {mcph_ppem_total:,}")
    print(f"MCPH added (vs baseline)= {mcph_total - baseline_total:,}")
    print(f"PPEM added (vs MCPH)    = {ppem_params:,}")
    print(f"Total added (vs baseline)= {mcph_ppem_total - baseline_total:,}")
    print(f"Parameter increase %     = {(mcph_ppem_total - baseline_total) / baseline_total * 100:.4f}%")
    print(f"可训练参数 = {n_trainable:,}")
    print("=" * 60)

    # 优化器（differential LR：backbone 1e-5，MCPH+PPEM 1e-4）
    optimizer = torch.optim.AdamW([
        {"params": backbone.parameters(), "lr": BACKBONE_LR},
        {"params": model.mcph.parameters(), "lr": HEAD_LR},
        {"params": model.ppem.parameters(), "lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=(amp_dtype == torch.float16))

    config = {
        "experiment": EXP_NAME,
        "model": "DINOv3-ViT-B16 + MCPH + PPEM",
        "innovation1": "Multi-Center Prototype Head (MCPH, K=2, 保持不变)",
        "innovation2": "Patch-to-Prototype Evidence Matching (PPEM)",
        "ppem": {
            "structure": "Patch LN -> L2 -> cosine with MCPH prototypes -> LogSumExp -> Top-K patch mean",
            "top_k_patches": TOP_K_PATCHES,
            "local_weight": LOCAL_WEIGHT,
            "scale": MCPH_SCALE,
            "fusion": "final_logits = global_logits + LOCAL_WEIGHT * local_logits（固定权重，非门控/特征融合）",
        },
        "model_dir": MODEL_DIR,
        "num_classes": NUM_CLASSES,
        "num_prototypes": NUM_PROTOTYPES,
        "seed": SEED,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "input_size": IMG_SIZE,
        "backbone_lr": BACKBONE_LR,
        "head_lr": HEAD_LR,
        "weight_decay": WEIGHT_DECAY,
        "warmup_epochs": WARMUP_EPOCHS,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "grad_clip_max_norm": GRAD_CLIP_MAX_NORM,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealing + LinearWarmup",
        "loss": "CrossEntropyLoss (仅 final_logits)",
        "amp": str(amp_dtype),
        "baseline_total_params": baseline_total,
        "mcph_total_params": mcph_total,
        "mcph_ppem_total_params": mcph_ppem_total,
        "mcph_added": mcph_total - baseline_total,
        "ppem_added": ppem_params,
        "total_added": mcph_ppem_total - baseline_total,
        "train_transform": "RandomResizedCrop + RandomRotation(±10) + ColorJitter",
        "eval_transform": "Resize(256) + CenterCrop(224)",
        "normalization_mean": mean,
        "normalization_std": std,
        "train_images": len(train_rows),
        "val_images": len(val_rows),
        "test_images": len(test_rows),
    }
    import yaml
    with open(os.path.join(EXP_DIR, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)

    # Smoke test
    if not smoke_test(model, train_loader, val_loader, optimizer, criterion, device, amp_dtype, scaler):
        print("MCPH + PPEM SMOKE TEST STATUS = FAIL，终止！")
        return

    if args.smoke:
        print("仅 smoke test，退出。")
        return

    # ================= 正式训练 =================
    print("=" * 60)
    print("正式训练开始")
    print("=" * 60)
    metrics_rows = []
    best_val_f1 = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    train_start = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, amp_dtype, scaler)
        val_metrics, _, _, _ = evaluate(model, val_loader, criterion, device, amp_dtype)
        scheduler.step()
        epoch_time = time.time() - epoch_start

        lr_backbone = optimizer.param_groups[0]["lr"]
        lr_head = optimizer.param_groups[1]["lr"]

        row = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "lr_backbone": lr_backbone,
            "lr_head": lr_head,
            "epoch_time_s": epoch_time,
        }
        metrics_rows.append(row)

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"train_loss {train_metrics['loss']:.4f} train_acc {train_metrics['accuracy']:.4f} | "
              f"val_loss {val_metrics['loss']:.4f} val_acc {val_metrics['accuracy']:.4f} "
              f"val_macro_f1 {val_metrics['macro_f1']:.4f} | lr_h {lr_head:.2e} | {epoch_time:.1f}s",
              flush=True)

        save_checkpoint(os.path.join(EXP_DIR, "last_model.pt"), model, optimizer, epoch + 1, best_val_f1, config)

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch + 1
            epochs_no_improve = 0
            save_checkpoint(os.path.join(EXP_DIR, "best_model.pt"), model, optimizer, epoch + 1, best_val_f1, config)
            print(f"  >>> 新的 best model (val_macro_f1={best_val_f1:.4f})", flush=True)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"Early stopping 触发（{EARLY_STOP_PATIENCE} epochs 无提升），停止于 epoch {epoch+1}", flush=True)
            break

    total_train_time = time.time() - train_start

    import csv as _csv
    with open(os.path.join(EXP_DIR, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
        w.writeheader()
        w.writerows(metrics_rows)

    # ================= Test 最终评估（唯一一次） =================
    print("=" * 60)
    print(f"训练结束，best_epoch={best_epoch}, best_val_macro_f1={best_val_f1:.4f}")
    print("加载 best_model.pt 进行 Test 评估（唯一一次）")
    print("=" * 60)

    ckpt = torch.load(os.path.join(EXP_DIR, "best_model.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics, y_true, y_pred, y_prob, global_preds, patch_scores, topk_all, paths_all = evaluate(
        model, test_loader, criterion, device, amp_dtype, collect_aux=True)

    per_class = precision_recall_fscore_support(y_true, y_pred, labels=list(range(10)), zero_division=0)
    p_class, r_class, f_class, support = per_class

    roc_macro = None
    roc_per_class = [None] * 10
    try:
        y_prob_norm = y_prob / y_prob.sum(axis=1, keepdims=True)
        roc_macro = float(roc_auc_score(y_true, y_prob_norm, multi_class="ovr", average="macro"))
        roc_per_class = [float(x) for x in roc_auc_score(y_true, y_prob_norm, multi_class="ovr", average=None)]
    except Exception as e:
        print(f"[WARN] ROC-AUC 计算失败: {e}")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    total_errors = int(cm.sum() - np.trace(cm))
    thalamus_to_ventricle = int(cm[0, 1])
    ventricle_to_thalamus = int(cm[1, 0])

    # ================= PPEM correction/harm 分析 =================
    final_pred = y_pred
    global_pred = global_preds
    corrected_mask = (global_pred != y_true) & (final_pred == y_true)
    harmed_mask = (global_pred == y_true) & (final_pred != y_true)
    n_corrected = int(corrected_mask.sum())
    n_harmed = int(harmed_mask.sum())

    # 保存 correction 分析
    with open(os.path.join(EXP_DIR, "ppem_correction_analysis.csv"), "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["filepath", "true_class_id", "true_class_name", "global_pred", "final_pred", "status"])
        for i in range(len(y_true)):
            if corrected_mask[i]:
                status = "corrected"
            elif harmed_mask[i]:
                status = "harmed"
            else:
                status = "unchanged"
            w.writerow([paths_all[i], int(y_true[i]), CLASS_NAMES[int(y_true[i])],
                        CLASS_NAMES[int(global_pred[i])], CLASS_NAMES[int(final_pred[i])], status])

    # ================= PPEM 证据热力图 =================
    ATTN_DIR = os.path.join(EXP_DIR, "ppem_evidence_examples")
    os.makedirs(ATTN_DIR, exist_ok=True)
    idx = np.arange(len(y_true))
    patch_grid = int(round(math.sqrt(patch_scores.shape[1])))

    selected = []
    c0 = idx[(y_true == 0) & (final_pred == 0)][:2]
    for k, i in enumerate(c0):
        selected.append((f"correct_class0_thalamus_{k}", paths_all[i], 0, 0,
                         patch_scores[i, :, 0], topk_all[i, 0]))
    c1 = idx[(y_true == 1) & (final_pred == 1)][:2]
    for k, i in enumerate(c1):
        selected.append((f"correct_class1_ventricle_{k}", paths_all[i], 1, 1,
                         patch_scores[i, :, 1], topk_all[i, 1]))
    e01 = idx[(y_true == 0) & (final_pred == 1)]
    for k, i in enumerate(e01):
        selected.append((f"error_thalamus_to_ventricle_{k}", paths_all[i], 0, 1,
                         patch_scores[i, :, 1], topk_all[i, 1]))
    e10 = idx[(y_true == 1) & (final_pred == 0)]
    for k, i in enumerate(e10):
        selected.append((f"error_ventricle_to_thalamus_{k}", paths_all[i], 1, 0,
                         patch_scores[i, :, 0], topk_all[i, 0]))
    cor = idx[corrected_mask]
    for k, i in enumerate(cor[:5]):
        selected.append((f"ppem_corrected_{k}", paths_all[i], int(y_true[i]), int(final_pred[i]),
                         patch_scores[i, :, int(final_pred[i])], topk_all[i, int(final_pred[i])]))
    har = idx[harmed_mask]
    for k, i in enumerate(har[:5]):
        selected.append((f"ppem_harmed_{k}", paths_all[i], int(y_true[i]), int(final_pred[i]),
                         patch_scores[i, :, int(final_pred[i])], topk_all[i, int(final_pred[i])]))

    save_ppem_examples(selected, ATTN_DIR)
    print(f"PPEM evidence examples 已保存到 {ATTN_DIR}（共 {len(selected)} 张）")

    # ================= 结果保存 =================
    test_result = {
        "best_epoch": best_epoch,
        "best_val_accuracy": metrics_rows[best_epoch - 1]["val_acc"] if best_epoch > 0 else None,
        "best_val_macro_f1": best_val_f1,
        "test_accuracy": float(test_metrics["accuracy"]),
        "macro_precision": float(test_metrics["macro_precision"]),
        "macro_recall": float(test_metrics["macro_recall"]),
        "macro_f1": float(test_metrics["macro_f1"]),
        "weighted_precision": float(test_metrics["weighted_precision"]),
        "weighted_recall": float(test_metrics["weighted_recall"]),
        "weighted_f1": float(test_metrics["weighted_f1"]),
        "macro_roc_auc": roc_macro,
        "per_class_precision": [float(x) for x in p_class],
        "per_class_recall": [float(x) for x in r_class],
        "per_class_f1": [float(x) for x in f_class],
        "per_class_support": [int(x) for x in support],
        "per_class_roc_auc": roc_per_class,
        "total_errors": total_errors,
        "thalamus_to_ventricle": thalamus_to_ventricle,
        "ventricle_to_thalamus": ventricle_to_thalamus,
        "ppem_corrected": n_corrected,
        "ppem_harmed": n_harmed,
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / 1024 / 1024),
        "total_train_time_s": float(total_train_time),
        "n_params": n_params,
        "n_trainable_params": n_trainable,
        "baseline_total_params": baseline_total,
        "mcph_total_params": mcph_total,
        "mcph_ppem_total_params": mcph_ppem_total,
        "mcph_added": mcph_total - baseline_total,
        "ppem_added": ppem_params,
        "total_added": mcph_ppem_total - baseline_total,
    }
    with open(os.path.join(EXP_DIR, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_result, f, ensure_ascii=False, indent=2)

    report_txt = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0)
    with open(os.path.join(EXP_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_txt)

    with open(os.path.join(EXP_DIR, "test_predictions.csv"), "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        header = ["filepath", "patient_id", "true_class_id", "true_class_name",
                  "pred_class_id", "pred_class_name", "pred_probability"] + \
                 [f"prob_class_{i}" for i in range(10)]
        w.writerow(header)
        for i, row in enumerate(test_rows):
            spath = to_server_path(row["filepath"], DATA_ROOT)
            prob = y_prob[i]
            pred_cid = int(y_pred[i])
            w.writerow([spath, row["patient_id"], int(y_true[i]), CLASS_NAMES[int(y_true[i])],
                        pred_cid, CLASS_NAMES[pred_cid], float(prob[pred_cid])] +
                       [float(p) for p in prob])

    # 混淆矩阵图
    def plot_cm(cm_data, path, title, fmt):
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(cm_data, cmap="Blues")
        ax.set_xticks(range(10)); ax.set_yticks(range(10))
        ax.set_xticklabels([f"{i}" for i in range(10)])
        ax.set_yticklabels([f"{i}-{CLASS_NAMES[i][:4]}" for i in range(10)])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(title)
        for i in range(10):
            for j in range(10):
                ax.text(j, i, fmt(cm_data[i, j]), ha="center", va="center",
                        fontsize=8, color="white" if cm_data[i, j] > cm_data.max() * 0.5 else "black")
        plt.colorbar(im)
        plt.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    plot_cm(cm, os.path.join(EXP_DIR, "confusion_matrix_raw.png"), "Confusion Matrix (Raw)", lambda x: f"{int(x)}")
    plot_cm(cm_norm, os.path.join(EXP_DIR, "confusion_matrix_normalized.png"),
            "Confusion Matrix (Row-Normalized)", lambda x: f"{x:.2f}")

    # 训练曲线
    epochs_list = [r["epoch"] for r in metrics_rows]
    def plot_curve(x, ys, labels, ylabel, path, title):
        fig, ax = plt.subplots(figsize=(10, 6))
        for y, lab in zip(ys, labels):
            ax.plot(x, y, marker="o", markersize=3, label=lab)
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.savefig(path, dpi=150); plt.close(fig)

    plot_curve(epochs_list,
               [[r["train_loss"] for r in metrics_rows], [r["val_loss"] for r in metrics_rows]],
               ["train_loss", "val_loss"], "Loss", os.path.join(EXP_DIR, "loss_curve.png"), "Loss Curve")
    plot_curve(epochs_list,
               [[r["train_acc"] for r in metrics_rows], [r["val_acc"] for r in metrics_rows]],
               ["train_acc", "val_acc"], "Accuracy", os.path.join(EXP_DIR, "accuracy_curve.png"), "Accuracy Curve")
    plot_curve(epochs_list,
               [[r["val_macro_f1"] for r in metrics_rows]],
               ["val_macro_f1"], "Macro F1", os.path.join(EXP_DIR, "macro_f1_curve.png"), "Val Macro F1 Curve")
    plot_curve(epochs_list,
               [[r["lr_backbone"] for r in metrics_rows], [r["lr_head"] for r in metrics_rows]],
               ["backbone_lr", "head_lr"], "Learning Rate", os.path.join(EXP_DIR, "learning_rate_curve.png"), "Learning Rate Curve")

    # ================= final_ablation.md =================
    def pct2(x): return f"{x * 100:.4f}%"
    def d(a, b): return f"{(a - b) * 100:+.4f}%"

    L = []
    L.append("# 最终消融：Baseline vs +MCPH vs +MCPH+PPEM\n")
    L.append("## 总体指标")
    L.append("| Metric | Baseline | +MCPH | +MCPH+PPEM |")
    L.append("|---|---|---|---|")
    L.append(f"| Accuracy | {pct2(BASELINE['test_accuracy'])} | {pct2(MCPH['test_accuracy'])} | {pct2(test_result['test_accuracy'])} |")
    L.append(f"| Macro Precision | {pct2(BASELINE['macro_precision'])} | {pct2(MCPH['macro_precision'])} | {pct2(test_result['macro_precision'])} |")
    L.append(f"| Macro Recall | {pct2(BASELINE['macro_recall'])} | {pct2(MCPH['macro_recall'])} | {pct2(test_result['macro_recall'])} |")
    L.append(f"| Macro F1 | {pct2(BASELINE['macro_f1'])} | {pct2(MCPH['macro_f1'])} | {pct2(test_result['macro_f1'])} |")
    L.append(f"| Weighted F1 | {pct2(BASELINE['weighted_f1'])} | {pct2(MCPH['weighted_f1'])} | {pct2(test_result['weighted_f1'])} |")
    L.append(f"| Macro ROC-AUC | {pct2(BASELINE['macro_roc_auc'])} | {pct2(MCPH['macro_roc_auc'])} | {pct2(test_result['macro_roc_auc'])} |")
    L.append("")
    L.append("## 目标类别")
    L.append("| 指标 | Baseline | +MCPH | +MCPH+PPEM |")
    L.append("|---|---|---|---|")
    L.append(f"| 丘脑 F1 | {pct2(BASELINE['class0_f1'], 2)} | {pct2(MCPH['class0_f1'], 2)} | {pct2(test_result['per_class_f1'][0], 2)} |")
    L.append(f"| 侧脑室 F1 | {pct2(BASELINE['class1_f1'], 2)} | {pct2(MCPH['class1_f1'], 2)} | {pct2(test_result['per_class_f1'][1], 2)} |")
    L.append("")
    L.append("## 混淆数量")
    L.append("| 混淆 | Baseline | +MCPH | +MCPH+PPEM |")
    L.append("|---|---|---|---|")
    L.append(f"| 丘脑→侧脑室 | {BASELINE['thalamus_to_ventricle']} | {MCPH['thalamus_to_ventricle']} | {test_result['thalamus_to_ventricle']} |")
    L.append(f"| 侧脑室→丘脑 | {BASELINE['ventricle_to_thalamus']} | {MCPH['ventricle_to_thalamus']} | {test_result['ventricle_to_thalamus']} |")
    L.append(f"| Total errors | {BASELINE['total_errors']} | {MCPH['total_errors']} | {test_result['total_errors']} |")
    L.append("")
    L.append("## 创新贡献")
    L.append("| Delta | 创新点一 (MCPH-Base) | 创新点二 (PPEM) | 完整 (MCPH+PPEM-Base) |")
    L.append("|---|---|---|---|")
    L.append(f"| Accuracy | {d(MCPH['test_accuracy'], BASELINE['test_accuracy'])} | {d(test_result['test_accuracy'], MCPH['test_accuracy'])} | {d(test_result['test_accuracy'], BASELINE['test_accuracy'])} |")
    L.append(f"| Macro F1 | {d(MCPH['macro_f1'], BASELINE['macro_f1'])} | {d(test_result['macro_f1'], MCPH['macro_f1'])} | {d(test_result['macro_f1'], BASELINE['macro_f1'])} |")
    L.append(f"| Macro Recall | {d(MCPH['macro_recall'], BASELINE['macro_recall'])} | {d(test_result['macro_recall'], MCPH['macro_recall'])} | {d(test_result['macro_recall'], BASELINE['macro_recall'])} |")
    L.append(f"| 丘脑 F1 | {d(MCPH['class0_f1'], BASELINE['class0_f1'])} | {d(test_result['per_class_f1'][0], MCPH['class0_f1'])} | {d(test_result['per_class_f1'][0], BASELINE['class0_f1'])} |")
    L.append(f"| 侧脑室 F1 | {d(MCPH['class1_f1'], BASELINE['class1_f1'])} | {d(test_result['per_class_f1'][1], MCPH['class1_f1'])} | {d(test_result['per_class_f1'][1], BASELINE['class1_f1'])} |")
    L.append("")
    L.append("## PPEM 分析")
    L.append(f"- Corrected samples = {n_corrected}")
    L.append(f"- Harmed samples = {n_harmed}")

    with open(os.path.join(EXP_DIR, "final_ablation.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    # ================= 最终打印 =================
    print("=" * 60)
    print("训练完成")
    print("=" * 60)
    print(f"Best epoch: {best_epoch}")
    print(f"Best Val Accuracy: {test_result['best_val_accuracy']:.4f}")
    print(f"Best Val Macro F1: {best_val_f1:.4f}")
    print(f"Test Accuracy: {test_result['test_accuracy']:.4f}")
    print(f"Test Macro Precision: {test_result['macro_precision']:.4f}")
    print(f"Test Macro Recall: {test_result['macro_recall']:.4f}")
    print(f"Test Macro F1: {test_result['macro_f1']:.4f}")
    print(f"Test Weighted F1: {test_result['weighted_f1']:.4f}")
    if roc_macro is not None:
        print(f"Test Macro ROC-AUC: {test_result['macro_roc_auc']:.4f}")
    print("Per-class F1:")
    for i in range(10):
        print(f"  class {i} ({CLASS_NAMES[i]}): f1={f_class[i]:.4f}, p={p_class[i]:.4f}, r={r_class[i]:.4f}, support={support[i]}")
    print(f"丘脑 → 侧脑室 = {thalamus_to_ventricle}")
    print(f"侧脑室 → 丘脑 = {ventricle_to_thalamus}")
    print(f"Total test errors = {total_errors}")
    print(f"PPEM corrected = {n_corrected}, harmed = {n_harmed}")
    print(f"Peak GPU memory: {test_result['peak_gpu_memory_mb']:.0f} MB")
    print(f"Total training time: {total_train_time:.1f}s")
    print(f"Experiment dir: {EXP_DIR}")
    print("DATA LEAKAGE CHECK = PASS")
    print("MCPH + PPEM SMOKE TEST STATUS = PASS")
    print("EXPERIMENT STATUS: PASS")


if __name__ == "__main__":
    main()
