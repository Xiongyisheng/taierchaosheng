#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DINOv3 ViT-B/16 + MCPH — 胎儿超声标准切面 10 分类
=====================================================
正式创新点一：Multi-Center Prototype Head（MCPH，多中心原型分类头）。

在 DINOv3_ViTB16_FullFT_Baseline_v1 基础上，仅将分类头
    CLS -> Dropout(0.1) -> Linear(768,10)
替换为
    CLS -> LayerNorm -> L2 Normalize -> Multi-Center Prototype Matching -> Logits

每个类别 K=2 个 learnable prototypes（共 20 个），用 cosine similarity 匹配，
类别得分为 LogSumExp 聚合两个原型。只修改分类头，其余与 baseline 完全一致。

注意：E1 必须重新从 facebook/dinov3-vitb16-pretrain-lvd1689m 官方权重开始，
不加载 baseline / ALGF 的 best_model.pt。
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
# 常量配置（与 baseline 完全一致）
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

CLASS_NAMES = ["丘脑横切面", "侧脑室横切面", "小脑横切面", "双眼横切面", "鼻唇冠状切面",
               "颜面正中矢状面", "脊柱纵切面", "双肾矢状切面", "双肾横切面", "腹围横切面"]

DATA_ROOT = "/root/autodl-tmp/fetal_ultrasound"
SPLIT_DIR = "/root/autodl-tmp/split"
MODEL_DIR = "/root/autodl-tmp/models/DINOv3-ViT-B16"
EXP_NAME = "DINOv3_ViTB16_MCPH_v1"
EXP_DIR = f"/root/autodl-tmp/experiments/{EXP_NAME}"

# baseline 参考指标（来自 DINOv3_ViTB16_FullFT_Baseline_v1）
BASELINE = {
    "test_accuracy": 0.985484,
    "macro_precision": 0.985137,
    "macro_recall": 0.983619,
    "macro_f1": 0.983957,
    "weighted_f1": 0.985441,
    "macro_roc_auc": 0.999721,
    "class0_precision": 0.9792, "class0_recall": 0.8868, "class0_f1": 0.9307,
    "class1_precision": 0.9032, "class1_recall": 0.9825, "class1_f1": 0.9412,
    "total_errors": 9,
    "thalamus_to_ventricle": 6,
    "ventricle_to_thalamus": 1,
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
# 数据一致性检查（与 baseline 完全一致）
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
    assert len(train) == 4906, "train 数量不符！"
    assert len(val) == 611, "val 数量不符！"
    assert len(test) == 620, "test 数量不符！"
    assert len(train) + len(val) + len(test) == 6137, "总数不符！"

    for split_name, split in [("train", train), ("val", val), ("test", test)]:
        cids = {r["class_id"] for r in split}
        assert cids == set(range(10)), f"{split_name} class_id 集合异常: {cids}"
        for r in split:
            assert 0 <= r["class_id"] <= 9, f"class_id 越界: {r['class_id']}"

    train_p = {r["patient_id"] for r in train}
    val_p = {r["patient_id"] for r in val}
    test_p = {r["patient_id"] for r in test}
    print(f"Train ∩ Val  = {len(train_p & val_p)}")
    print(f"Train ∩ Test = {len(train_p & test_p)}")
    print(f"Val ∩ Test   = {len(val_p & test_p)}")
    assert train_p.isdisjoint(val_p), "Train/Val 患者泄漏！"
    assert train_p.isdisjoint(test_p), "Train/Test 患者泄漏！"
    assert val_p.isdisjoint(test_p), "Val/Test 患者泄漏！"

    all_rows = train + val + test
    missing = []
    corrupted = []
    print(f"检查 {len(all_rows)} 个文件存在性/完整性 ...")
    for r in all_rows:
        spath = to_server_path(r["filepath"], DATA_ROOT)
        if not os.path.exists(spath):
            missing.append(spath)
            continue
        try:
            with Image.open(spath) as im:
                im.verify()
        except Exception:
            corrupted.append(spath)

    print(f"missing files = {len(missing)}")
    print(f"corrupted files = {len(corrupted)}")
    if missing:
        for m in missing[:20]:
            print(f"  [MISSING] {m}")
    if corrupted:
        with open(os.path.join(EXP_DIR, "corrupted_images.csv"), "w", encoding="utf-8") as f:
            f.write("filepath\n")
            for c in corrupted:
                f.write(c + "\n")

    assert len(missing) == 0, "存在缺失文件，终止训练！"

    print("DATA LEAKAGE CHECK = PASS")
    print("=" * 60)
    return train, val, test


# ====================================================================
# Dataset（与 baseline 完全一致）
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
# MCPH（正式创新点一）
# ====================================================================
class MultiCenterPrototypeHead(nn.Module):
    """Multi-Center Prototype Head

    CLS -> LayerNorm -> L2 Normalize -> 与 10×K 个原型做 cosine 匹配
         -> LogSumExp 聚合同类别的 K 个原型 -> logits。
    每个类别 K 个 learnable prototypes（默认 K=2），无 MLP projection，无 Dropout。
    """

    def __init__(self, dim=768, num_classes=10, num_prototypes=2, scale=20.0):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.scale = scale
        self.norm = nn.LayerNorm(dim)
        self.prototypes = nn.Parameter(torch.randn(num_classes, num_prototypes, dim))
        # 小幅初始化（方向随机即可，forward 内会做 L2 归一化）
        nn.init.trunc_normal_(self.prototypes, std=0.02)

    def forward(self, x):
        # x: [B, dim] = CLS token
        x = self.norm(x)                        # [B, dim]
        x = F.normalize(x, dim=-1)              # L2 归一化特征
        p = F.normalize(self.prototypes, dim=-1)  # [C, K, dim] L2 归一化原型
        similarity = torch.einsum('bd,ckd->bck', x, p)   # [B, C, K] cosine
        scaled = self.scale * similarity
        logits = torch.logsumexp(scaled, dim=-1) - math.log(self.num_prototypes)  # [B, C]
        return logits, similarity


# ====================================================================
# 模型：DINOv3 + MCPH
# ====================================================================
class DINOv3MCPHClassifier(nn.Module):
    def __init__(self, backbone, hidden_size, num_classes, num_prototypes=2, scale=20.0):
        super().__init__()
        self.backbone = backbone
        self.head = MultiCenterPrototypeHead(hidden_size, num_classes, num_prototypes, scale)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0]  # 最终 CLS token
        logits, similarity = self.head(cls_token)
        return logits, similarity


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
# 评估（可选收集 similarity，供原型使用分析）
# ====================================================================
@torch.no_grad()
def evaluate(model, loader, criterion, device, amp_dtype, collect_sim=False):
    model.eval()
    total_loss = 0.0
    y_true, y_pred, y_prob = [], [], []
    sims_all = []
    for images, labels, pids, paths in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits, similarity = model(images)
            loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        probs = torch.softmax(logits, dim=1).float()
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(probs.argmax(1).cpu().tolist())
        y_prob.append(probs.cpu().numpy())
        if collect_sim:
            sims_all.append(similarity.float().cpu().numpy())
    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)
    y_prob = np.concatenate(y_prob, axis=0).astype(np.float32)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics["loss"] = total_loss / len(loader.dataset)
    if collect_sim:
        sims_all = np.concatenate(sims_all, axis=0).astype(np.float32)
        return metrics, y_true, y_pred, y_prob, sims_all
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
            logits, _ = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(logits.argmax(1).cpu().tolist())
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
        "model_name": "DINOv3-ViT-B16-MCPH",
        "input_size": IMG_SIZE,
        "seed": SEED,
        "training_config": config,
    }, path)


# ====================================================================
# Smoke Test（重点检查 MCPH 形状与归一化）
# ====================================================================
def smoke_test(model, train_loader, val_loader, optimizer, criterion, device, amp_dtype, scaler):
    print("=" * 60)
    print("SMOKE TEST (MCPH)")
    print("=" * 60)
    model.train()
    try:
        batch = next(iter(train_loader))
        images, labels, pids, paths = batch
        print(f"1. 数据读取: OK (batch {tuple(images.shape)}, labels {labels[:5].tolist()})")
        images = images.to(device)
        labels = labels.to(device)

        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            outputs = model.backbone(pixel_values=images)
            cls_token = outputs.last_hidden_state[:, 0]
            normed = model.head.norm(cls_token)
            normed = F.normalize(normed, dim=-1)
            proto = F.normalize(model.head.prototypes, dim=-1)
            similarity = torch.einsum('bd,ckd->bck', normed, proto)
            logits, _ = model(images)
            loss = criterion(logits, labels)

        feat_norm = normed.norm(dim=-1)
        proto_norm = proto.norm(dim=-1)
        print(f"2. CLS shape = {tuple(cls_token.shape)}")
        print(f"3. Normalized feature shape = {tuple(normed.shape)}")
        print(f"   feature L2 norm: min={feat_norm.min().item():.6f}, max={feat_norm.max().item():.6f} (应 ≈1)")
        print(f"4. Prototype shape = {tuple(proto.shape)}")
        print(f"   prototype L2 norm: min={proto_norm.min().item():.6f}, max={proto_norm.max().item():.6f} (应 ≈1)")
        print(f"5. Similarity shape = {tuple(similarity.shape)}")
        print(f"6. Logits shape = {tuple(logits.shape)}")
        print(f"7. loss = {loss.item():.4f} (finite={math.isfinite(loss.item())})")

        assert tuple(cls_token.shape) == (images.shape[0], 768), "CLS shape 错误"
        assert tuple(proto.shape) == (NUM_CLASSES, NUM_PROTOTYPES, 768), "Prototype shape 错误"
        assert tuple(similarity.shape) == (images.shape[0], NUM_CLASSES, NUM_PROTOTYPES), "Similarity shape 错误"
        assert tuple(logits.shape) == (images.shape[0], NUM_CLASSES), "Logits shape 错误"
        assert float(feat_norm.min().item()) > 0.999 and float(feat_norm.max().item()) < 1.001, "feature 未归一化"
        assert float(proto_norm.min().item()) > 0.999 and float(proto_norm.max().item()) < 1.001, "prototype 未归一化"
        assert math.isfinite(loss.item()), "loss 非有限"

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
        scaler.step(optimizer)
        scaler.update()
        print("8. backward + optimizer.step: OK")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[FAIL] 前向/反向失败: {e}")
        return False

    mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"9. GPU 显存峰值(allocated): {mem:.0f} MB")

    try:
        save_checkpoint(os.path.join(EXP_DIR, "best_model.pt"), model, optimizer, 0, 0.0, {})
        print("10. checkpoint 保存: OK")
    except Exception as e:
        print(f"[FAIL] checkpoint 保存失败: {e}")
        return False

    try:
        metrics, _, _, _ = evaluate(model, val_loader, criterion, device, amp_dtype)
        print(f"11. validation metrics: OK (val acc {metrics['accuracy']:.4f}, val macro_f1 {metrics['macro_f1']:.4f})")
    except Exception as e:
        print(f"[FAIL] validation 失败: {e}")
        return False

    print("MCPH SMOKE TEST STATUS = PASS")
    print("=" * 60)
    return True


# ====================================================================
# 原型分析
# ====================================================================
def prototype_analysis(model, sims_all, y_true, device):
    """计算原型相似度矩阵；K>=2 时额外计算同类两原型相似度与使用情况。"""
    proto = F.normalize(model.head.prototypes.detach(), dim=-1)  # [C, K, D]
    proto_flat = proto.reshape(NUM_CLASSES * NUM_PROTOTYPES, -1)  # [C*K, D]
    sim_matrix = (proto_flat @ proto_flat.T).float().cpu().numpy()  # [C*K, C*K]

    same_class_sim = []
    mean_same_class_sim = None
    if NUM_PROTOTYPES >= 2:
        for c in range(NUM_CLASSES):
            s = float(torch.dot(proto[c, 0], proto[c, 1]).item())
            same_class_sim.append(s)
        mean_same_class_sim = float(np.mean(same_class_sim))

    usage = {}
    for c in range(NUM_CLASSES):
        mask = y_true == c
        cls_sims = sims_all[mask, c, :]  # [n_c, K]
        n = int(cls_sims.shape[0])
        if n == 0:
            usage[c] = {"P1": 0, "P2": 0, "support": 0}
            continue
        if NUM_PROTOTYPES >= 2:
            sel = cls_sims.argmax(axis=1)
            usage[c] = {"P1": int((sel == 0).sum()), "P2": int((sel == 1).sum()), "support": n}
        else:
            usage[c] = {"P1": n, "P2": 0, "support": n}

    return sim_matrix, same_class_sim, mean_same_class_sim, usage


# ====================================================================
# 主流程
# ====================================================================
def main():
    global EXP_DIR, EXP_NAME, NUM_PROTOTYPES

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="仅运行 smoke test")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--num_prototypes", type=int, default=NUM_PROTOTYPES, help="每个类别的原型数 K")
    args = parser.parse_args()

    NUM_PROTOTYPES = args.num_prototypes
    if NUM_PROTOTYPES == 1:
        EXP_NAME = "DINOv3_ViTB16_PMCH_K1_v1"
    elif NUM_PROTOTYPES == 2:
        EXP_NAME = "DINOv3_ViTB16_MCPH_v1"
    else:
        EXP_NAME = f"DINOv3_ViTB16_MCPH_K{NUM_PROTOTYPES}_v1"
    EXP_DIR = f"/root/autodl-tmp/experiments/{EXP_NAME}"
    os.makedirs(EXP_DIR, exist_ok=True)
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("环境信息")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    import transformers
    print(f"transformers: {transformers.__version__}")
    import torchvision
    print(f"torchvision: {torchvision.__version__}")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"AMP dtype: {amp_dtype}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)

    env_info = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "torchvision": torchvision.__version__,
        "python": os.sys.version,
        "amp_dtype": str(amp_dtype),
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(EXP_DIR, "environment.txt"), "w", encoding="utf-8") as f:
        for k, v in env_info.items():
            f.write(f"{k}: {v}\n")

    train_rows, val_rows, test_rows = data_consistency_check()

    # 图像预处理（与 baseline 完全一致）
    processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
    mean = getattr(processor, "image_mean", None)
    std = getattr(processor, "image_std", None)
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    mean = list(mean)
    std = list(std)
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

    # 模型（重新从官方 pretrained 开始，不加载 baseline/ALGF checkpoint）
    backbone = AutoModel.from_pretrained(MODEL_DIR)
    hidden_size = backbone.config.hidden_size
    print(f"backbone hidden_size = {hidden_size}")
    model = DINOv3MCPHClassifier(backbone, hidden_size, NUM_CLASSES,
                                 num_prototypes=NUM_PROTOTYPES, scale=MCPH_SCALE).to(device)

    # 参数量统计
    backbone_params = sum(p.numel() for p in backbone.parameters())
    mcph_head_params = sum(p.numel() for p in model.head.parameters())
    linear_head_params = sum(p.numel() for p in nn.Linear(hidden_size, NUM_CLASSES).parameters())
    baseline_total = backbone_params + linear_head_params
    mcph_total = backbone_params + mcph_head_params
    n_params = mcph_total
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 60)
    print("参数量统计")
    print("=" * 60)
    print(f"Baseline total parameters = {baseline_total:,}")
    print(f"MCPH total parameters     = {mcph_total:,}")
    print(f"MCPH head parameters      = {mcph_head_params:,}")
    print(f"Parameter Delta           = {mcph_total - baseline_total:,}")
    print(f"Parameter Increase %      = {(mcph_total - baseline_total) / baseline_total * 100:.4f}%")
    print(f"可训练参数 = {n_trainable:,}")
    print("=" * 60)

    # 优化器（differential LR：backbone 1e-5，MCPH head 1e-4）
    optimizer = torch.optim.AdamW([
        {"params": backbone.parameters(), "lr": BACKBONE_LR},
        {"params": model.head.parameters(), "lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=(amp_dtype == torch.float16))

    config = {
        "experiment": EXP_NAME,
        "model": "DINOv3-ViT-B16 + MCPH",
        "innovation": "Multi-Center Prototype Head (MCPH)",
        "head": {
            "structure": "CLS -> LayerNorm -> L2 Normalize -> Multi-Center Prototype Matching -> Logits",
            "num_prototypes_per_class": NUM_PROTOTYPES,
            "total_prototypes": NUM_CLASSES * NUM_PROTOTYPES,
            "scale": MCPH_SCALE,
            "aggregation": "LogSumExp - log(K)",
            "similarity": "cosine",
            "init": "trunc_normal_(std=0.02)",
        },
        "model_dir": MODEL_DIR,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
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
        "loss": "CrossEntropyLoss",
        "amp": str(amp_dtype),
        "baseline_total_params": baseline_total,
        "mcph_total_params": mcph_total,
        "mcph_head_params": mcph_head_params,
        "parameter_delta": mcph_total - baseline_total,
        "parameter_increase_percentage": (mcph_total - baseline_total) / baseline_total * 100,
        "train_transform": "RandomResizedCrop + RandomRotation(±10) + ColorJitter(brightness/contrast 0.2)",
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
        print("MCPH SMOKE TEST STATUS = FAIL，终止！")
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

    test_metrics, y_true, y_pred, y_prob, sims_all = evaluate(
        model, test_loader, criterion, device, amp_dtype, collect_sim=True)

    per_class = precision_recall_fscore_support(y_true, y_pred, labels=list(range(10)), zero_division=0)
    p_class, r_class, f_class, support = per_class

    # ROC-AUC（对 BF16 softmax 概率归一化后计算）
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

    # ================= 原型分析 =================
    sim_matrix, same_class_sim, mean_same_class_sim, usage = prototype_analysis(model, sims_all, y_true, device)

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
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / 1024 / 1024),
        "total_train_time_s": float(total_train_time),
        "n_params": n_params,
        "n_trainable_params": n_trainable,
        "baseline_total_params": baseline_total,
        "mcph_total_params": mcph_total,
        "mcph_head_params": mcph_head_params,
        "parameter_delta": mcph_total - baseline_total,
        "parameter_increase_percentage": (mcph_total - baseline_total) / baseline_total * 100,
        "mean_same_class_prototype_similarity": mean_same_class_sim,
        "same_class_prototype_similarity": [float(x) for x in same_class_sim],
        "prototype_usage": usage,
    }
    with open(os.path.join(EXP_DIR, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_result, f, ensure_ascii=False, indent=2)

    # 保存原型相似度矩阵 + 使用情况
    if NUM_PROTOTYPES >= 2:
        proto_sim_data = {
            "same_class_prototype_similarity": {f"class_{c}": float(same_class_sim[c]) for c in range(NUM_CLASSES)},
            "mean_same_class_prototype_similarity": mean_same_class_sim,
        }
    else:
        proto_sim_data = {
            "note": "K=1 单原型，无同类两原型；矩阵为类间原型相似度",
            "inter_class_prototype_similarity_matrix": sim_matrix.tolist(),
        }
    with open(os.path.join(EXP_DIR, "prototype_similarity.json"), "w", encoding="utf-8") as f:
        json.dump(proto_sim_data, f, ensure_ascii=False, indent=2)
    with open(os.path.join(EXP_DIR, "prototype_usage.json"), "w", encoding="utf-8") as f:
        json.dump(usage, f, ensure_ascii=False, indent=2)

    # 20×20 原型相似度矩阵图
    n_proto_total = NUM_CLASSES * NUM_PROTOTYPES
    if NUM_PROTOTYPES >= 2:
        labels_proto = [f"c{c}P{k}" for c in range(NUM_CLASSES) for k in range(NUM_PROTOTYPES)]
    else:
        labels_proto = [f"c{c}" for c in range(NUM_CLASSES)]
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(sim_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(n_proto_total)); ax.set_yticks(range(n_proto_total))
    ax.set_xticklabels(labels_proto, rotation=90, fontsize=8)
    ax.set_yticklabels(labels_proto, fontsize=8)
    ax.set_title(f"Prototype Cosine Similarity Matrix ({n_proto_total}×{n_proto_total})")
    plt.colorbar(im)
    plt.tight_layout()
    fig.savefig(os.path.join(EXP_DIR, "prototype_similarity_matrix.png"), dpi=150)
    plt.close(fig)

    # classification_report.txt
    report_txt = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0)
    with open(os.path.join(EXP_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_txt)

    # test_predictions.csv
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

    # ================= ablation_vs_baseline.md =================
    def pct2(x): return f"{x * 100:.4f}%"
    def delta(a, b): return f"{(a - b) * 100:+.4f}%"

    L = []
    L.append("# 消融对比：Baseline vs MCPH（正式创新点一）\n")
    L.append("> DINOv3_ViTB16_FullFT_Baseline_v1（Dropout+Linear） vs DINOv3_ViTB16_MCPH_v1（Multi-Center Prototype Head）\n")
    L.append("## 总体指标")
    L.append("| Metric | Baseline | MCPH | Delta |")
    L.append("|---|---|---|---|")
    L.append(f"| Accuracy | {pct2(BASELINE['test_accuracy'])} | {pct2(test_result['test_accuracy'])} | {delta(test_result['test_accuracy'], BASELINE['test_accuracy'])} |")
    L.append(f"| Macro Precision | {pct2(BASELINE['macro_precision'])} | {pct2(test_result['macro_precision'])} | {delta(test_result['macro_precision'], BASELINE['macro_precision'])} |")
    L.append(f"| Macro Recall | {pct2(BASELINE['macro_recall'])} | {pct2(test_result['macro_recall'])} | {delta(test_result['macro_recall'], BASELINE['macro_recall'])} |")
    L.append(f"| Macro F1 | {pct2(BASELINE['macro_f1'])} | {pct2(test_result['macro_f1'])} | {delta(test_result['macro_f1'], BASELINE['macro_f1'])} |")
    L.append(f"| Weighted F1 | {pct2(BASELINE['weighted_f1'])} | {pct2(test_result['weighted_f1'])} | {delta(test_result['weighted_f1'], BASELINE['weighted_f1'])} |")
    L.append(f"| Macro ROC-AUC | {pct2(BASELINE['macro_roc_auc'])} | {pct2(test_result['macro_roc_auc'])} | {delta(test_result['macro_roc_auc'], BASELINE['macro_roc_auc'])} |")
    L.append("")
    L.append("## 目标类别")
    L.append("| 丘脑 F1 | Baseline | MCPH | Delta |")
    L.append("|---|---|---|---|")
    L.append(f"| F1 | {pct2(BASELINE['class0_f1'])} | {pct2(test_result['per_class_f1'][0])} | {delta(test_result['per_class_f1'][0], BASELINE['class0_f1'])} |")
    L.append("| 侧脑室 F1 | Baseline | MCPH | Delta |")
    L.append("|---|---|---|---|")
    L.append(f"| F1 | {pct2(BASELINE['class1_f1'])} | {pct2(test_result['per_class_f1'][1])} | {delta(test_result['per_class_f1'][1], BASELINE['class1_f1'])} |")
    L.append("")
    L.append("## 混淆数量")
    L.append("| 混淆 | Baseline | MCPH |")
    L.append("|---|---|---|")
    L.append(f"| 丘脑→侧脑室 | {BASELINE['thalamus_to_ventricle']} | {test_result['thalamus_to_ventricle']} |")
    L.append(f"| 侧脑室→丘脑 | {BASELINE['ventricle_to_thalamus']} | {test_result['ventricle_to_thalamus']} |")
    L.append(f"| Total errors | {BASELINE['total_errors']} | {test_result['total_errors']} |")
    L.append("")
    L.append("## 参数量")
    L.append(f"- Baseline total = {baseline_total:,}")
    L.append(f"- MCPH total = {mcph_total:,}")
    L.append(f"- MCPH head = {mcph_head_params:,}")
    L.append(f"- Delta = {mcph_total - baseline_total:,}")
    L.append(f"- Increase % = {(mcph_total - baseline_total) / baseline_total * 100:.4f}%")
    L.append("")
    L.append("## 原型分析")
    if NUM_PROTOTYPES >= 2:
        L.append(f"- Mean same-class prototype similarity = {mean_same_class_sim:.4f}")
        for c in range(NUM_CLASSES):
            L.append(f"- Class {c} ({CLASS_NAMES[c]}) prototype similarity = {same_class_sim[c]:.4f} | usage P1={usage[c]['P1']}, P2={usage[c]['P2']}")
    else:
        L.append("- K=1 单原型（无同类两原型），类间原型相似度矩阵见 prototype_similarity_matrix.png")

    with open(os.path.join(EXP_DIR, "ablation_vs_baseline.md"), "w", encoding="utf-8") as f:
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
    print("Prototype analysis:")
    if NUM_PROTOTYPES >= 2:
        print(f"  Mean same-class prototype similarity = {mean_same_class_sim:.4f}")
        for c in range(NUM_CLASSES):
            print(f"  class {c} ({CLASS_NAMES[c]}): proto_sim={same_class_sim[c]:.4f}, usage P1={usage[c]['P1']} P2={usage[c]['P2']}")
    else:
        print(f"  K=1（单原型），类间原型相似度矩阵见 prototype_similarity_matrix.png")
    print(f"Peak GPU memory: {test_result['peak_gpu_memory_mb']:.0f} MB")
    print(f"Total training time: {total_train_time:.1f}s")
    print(f"Experiment dir: {EXP_DIR}")
    print("DATA LEAKAGE CHECK = PASS")
    print("MCPH SMOKE TEST STATUS = PASS")
    print("EXPERIMENT STATUS: PASS")


if __name__ == "__main__":
    main()
