#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DINOv3 ViT-B/16 Full Fine-Tuning Baseline — 胎儿超声标准切面 10 分类
=====================================================================
正式科研实验。固定患者级划分，禁止重新划分数据。
基线：DINOv3 backbone + 10-class linear head，Full Fine-Tuning + differential LR。
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
# 常量配置（固定，禁止随意改动）
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
HEAD_DROPOUT = 0.1

CLASS_NAMES = ["丘脑横切面", "侧脑室横切面", "小脑横切面", "双眼横切面", "鼻唇冠状切面",
               "颜面正中矢状面", "脊柱纵切面", "双肾矢状切面", "双肾横切面", "腹围横切面"]

DATA_ROOT = "/root/autodl-tmp/fetal_ultrasound"
SPLIT_DIR = "/root/autodl-tmp/split"
MODEL_DIR = "/root/autodl-tmp/models/DINOv3-ViT-B16"
EXP_NAME = "DINOv3_ViTB16_FullFT_Baseline_v1"
EXP_DIR = f"/root/autodl-tmp/experiments/{EXP_NAME}"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_server_path(win_path, data_root):
    """将 CSV 中的 Windows 路径转换为服务器路径。
    例如 D:\\...\\10腹围横切面\\丘凤娇_001_192702.jpg -> /root/.../fetal_ultrasound/10腹围横切面/丘凤娇_001_192702.jpg
    """
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
# 数据一致性检查
# ====================================================================
def data_consistency_check():
    print("=" * 60)
    print("数据一致性检查")
    print("=" * 60)

    train = load_csv(os.path.join(SPLIT_DIR, "train.csv"))
    val = load_csv(os.path.join(SPLIT_DIR, "val.csv"))
    test = load_csv(os.path.join(SPLIT_DIR, "test.csv"))

    # 1. 数量检查
    print(f"train.csv = {len(train)} images (期望 4906)")
    print(f"val.csv   = {len(val)} images (期望 611)")
    print(f"test.csv  = {len(test)} images (期望 620)")
    assert len(train) == 4906, "train 数量不符！"
    assert len(val) == 611, "val 数量不符！"
    assert len(test) == 620, "test 数量不符！"
    assert len(train) + len(val) + len(test) == 6137, "总数不符！"

    # 2. class_id 唯一性/范围检查
    for split_name, split in [("train", train), ("val", val), ("test", test)]:
        cids = {r["class_id"] for r in split}
        assert cids == set(range(10)), f"{split_name} class_id 集合异常: {cids}"
        for r in split:
            assert 0 <= r["class_id"] <= 9, f"class_id 越界: {r['class_id']}"

    # 3. 患者泄漏检查
    train_p = {r["patient_id"] for r in train}
    val_p = {r["patient_id"] for r in val}
    test_p = {r["patient_id"] for r in test}
    print(f"Train ∩ Val  = {len(train_p & val_p)}")
    print(f"Train ∩ Test = {len(train_p & test_p)}")
    print(f"Val ∩ Test   = {len(val_p & test_p)}")
    assert train_p.isdisjoint(val_p), "Train/Val 患者泄漏！"
    assert train_p.isdisjoint(test_p), "Train/Test 患者泄漏！"
    assert val_p.isdisjoint(test_p), "Val/Test 患者泄漏！"

    # 4. 文件存在性 + 损坏检查
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
        print(f"  corrupted 已记录到 corrupted_images.csv")

    assert len(missing) == 0, "存在缺失文件，终止训练！"

    # 5. 统计 class_id -> class_name 映射
    cid_to_name = {}
    for r in all_rows:
        cid_to_name[r["class_id"]] = r["class_name"]
    for i in range(10):
        print(f"  class {i} = {cid_to_name.get(i)}")
    assert len(cid_to_name) == 10, "class_id 映射异常！"

    print("DATA LEAKAGE CHECK = PASS")
    print("=" * 60)
    return train, val, test


# ====================================================================
# Dataset
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
# 模型
# ====================================================================
class DINOv3Classifier(nn.Module):
    def __init__(self, backbone, hidden_size, num_classes, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        if getattr(outputs, "pooler_output", None) is not None:
            x = outputs.pooler_output
        else:
            x = outputs.last_hidden_state[:, 0]  # CLS token
        x = self.dropout(x)
        logits = self.head(x)
        return logits


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
# 评估
# ====================================================================
@torch.no_grad()
def evaluate(model, loader, criterion, device, amp_dtype):
    model.eval()
    total_loss = 0.0
    y_true, y_pred, y_prob = [], [], []
    for images, labels, pids, paths in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits = model(images)
            loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        probs = torch.softmax(logits, dim=1).float()
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(probs.argmax(1).cpu().tolist())
        y_prob.append(probs.cpu().numpy())
    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)
    y_prob = np.concatenate(y_prob, axis=0).astype(np.float32)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics["loss"] = total_loss / len(loader.dataset)
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
            logits = model(images)
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
        "model_name": "DINOv3-ViT-B16",
        "input_size": IMG_SIZE,
        "seed": SEED,
        "training_config": config,
    }, path)


# ====================================================================
# Smoke Test
# ====================================================================
def smoke_test(model, train_loader, val_loader, optimizer, criterion, device, amp_dtype, scaler):
    print("=" * 60)
    print("SMOKE TEST")
    print("=" * 60)
    model.train()
    # 1. 数据读取 + forward + loss + backward + AMP
    try:
        batch = next(iter(train_loader))
        images, labels, pids, paths = batch
        print(f"1. 数据读取: OK (batch shape {tuple(images.shape)}, labels {labels[:5].tolist()})")
        images = images.to(device)
        labels = labels.to(device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits = model(images)
            loss = criterion(logits, labels)
        print(f"2. DINOv3 forward: OK (logits {tuple(logits.shape)})")
        print(f"3. loss: OK ({loss.item():.4f})")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
        scaler.step(optimizer)
        scaler.update()
        print("4. backward + optimizer.step: OK")
        # 检查梯度
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        print(f"   gradient norm sum: {grad_norm:.4f}")
    except Exception as e:
        print(f"[FAIL] 前向/反向失败: {e}")
        return False

    # 2. GPU 显存
    mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"5. GPU 显存峰值(allocated): {mem:.0f} MB")

    # 3. checkpoint 保存
    try:
        save_checkpoint(os.path.join(EXP_DIR, "best_model.pt"), model, optimizer, 0, 0.0, {})
        print("6. checkpoint 保存: OK")
    except Exception as e:
        print(f"[FAIL] checkpoint 保存失败: {e}")
        return False

    # 4. validation metrics
    try:
        metrics, _, _, _ = evaluate(model, val_loader, criterion, device, amp_dtype)
        print(f"7. validation metrics: OK (val acc {metrics['accuracy']:.4f}, val macro_f1 {metrics['macro_f1']:.4f})")
    except Exception as e:
        print(f"[FAIL] validation 失败: {e}")
        return False

    print("SMOKE TEST STATUS = PASS")
    print("=" * 60)
    return True


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
    print(f"CUDA: {torch.version.cuda}")
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
    import transformers
    print(f"transformers: {transformers.__version__}")
    import torchvision
    print(f"torchvision: {torchvision.__version__}")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"AMP dtype: {amp_dtype}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)

    # 环境信息保存
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

    # 数据一致性检查
    train_rows, val_rows, test_rows = data_consistency_check()

    # 图像预处理
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

    # 模型
    backbone = AutoModel.from_pretrained(MODEL_DIR)
    hidden_size = backbone.config.hidden_size
    print(f"backbone hidden_size = {hidden_size}")
    model = DINOv3Classifier(backbone, hidden_size, NUM_CLASSES, dropout=HEAD_DROPOUT).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {n_params:,}, 可训练参数: {n_trainable:,}")

    # 优化器（differential LR）
    optimizer = torch.optim.AdamW([
        {"params": backbone.parameters(), "lr": BACKBONE_LR},
        {"params": model.head.parameters(), "lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=(amp_dtype == torch.float16))

    # config.yaml
    config = {
        "experiment": EXP_NAME,
        "model": "DINOv3-ViT-B16",
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
        "head_dropout": HEAD_DROPOUT,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealing + LinearWarmup",
        "loss": "CrossEntropyLoss",
        "amp": str(amp_dtype),
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
        print("SMOKE TEST STATUS = FAIL，终止！")
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

        # 保存 last_model
        save_checkpoint(os.path.join(EXP_DIR, "last_model.pt"), model, optimizer, epoch + 1, best_val_f1, config)

        # best model（按 Val Macro F1）
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch + 1
            epochs_no_improve = 0
            save_checkpoint(os.path.join(EXP_DIR, "best_model.pt"), model, optimizer, epoch + 1, best_val_f1, config)
            print(f"  >>> 新的 best model (val_macro_f1={best_val_f1:.4f})", flush=True)
        else:
            epochs_no_improve += 1

        # early stopping
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"Early stopping 触发（{EARLY_STOP_PATIENCE} epochs 无提升），停止于 epoch {epoch+1}", flush=True)
            break

    total_train_time = time.time() - train_start

    # metrics.csv
    import csv as _csv
    with open(os.path.join(EXP_DIR, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
        w.writeheader()
        w.writerows(metrics_rows)

    # ================= Test 最终评估（只一次） =================
    print("=" * 60)
    print(f"训练结束，best_epoch={best_epoch}, best_val_macro_f1={best_val_f1:.4f}")
    print("加载 best_model.pt 进行 Test 评估（唯一一次）")
    print("=" * 60)

    ckpt = torch.load(os.path.join(EXP_DIR, "best_model.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics, y_true, y_pred, y_prob = evaluate(model, test_loader, criterion, device, amp_dtype)

    # per-class 指标
    per_class = precision_recall_fscore_support(y_true, y_pred, labels=list(range(10)), zero_division=0)
    p_class, r_class, f_class, support = per_class

    # ROC-AUC (one-vs-rest)
    try:
        roc_macro = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        roc_per_class = roc_auc_score(y_true, y_prob, multi_class="ovr", average=None)
    except Exception as e:
        roc_macro = None
        roc_per_class = [None] * 10
        print(f"[WARN] ROC-AUC 计算失败: {e}")

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

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
        "macro_roc_auc": float(roc_macro) if roc_macro is not None else None,
        "per_class_precision": [float(x) for x in p_class],
        "per_class_recall": [float(x) for x in r_class],
        "per_class_f1": [float(x) for x in f_class],
        "per_class_support": [int(x) for x in support],
        "per_class_roc_auc": [float(x) if x is not None else None for x in roc_per_class],
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / 1024 / 1024),
        "total_train_time_s": float(total_train_time),
        "n_params": n_params,
        "n_trainable_params": n_trainable,
    }
    with open(os.path.join(EXP_DIR, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_result, f, ensure_ascii=False, indent=2)

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

    # 最终打印
    print("=" * 60)
    print("训练完成")
    print("=" * 60)
    print(f"Best epoch: {best_epoch}")
    print(f"Best Val Accuracy: {test_result['best_val_accuracy']:.4f}")
    print(f"Best Val Macro F1: {best_val_f1:.4f}")
    print(f"Test Accuracy: {test_result['test_accuracy']:.4f}")
    print(f"Test Macro F1: {test_result['macro_f1']:.4f}")
    print(f"Test Weighted F1: {test_result['weighted_f1']:.4f}")
    print(f"Test Macro ROC-AUC: {test_result['macro_roc_auc']:.4f}" if roc_macro is not None else "ROC-AUC: N/A")
    print("Per-class F1:")
    for i in range(10):
        print(f"  class {i} ({CLASS_NAMES[i]}): f1={f_class[i]:.4f}, p={p_class[i]:.4f}, r={r_class[i]:.4f}, support={support[i]}")
    print(f"Peak GPU memory: {test_result['peak_gpu_memory_mb']:.0f} MB")
    print(f"Total training time: {total_train_time:.1f}s")
    print(f"Experiment dir: {EXP_DIR}")
    print("EXPERIMENT STATUS: PASS")


if __name__ == "__main__":
    main()
