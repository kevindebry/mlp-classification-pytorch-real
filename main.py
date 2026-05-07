import json
import os
import random
import sys
import time

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sklearn.utils.class_weight import compute_class_weight
except ImportError:
    compute_class_weight = None

from config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    CLASS_DISTRIBUTION_REPORT_PATH,
    CONFUSION_MATRIX_PLOT_PATH,
    DEVICE,
    DROPOUT,
    EMBED_DIM,
    EPOCHS,
    ERROR_SAMPLES_PATH,
    HIDDEN_DIM,
    KERNEL_SIZES,
    LABEL_MAP,
    LABEL_MAP_PATH,
    LOW_MISSED_ERRORS_PATH,
    LR,
    LR_FACTOR,
    LR_PATIENCE,
    MAX_GRAD_NORM,
    MAX_LEN,
    NUM_CLASSES,
    NUM_FILTERS,
    NUM_WORKERS,
    OUTPUT_DIR,
    PIN_MEMORY,
    PREDICTED_LOW_SAMPLES_PATH,
    SEED,
    TEST_METRICS_PATH,
    TEST_PATH,
    TRAINING_HISTORY_PLOT_PATH,
    TRAIN_PATH,
    USE_CLASS_WEIGHT,
    VAL_PATH,
    VOCAB_PATH,
    WEIGHT_DECAY,
    WEIGHTED_BEST_MODEL_PATH,
    WEIGHTED_TEST_METRICS_PATH,
)
from data_utils import TextDataset, build_vocab, clean_data, encode_text, load_data
from model import TextCNN
from tokenizer import char_tokenize, print_tokenization_examples


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def print_time(name, start):
    elapsed = time.perf_counter() - start
    print(f"[TIME] {name}: {elapsed:.2f}s", flush=True)
    return elapsed


def make_dataloader(df, token_to_id, shuffle):
    dataset = TextDataset(df, token_to_id, MAX_LEN)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )


def build_model(vocab_size):
    return TextCNN(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        num_filters=NUM_FILTERS,
        kernel_sizes=KERNEL_SIZES,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ).to(DEVICE)


def get_model_path():
    return WEIGHTED_BEST_MODEL_PATH if USE_CLASS_WEIGHT else BEST_MODEL_PATH


def get_test_metrics_path():
    return WEIGHTED_TEST_METRICS_PATH if USE_CLASS_WEIGHT else TEST_METRICS_PATH


def get_label_counts(df):
    return {
        str(label_id): int((df["label_id"] == label_id).sum())
        for label_id in sorted(LABEL_MAP.keys())
    }


def compute_balanced_class_weights(train_df, label_map):
    labels = train_df["label_id"].astype(int).values
    class_ids = np.array(sorted(label_map.keys()))
    if compute_class_weight is not None:
        weights = compute_class_weight(
            class_weight="balanced",
            classes=class_ids,
            y=labels,
        )
    else:
        total_count = len(labels)
        class_count = len(class_ids)
        weights = []
        for class_id in class_ids:
            count = int((labels == class_id).sum())
            weight = 0.0 if count == 0 else total_count / (class_count * count)
            weights.append(weight)

    print("class_weight method: sklearn balanced" if compute_class_weight is not None else "class_weight method: manual balanced")
    return torch.tensor(weights, dtype=torch.float, device=DEVICE)


def print_class_weights(class_weights, label_map):
    print("\n====== Class Weights ======")
    for label_id, weight in zip(sorted(label_map.keys()), class_weights.detach().cpu().tolist()):
        print(f"{label_id} {label_map[label_id]} weight = {weight:.6f}")


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for input_ids, labels in loader:
        input_ids = input_ids.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()

        if MAX_GRAD_NORM is not None:
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

        optimizer.step()

        batch_size = input_ids.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
        total_count += batch_size

    return {
        "loss": total_loss / total_count,
        "accuracy": total_correct / total_count,
    }


def compute_classification_metrics(all_labels, all_preds, label_map):
    class_ids = sorted(label_map.keys())
    class_to_index = {class_id: idx for idx, class_id in enumerate(class_ids)}
    cm = np.zeros((len(class_ids), len(class_ids)), dtype=np.int64)

    for true_id, pred_id in zip(all_labels, all_preds):
        if true_id in class_to_index and pred_id in class_to_index:
            cm[class_to_index[true_id], class_to_index[pred_id]] += 1

    true_positive = np.diag(cm).astype(np.float64)
    pred_count = cm.sum(axis=0).astype(np.float64)
    true_count = cm.sum(axis=1).astype(np.float64)
    total = cm.sum()

    precision = np.divide(
        true_positive,
        pred_count,
        out=np.zeros_like(true_positive),
        where=pred_count != 0,
    )
    recall = np.divide(
        true_positive,
        true_count,
        out=np.zeros_like(true_positive),
        where=true_count != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    per_class = {}
    for idx, class_id in enumerate(class_ids):
        per_class[str(class_id)] = {
            "label": label_map[class_id],
            "true_count": int(true_count[idx]),
            "pred_count": int(pred_count[idx]),
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(true_count[idx]),
        }

    return {
        "accuracy": float(true_positive.sum() / total) if total else 0.0,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def evaluate(model, loader, criterion, label_map):
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_labels = []
    all_preds = []
    all_confidences = []

    with torch.inference_mode():
        for input_ids, labels in loader:
            input_ids = input_ids.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            logits = model(input_ids)
            loss = criterion(logits, labels)
            probabilities = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            confidences = probabilities.gather(1, preds.unsqueeze(1)).squeeze(1)

            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_confidences.extend(confidences.cpu().tolist())

    metrics = compute_classification_metrics(all_labels, all_preds, label_map)
    metrics.update({
        "loss": total_loss / total_count,
        "labels": all_labels,
        "preds": all_preds,
        "confidences": all_confidences,
    })
    return metrics


def print_metrics(name, metrics):
    print(f"\n====== {name} ======")
    print(
        f"loss={metrics['loss']:.4f} | "
        f"accuracy={metrics['accuracy']:.4f} | "
        f"macro_precision={metrics['macro_precision']:.4f} | "
        f"macro_recall={metrics['macro_recall']:.4f} | "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )
    print("\n每类指标:")
    print("class_id | label | precision | recall | f1 | support")
    for class_id, item in metrics["per_class"].items():
        print(
            f"{int(class_id):>8} | {item['label']:<4} | "
            f"{item['precision']:.4f}    | {item['recall']:.4f} | "
            f"{item['f1']:.4f} | {item['support']}"
        )
    print("\nConfusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))


def print_class_diagnosis(metrics, label_map):
    print("\n====== Class Diagnosis ======")
    print("label_id | label | true_count | pred_count | precision | recall | f1 | support")
    for label_id in sorted(label_map.keys()):
        item = metrics["per_class"][str(label_id)]
        print(
            f"{label_id:<8} | {item['label']:<4} | "
            f"{item['true_count']:<10} | {item['pred_count']:<10} | "
            f"{item['precision']:.4f}    | {item['recall']:.4f} | "
            f"{item['f1']:.4f} | {item['support']}"
        )


def print_low_class_diagnosis(metrics, label_map):
    low_ids = [label_id for label_id, label in label_map.items() if label == "低落"]
    if not low_ids:
        return

    low_id = low_ids[0]
    item = metrics["per_class"][str(low_id)]
    print("\n====== Low Class Diagnosis ======")
    print(f"测试集中真实低落数量: {item['true_count']}")
    print(f"模型预测为低落数量: {item['pred_count']}")
    print(f"低落 precision: {item['precision']:.4f}")
    print(f"低落 recall: {item['recall']:.4f}")
    print(f"低落 f1: {item['f1']:.4f}")


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def plot_training_history(history, path):
    if not history:
        return

    epochs = [item["epoch"] for item in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    axes[0].plot(epochs, [item["train_loss"] for item in history], marker="o", label="train_loss")
    axes[0].plot(epochs, [item["val_loss"] for item in history], marker="o", label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, [item["train_accuracy"] for item in history], marker="o", label="train_accuracy")
    axes[1].plot(epochs, [item["val_accuracy"] for item in history], marker="o", label="val_accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(epochs, [item["val_macro_f1"] for item in history], marker="o", color="tab:purple")
    axes[2].set_title("Validation Macro-F1")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro-F1")
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(epochs, [item["lr"] for item in history], marker="o", color="tab:green")
    axes[3].set_title("Learning Rate")
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("LR")
    axes[3].grid(True, alpha=0.3)

    plt.suptitle("Char-only TextCNN Training History")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"训练曲线图已保存到 {path}")


def plot_confusion_matrix(metrics, label_map, path):
    cm = np.array(metrics["confusion_matrix"])
    labels = [label_map[class_id] for class_id in sorted(label_map.keys())]
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=np.float64),
        where=row_sums != 0,
    )

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_title("Char-only TextCNN Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    fig.colorbar(im, ax=ax, label="Recall Ratio")

    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm_normalized[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]}\n{cm_normalized[i, j]:.2f}", ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"混淆矩阵图已保存到 {path}")


def save_artifacts(token_to_id, id_to_token, label_map, test_metrics):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_json(
        VOCAB_PATH,
        {
            "token_to_id": token_to_id,
            "id_to_token": id_to_token,
            "pad_token": "<PAD>",
            "unk_token": "<UNK>",
            "max_len": MAX_LEN,
        },
    )
    save_json(LABEL_MAP_PATH, {str(k): v for k, v in label_map.items()})

    metrics_to_save = {
        key: value
        for key, value in test_metrics.items()
        if key not in {"labels", "preds", "confidences"}
    }
    save_json(get_test_metrics_path(), metrics_to_save)


def save_class_distribution_report(
    train_df,
    val_df,
    test_df,
    test_metrics,
    class_weights,
):
    report = {
        "train_label_counts": get_label_counts(train_df),
        "val_label_counts": get_label_counts(val_df),
        "test_label_counts": get_label_counts(test_df),
        "pred_label_counts": {
            str(label_id): int(test_metrics["per_class"][str(label_id)]["pred_count"])
            for label_id in sorted(LABEL_MAP.keys())
        },
        "per_class_metrics": test_metrics["per_class"],
        "class_weights": {
            str(label_id): float(weight)
            for label_id, weight in zip(sorted(LABEL_MAP.keys()), class_weights.detach().cpu().tolist())
        } if class_weights is not None else None,
        "use_class_weight": USE_CLASS_WEIGHT,
    }
    save_json(CLASS_DISTRIBUTION_REPORT_PATH, report)
    print(f"类别诊断报告已保存到 {CLASS_DISTRIBUTION_REPORT_PATH}")


def load_artifacts():
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    token_to_id = vocab_data["token_to_id"]
    try:
        checkpoint = torch.load(get_model_path(), map_location=DEVICE, weights_only=False)
    except TypeError:
        checkpoint = torch.load(get_model_path(), map_location=DEVICE)
    model = build_model(vocab_size=len(token_to_id))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model, token_to_id, label_map


def predict_text(model, text, token_to_id, label_map):
    model.eval()
    with torch.inference_mode():
        input_ids = encode_text(text, token_to_id, MAX_LEN).unsqueeze(0).to(DEVICE, non_blocking=True)
        logits = model(input_ids)
        pred_id = int(torch.argmax(logits, dim=1).item())
    return pred_id, label_map[pred_id]


def export_error_samples(df, metrics, label_map):
    rows = []
    for sample_index, (text, true_id, pred_id) in enumerate(
        zip(df["text"].tolist(), metrics["labels"], metrics["preds"])
    ):
        if true_id == pred_id:
            continue
        rows.append({
            "sample_index": sample_index,
            "text": text,
            "true_label": label_map[int(true_id)],
            "true_id": int(true_id),
            "pred_label": label_map[int(pred_id)],
            "pred_id": int(pred_id),
        })

    error_df = pd.DataFrame(
        rows,
        columns=["sample_index", "text", "true_label", "true_id", "pred_label", "pred_id"],
    )
    error_df.to_csv(ERROR_SAMPLES_PATH, index=False, encoding="utf-8-sig")
    print(f"错分样本已保存到 {ERROR_SAMPLES_PATH}，共 {len(error_df)} 条。")


def build_prediction_rows(df, metrics, label_map):
    rows = []
    for sample_index, (text, true_id, pred_id, confidence) in enumerate(
        zip(
            df["text"].tolist(),
            metrics["labels"],
            metrics["preds"],
            metrics["confidences"],
        )
    ):
        rows.append({
            "sample_index": sample_index,
            "text": text,
            "true_label": label_map[int(true_id)],
            "true_id": int(true_id),
            "pred_label": label_map[int(pred_id)],
            "pred_id": int(pred_id),
            "confidence": float(confidence),
        })
    return rows


def export_low_diagnosis_samples(df, metrics, label_map):
    low_ids = [label_id for label_id, label in label_map.items() if label == "低落"]
    if not low_ids:
        return

    low_id = low_ids[0]
    rows = build_prediction_rows(df, metrics, label_map)

    low_missed_rows = [
        row
        for row in rows
        if row["true_id"] == low_id and row["pred_id"] != low_id
    ]
    predicted_low_rows = [
        row
        for row in rows
        if row["pred_id"] == low_id
    ]

    columns = ["sample_index", "text", "true_label", "true_id", "pred_label", "pred_id", "confidence"]
    low_missed_df = pd.DataFrame(low_missed_rows, columns=columns).sort_values(
        "confidence",
        ascending=False,
    )
    predicted_low_df = pd.DataFrame(predicted_low_rows, columns=columns).sort_values(
        "confidence",
        ascending=False,
    )

    low_missed_df.to_csv(LOW_MISSED_ERRORS_PATH, index=False, encoding="utf-8-sig")
    predicted_low_df.to_csv(PREDICTED_LOW_SAMPLES_PATH, index=False, encoding="utf-8-sig")
    print(f"低落漏判样本已保存到 {LOW_MISSED_ERRORS_PATH}，共 {len(low_missed_df)} 条。")
    print(f"预测为低落样本已保存到 {PREDICTED_LOW_SAMPLES_PATH}，共 {len(predicted_low_df)} 条。")


def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    start = time.perf_counter()
    train_df = clean_data(load_data(TRAIN_PATH), valid_labels=LABEL_MAP.keys())
    val_df = clean_data(load_data(VAL_PATH), valid_labels=LABEL_MAP.keys())
    test_df = clean_data(load_data(TEST_PATH), valid_labels=LABEL_MAP.keys())
    print_time("load_data + clean_data", start)

    print("训练集前五行")
    print(train_df.head(), "\n")
    print("训练集类别分布")
    print(train_df["label_id"].value_counts().sort_index(), "\n")
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    print(f"测试集大小: {len(test_df)}\n")
    print_tokenization_examples()

    start = time.perf_counter()
    token_to_id, id_to_token = build_vocab(train_df["text"])
    print_time("build char vocab", start)
    print(f"char vocab size: {len(token_to_id)}\n")

    start = time.perf_counter()
    train_loader = make_dataloader(train_df, token_to_id, shuffle=True)
    val_loader = make_dataloader(val_df, token_to_id, shuffle=False)
    test_loader = make_dataloader(test_df, token_to_id, shuffle=False)
    print_time("Dataset + DataLoader", start)

    model = build_model(vocab_size=len(token_to_id))
    print(model)
    print(f"\n运行设备: {DEVICE}\n")

    class_weights = compute_balanced_class_weights(train_df, LABEL_MAP)
    print_class_weights(class_weights, LABEL_MAP)
    if USE_CLASS_WEIGHT:
        print("\nUSE_CLASS_WEIGHT = True，使用带类别权重的 CrossEntropyLoss。")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        print("\nUSE_CLASS_WEIGHT = False，使用普通 CrossEntropyLoss。")
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
    )

    best_val_f1 = -1.0
    best_epoch = 0
    history = []

    start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer)
        val_metrics = evaluate(model, val_loader, criterion, LABEL_MAP)
        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": current_lr,
        })

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} | "
            f"lr={current_lr:.6f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            torch.save(
                {
                    "model_name": "CharOnlyTextCNN",
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "vocab_size": len(token_to_id),
                        "embed_dim": EMBED_DIM,
                        "num_filters": NUM_FILTERS,
                        "kernel_sizes": list(KERNEL_SIZES),
                        "hidden_dim": HIDDEN_DIM,
                        "num_classes": NUM_CLASSES,
                        "dropout": DROPOUT,
                        "max_len": MAX_LEN,
                    },
                    "token_to_id": token_to_id,
                    "id_to_token": id_to_token,
                    "label_map": LABEL_MAP,
                    "use_class_weight": USE_CLASS_WEIGHT,
                    "class_weights": class_weights.detach().cpu().tolist(),
                    "history": history,
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_val_f1,
                },
                get_model_path(),
            )
            print(f"保存新的 best model: epoch={best_epoch}, val_macro_f1={best_val_f1:.4f}")

    print_time("train", start)

    try:
        checkpoint = torch.load(get_model_path(), map_location=DEVICE, weights_only=False)
    except TypeError:
        checkpoint = torch.load(get_model_path(), map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    start = time.perf_counter()
    test_metrics = evaluate(model, test_loader, criterion, LABEL_MAP)
    print_time("test evaluate", start)
    print_metrics("测试集结果", test_metrics)
    print_class_diagnosis(test_metrics, LABEL_MAP)
    print_low_class_diagnosis(test_metrics, LABEL_MAP)

    save_artifacts(token_to_id, id_to_token, LABEL_MAP, test_metrics)
    save_class_distribution_report(train_df, val_df, test_df, test_metrics, class_weights)
    export_error_samples(test_df, test_metrics, LABEL_MAP)
    export_low_diagnosis_samples(test_df, test_metrics, LABEL_MAP)
    plot_training_history(history, TRAINING_HISTORY_PLOT_PATH)
    plot_confusion_matrix(test_metrics, LABEL_MAP, CONFUSION_MATRIX_PLOT_PATH)

    print(f"\n最佳验证集 macro F1: {best_val_f1:.4f} (epoch {best_epoch})")
    print(f"模型已保存到 {get_model_path()}")
    print(f"词表已保存到 {VOCAB_PATH}")
    print(f"标签映射已保存到 {LABEL_MAP_PATH}")
    print(f"测试指标已保存到 {get_test_metrics_path()}")
    print(f"类别诊断报告已保存到 {CLASS_DISTRIBUTION_REPORT_PATH}")
    print(f"训练曲线图已保存到 {TRAINING_HISTORY_PLOT_PATH}")
    print(f"混淆矩阵图已保存到 {CONFUSION_MATRIX_PLOT_PATH}")

    if sys.stdin.isatty():
        print("\n输入一句中文文本预测情绪，直接回车退出。")
        while True:
            text = input("请输入文本: ").strip()
            if not text:
                break
            pred_id, pred_label = predict_text(model, text, token_to_id, LABEL_MAP)
            print(f"预测类别: {pred_label} ({pred_id})")


if __name__ == "__main__":
    main()
