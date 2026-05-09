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
    BRANCH_DROPOUT,
    CACHE_DIR,
    CHAR_EMBED_DIM,
    CHAR_KERNEL_SIZES,
    CHAR_MAX_LEN,
    CHAR_NUM_FILTERS,
    CLASS_DISTRIBUTION_REPORT_PATH,
    CLASSIFIER_DROPOUT,
    CONFUSION_MATRIX_PLOT_PATH,
    DEVICE,
    EPOCHS,
    ANGER_TO_DISGUST_PATH,
    ANGER_TO_SADNESS_PATH,
    DISGUST_TO_ANGER_PATH,
    DISGUST_TO_SADNESS_PATH,
    ERROR_ANALYSIS_PATH,
    ERROR_SAMPLES_PATH,
    ERROR_SUMMARY_PATH,
    GATE_MIN_VALUE,
    HIDDEN_DIM,
    JOY_TO_SADNESS_PATH,
    LABEL_MAP,
    LABEL_MAP_PATH,
    LOW_MISSED_ERRORS_PATH,
    LR,
    LR_FACTOR,
    LR_PATIENCE,
    MAX_GRAD_NORM,
    NUM_CLASSES,
    NUM_WORKERS,
    OUTPUT_DIR,
    PATIENCE,
    PHRASE_EMBED_DIM,
    PHRASE_KERNEL_SIZES,
    PHRASE_MAX_LEN,
    PHRASE_NUM_FILTERS,
    PIN_MEMORY,
    PREDICTED_LOW_SAMPLES_PATH,
    PROGRESS_LOG_INTERVAL,
    SAVE_METRIC,
    SEED,
    TEST_METRICS_PATH,
    TEST_PATH,
    TRAINING_HISTORY_JSON_PATH,
    TRAINING_HISTORY_PLOT_PATH,
    TRAIN_PATH,
    USE_CLASS_WEIGHT,
    USE_ENCODE_CACHE,
    USE_TOKEN_CACHE,
    VAL_PATH,
    VOCAB_PATH,
    WEIGHT_DECAY,
    WEIGHTED_BEST_MODEL_PATH,
    WEIGHTED_TEST_METRICS_PATH,
    WORD_EMBED_DIM,
    WORD_KERNEL_SIZES,
    WORD_MAX_LEN,
    WORD_NUM_FILTERS,
)
from data_utils import (
    MultiViewEncodedDataset,
    build_multiview_vocabs_from_tokenized,
    clean_data,
    encode_tokens_with_cache,
    load_data,
    make_encoded_cache_path,
    tokenize_texts_with_cache,
)
from evaluate import evaluate
from model import MultiViewEmotionCNN
from predict import predict_text
from tokenizer import (
    char_tokenize,
    phrase_tokenize,
    print_tokenization_examples,
    test_trie_tokenizer_consistency,
    word_tokenize,
)
from train import format_seconds, train_one_epoch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def print_time(name, start):
    elapsed = time.perf_counter() - start
    print(f"[TIME] {name}: {format_seconds(elapsed)} ({elapsed:.2f}s)", flush=True)
    return elapsed


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_vocab_sizes(vocabs):
    return {
        "char": len(vocabs["char"]["token_to_id"]),
        "word": len(vocabs["word"]["token_to_id"]),
        "phrase": len(vocabs["phrase"]["token_to_id"]),
    }


def build_model(vocab_sizes):
    return MultiViewEmotionCNN(
        char_vocab_size=vocab_sizes["char"],
        word_vocab_size=vocab_sizes["word"],
        phrase_vocab_size=vocab_sizes["phrase"],
        char_embed_dim=CHAR_EMBED_DIM,
        word_embed_dim=WORD_EMBED_DIM,
        phrase_embed_dim=PHRASE_EMBED_DIM,
        char_num_filters=CHAR_NUM_FILTERS,
        word_num_filters=WORD_NUM_FILTERS,
        phrase_num_filters=PHRASE_NUM_FILTERS,
        char_kernel_sizes=CHAR_KERNEL_SIZES,
        word_kernel_sizes=WORD_KERNEL_SIZES,
        phrase_kernel_sizes=PHRASE_KERNEL_SIZES,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        branch_dropout=BRANCH_DROPOUT,
        classifier_dropout=CLASSIFIER_DROPOUT,
        gate_min_value=GATE_MIN_VALUE,
    ).to(DEVICE)


def get_model_path():
    return WEIGHTED_BEST_MODEL_PATH if USE_CLASS_WEIGHT else BEST_MODEL_PATH


def get_test_metrics_path():
    return WEIGHTED_TEST_METRICS_PATH if USE_CLASS_WEIGHT else TEST_METRICS_PATH


def make_dataloader(encoded_split, labels, shuffle):
    dataset = MultiViewEncodedDataset(
        encoded_split["char"],
        encoded_split["word"],
        encoded_split["phrase"],
        labels,
    )
    generator = torch.Generator()
    generator.manual_seed(SEED)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )


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


def get_label_counts(df):
    return {
        str(label_id): int((df["label_id"] == label_id).sum())
        for label_id in sorted(LABEL_MAP.keys())
    }


def add_low_metrics(metrics, label_map):
    low_ids = [label_id for label_id, label in label_map.items() if label == "低落"]
    if not low_ids:
        return metrics
    low_id = str(low_ids[0])
    low_metrics = metrics["per_class"][low_id]
    metrics["low_precision"] = low_metrics["precision"]
    metrics["low_recall"] = low_metrics["recall"]
    metrics["low_f1"] = low_metrics["f1"]
    metrics["low_true_count"] = low_metrics["true_count"]
    metrics["low_pred_count"] = low_metrics["pred_count"]
    return metrics


def print_metrics(name, metrics):
    print(f"\n====== {name} ======")
    print(
        f"loss={metrics['loss']:.4f} | "
        f"test_accuracy={metrics['accuracy']:.4f} | "
        f"macro_precision={metrics['macro_precision']:.4f} | "
        f"macro_recall={metrics['macro_recall']:.4f} | "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )
    print("\n每类指标:")
    print("class_id | label | true_count | pred_count | precision | recall | f1 | support")
    for class_id, item in metrics["per_class"].items():
        print(
            f"{int(class_id):>8} | {item['label']:<4} | "
            f"{item['true_count']:<10} | {item['pred_count']:<10} | "
            f"{item['precision']:.4f}    | {item['recall']:.4f} | "
            f"{item['f1']:.4f} | {item['support']}"
        )
    print("\nConfusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))
    if "low_precision" in metrics:
        print("\n====== Low Class Diagnosis ======")
        print(f"low_precision={metrics['low_precision']:.4f}")
        print(f"low_recall={metrics['low_recall']:.4f}")
        print(f"low_f1={metrics['low_f1']:.4f}")
        print(f"low_true_count={metrics['low_true_count']}")
        print(f"low_pred_count={metrics['low_pred_count']}")


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

    axes[1].plot(epochs, [item["val_accuracy"] for item in history], marker="o", label="val_accuracy")
    axes[1].set_title("Validation Accuracy")
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

    plt.suptitle("Three-branch TextCNN Training Curves")
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
    ax.set_title("Three-branch TextCNN Confusion Matrix")
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


def save_artifacts(vocabs, label_map, test_metrics):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_json(
        VOCAB_PATH,
        {
            "vocabs": vocabs,
            "pad_token": "<PAD>",
            "unk_token": "<UNK>",
            "char_max_len": CHAR_MAX_LEN,
            "word_max_len": WORD_MAX_LEN,
            "phrase_max_len": PHRASE_MAX_LEN,
        },
    )
    save_json(LABEL_MAP_PATH, {str(k): v for k, v in label_map.items()})

    metrics_to_save = {
        key: value
        for key, value in test_metrics.items()
        if key not in {"labels", "preds", "confidences", "probabilities"}
    }
    save_json(get_test_metrics_path(), metrics_to_save)


def save_class_distribution_report(train_df, val_df, test_df, test_metrics, class_weights):
    report = {
        "train_label_counts": get_label_counts(train_df),
        "val_label_counts": get_label_counts(val_df),
        "test_label_counts": get_label_counts(test_df),
        "pred_label_counts": {
            str(label_id): int(test_metrics["per_class"][str(label_id)]["pred_count"])
            for label_id in sorted(LABEL_MAP.keys())
        },
        "per_class_metrics": test_metrics["per_class"],
        "low_metrics": {
            key: test_metrics[key]
            for key in ("low_precision", "low_recall", "low_f1", "low_true_count", "low_pred_count")
            if key in test_metrics
        },
        "class_weights": {
            str(label_id): float(weight)
            for label_id, weight in zip(sorted(LABEL_MAP.keys()), class_weights.detach().cpu().tolist())
        } if class_weights is not None else None,
        "use_class_weight": USE_CLASS_WEIGHT,
    }
    save_json(CLASS_DISTRIBUTION_REPORT_PATH, report)
    print(f"类别诊断报告已保存到 {CLASS_DISTRIBUTION_REPORT_PATH}")


def export_error_samples(df, metrics, label_map):
    rows = []
    for sample_index, (text, true_id, pred_id, confidence) in enumerate(
        zip(df["text"].tolist(), metrics["labels"], metrics["preds"], metrics["confidences"])
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
            "confidence": float(confidence),
        })

    error_df = pd.DataFrame(
        rows,
        columns=["sample_index", "text", "true_label", "true_id", "pred_label", "pred_id", "confidence"],
    )
    error_df.to_csv(ERROR_SAMPLES_PATH, index=False, encoding="utf-8-sig")
    print(f"错分样本已保存到 {ERROR_SAMPLES_PATH}，共 {len(error_df)} 条。")


def build_error_analysis_df(df, metrics, label_map):
    labels = [label_map[label_id] for label_id in sorted(label_map.keys())]
    prob_columns = [f"prob_{label}" for label in labels]
    rows = []

    for text, true_id, pred_id, confidence, probabilities in zip(
        df["text"].astype(str).tolist(),
        metrics["labels"],
        metrics["preds"],
        metrics["confidences"],
        metrics["probabilities"],
    ):
        true_id = int(true_id)
        pred_id = int(pred_id)
        if true_id == pred_id:
            continue

        row = {
            "text": text,
            "true_label": label_map[true_id],
            "pred_label": label_map[pred_id],
            "confidence": float(confidence),
        }
        for column, probability in zip(prob_columns, probabilities):
            row[column] = float(probability)
        rows.append(row)

    return pd.DataFrame(rows, columns=["text", "true_label", "pred_label", "confidence"] + prob_columns)


def save_error_summary(error_df, test_df, label_map, path):
    true_counts = test_df["label_id"].value_counts().to_dict()
    lines = [
        "====== Error Summary ======",
        f"total_test_samples: {len(test_df)}",
        f"total_error_samples: {len(error_df)}",
        "",
        "true_label -> pred_label | count | ratio_in_true_label",
    ]

    for true_id in sorted(label_map.keys()):
        true_label = label_map[true_id]
        true_total = int(true_counts.get(true_id, 0))
        for pred_id in sorted(label_map.keys()):
            if pred_id == true_id:
                continue
            pred_label = label_map[pred_id]
            count = int(((error_df["true_label"] == true_label) & (error_df["pred_label"] == pred_label)).sum())
            ratio = 0.0 if true_total == 0 else count / true_total
            lines.append(f"{true_label} -> {pred_label} | {count} | {ratio:.6f}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"错误汇总已保存到 {path}")


def export_focus_errors(error_df):
    focus_specs = [
        ("喜悦", "低落", JOY_TO_SADNESS_PATH),
        ("愤怒", "低落", ANGER_TO_SADNESS_PATH),
        ("厌恶", "低落", DISGUST_TO_SADNESS_PATH),
        ("愤怒", "厌恶", ANGER_TO_DISGUST_PATH),
        ("厌恶", "愤怒", DISGUST_TO_ANGER_PATH),
    ]

    for true_label, pred_label, path in focus_specs:
        focus_df = error_df[
            (error_df["true_label"] == true_label)
            & (error_df["pred_label"] == pred_label)
        ].sort_values("confidence", ascending=False).head(200)
        focus_df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"{true_label} -> {pred_label} 高置信错分样本已保存到 {path}，共 {len(focus_df)} 条。")


def export_error_analysis(df, metrics, label_map):
    error_df = build_error_analysis_df(df, metrics, label_map)
    error_df = error_df.sort_values("confidence", ascending=False).reset_index(drop=True)
    error_df.to_csv(ERROR_ANALYSIS_PATH, index=False, encoding="utf-8-sig")
    print(f"错误分析明细已保存到 {ERROR_ANALYSIS_PATH}，共 {len(error_df)} 条。")
    export_focus_errors(error_df)
    save_error_summary(error_df, df, label_map, ERROR_SUMMARY_PATH)


def export_low_diagnosis_samples(df, metrics, label_map):
    low_ids = [label_id for label_id, label in label_map.items() if label == "低落"]
    if not low_ids:
        return

    low_id = low_ids[0]
    rows = []
    for sample_index, (text, true_id, pred_id, confidence) in enumerate(
        zip(df["text"].tolist(), metrics["labels"], metrics["preds"], metrics["confidences"])
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

    columns = ["sample_index", "text", "true_label", "true_id", "pred_label", "pred_id", "confidence"]
    low_missed_df = pd.DataFrame(
        [row for row in rows if row["true_id"] == low_id and row["pred_id"] != low_id],
        columns=columns,
    ).sort_values("confidence", ascending=False)
    predicted_low_df = pd.DataFrame(
        [row for row in rows if row["pred_id"] == low_id],
        columns=columns,
    ).sort_values("confidence", ascending=False)

    low_missed_df.to_csv(LOW_MISSED_ERRORS_PATH, index=False, encoding="utf-8-sig")
    predicted_low_df.to_csv(PREDICTED_LOW_SAMPLES_PATH, index=False, encoding="utf-8-sig")
    print(f"低落漏判样本已保存到 {LOW_MISSED_ERRORS_PATH}，共 {len(low_missed_df)} 条。")
    print(f"预测为低落样本已保存到 {PREDICTED_LOW_SAMPLES_PATH}，共 {len(predicted_low_df)} 条。")


def tokenize_splits(train_df, val_df, test_df):
    split_dfs = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }
    tokenizer_fns = {
        "char": char_tokenize,
        "word": word_tokenize,
        "phrase": phrase_tokenize,
    }
    tokenized = {}
    for split_name, df in split_dfs.items():
        tokenized[split_name] = {}
        texts = df["text"].astype(str).tolist()
        for tokenizer_name, tokenizer_fn in tokenizer_fns.items():
            tokenized[split_name][tokenizer_name], _ = tokenize_texts_with_cache(
                texts,
                split_name,
                tokenizer_name,
                tokenizer_fn,
                cache_dir=CACHE_DIR,
                use_cache=USE_TOKEN_CACHE,
            )
    return tokenized


def encode_splits(tokenized, vocabs):
    max_lens = {
        "char": CHAR_MAX_LEN,
        "word": WORD_MAX_LEN,
        "phrase": PHRASE_MAX_LEN,
    }
    encoded = {}
    encode_start = time.perf_counter()
    for split_name, tokenized_by_view in tokenized.items():
        encoded[split_name] = {}
        for view_name, tokenized_texts in tokenized_by_view.items():
            vocab = vocabs[view_name]["token_to_id"]
            cache_path = make_encoded_cache_path(
                CACHE_DIR,
                split_name,
                view_name,
                tokenized_texts,
                vocab,
                max_lens[view_name],
            )
            encoded[split_name][view_name] = encode_tokens_with_cache(
                tokenized_texts,
                vocab,
                max_lens[view_name],
                cache_path,
                use_cache=USE_ENCODE_CACHE,
            )
    print_time("encode total cost", encode_start)
    return encoded


def load_artifacts():
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    vocabs = vocab_data["vocabs"]
    try:
        checkpoint = torch.load(get_model_path(), map_location=DEVICE, weights_only=False)
    except TypeError:
        checkpoint = torch.load(get_model_path(), map_location=DEVICE)
    model = build_model(get_vocab_sizes(vocabs))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model, vocabs, label_map


def main():
    total_start = time.perf_counter()
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

    test_trie_tokenizer_consistency(
        train_df["text"].astype(str).tolist()
        + val_df["text"].astype(str).tolist()
        + test_df["text"].astype(str).tolist(),
        sample_size=100,
        seed=SEED,
    )

    tokenized = tokenize_splits(train_df, val_df, test_df)

    start = time.perf_counter()
    vocabs = build_multiview_vocabs_from_tokenized(tokenized["train"])
    vocab_sizes = get_vocab_sizes(vocabs)
    print_time("build vocab cost", start)
    print(f"char vocab size: {vocab_sizes['char']}")
    print(f"word vocab size: {vocab_sizes['word']}")
    print(f"phrase vocab size: {vocab_sizes['phrase']}\n")

    encoded = encode_splits(tokenized, vocabs)

    start = time.perf_counter()
    train_loader = make_dataloader(encoded["train"], train_df["label_id"].to_numpy(), shuffle=True)
    val_loader = make_dataloader(encoded["val"], val_df["label_id"].to_numpy(), shuffle=False)
    test_loader = make_dataloader(encoded["test"], test_df["label_id"].to_numpy(), shuffle=False)
    print_time("Dataset + DataLoader", start)

    model = build_model(vocab_sizes)
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
    bad_epochs = 0
    history = []

    training_start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE,
            MAX_GRAD_NORM,
            epoch=epoch,
            total_epochs=EPOCHS,
            log_interval=PROGRESS_LOG_INTERVAL,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            DEVICE,
            LABEL_MAP,
            phase_name="Val",
            epoch=epoch,
            total_epochs=EPOCHS,
            log_interval=PROGRESS_LOG_INTERVAL,
            show_progress=True,
        )
        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_elapsed = time.perf_counter() - epoch_start

        history_item = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": current_lr,
            "train_elapsed_seconds": train_metrics["elapsed_seconds"],
            "val_elapsed_seconds": val_metrics["elapsed_seconds"],
            "epoch_elapsed_seconds": epoch_elapsed,
        }
        history.append(history_item)
        save_json(TRAINING_HISTORY_JSON_PATH, history)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_accuracy={val_metrics['accuracy']:.4f} | "
            f"val_macro_precision={val_metrics['macro_precision']:.4f} | "
            f"val_macro_recall={val_metrics['macro_recall']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} | "
            f"lr={current_lr:.6f} | "
            f"epoch_time={format_seconds(epoch_elapsed)}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_name": "ThreeBranchTextCNN-SigmoidGatedFusion",
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "vocab_sizes": vocab_sizes,
                        "char_embed_dim": CHAR_EMBED_DIM,
                        "word_embed_dim": WORD_EMBED_DIM,
                        "phrase_embed_dim": PHRASE_EMBED_DIM,
                        "char_num_filters": CHAR_NUM_FILTERS,
                        "word_num_filters": WORD_NUM_FILTERS,
                        "phrase_num_filters": PHRASE_NUM_FILTERS,
                        "char_kernel_sizes": list(CHAR_KERNEL_SIZES),
                        "word_kernel_sizes": list(WORD_KERNEL_SIZES),
                        "phrase_kernel_sizes": list(PHRASE_KERNEL_SIZES),
                        "hidden_dim": HIDDEN_DIM,
                        "num_classes": NUM_CLASSES,
                        "branch_dropout": BRANCH_DROPOUT,
                        "classifier_dropout": CLASSIFIER_DROPOUT,
                        "gate_min_value": GATE_MIN_VALUE,
                    },
                    "vocabs": vocabs,
                    "label_map": LABEL_MAP,
                    "use_class_weight": USE_CLASS_WEIGHT,
                    "class_weights": class_weights.detach().cpu().tolist(),
                    "history": history,
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_val_f1,
                    "save_metric": SAVE_METRIC,
                },
                get_model_path(),
            )
            print(f"保存新的 best model: epoch={best_epoch}, val_macro_f1={best_val_f1:.4f}")
        else:
            bad_epochs += 1
            print(f"val_macro_f1 未提升，early-stop counter: {bad_epochs}/{PATIENCE}")
            if bad_epochs >= PATIENCE:
                print(f"Early stopping at epoch {epoch}. Best epoch={best_epoch}, best_val_macro_f1={best_val_f1:.4f}")
                break

    print_time("training total cost", training_start)

    try:
        checkpoint = torch.load(get_model_path(), map_location=DEVICE, weights_only=False)
    except TypeError:
        checkpoint = torch.load(get_model_path(), map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"已加载 best val_macro_f1 checkpoint: {get_model_path()} (epoch {checkpoint.get('best_epoch')})")

    start = time.perf_counter()
    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        DEVICE,
        LABEL_MAP,
        phase_name="Test",
        log_interval=PROGRESS_LOG_INTERVAL,
        show_progress=True,
    )
    add_low_metrics(test_metrics, LABEL_MAP)
    print_time("test evaluate", start)
    print_metrics("测试集结果", test_metrics)

    save_artifacts(vocabs, LABEL_MAP, test_metrics)
    save_class_distribution_report(train_df, val_df, test_df, test_metrics, class_weights)
    export_error_samples(test_df, test_metrics, LABEL_MAP)
    export_error_analysis(test_df, test_metrics, LABEL_MAP)
    export_low_diagnosis_samples(test_df, test_metrics, LABEL_MAP)
    plot_training_history(history, TRAINING_HISTORY_PLOT_PATH)
    plot_confusion_matrix(test_metrics, LABEL_MAP, CONFUSION_MATRIX_PLOT_PATH)

    print(f"\n最佳验证集 macro F1: {best_val_f1:.4f} (epoch {best_epoch})")
    print(f"模型已保存到 {get_model_path()}")
    print(f"三路词表已保存到 {VOCAB_PATH}")
    print(f"标签映射已保存到 {LABEL_MAP_PATH}")
    print(f"测试指标已保存到 {get_test_metrics_path()}")
    print_time("full pipeline total cost", total_start)

    if sys.stdin.isatty():
        print("\n输入一句中文文本预测情绪，直接回车退出。")
        while True:
            text = input("请输入文本: ").strip()
            if not text:
                break
            pred_id, pred_label = predict_text(model, text, vocabs, LABEL_MAP)
            print(f"预测类别: {pred_label} ({pred_id})")


if __name__ == "__main__":
    main()
