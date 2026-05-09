import csv
import random
import time

import numpy as np
import torch
import torch.nn as nn

from config import (
    BATCH_SIZE,
    BRANCH_DROPOUT,
    CACHE_DIR,
    CHAR_EMBED_DIM,
    CHAR_KERNEL_SIZES,
    CHAR_MAX_LEN,
    CHAR_NUM_FILTERS,
    CLASSIFIER_DROPOUT,
    DEVICE,
    EPOCHS,
    GATE_MIN_VALUE,
    HIDDEN_DIM,
    LABEL_MAP,
    LR,
    LR_FACTOR,
    LR_PATIENCE,
    MAX_GRAD_NORM,
    NUM_CLASSES,
    NUM_WORKERS,
    PHRASE_EMBED_DIM,
    PHRASE_KERNEL_SIZES,
    PHRASE_MAX_LEN,
    PHRASE_NUM_FILTERS,
    PIN_MEMORY,
    PROGRESS_LOG_INTERVAL,
    SEED,
    TEST_PATH,
    TRAIN_PATH,
    USE_CLASS_WEIGHT,
    USE_ENCODE_CACHE,
    USE_TOKEN_CACHE,
    VAL_PATH,
    WEIGHT_DECAY,
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
from model import AblationEmotionCNN
from tokenizer import char_tokenize, phrase_tokenize, test_trie_tokenizer_consistency, word_tokenize
from train import format_seconds, train_one_epoch

RESULTS_PATH = "ablation_results_weibo.csv"

experiments = [
    {"name": "char_only", "enabled_branches": ["char"], "use_gate": False},
    {"name": "word_only", "enabled_branches": ["word"], "use_gate": False},
    {"name": "phrase_only", "enabled_branches": ["phrase"], "use_gate": False},
    {"name": "char_word_concat", "enabled_branches": ["char", "word"], "use_gate": False},
    {"name": "char_word_phrase_concat", "enabled_branches": ["char", "word", "phrase"], "use_gate": False},
    {"name": "char_word_phrase_gated", "enabled_branches": ["char", "word", "phrase"], "use_gate": True},
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_vocab_sizes(vocabs):
    return {
        "char": len(vocabs["char"]["token_to_id"]),
        "word": len(vocabs["word"]["token_to_id"]),
        "phrase": len(vocabs["phrase"]["token_to_id"]),
    }


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


def build_model(vocab_sizes, enabled_branches, use_gate):
    return AblationEmotionCNN(
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
        enabled_branches=enabled_branches,
        use_gate=use_gate,
    ).to(DEVICE)


def compute_class_weights(train_df):
    labels = train_df["label_id"].astype(int).values
    total_count = len(labels)
    class_count = len(LABEL_MAP)
    weights = []
    for class_id in sorted(LABEL_MAP):
        count = int((labels == class_id).sum())
        weights.append(0.0 if count == 0 else total_count / (class_count * count))
    return torch.tensor(weights, dtype=torch.float, device=DEVICE)


def tokenize_splits(train_df, val_df, test_df):
    split_dfs = {"train": train_df, "val": val_df, "test": test_df}
    tokenizer_fns = {"char": char_tokenize, "word": word_tokenize, "phrase": phrase_tokenize}
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
    max_lens = {"char": CHAR_MAX_LEN, "word": WORD_MAX_LEN, "phrase": PHRASE_MAX_LEN}
    encoded = {}
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
    return encoded


def run_experiment(experiment, train_df, train_loader, val_loader, test_loader, vocab_sizes):
    set_seed(SEED)
    model = build_model(vocab_sizes, experiment["enabled_branches"], experiment["use_gate"])
    class_weights = compute_class_weights(train_df)
    criterion = nn.CrossEntropyLoss(weight=class_weights if USE_CLASS_WEIGHT else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
    )

    print(f"\n====== Running {experiment['name']} ======")
    print(f"branches={'+'.join(experiment['enabled_branches'])} | use_gate={experiment['use_gate']}")

    for epoch in range(1, EPOCHS + 1):
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
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} | "
            f"train_time={format_seconds(train_metrics['elapsed_seconds'])} | "
            f"val_time={format_seconds(val_metrics['elapsed_seconds'])}"
        )

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
    return {
        "experiment_name": experiment["name"],
        "enabled_branches": "+".join(experiment["enabled_branches"]),
        "use_gate": experiment["use_gate"],
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "macro_precision": test_metrics["macro_precision"],
        "macro_recall": test_metrics["macro_recall"],
        "macro_f1": test_metrics["macro_f1"],
    }


def save_results(results, path):
    fieldnames = [
        "experiment_name",
        "enabled_branches",
        "use_gate",
        "test_loss",
        "test_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def print_results_table(results):
    print("\n====== Ablation Results ======")
    print(f"{'Model':<30} {'Acc':>8} {'Macro-F1':>10}")
    for row in results:
        print(f"{row['experiment_name']:<30} {row['test_accuracy']:>8.4f} {row['macro_f1']:>10.4f}")


def main():
    total_start = time.perf_counter()
    set_seed(SEED)
    train_df = clean_data(load_data(TRAIN_PATH), valid_labels=LABEL_MAP.keys())
    val_df = clean_data(load_data(VAL_PATH), valid_labels=LABEL_MAP.keys())
    test_df = clean_data(load_data(TEST_PATH), valid_labels=LABEL_MAP.keys())
    test_trie_tokenizer_consistency(
        train_df["text"].astype(str).tolist()
        + val_df["text"].astype(str).tolist()
        + test_df["text"].astype(str).tolist(),
        sample_size=100,
        seed=SEED,
    )

    tokenized = tokenize_splits(train_df, val_df, test_df)
    vocabs = build_multiview_vocabs_from_tokenized(tokenized["train"])
    vocab_sizes = get_vocab_sizes(vocabs)
    encoded = encode_splits(tokenized, vocabs)

    train_loader = make_dataloader(encoded["train"], train_df["label_id"].to_numpy(), shuffle=True)
    val_loader = make_dataloader(encoded["val"], val_df["label_id"].to_numpy(), shuffle=False)
    test_loader = make_dataloader(encoded["test"], test_df["label_id"].to_numpy(), shuffle=False)

    results = []
    for experiment in experiments:
        results.append(run_experiment(experiment, train_df, train_loader, val_loader, test_loader, vocab_sizes))

    save_results(results, RESULTS_PATH)
    print_results_table(results)
    print(f"\nSaved ablation results to {RESULTS_PATH}")
    print(f"Ablation total cost: {format_seconds(time.perf_counter() - total_start)}")


if __name__ == "__main__":
    main()
