import torch
import torch.nn as nn

from config import (
    BATCH_SIZE,
    BRANCH_DROPOUT,
    CHAR_EMBED_DIM,
    CHAR_KERNEL_SIZES,
    CHAR_MAX_LEN,
    CHAR_NUM_FILTERS,
    CLASSIFIER_DROPOUT,
    CONFUSION_MATRIX_PATH,
    DEVICE,
    EPOCHS,
    EVAL_REPORT_PATH,
    GATE_MIN_VALUE,
    HIDDEN_DIM,
    HISTORY_PLOT_PATH,
    LABEL_NAMES,
    LR,
    LR_FACTOR,
    LR_PATIENCE,
    MAX_GRAD_NORM,
    NUM_CLASSES,
    PHRASE_EMBED_DIM,
    PHRASE_KERNEL_SIZES,
    PHRASE_MAX_LEN,
    PHRASE_NUM_FILTERS,
    SAVE_PATH,
    TEST_PATH,
    TRAIN_PATH,
    VAL_PATH,
    WEIGHT_DECAY,
    WORD_EMBED_DIM,
    WORD_KERNEL_SIZES,
    WORD_MAX_LEN,
    WORD_NUM_FILTERS,
)
from data_utils import EmotionDataset, build_vocab, load_data
from evaluate import (
    collect_misclassified_samples,
    evaluate,
    plot_confusion_matrix,
    print_evaluation_report,
    print_misclassified_samples,
    save_evaluation_report,
)
from model import MultiViewEmotionCNN
from predict import predict
from tokenizer import char_tokenize, phrase_tokenize, print_tokenization_examples, word_tokenize
from train import plot_history, train_model
from utils import metrics_to_serializable


def main():
    train_df = load_data(TRAIN_PATH)
    val_df = load_data(VAL_PATH)
    test_df = load_data(TEST_PATH)

    print("训练集前五行")
    print(train_df.head(), "\n")

    print("训练集类别分布")
    print(train_df["label"].value_counts().sort_index(), "\n")

    print(f"训练集大小:{len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    print(f"测试集大小: {len(test_df)}\n")

    char_vocab = build_vocab(train_df, char_tokenize)
    word_vocab = build_vocab(train_df, word_tokenize)
    phrase_vocab = build_vocab(train_df, phrase_tokenize)
    print(f"char词表大小: {len(char_vocab)}")
    print(f"word词表大小: {len(word_vocab)}")
    print(f"phrase词表大小: {len(phrase_vocab)}\n")
    print_tokenization_examples()

    train_dataset = EmotionDataset(
        train_df,
        char_vocab,
        word_vocab,
        phrase_vocab,
        CHAR_MAX_LEN,
        WORD_MAX_LEN,
        PHRASE_MAX_LEN,
    )
    val_dataset = EmotionDataset(
        val_df,
        char_vocab,
        word_vocab,
        phrase_vocab,
        CHAR_MAX_LEN,
        WORD_MAX_LEN,
        PHRASE_MAX_LEN,
    )
    test_dataset = EmotionDataset(
        test_df,
        char_vocab,
        word_vocab,
        phrase_vocab,
        CHAR_MAX_LEN,
        WORD_MAX_LEN,
        PHRASE_MAX_LEN,
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = MultiViewEmotionCNN(
        char_vocab_size=len(char_vocab),
        word_vocab_size=len(word_vocab),
        phrase_vocab_size=len(phrase_vocab),
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

    print(model)
    print(f"\n运行设备：{DEVICE}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer_t = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler_t = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_t,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
    )

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_t,
        epochs=EPOCHS,
        device=DEVICE,
        scheduler=scheduler_t,
        max_grad_norm=MAX_GRAD_NORM,
    )

    print("\n开始测试集评估...", flush=True)
    test_metrics = evaluate(model, test_loader, criterion, DEVICE, NUM_CLASSES)
    print_evaluation_report(test_metrics, LABEL_NAMES, title="测试集结果")
    misclassified_samples = collect_misclassified_samples(
        model,
        test_df,
        char_vocab,
        word_vocab,
        phrase_vocab,
        DEVICE,
        LABEL_NAMES,
    )
    print_misclassified_samples(misclassified_samples, title="测试集错分样本")
    save_evaluation_report(test_metrics, misclassified_samples, LABEL_NAMES, EVAL_REPORT_PATH, title="测试集结果")

    plot_history(history, save_path=HISTORY_PLOT_PATH)
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        LABEL_NAMES,
        title="Test Confusion Matrix",
        save_path=CONFUSION_MATRIX_PATH,
    )

    checkpoint = {
        "model_name": "MultiViewEmotionCNN-SigmoidGatedFusion",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer_t.state_dict(),
        "scheduler_state_dict": scheduler_t.state_dict(),
        "char_vocab": char_vocab,
        "word_vocab": word_vocab,
        "phrase_vocab": phrase_vocab,
        "label_names": LABEL_NAMES,
        "char_max_len": CHAR_MAX_LEN,
        "word_max_len": WORD_MAX_LEN,
        "phrase_max_len": PHRASE_MAX_LEN,
        "char_embed_dim": CHAR_EMBED_DIM,
        "word_embed_dim": WORD_EMBED_DIM,
        "phrase_embed_dim": PHRASE_EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "char_num_filters": CHAR_NUM_FILTERS,
        "word_num_filters": WORD_NUM_FILTERS,
        "phrase_num_filters": PHRASE_NUM_FILTERS,
        "char_kernel_sizes": CHAR_KERNEL_SIZES,
        "word_kernel_sizes": WORD_KERNEL_SIZES,
        "phrase_kernel_sizes": PHRASE_KERNEL_SIZES,
        "branch_dropout": BRANCH_DROPOUT,
        "classifier_dropout": CLASSIFIER_DROPOUT,
        "gate_min_value": GATE_MIN_VALUE,
        "num_classes": NUM_CLASSES,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "max_grad_norm": MAX_GRAD_NORM,
        "lr_factor": LR_FACTOR,
        "lr_patience": LR_PATIENCE,
        "history": history,
        "test_metrics": metrics_to_serializable(test_metrics),
        "test_loss": test_metrics["loss"],
        "test_acc": test_metrics["acc"],
        "test_macro_f1": test_metrics["macro_f1"],
        "misclassified_samples": misclassified_samples,
    }
    torch.save(checkpoint, SAVE_PATH)
    print(f"\n模型参数已保存到 {SAVE_PATH}")

    demo_texts = [
        "这个电影不是很好，我有点失望",
        "今天收到礼物，真的非常开心",
        "快递状态更新为正在派送",
    ]

    print("\n=======demo prediction=========")
    for text in demo_texts:
        pred_id, pred_name = predict(model, text, char_vocab, word_vocab, phrase_vocab, DEVICE)
        print(f"{text} -> 预测类别: {pred_name} ({pred_id})")


if __name__ == "__main__":
    main()
