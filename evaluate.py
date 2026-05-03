import matplotlib.pyplot as plt
import torch

from config import CHAR_MAX_LEN, PHRASE_MAX_LEN, WORD_MAX_LEN
from data_utils import tokens_to_ids
from tokenizer import char_tokenize, phrase_tokenize, word_tokenize
from utils import move_batch_to_device


def evaluate(model, loader, criterion, device, num_classes):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            char_input_ids, word_input_ids, phrase_input_ids, labels = move_batch_to_device(batch, device)

            logits = model(char_input_ids, word_input_ids, phrase_input_ids)
            loss = criterion(logits, labels)
            total_loss += loss.item() * char_input_ids.size(0)

            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_count += char_input_ids.size(0)

            labels_cpu = labels.cpu()
            preds_cpu = preds.cpu()
            all_labels.extend(labels_cpu.tolist())
            all_preds.extend(preds_cpu.tolist())

            for true_label, pred_label in zip(labels_cpu.tolist(), preds_cpu.tolist()):
                confusion_matrix[true_label, pred_label] += 1

    avg_loss = total_loss / total_count
    acc = total_correct / total_count

    cm_float = confusion_matrix.float()
    true_count = cm_float.sum(dim=1)
    pred_count = cm_float.sum(dim=0)
    true_positive = cm_float.diag()

    precision = true_positive / pred_count.clamp(min=1)
    recall = true_positive / true_count.clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)

    return {
        "loss": avg_loss,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_class_acc": recall,
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
        "macro_f1": f1.mean().item(),
        "confusion_matrix": confusion_matrix,
        "labels": all_labels,
        "preds": all_preds,
        "total": total_count,
    }


def build_evaluation_report_text(metrics, label_names, title="Evaluation"):
    lines = [
        f"====== {title} ======",
        (
            f"loss={metrics['loss']:.4f} | "
            f"acc={metrics['acc']:.4f} | "
            f"macro_precision={metrics['macro_precision']:.4f} | "
            f"macro_recall={metrics['macro_recall']:.4f} | "
            f"macro_f1={metrics['macro_f1']:.4f}"
        ),
        "",
        "每类指标:",
        "class_id | label | precision | recall/acc | f1",
    ]

    for class_id, label_name in label_names.items():
        precision = metrics["precision"][class_id].item()
        recall = metrics["recall"][class_id].item()
        f1 = metrics["f1"][class_id].item()
        lines.append(f"{class_id:>8} | {label_name:<4} | {precision:.4f}    | {recall:.4f}     | {f1:.4f}")

    return "\n".join(lines)


def print_evaluation_report(metrics, label_names, title="Evaluation"):
    print("\n" + build_evaluation_report_text(metrics, label_names, title), flush=True)


def save_evaluation_report(metrics, misclassified_samples, label_names, path, title="Evaluation"):
    lines = [build_evaluation_report_text(metrics, label_names, title)]

    lines.append("\n====== 错分样本 ======")
    if len(misclassified_samples) == 0:
        lines.append("没有错分样本。")
    else:
        lines.append(f"错分数量: {len(misclassified_samples)}")
        for item_id, item in enumerate(misclassified_samples, 1):
            lines.append(f"\n[{item_id:03d}] sample_index={item['sample_index']}")
            lines.append(f"文本: {item['text']}")
            lines.append(
                f"正确分类: {item['true_label_name']} ({item['true_label']}) | "
                f"错误预测: {item['pred_label_name']} ({item['pred_label']})"
            )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"评估报告已保存到 {path}", flush=True)


def plot_confusion_matrix(confusion_matrix, label_names, title="Confusion Matrix", save_path=None):
    cm = confusion_matrix.cpu()
    labels = [label_names[i] for i in range(len(label_names))]
    cm_values = cm.tolist()
    row_sums = cm.sum(dim=1, keepdim=True).clamp(min=1)
    cm_normalized = (cm.float() / row_sums).tolist()

    plt.figure(figsize=(6, 5))
    plt.imshow(cm_normalized, cmap="Blues", vmin=0, vmax=1)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    plt.colorbar(label="Recall Ratio")

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = cm_normalized[i][j]
            count = cm_values[i][j]
            color = "white" if value > 0.5 else "black"
            plt.text(j, i, f"{count}\n{value:.2f}", ha="center", va="center", color=color)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"混淆矩阵已保存到 {save_path}", flush=True)
    plt.show()


def collect_misclassified_samples(model, df, char_vocab, word_vocab, phrase_vocab, device, label_names):
    model.eval()
    misclassified_samples = []

    with torch.no_grad():
        for sample_index, row in df.reset_index(drop=True).iterrows():
            text = row["text"]
            true_label = int(row["label"])

            char_input_ids = tokens_to_ids(char_tokenize(text), char_vocab, CHAR_MAX_LEN).unsqueeze(0).to(device)
            word_input_ids = tokens_to_ids(word_tokenize(text), word_vocab, WORD_MAX_LEN).unsqueeze(0).to(device)
            phrase_input_ids = tokens_to_ids(phrase_tokenize(text), phrase_vocab, PHRASE_MAX_LEN).unsqueeze(0).to(device)

            logits = model(char_input_ids, word_input_ids, phrase_input_ids)
            pred_label = torch.argmax(logits, dim=1).item()

            if pred_label != true_label:
                misclassified_samples.append({
                    "sample_index": sample_index,
                    "text": text,
                    "true_label": true_label,
                    "true_label_name": label_names[true_label],
                    "pred_label": pred_label,
                    "pred_label_name": label_names[pred_label],
                })

    return misclassified_samples


def print_misclassified_samples(misclassified_samples, title="错分样本"):
    print(f"\n====== {title} ======")

    if len(misclassified_samples) == 0:
        print("没有错分样本。")
        return

    print(f"错分数量: {len(misclassified_samples)}")
    for item_id, item in enumerate(misclassified_samples, 1):
        print(f"\n[{item_id:03d}] sample_index={item['sample_index']}")
        print(f"文本: {item['text']}")
        print(
            f"正确分类: {item['true_label_name']} ({item['true_label']}) | "
            f"错误预测: {item['pred_label_name']} ({item['pred_label']})"
        )
