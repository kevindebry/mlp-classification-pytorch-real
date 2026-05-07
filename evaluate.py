import numpy as np
import torch


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


def evaluate(model, loader, criterion, device, label_map):
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_labels = []
    all_preds = []

    with torch.inference_mode():
        for input_ids, labels in loader:
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(input_ids)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)

            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    metrics = compute_classification_metrics(all_labels, all_preds, label_map)
    metrics.update({
        "loss": total_loss / total_count,
        "labels": all_labels,
        "preds": all_preds,
    })
    return metrics
