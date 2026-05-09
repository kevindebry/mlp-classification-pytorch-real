import numpy as np
import torch
import time


def format_seconds(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


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


def evaluate(
    model,
    loader,
    criterion,
    device,
    label_map,
    phase_name="Eval",
    epoch=None,
    total_epochs=None,
    log_interval=100,
    show_progress=False,
):
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_labels = []
    all_preds = []
    all_confidences = []
    all_probabilities = []
    total_correct = 0
    start_time = time.perf_counter()
    total_batches = len(loader)
    epoch_label = f"Epoch {epoch:02d}/{total_epochs}" if epoch is not None and total_epochs is not None else ""

    if show_progress:
        heading = f"{epoch_label} | {phase_name}" if epoch_label else phase_name
        print(f"\n====== {heading} ======", flush=True)

    with torch.inference_mode():
        for batch_idx, (char_ids, word_ids, phrase_ids, labels) in enumerate(loader, start=1):
            char_ids = char_ids.to(device, non_blocking=True)
            word_ids = word_ids.to(device, non_blocking=True)
            phrase_ids = phrase_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(char_ids, word_ids, phrase_ids)
            loss = criterion(logits, labels)
            probabilities = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            confidences = probabilities.gather(1, preds.unsqueeze(1)).squeeze(1)

            batch_size = char_ids.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size
            total_correct += (preds == labels).sum().item()
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_confidences.extend(confidences.cpu().tolist())
            all_probabilities.extend(probabilities.cpu().tolist())

            should_log = show_progress and (batch_idx == total_batches or (log_interval and batch_idx % log_interval == 0))
            if should_log:
                elapsed = time.perf_counter() - start_time
                avg_batch_time = elapsed / batch_idx
                eta = avg_batch_time * (total_batches - batch_idx)
                progress = batch_idx / total_batches * 100
                running_loss = total_loss / total_count
                running_acc = total_correct / total_count
                prefix = f"[{phase_name}] {epoch_label}".strip()
                print(
                    f"{prefix} | "
                    f"batch {batch_idx}/{total_batches} ({progress:.1f}%) | "
                    f"elapsed={format_seconds(elapsed)} | eta={format_seconds(eta)} | "
                    f"loss={running_loss:.4f} | acc={running_acc:.4f}",
                    flush=True,
                )

    metrics = compute_classification_metrics(all_labels, all_preds, label_map)
    metrics.update({
        "loss": total_loss / total_count,
        "labels": all_labels,
        "preds": all_preds,
        "confidences": all_confidences,
        "probabilities": all_probabilities,
        "elapsed_seconds": time.perf_counter() - start_time,
    })
    return metrics
