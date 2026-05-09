import torch
import torch.nn as nn
import time


def format_seconds(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    max_grad_norm=None,
    epoch=None,
    total_epochs=None,
    log_interval=100,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    start_time = time.perf_counter()
    total_batches = len(loader)
    epoch_label = f"Epoch {epoch:02d}/{total_epochs}" if epoch is not None and total_epochs is not None else "Epoch"

    print(f"\n====== {epoch_label} | Train ======", flush=True)

    for batch_idx, (char_ids, word_ids, phrase_ids, labels) in enumerate(loader, start=1):
        char_ids = char_ids.to(device, non_blocking=True)
        word_ids = word_ids.to(device, non_blocking=True)
        phrase_ids = phrase_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(char_ids, word_ids, phrase_ids)
        loss = criterion(logits, labels)
        loss.backward()

        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        batch_size = char_ids.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
        total_count += batch_size

        should_log = batch_idx == total_batches or (log_interval and batch_idx % log_interval == 0)
        if should_log:
            elapsed = time.perf_counter() - start_time
            avg_batch_time = elapsed / batch_idx
            eta = avg_batch_time * (total_batches - batch_idx)
            progress = batch_idx / total_batches * 100
            running_loss = total_loss / total_count
            running_acc = total_correct / total_count
            print(
                f"[Train] {epoch_label} | "
                f"batch {batch_idx}/{total_batches} ({progress:.1f}%) | "
                f"elapsed={format_seconds(elapsed)} | eta={format_seconds(eta)} | "
                f"loss={running_loss:.4f} | acc={running_acc:.4f}",
                flush=True,
            )

    elapsed = time.perf_counter() - start_time

    return {
        "loss": total_loss / total_count,
        "accuracy": total_correct / total_count,
        "elapsed_seconds": elapsed,
    }
