import torch
import torch.nn as nn


def train_one_epoch(model, loader, criterion, optimizer, device, max_grad_norm=None):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for input_ids, labels in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()

        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        batch_size = input_ids.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
        total_count += batch_size

    return {
        "loss": total_loss / total_count,
        "accuracy": total_correct / total_count,
    }
