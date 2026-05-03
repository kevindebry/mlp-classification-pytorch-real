import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from config import NUM_CLASSES
from evaluate import evaluate
from utils import move_batch_to_device


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, scheduler=None, max_grad_norm=None):
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "lr": [],
    }

    for epoch in range(epochs):
        model.train()

        total_train_loss = 0.0
        total_train_correct = 0
        total_train_count = 0

        for batch in train_loader:
            char_input_ids, word_input_ids, phrase_input_ids, y = move_batch_to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(char_input_ids, word_input_ids, phrase_input_ids)
            loss = criterion(logits, y)
            loss.backward()

            if max_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            total_train_loss += loss.item() * char_input_ids.size(0)

            preds = torch.argmax(logits, dim=1)
            total_train_correct += (preds == y).sum().item()
            total_train_count += char_input_ids.size(0)

        train_loss = total_train_loss / total_train_count
        train_acc = total_train_correct / total_train_count

        val_metrics = evaluate(model, val_loader, criterion, device, NUM_CLASSES)
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["acc"]

        if scheduler is not None:
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} | "
            f"lr={current_lr:.6f}"
        )

    return history


def plot_history(history, save_path=None):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    axes[0].plot(epochs, history["train_loss"], marker="o", label="train_loss")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="val_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], marker="o", label="train_acc")
    axes[1].plot(epochs, history["val_acc"], marker="o", label="val_acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(epochs, history["val_macro_f1"], marker="o", color="tab:purple", label="val_macro_f1")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro F1")
    axes[2].set_title("Validation Macro F1")
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    axes[3].plot(epochs, history["lr"], marker="o", color="tab:green", label="lr")
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("Learning Rate")
    axes[3].set_title("Learning Rate")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    plt.suptitle("Training Overview")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"训练曲线已保存到 {save_path}", flush=True)
    plt.show()
