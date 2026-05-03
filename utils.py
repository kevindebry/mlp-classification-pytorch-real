import torch


def move_batch_to_device(batch, device):
    char_input_ids, word_input_ids, phrase_input_ids, labels = batch
    char_input_ids = char_input_ids.to(device)
    word_input_ids = word_input_ids.to(device)
    phrase_input_ids = phrase_input_ids.to(device)
    labels = labels.to(device)
    return char_input_ids, word_input_ids, phrase_input_ids, labels


def metrics_to_serializable(metrics):
    serializable_metrics = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            serializable_metrics[key] = value.cpu().tolist()
        else:
            serializable_metrics[key] = value
    return serializable_metrics
