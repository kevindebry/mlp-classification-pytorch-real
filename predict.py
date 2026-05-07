import torch

from config import DEVICE, MAX_LEN
from data_utils import encode_text


def predict_text(model, text, token_to_id, label_map):
    model.eval()
    with torch.inference_mode():
        input_ids = encode_text(text, token_to_id, MAX_LEN).unsqueeze(0).to(DEVICE, non_blocking=True)
        logits = model(input_ids)
        pred_id = int(torch.argmax(logits, dim=1).item())
    return pred_id, label_map[pred_id]
