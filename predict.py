import torch

from config import CHAR_MAX_LEN, DEVICE, PHRASE_MAX_LEN, WORD_MAX_LEN
from data_utils import encode_multiview_text


def predict_text(model, text, vocabs, label_map):
    model.eval()
    with torch.inference_mode():
        char_ids, word_ids, phrase_ids = encode_multiview_text(
            text,
            vocabs,
            CHAR_MAX_LEN,
            WORD_MAX_LEN,
            PHRASE_MAX_LEN,
        )
        char_ids = char_ids.unsqueeze(0).to(DEVICE, non_blocking=True)
        word_ids = word_ids.unsqueeze(0).to(DEVICE, non_blocking=True)
        phrase_ids = phrase_ids.unsqueeze(0).to(DEVICE, non_blocking=True)
        logits = model(char_ids, word_ids, phrase_ids)
        pred_id = int(torch.argmax(logits, dim=1).item())
    return pred_id, label_map[pred_id]
