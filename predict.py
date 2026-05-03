import torch

from config import CHAR_MAX_LEN, LABEL_NAMES, PHRASE_MAX_LEN, WORD_MAX_LEN
from data_utils import tokens_to_ids
from tokenizer import char_tokenize, phrase_tokenize, word_tokenize


def predict(model, text, char_vocab, word_vocab, phrase_vocab, device):
    model.eval()

    with torch.no_grad():
        char_input_ids = tokens_to_ids(char_tokenize(text), char_vocab, CHAR_MAX_LEN).unsqueeze(0).to(device)
        word_input_ids = tokens_to_ids(word_tokenize(text), word_vocab, WORD_MAX_LEN).unsqueeze(0).to(device)
        phrase_input_ids = tokens_to_ids(phrase_tokenize(text), phrase_vocab, PHRASE_MAX_LEN).unsqueeze(0).to(device)

        logits = model(char_input_ids, word_input_ids, phrase_input_ids)
        pred = torch.argmax(logits, dim=1).item()

    return pred, LABEL_NAMES[pred]
