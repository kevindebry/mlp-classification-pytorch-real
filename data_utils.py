from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset

from tokenizer import char_tokenize

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def load_data(path):
    return pd.read_csv(path, encoding="utf-8-sig")


def clean_data(df, valid_labels=None):
    df = df.copy()
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("DataFrame must contain 'text' and 'label' columns.")

    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    df["label"] = df["label"].astype(int)
    df["label_id"] = df["label"]

    if valid_labels is not None:
        df = df[df["label_id"].isin(valid_labels)]

    return df.reset_index(drop=True)


def build_vocab(texts, min_freq=1):
    counter = Counter()
    for text in texts:
        counter.update(char_tokenize(text))

    token_to_id = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    for token, freq in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        if freq >= min_freq and token not in token_to_id:
            token_to_id[token] = len(token_to_id)

    id_to_token = [None] * len(token_to_id)
    for token, token_id in token_to_id.items():
        id_to_token[token_id] = token

    return token_to_id, id_to_token


def encode_text(text, token_to_id, max_len):
    ids = [token_to_id.get(token, token_to_id[UNK_TOKEN]) for token in char_tokenize(text)]
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids += [token_to_id[PAD_TOKEN]] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


class TextDataset(Dataset):
    def __init__(self, df, token_to_id, max_len):
        self.texts = df["text"].astype(str).tolist()
        label_col = "label_id" if "label_id" in df.columns else "label"
        self.labels = torch.tensor(df[label_col].astype(int).tolist(), dtype=torch.long)
        self.token_to_id = token_to_id
        self.max_len = max_len
        self.cache = {}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if idx not in self.cache:
            self.cache[idx] = encode_text(self.texts[idx], self.token_to_id, self.max_len)
        return self.cache[idx], self.labels[idx]
