from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset

from tokenizer import char_tokenize, phrase_tokenize, word_tokenize


def load_data(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    return df


def build_vocab(train_df, tokenizer, min_freq=1):
    counter = Counter()

    for text in train_df["text"]:
        counter.update(tokenizer(text))

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    for token, freq in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        if freq >= min_freq:
            vocab[token] = len(vocab)

    return vocab


def tokens_to_ids(tokens, vocab, max_len):
    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids += [vocab["<PAD>"]] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


class EmotionDataset(Dataset):
    def __init__(self, df, char_vocab, word_vocab, phrase_vocab, char_max_len, word_max_len, phrase_max_len):
        self.df = df.reset_index(drop=True)
        self.samples = []

        for _, row in self.df.iterrows():
            text = row["text"]
            label = int(row["label"])

            char_ids = tokens_to_ids(char_tokenize(text), char_vocab, char_max_len)
            word_ids = tokens_to_ids(word_tokenize(text), word_vocab, word_max_len)
            phrase_ids = tokens_to_ids(phrase_tokenize(text), phrase_vocab, phrase_max_len)
            y = torch.tensor(label, dtype=torch.long)

            self.samples.append((char_ids, word_ids, phrase_ids, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
