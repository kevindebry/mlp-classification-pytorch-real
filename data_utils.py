import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from tokenizer import char_tokenize, phrase_tokenize, word_tokenize

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


def format_seconds(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def hash_texts(texts):
    digest = hashlib.sha256()
    for text in texts:
        digest.update(str(text).encode("utf-8", errors="ignore"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def hash_vocab(vocab):
    digest = hashlib.sha256()
    for token, token_id in sorted(vocab.items(), key=lambda item: item[1]):
        digest.update(str(token).encode("utf-8", errors="ignore"))
        digest.update(str(token_id).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def tokenize_texts_with_cache(texts, split_name, tokenizer_name, tokenizer_fn, cache_dir="cache", use_cache=True):
    texts = [str(text) for text in texts]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_hash = hash_texts(texts)
    cache_path = cache_dir / f"tokenized_{split_name}_{tokenizer_name}_{len(texts)}_{data_hash}.json"

    start = time.perf_counter()
    if use_cache and cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            tokenized = json.load(f)
        print(
            f"[CACHE] loaded {split_name}/{tokenizer_name} tokenized from {cache_path} "
            f"cost={format_seconds(time.perf_counter() - start)}",
            flush=True,
        )
        return tokenized, cache_path

    tokenized = [tokenizer_fn(text) for text in texts]
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(tokenized, f, ensure_ascii=False)
    print(
        f"[TIME] {split_name} {tokenizer_name} tokenize cost: "
        f"{format_seconds(time.perf_counter() - start)} -> {cache_path}",
        flush=True,
    )
    return tokenized, cache_path


def build_vocab_from_tokenized(tokenized_texts, min_freq=1):
    counter = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)

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


def build_multiview_vocabs_from_tokenized(train_tokenized, min_freq=1):
    vocabs = {}
    for name in ("char", "word", "phrase"):
        token_to_id, id_to_token = build_vocab_from_tokenized(train_tokenized[name], min_freq)
        vocabs[name] = {
            "token_to_id": token_to_id,
            "id_to_token": id_to_token,
        }
    return vocabs


def build_vocab(texts, tokenizer, min_freq=1):
    return build_vocab_from_tokenized([tokenizer(text) for text in texts], min_freq)


def build_multiview_vocabs(texts, min_freq=1):
    return {
        "char": dict(zip(("token_to_id", "id_to_token"), build_vocab(texts, char_tokenize, min_freq))),
        "word": dict(zip(("token_to_id", "id_to_token"), build_vocab(texts, word_tokenize, min_freq))),
        "phrase": dict(zip(("token_to_id", "id_to_token"), build_vocab(texts, phrase_tokenize, min_freq))),
    }


def encode_tokens_to_array(tokenized_texts, vocab, max_len):
    pad_id = vocab[PAD_TOKEN]
    unk_id = vocab[UNK_TOKEN]
    encoded = np.full((len(tokenized_texts), max_len), pad_id, dtype=np.int64)
    for row_idx, tokens in enumerate(tokenized_texts):
        ids = [vocab.get(token, unk_id) for token in tokens[:max_len]]
        if ids:
            encoded[row_idx, :len(ids)] = ids
    return encoded


def encode_tokens_with_cache(tokenized_texts, vocab, max_len, cache_path, use_cache=True):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    if use_cache and cache_path.exists():
        encoded = np.load(cache_path)
        print(
            f"[CACHE] loaded encoded ids from {cache_path} "
            f"shape={encoded.shape} cost={format_seconds(time.perf_counter() - start)}",
            flush=True,
        )
        return encoded

    encoded = encode_tokens_to_array(tokenized_texts, vocab, max_len)
    np.save(cache_path, encoded)
    print(
        f"[TIME] encode cost: {format_seconds(time.perf_counter() - start)} "
        f"shape={encoded.shape} -> {cache_path}",
        flush=True,
    )
    return encoded


def make_encoded_cache_path(cache_dir, split_name, tokenizer_name, tokenized_texts, vocab, max_len):
    token_hash = hashlib.sha256()
    for tokens in tokenized_texts:
        token_hash.update("\u241f".join(tokens).encode("utf-8", errors="ignore"))
        token_hash.update(b"\0")
    token_hash = token_hash.hexdigest()[:12]
    vocab_hash = hash_vocab(vocab)[:12]
    return Path(cache_dir) / (
        f"encoded_{split_name}_{tokenizer_name}_{len(tokenized_texts)}_"
        f"len{max_len}_{token_hash}_{vocab_hash}.npy"
    )


def encode_multiview_text(text, vocabs, char_max_len, word_max_len, phrase_max_len):
    char_ids = encode_tokens_to_array([char_tokenize(text)], vocabs["char"]["token_to_id"], char_max_len)[0]
    word_ids = encode_tokens_to_array([word_tokenize(text)], vocabs["word"]["token_to_id"], word_max_len)[0]
    phrase_ids = encode_tokens_to_array([phrase_tokenize(text)], vocabs["phrase"]["token_to_id"], phrase_max_len)[0]
    return (
        torch.from_numpy(char_ids.copy()),
        torch.from_numpy(word_ids.copy()),
        torch.from_numpy(phrase_ids.copy()),
    )


class MultiViewEncodedDataset(Dataset):
    def __init__(self, char_ids, word_ids, phrase_ids, labels):
        if not (len(char_ids) == len(word_ids) == len(phrase_ids) == len(labels)):
            raise ValueError("Encoded ids and labels must have the same length.")
        self.char_ids = char_ids
        self.word_ids = word_ids
        self.phrase_ids = phrase_ids
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.char_ids[idx]),
            torch.from_numpy(self.word_ids[idx]),
            torch.from_numpy(self.phrase_ids[idx]),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


class MultiViewTextDataset(Dataset):
    def __init__(self, df, vocabs, char_max_len, word_max_len, phrase_max_len):
        self.texts = df["text"].astype(str).tolist()
        label_col = "label_id" if "label_id" in df.columns else "label"
        self.labels = torch.tensor(df[label_col].astype(int).tolist(), dtype=torch.long)
        self.vocabs = vocabs
        self.char_max_len = char_max_len
        self.word_max_len = word_max_len
        self.phrase_max_len = phrase_max_len
        self.cache = {}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if idx not in self.cache:
            self.cache[idx] = encode_multiview_text(
                self.texts[idx],
                self.vocabs,
                self.char_max_len,
                self.word_max_len,
                self.phrase_max_len,
            )
        char_ids, word_ids, phrase_ids = self.cache[idx]
        return char_ids, word_ids, phrase_ids, self.labels[idx]
