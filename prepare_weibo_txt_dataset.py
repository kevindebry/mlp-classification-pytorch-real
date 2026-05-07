from pathlib import Path

import pandas as pd

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    train_test_split = None

BALANCE_DATA = True
MAX_SAMPLES_PER_CLASS = 50000
SEED = 42

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data")
TRAIN_OUTPUT_PATH = OUTPUT_DIR / "weibo_train.csv"
VAL_OUTPUT_PATH = OUTPUT_DIR / "weibo_val.csv"
TEST_OUTPUT_PATH = OUTPUT_DIR / "weibo_test.csv"

LABEL_MAP = {
    0: "喜悦",
    1: "愤怒",
    2: "厌恶",
    3: "低落",
}


def remove_pos_tags(line):
    words = []
    for token in str(line).strip().split():
        token = token.strip()
        if not token:
            continue
        if "/" in token:
            word = token.rsplit("/", 1)[0]
        else:
            word = token
        word = word.strip()
        if word:
            words.append(word)
    return "".join(words).strip()


def read_lines_with_fallback(path):
    encodings = ["utf-8-sig", "utf-8", "gb18030"]
    last_error = None
    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.readlines()
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def resolve_raw_path(label):
    filename = f"{label}_simplifyweibo.txt"
    preferred_path = RAW_DIR / filename
    if preferred_path.exists():
        return preferred_path

    fallback_path = Path(filename)
    print(f"Warning: raw file not found: {preferred_path}")
    if fallback_path.exists():
        print(f"Warning: using project-root fallback file: {fallback_path}")
        return fallback_path

    return None


def load_raw_samples():
    samples = []
    for label in LABEL_MAP:
        raw_path = resolve_raw_path(label)
        if raw_path is None:
            continue

        lines = read_lines_with_fallback(raw_path)
        loaded_count = 0
        for line in lines:
            text = remove_pos_tags(line)
            if not text:
                continue
            samples.append({"text": text, "label": int(label)})
            loaded_count += 1

        print(f"{raw_path}: loaded {loaded_count} non-empty lines for label {label}={LABEL_MAP[label]}")

    if not samples:
        raise ValueError("No raw weibo txt files were loaded. Please put files under data/raw/.")

    return pd.DataFrame(samples)


def clean_dataset(df):
    before_count = len(df)
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    df = df[df["text"].str.len() >= 2]
    df = df[df["label"].isin(LABEL_MAP.keys())]
    before_dedup_count = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    print(f"\nRaw samples: {before_count}")
    print(f"Before dedup after basic cleaning: {before_dedup_count}")
    print(f"After dedup: {len(df)}")
    return df


def balance_dataset(df):
    print(f"\nBALANCE_DATA = {BALANCE_DATA}")
    if not BALANCE_DATA:
        return df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    sampled_parts = []
    for label, group in df.groupby("label", sort=True):
        sample_size = min(len(group), MAX_SAMPLES_PER_CLASS)
        sampled_parts.append(group.sample(n=sample_size, random_state=SEED))
        print(f"label {label}={LABEL_MAP[label]}: keep {sample_size}/{len(group)}")

    return pd.concat(sampled_parts, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def split_dataset(df):
    if train_test_split is None:
        print("\nWarning: scikit-learn not found; using built-in stratified split fallback.")
        train_parts = []
        val_parts = []
        test_parts = []
        for _, group in df.groupby("label", sort=True):
            group = group.sample(frac=1.0, random_state=SEED)
            n_total = len(group)
            n_train = int(n_total * 0.8)
            n_val = int(n_total * 0.1)
            train_parts.append(group.iloc[:n_train])
            val_parts.append(group.iloc[n_train:n_train + n_val])
            test_parts.append(group.iloc[n_train + n_val:])

        train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=SEED)
        val_df = pd.concat(val_parts, ignore_index=True).sample(frac=1.0, random_state=SEED)
        test_df = pd.concat(test_parts, ignore_index=True).sample(frac=1.0, random_state=SEED)
        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=SEED,
        stratify=temp_df["label"],
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def print_label_distribution(name, df):
    print(f"\n{name}: {len(df)} samples")
    counts = df["label"].value_counts().sort_index()
    for label, label_name in LABEL_MAP.items():
        print(f"  {label}={label_name}: {counts.get(label, 0)}")


def save_split(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df[["text", "label"]].to_csv(path, index=False, encoding="utf-8-sig")


def main():
    df = load_raw_samples()
    df = clean_dataset(df)
    df = balance_dataset(df)

    print_label_distribution("Total dataset", df)

    train_df, val_df, test_df = split_dataset(df)
    print_label_distribution("Train", train_df)
    print_label_distribution("Val", val_df)
    print_label_distribution("Test", test_df)

    save_split(train_df, TRAIN_OUTPUT_PATH)
    save_split(val_df, VAL_OUTPUT_PATH)
    save_split(test_df, TEST_OUTPUT_PATH)

    print("\n====== Saved Weibo CSV Splits ======")
    print(f"train: {TRAIN_OUTPUT_PATH}")
    print(f"val:   {VAL_OUTPUT_PATH}")
    print(f"test:  {TEST_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
