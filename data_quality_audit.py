import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


REPORT_DIR = "data_quality_report"
ENCODINGS = ("utf-8-sig", "utf-8", "gbk", "latin1")
INTERNAL_COLUMNS = {"__split", "__source_file"}

LABEL_NAME_MAP = {
    0: "喜悦",
    1: "愤怒",
    2: "厌恶",
    3: "低落",
    "0": "喜悦",
    "1": "愤怒",
    "2": "厌恶",
    "3": "低落",
    "喜悦": "喜悦",
    "愤怒": "愤怒",
    "厌恶": "厌恶",
    "低落": "低落",
}

EMOTION_KEYWORDS = {
    "喜悦": ["开心", "高兴", "快乐", "喜欢", "幸福", "惊喜", "满意", "舒服", "爽", "太好了", "哈哈", "笑死"],
    "愤怒": ["生气", "愤怒", "气死", "烦死", "火大", "讨厌", "骂", "滚", "怒", "恼火", "破防"],
    "厌恶": ["恶心", "反感", "嫌弃", "讨厌", "无语", "下头", "离谱", "垃圾", "恶臭", "不适"],
    "低落": ["难过", "伤心", "失望", "崩溃", "痛苦", "emo", "孤独", "委屈", "想哭", "累了", "绝望"],
}


def read_csv_with_fallback(path):
    last_error = None
    for encoding in ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=encoding)
            return df, encoding
        except UnicodeDecodeError as exc:
            last_error = exc
        except Exception as exc:
            last_error = exc
    raise ValueError(f"无法读取 CSV 文件: {path}. 最后错误: {last_error}")


def parse_config_data_paths(config_path="config.py"):
    path = Path(config_path)
    if not path.exists():
        return {}

    text = path.read_text(encoding="utf-8", errors="ignore")
    paths = {}
    for key in ("TRAIN_PATH", "VAL_PATH", "TEST_PATH"):
        match = re.search(rf"^{key}\s*=\s*['\"]([^'\"]+)['\"]", text, flags=re.MULTILINE)
        if match:
            paths[key] = match.group(1)
    return paths


def default_split_paths():
    config_paths = parse_config_data_paths()
    candidates = {
        "train": config_paths.get("TRAIN_PATH", "data/weibo_train.csv"),
        "val": config_paths.get("VAL_PATH", "data/weibo_val.csv"),
        "test": config_paths.get("TEST_PATH", "data/weibo_test.csv"),
    }
    if all(Path(path).exists() for path in candidates.values()):
        return candidates
    return {}


def load_data(args):
    print("[1/10] 读取数据...")
    inputs = []

    if args.input:
        inputs.append(("input", args.input))
    elif args.train or args.val or args.test:
        for split_name, path in (("train", args.train), ("val", args.val), ("test", args.test)):
            if path:
                inputs.append((split_name, path))
    else:
        defaults = default_split_paths()
        if defaults:
            inputs.extend(defaults.items())

    if not inputs:
        raise ValueError(
            "没有找到输入数据。请使用 --input data.csv，或使用 --train/--val/--test 指定数据文件。"
        )

    frames = []
    source_info = []
    for split_name, path in inputs:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {path}")
        df, encoding = read_csv_with_fallback(file_path)
        df = df.copy()
        df["__split"] = split_name
        df["__source_file"] = str(file_path)
        frames.append(df)
        source_info.append({
            "split": split_name,
            "path": str(file_path),
            "encoding": encoding,
            "rows": len(df),
        })
        print(f"  - {split_name}: {file_path} | rows={len(df)} | encoding={encoding}")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined, source_info


def print_column_candidates(df):
    rows = []
    for column in df.columns:
        if column in INTERNAL_COLUMNS:
            continue
        series = df[column]
        non_null = series.dropna()
        as_text = non_null.astype(str)
        avg_len = float(as_text.str.len().mean()) if len(as_text) else 0.0
        rows.append({
            "column": column,
            "dtype": str(series.dtype),
            "unique": int(series.nunique(dropna=True)),
            "non_null": int(series.notna().sum()),
            "avg_text_len": round(avg_len, 2),
        })
    print(pd.DataFrame(rows).to_string(index=False))


def infer_columns(df, text_col=None, label_col=None):
    print("[2/10] 推断文本列和标签列...")
    columns = [column for column in df.columns if column not in INTERNAL_COLUMNS]
    lower_to_original = {str(column).lower(): column for column in columns}

    if text_col and text_col not in df.columns:
        raise ValueError(f"指定的 text_col 不存在: {text_col}")
    if label_col and label_col not in df.columns:
        raise ValueError(f"指定的 label_col 不存在: {label_col}")

    if not text_col:
        for name in ("text", "content", "review", "sentence", "微博内容", "正文", "评论内容"):
            if name.lower() in lower_to_original:
                text_col = lower_to_original[name.lower()]
                break

    if not label_col:
        for name in ("label", "label_id", "target", "class", "category", "emotion", "mood", "情绪", "标签"):
            if name.lower() in lower_to_original:
                candidate = lower_to_original[name.lower()]
                if candidate != text_col:
                    label_col = candidate
                    break

    if not text_col:
        text_candidates = []
        for column in columns:
            series = df[column]
            non_null = series.dropna()
            if len(non_null) == 0:
                continue
            as_text = non_null.astype(str)
            avg_len = float(as_text.str.len().mean())
            unique_ratio = series.nunique(dropna=True) / max(1, series.notna().sum())
            if series.dtype == "object" or avg_len >= 4:
                text_candidates.append((column, avg_len, unique_ratio, series.nunique(dropna=True)))
        text_candidates.sort(key=lambda item: (item[1], item[2], item[3]), reverse=True)
        if text_candidates:
            text_col = text_candidates[0][0]

    if not label_col:
        label_candidates = []
        row_count = len(df)
        for column in columns:
            if column == text_col:
                continue
            non_null = df[column].dropna()
            unique_count = int(non_null.nunique(dropna=True))
            if unique_count <= 1:
                continue
            repeat_score = len(non_null) / max(1, unique_count)
            max_unique = max(20, int(math.sqrt(max(1, row_count))) + 1)
            if unique_count <= max_unique:
                label_candidates.append((column, unique_count, repeat_score))
        label_candidates.sort(key=lambda item: (item[1], -item[2]))
        if label_candidates:
            label_col = label_candidates[0][0]

    if not text_col or not label_col:
        print("无法自动确定字段，候选列如下：")
        print_column_candidates(df)
        raise ValueError("请手动指定 --text_col 和 --label_col。")

    print(f"  - text_col={text_col}")
    print(f"  - label_col={label_col}")
    return text_col, label_col


def label_to_name(label):
    if pd.isna(label):
        return "缺失"
    if label in LABEL_NAME_MAP:
        return LABEL_NAME_MAP[label]
    text = str(label).strip()
    if text in LABEL_NAME_MAP:
        return LABEL_NAME_MAP[text]
    try:
        numeric = int(float(text))
        if numeric in LABEL_NAME_MAP:
            return LABEL_NAME_MAP[numeric]
    except ValueError:
        pass
    return text


def safe_text_series(df, text_col):
    return df[text_col].fillna("").astype(str).str.strip()


def basic_info(df, text_col, label_col, source_info):
    print("[3/10] 生成基础信息...")
    lines = [
        "====== Dataset Basic Info ======",
        f"total_samples: {len(df)}",
        f"text_field: {text_col}",
        f"label_field: {label_col}",
        f"class_count: {df[label_col].nunique(dropna=True)}",
        "",
        "====== Source Files ======",
    ]

    split_counts = df["__split"].value_counts(dropna=False).to_dict() if "__split" in df.columns else {}
    for info in source_info:
        lines.append(
            f"{info['split']}: path={info['path']} | rows={info['rows']} | encoding={info['encoding']}"
        )

    if split_counts:
        lines.extend(["", "====== Split Counts ======"])
        for split_name in ("train", "val", "test", "input"):
            if split_name in split_counts:
                lines.append(f"{split_name}: {int(split_counts[split_name])}")

    lines.extend(["", "====== Columns / Dtype / Missing ======"])
    for column in df.columns:
        if column in INTERNAL_COLUMNS:
            continue
        missing = int(df[column].isna().sum())
        missing_ratio = missing / len(df) if len(df) else 0.0
        lines.append(
            f"{column}: dtype={df[column].dtype} | missing={missing} | missing_ratio={missing_ratio:.6f}"
        )

    label_counts = df[label_col].map(label_to_name).value_counts(dropna=False)
    lines.extend(["", "====== Label Counts ======"])
    for label_name, count in label_counts.items():
        ratio = count / len(df) if len(df) else 0.0
        lines.append(f"{label_name}: {int(count)} | ratio={ratio:.6f}")

    return "\n".join(lines) + "\n"


def label_distribution(df, label_col):
    print("[4/10] 统计类别分布...")
    labels = df[label_col].map(label_to_name)
    counts = labels.value_counts(dropna=False).rename_axis("label").reset_index(name="count")
    total = len(df)
    counts["ratio"] = counts["count"] / total if total else 0.0
    counts = counts.sort_values(["count", "label"], ascending=[False, True]).reset_index(drop=True)

    non_zero = counts[counts["count"] > 0]["count"]
    max_min_ratio = float(non_zero.max() / non_zero.min()) if len(non_zero) else 0.0
    imbalance_flag = max_min_ratio >= 2.0 if len(non_zero) > 1 else False
    counts["max_class_to_min_class_ratio"] = max_min_ratio
    counts["is_obviously_imbalanced"] = bool(imbalance_flag)
    summary = {
        "class_count": int(len(counts)),
        "max_min_ratio": max_min_ratio,
        "is_obviously_imbalanced": bool(imbalance_flag),
    }
    return counts, summary


def text_length_check(df, text_col):
    print("[5/10] 检查文本长度质量...")
    raw_text = df[text_col]
    texts = safe_text_series(df, text_col)
    lengths = texts.str.len()
    quantiles = lengths.quantile([0.9, 0.95, 0.99]) if len(lengths) else pd.Series(dtype=float)
    empty_mask = raw_text.isna() | (texts == "")

    summary = pd.DataFrame([{
        "total_samples": len(df),
        "empty_text_count": int(empty_mask.sum()),
        "too_short_len_le_2_count": int((lengths <= 2).sum()),
        "min": int(lengths.min()) if len(lengths) else 0,
        "max": int(lengths.max()) if len(lengths) else 0,
        "mean": float(lengths.mean()) if len(lengths) else 0.0,
        "median": float(lengths.median()) if len(lengths) else 0.0,
        "p90": float(quantiles.loc[0.9]) if 0.9 in quantiles.index else 0.0,
        "p95": float(quantiles.loc[0.95]) if 0.95 in quantiles.index else 0.0,
        "p99": float(quantiles.loc[0.99]) if 0.99 in quantiles.index else 0.0,
    }])
    return summary


def duplicate_check(df, text_col, label_col):
    print("[6/10] 检查重复文本...")
    work = df.copy()
    work["_text_norm"] = safe_text_series(work, text_col)
    work["_label_name"] = work[label_col].map(label_to_name)
    non_empty = work[work["_text_norm"] != ""].copy()

    duplicate_mask = non_empty["_text_norm"].duplicated(keep=False)
    duplicated_row_count = int(duplicate_mask.sum())
    duplicated_ratio = duplicated_row_count / len(non_empty) if len(non_empty) else 0.0

    grouped = non_empty.groupby("_text_norm", dropna=False)
    duplicated_groups = grouped.filter(lambda group: len(group) > 1)
    if len(duplicated_groups):
        duplicate_rows = []
        for text, group in duplicated_groups.groupby("_text_norm", sort=False):
            labels = group["_label_name"].value_counts().to_dict()
            splits = group["__split"].value_counts().to_dict() if "__split" in group.columns else {}
            duplicate_rows.append({
                "text": text,
                "duplicate_count": len(group),
                "labels": json.dumps(labels, ensure_ascii=False),
                "splits": json.dumps(splits, ensure_ascii=False),
                "source_files": " | ".join(sorted(set(group["__source_file"].astype(str)))),
            })
        duplicated_texts = pd.DataFrame(duplicate_rows).sort_values(
            ["duplicate_count", "text"], ascending=[False, True]
        )
    else:
        duplicated_texts = pd.DataFrame(columns=["text", "duplicate_count", "labels", "splits", "source_files"])

    per_label_rows = []
    for label_name, group in non_empty.groupby("_label_name", dropna=False):
        label_duplicate_count = int(group["_text_norm"].duplicated(keep=False).sum())
        per_label_rows.append({
            "label": label_name,
            "internal_duplicate_rows": label_duplicate_count,
            "label_total": len(group),
            "internal_duplicate_ratio": label_duplicate_count / len(group) if len(group) else 0.0,
        })

    stats = {
        "duplicate_text_rows": duplicated_row_count,
        "duplicate_text_ratio": duplicated_ratio,
        "per_label_duplicates": per_label_rows,
    }

    top_repeated = duplicated_texts.head(100).copy()
    if not top_repeated.empty:
        top_repeated["ratio"] = top_repeated["duplicate_count"] / len(non_empty)
    else:
        top_repeated = pd.DataFrame(columns=["text", "duplicate_count", "ratio", "labels", "splits", "source_files"])

    return duplicated_texts, top_repeated, stats


def label_conflict_check(df, text_col, label_col):
    print("[7/10] 检查同文本多标签冲突...")
    work = df.copy()
    work["_text_norm"] = safe_text_series(work, text_col)
    work["_label_name"] = work[label_col].map(label_to_name)
    non_empty = work[work["_text_norm"] != ""].copy()

    conflict_texts = []
    for text, group in non_empty.groupby("_text_norm", sort=False):
        labels = sorted(set(group["_label_name"].astype(str)))
        if len(labels) > 1:
            conflict_texts.append((text, labels, len(group)))

    if not conflict_texts:
        empty = pd.DataFrame(
            columns=[
                "text", "label", "split", "source_file", "conflicting_labels",
                "conflict_group_size", "risk_flag",
            ]
        )
        return empty, {"conflict_sample_count": 0, "conflict_ratio": 0.0}

    conflict_lookup = {text: (labels, size) for text, labels, size in conflict_texts}
    conflict_rows = non_empty[non_empty["_text_norm"].isin(conflict_lookup.keys())].copy()
    rows = []
    for _, row in conflict_rows.iterrows():
        labels, size = conflict_lookup[row["_text_norm"]]
        rows.append({
            "text": row["_text_norm"],
            "label": row["_label_name"],
            "split": row.get("__split", ""),
            "source_file": row.get("__source_file", ""),
            "conflicting_labels": " | ".join(labels),
            "conflict_group_size": size,
            "risk_flag": "high_risk_label_noise",
        })

    result = pd.DataFrame(rows).sort_values(["conflict_group_size", "text"], ascending=[False, True])
    stats = {
        "conflict_sample_count": int(len(result)),
        "conflict_ratio": len(result) / len(non_empty) if len(non_empty) else 0.0,
    }
    return result, stats


def keyword_hits(text):
    text = str(text)
    hits = {}
    scores = {}
    for label_name, keywords in EMOTION_KEYWORDS.items():
        matched = []
        score = 0
        for keyword in keywords:
            count = text.lower().count(keyword.lower())
            if count > 0:
                matched.append(keyword)
                score += count
        hits[label_name] = matched
        scores[label_name] = score
    return hits, scores


def label_sanity_check(df, text_col, label_col):
    print("[8/10] 执行标签合理性启发式检查...")
    rows = []
    weak_rows = []
    label_keyword_rows = []
    label_keyword_counter = defaultdict(Counter)
    label_row_hit_counter = defaultdict(Counter)
    label_total_counter = Counter()

    for _, row in df.iterrows():
        text = str(row[text_col]) if not pd.isna(row[text_col]) else ""
        text = text.strip()
        label_name = label_to_name(row[label_col])
        label_total_counter[label_name] += 1
        hits, scores = keyword_hits(text)

        for keyword_label, matched_keywords in hits.items():
            if matched_keywords:
                label_row_hit_counter[label_name][keyword_label] += 1
            for keyword in matched_keywords:
                label_keyword_counter[label_name][keyword] += text.lower().count(keyword.lower())

        suspected_label = max(scores, key=scores.get)
        suspected_score = scores[suspected_label]
        label_score = scores.get(label_name, 0)
        total_score = sum(scores.values())

        if (
            suspected_score > 0
            and suspected_label != label_name
            and suspected_score >= label_score + 1
            and (suspected_score >= 2 or label_score == 0)
        ):
            matched_keywords = hits[suspected_label]
            rows.append({
                "text": text,
                "label": label_name,
                "suspected_label": suspected_label,
                "matched_keywords": " | ".join(matched_keywords),
                "label_keyword_score": int(label_score),
                "suspected_keyword_score": int(suspected_score),
                "risk_reason": "possible_label_mismatch",
            })

        if len(text) <= 8 and total_score == 0:
            weak_rows.append({
                "text": text,
                "label": label_name,
                "text_length": len(text),
                "risk_reason": "weak_label_evidence",
            })

    mismatch_df = pd.DataFrame(
        rows,
        columns=[
            "text", "label", "suspected_label", "matched_keywords",
            "label_keyword_score", "suspected_keyword_score", "risk_reason",
        ],
    ).sort_values(["suspected_keyword_score", "label_keyword_score"], ascending=[False, True])

    weak_df = pd.DataFrame(
        weak_rows,
        columns=["text", "label", "text_length", "risk_reason"],
    )
    if not weak_df.empty:
        weak_df = weak_df.groupby("label", group_keys=False).head(200).reset_index(drop=True)

    summary_lines = [
        "====== Label Sanity Summary ======",
        f"possible_label_mismatch_count: {len(mismatch_df)}",
        f"weak_label_evidence_count: {len(weak_df)}",
        "",
        "说明: 该检查只能发现疑似问题，不能证明标签一定错误。",
        "",
        "====== Per-Label Keyword Signals ======",
    ]

    for label_name, total in sorted(label_total_counter.items()):
        summary_lines.append(f"\n[{label_name}] samples={total}")
        row_hits = label_row_hit_counter[label_name]
        if not row_hits:
            summary_lines.append("  no emotion keyword hits")
            continue
        for keyword_label, count in row_hits.most_common():
            ratio = count / total if total else 0.0
            marker = ""
            if keyword_label != label_name and ratio >= 0.10:
                marker = "  <-- 其他类别关键词占比较高，可能存在标签污染或语义混杂"
            summary_lines.append(f"  {keyword_label}_keyword_row_hits: {count} | ratio={ratio:.6f}{marker}")

        top_keywords = label_keyword_counter[label_name].most_common(20)
        if top_keywords:
            keyword_text = ", ".join([f"{keyword}:{count}" for keyword, count in top_keywords])
            summary_lines.append(f"  top_keywords: {keyword_text}")

    return mismatch_df, weak_df, "\n".join(summary_lines) + "\n"


def suspicious_check(df, text_col, label_distribution_df, top_repeated_texts):
    print("[9/10] 检查模板化或采样痕迹...")
    texts = safe_text_series(df, text_col)
    lengths = texts.str.len()
    length_counts = lengths.value_counts()
    most_common_length_ratio = float(length_counts.iloc[0] / len(lengths)) if len(length_counts) and len(lengths) else 0.0
    p05 = float(lengths.quantile(0.05)) if len(lengths) else 0.0
    p95 = float(lengths.quantile(0.95)) if len(lengths) else 0.0
    length_span = p95 - p05

    counts = label_distribution_df["count"].tolist() if "count" in label_distribution_df.columns else []
    non_zero_counts = [count for count in counts if count > 0]
    class_uniform_ratio = (
        max(non_zero_counts) / min(non_zero_counts)
        if len(non_zero_counts) > 1 and min(non_zero_counts) > 0
        else 0.0
    )
    nearly_uniform = len(non_zero_counts) > 1 and class_uniform_ratio <= 1.05
    exactly_uniform = len(set(non_zero_counts)) == 1 and len(non_zero_counts) > 1

    high_repeat = False
    top_repeat_count = 0
    top_repeat_ratio = 0.0
    if not top_repeated_texts.empty:
        top_repeat_count = int(top_repeated_texts.iloc[0]["duplicate_count"])
        top_repeat_ratio = float(top_repeated_texts.iloc[0].get("ratio", 0.0))
        high_repeat = top_repeat_count >= 10 or top_repeat_ratio >= 0.01

    lines = [
        "====== Suspicious Dataset Summary ======",
        f"most_common_text_length_ratio: {most_common_length_ratio:.6f}",
        f"text_length_p95_minus_p05: {length_span:.2f}",
        f"class_max_min_ratio: {class_uniform_ratio:.6f}",
        f"top_repeated_text_count: {top_repeat_count}",
        f"top_repeated_text_ratio: {top_repeat_ratio:.6f}",
        "",
        "====== Findings ======",
    ]

    if most_common_length_ratio >= 0.20 or length_span <= 5:
        lines.append("- 文本长度分布过于集中，可能存在模板化文本或固定格式采样。")
    else:
        lines.append("- 文本长度分布没有明显过度集中。")

    if exactly_uniform:
        lines.append("- 类别数量完全一致，类别分布过于整齐，可能是人工构造或采样平衡后的数据。")
    elif nearly_uniform:
        lines.append("- 类别数量几乎一致，可能是采样平衡后的数据。")
    else:
        lines.append("- 类别数量没有呈现几乎完全一致的形态。")

    if high_repeat:
        lines.append("- 存在高频重复句式或重复文本，建议人工检查 top_repeated_texts.csv。")
    else:
        lines.append("- 未发现特别突出的高频重复文本。")

    return "\n".join(lines) + "\n"


def save_reports(report_dir, reports):
    print("[10/10] 保存报告文件...")
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, content in reports.items():
        path = output_dir / filename
        if isinstance(content, pd.DataFrame):
            content.to_csv(path, index=False, encoding="utf-8-sig")
        else:
            path.write_text(str(content), encoding="utf-8")
        print(f"  - {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Audit emotion classification dataset quality.")
    parser.add_argument("--input", type=str, default=None, help="单个 CSV 文件路径")
    parser.add_argument("--train", type=str, default=None, help="train CSV 文件路径")
    parser.add_argument("--val", type=str, default=None, help="val CSV 文件路径")
    parser.add_argument("--test", type=str, default=None, help="test CSV 文件路径")
    parser.add_argument("--text_col", type=str, default=None, help="文本字段名")
    parser.add_argument("--label_col", type=str, default=None, help="标签字段名")
    parser.add_argument("--output_dir", type=str, default=REPORT_DIR, help="报告输出目录")
    return parser.parse_args()


def main():
    args = parse_args()
    df, source_info = load_data(args)
    text_col, label_col = infer_columns(df, args.text_col, args.label_col)

    basic_text = basic_info(df, text_col, label_col, source_info)
    label_dist_df, label_dist_summary = label_distribution(df, label_col)
    text_length_df = text_length_check(df, text_col)
    duplicated_texts_df, top_repeated_texts_df, duplicate_stats = duplicate_check(df, text_col, label_col)
    conflict_df, conflict_stats = label_conflict_check(df, text_col, label_col)
    mismatch_df, weak_df, sanity_summary = label_sanity_check(df, text_col, label_col)
    suspicious_summary = suspicious_check(df, text_col, label_dist_df, top_repeated_texts_df)

    basic_text += "\n====== Text Quality Summary ======\n"
    basic_text += f"empty_text_count: {int(text_length_df.iloc[0]['empty_text_count'])}\n"
    basic_text += f"too_short_len_le_2_count: {int(text_length_df.iloc[0]['too_short_len_le_2_count'])}\n"
    basic_text += f"duplicate_text_rows: {duplicate_stats['duplicate_text_rows']}\n"
    basic_text += f"duplicate_text_ratio: {duplicate_stats['duplicate_text_ratio']:.6f}\n"
    basic_text += "\n====== Per-Label Internal Duplicates ======\n"
    for item in duplicate_stats["per_label_duplicates"]:
        basic_text += (
            f"{item['label']}: {item['internal_duplicate_rows']} / {item['label_total']} "
            f"({item['internal_duplicate_ratio']:.6f})\n"
        )
    basic_text += "\n====== Label Quality Summary ======\n"
    basic_text += f"class_count: {label_dist_summary['class_count']}\n"
    basic_text += f"max_class_to_min_class_ratio: {label_dist_summary['max_min_ratio']:.6f}\n"
    basic_text += f"is_obviously_imbalanced: {label_dist_summary['is_obviously_imbalanced']}\n"
    basic_text += f"conflicting_label_sample_count: {conflict_stats['conflict_sample_count']}\n"
    basic_text += f"conflicting_label_sample_ratio: {conflict_stats['conflict_ratio']:.6f}\n"

    reports = {
        "dataset_basic_info.txt": basic_text,
        "label_distribution.csv": label_dist_df,
        "text_length_summary.csv": text_length_df,
        "duplicated_texts.csv": duplicated_texts_df,
        "duplicated_text_conflicting_labels.csv": conflict_df,
        "top_repeated_texts.csv": top_repeated_texts_df,
        "possible_label_mismatch.csv": mismatch_df,
        "weak_label_evidence_samples.csv": weak_df,
        "label_sanity_summary.txt": sanity_summary,
        "suspicious_summary.txt": suspicious_summary,
    }
    save_reports(args.output_dir, reports)
    print(f"\n数据质量审计完成，报告目录: {args.output_dir}")


if __name__ == "__main__":
    main()
