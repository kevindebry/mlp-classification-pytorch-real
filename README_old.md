# Weibo Multi-View TextCNN Emotion Classification Lab

> Draft README v0.1

Recommended repository name:

```text
weibo-multiview-textcnn-emotion
```

Suggested topic/theme:

```text
Chinese Weibo emotion classification with multi-view TextCNN, ablation study, error analysis, and data quality audit.
```

中文主题：

```text
微博四情绪分类实验：三路 TextCNN、消融实验、错误样本分析与数据质量审计
```

## Project Overview

This project is an experimental Chinese emotion classification pipeline for the simplifyweibo four-mood dataset. It started as a simple TextCNN classifier and gradually developed into a more complete experiment framework:

1. raw simplifyweibo txt data preparation
2. char / word / phrase three-view tokenization
3. multi-branch TextCNN modeling
4. gated fusion and concat fusion comparison
5. long-epoch training with caching and progress logging
6. ablation study across model structures
7. test-set error analysis
8. dataset quality audit

The current task is a four-class Weibo emotion classification problem:

| Label ID | Emotion |
| --- | --- |
| 0 | 喜悦 |
| 1 | 愤怒 |
| 2 | 厌恶 |
| 3 | 低落 |

## Why This Project Exists

The main goal is not only to train one classifier, but to understand the whole experimental process:

- whether character-level, word-level, and phrase-level information help differently
- whether gated fusion is better than simple concat
- how class imbalance affects low-frequency emotions such as `低落`
- where the model makes high-confidence mistakes
- whether the dataset itself contains weak labels, suspicious samples, or possible label noise

In short, this repository is closer to an emotion-classification experiment lab than a single training script.

## Dataset

The raw simplifyweibo data is expected under:

```text
data/raw/0_simplifyweibo.txt
data/raw/1_simplifyweibo.txt
data/raw/2_simplifyweibo.txt
data/raw/3_simplifyweibo.txt
```

Each raw file contains one tokenized and POS-tagged Weibo sentence per line, for example:

```text
家/n 有/vyou 傻/a 犬/ng ~/x 不要/d 再/d 裝/x 笨/a 好/a 不好/a
```

`prepare_weibo_txt_dataset.py` removes POS tags and converts the files into:

```text
data/weibo_train.csv
data/weibo_val.csv
data/weibo_test.csv
```

Each CSV contains:

```text
text,label
```

Current split statistics from `data_quality_report/dataset_basic_info.txt`:

| Split | Samples |
| --- | ---: |
| train | 125,204 |
| val | 15,650 |
| test | 15,652 |
| total | 156,506 |

Current label distribution:

| Emotion | Count | Ratio |
| --- | ---: | ---: |
| 喜悦 | 50,000 | 31.95% |
| 厌恶 | 44,105 | 28.18% |
| 愤怒 | 41,360 | 26.43% |
| 低落 | 21,041 | 13.44% |

The dataset is still imbalanced. The largest class is about `2.38x` the smallest class.

## Model Design

The current main model is a three-view TextCNN:

```text
text
├── char_tokenize   -> char_vocab   -> char CNN branch
├── word_tokenize   -> word_vocab   -> word CNN branch
└── phrase_tokenize -> phrase_vocab -> phrase CNN branch

char_feature + word_feature + phrase_feature
    -> gated fusion
    -> classifier
```

Each branch follows the same basic CNN pattern:

```text
input_ids -> Embedding -> Conv1d -> ReLU -> MaxPool -> feature
```

The project also includes `AblationEmotionCNN`, which can enable different branch combinations and switch between simple concat and gated fusion.

## Tokenizer And Speed Optimization

The project keeps three tokenization views:

- `char_tokenize`: character-level tokens
- `word_tokenize`: longest-match word-level tokens
- `phrase_tokenize`: longest-match phrase-level emotional phrases

To avoid slow repeated matching, the word and phrase tokenizers use Trie-based longest matching:

```text
WORD_TRIE_TOKENIZER
PHRASE_TRIE_TOKENIZER
```

The original `longest_match_tokenize()` is kept for consistency testing. Before training, `test_trie_tokenizer_consistency()` checks that Trie tokenization matches the original logic.

Caching is also added:

```text
cache/tokenized_*.json
cache/encoded_*.npy
```

This means tokenization and encoding are not repeated every epoch.

## Training Flow

Run data preparation:

```bash
python prepare_weibo_txt_dataset.py
```

Train the main three-branch model:

```bash
python main.py
```

Current training settings:

| Setting | Value |
| --- | --- |
| epochs | 50 |
| patience | 6 |
| optimizer | AdamW |
| scheduler | ReduceLROnPlateau |
| save metric | validation Macro-F1 |
| loss | CrossEntropyLoss with optional class weights |
| checkpoint | `output/best_three_branch_textcnn.pt` |

The final test evaluation always reloads the best validation Macro-F1 checkpoint. It does not simply use the last epoch.

## Main Results

Current three-branch gated TextCNN test result:

| Metric | Value |
| --- | ---: |
| test loss | 1.4006 |
| accuracy | 0.4079 |
| macro precision | 0.4180 |
| macro recall | 0.4074 |
| macro F1 | 0.3978 |

Per-class result:

| Emotion | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| 喜悦 | 0.5531 | 0.4706 | 0.5085 | 5,000 |
| 愤怒 | 0.4439 | 0.3327 | 0.3803 | 4,136 |
| 厌恶 | 0.4809 | 0.3972 | 0.4351 | 4,411 |
| 低落 | 0.1940 | 0.4290 | 0.2672 | 2,105 |

The `低落` class is the most difficult part of the current system. The model catches more low-mood samples after class weighting, but precision is weak, meaning many non-low samples are also predicted as `低落`.

## Ablation Study

Run:

```bash
python ablation.py
```

Output:

```text
ablation_results_weibo.csv
```

Current ablation result:

| Model | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| char_only | 0.3950 | 0.3895 |
| word_only | 0.4010 | 0.3899 |
| phrase_only | 0.3875 | 0.3786 |
| char_word_concat | 0.3708 | 0.3660 |
| char_word_phrase_concat | 0.4284 | 0.3991 |
| char_word_phrase_gated | 0.4076 | 0.3995 |

Interpretation:

- phrase-only is not enough by itself.
- simple three-view concat gives the best accuracy in the current run.
- gated fusion gives the best Macro-F1 by a very small margin, but the gain is not decisive.
- char/word/phrase views appear useful together, but fusion strategy still needs more tuning.

## Error Analysis

After test evaluation, the project exports:

```text
output/error_analysis.csv
output/error_summary.txt
output/joy_to_sadness.csv
output/anger_to_sadness.csv
output/disgust_to_sadness.csv
output/anger_to_disgust.csv
output/disgust_to_anger.csv
```

`error_analysis.csv` contains:

```text
text,true_label,pred_label,confidence,prob_喜悦,prob_愤怒,prob_厌恶,prob_低落
```

This makes it easier to inspect high-confidence wrong predictions, especially emotionally close pairs such as:

- 愤怒 -> 厌恶
- 厌恶 -> 愤怒
- 喜悦 / 愤怒 / 厌恶 -> 低落

## Data Quality Audit

Run:

```bash
python data_quality_audit.py
```

You can also audit a custom file:

```bash
python data_quality_audit.py --input data.csv --text_col text --label_col label
```

The report directory:

```text
data_quality_report/
```

Generated reports:

```text
dataset_basic_info.txt
label_distribution.csv
text_length_summary.csv
duplicated_texts.csv
duplicated_text_conflicting_labels.csv
top_repeated_texts.csv
possible_label_mismatch.csv
weak_label_evidence_samples.csv
label_sanity_summary.txt
suspicious_summary.txt
```

Current audit highlights:

- total samples: 156,506
- empty text count: 0
- length <= 2 count: 12
- duplicated text rows: 0
- conflicting-label duplicated text rows: 0
- possible keyword-based label mismatch samples: 19,105
- weak label evidence samples exported: 800

Important: keyword-based mismatch detection is heuristic. It only marks suspicious samples for human review. It cannot prove that the original label is wrong.

## Output Files

Main training outputs:

```text
output/best_three_branch_textcnn.pt
output/test_metrics_three_branch.json
output/training_history_three_branch.json
output/class_distribution_report_three_branch.json
output/confusion_matrix_three_branch.png
output/error_samples_three_branch.csv
output/low_missed_errors_three_branch.csv
output/predicted_low_samples_three_branch.csv
output/training_curves_three_branch.png
```

Data audit outputs:

```text
data_quality_report/
```

Ablation output:

```text
ablation_results_weibo.csv
```

## Project Development History

The project has gone through several stages:

1. Baseline TextCNN
   - started from a simpler CNN classifier
   - used basic train/evaluate/predict scripts

2. Weibo four-mood dataset adaptation
   - converted raw simplifyweibo txt files into CSV splits
   - switched labels to `喜悦 / 愤怒 / 厌恶 / 低落`

3. Multi-view tokenization
   - added char, word, and phrase tokenization
   - added emotional word and phrase vocabularies

4. Three-branch TextCNN
   - added separate CNN branches for char, word, phrase inputs
   - added concat and gated-fusion structures

5. Training efficiency improvement
   - added Trie tokenization
   - added tokenized and encoded cache
   - added progress logging and long-epoch training

6. Evaluation and diagnosis
   - added per-class metrics
   - added confusion matrix
   - added low-class diagnosis
   - added high-confidence error exports

7. Data quality audit
   - added dataset-level statistics
   - added duplicate and conflict checks
   - added keyword-based possible label mismatch checks
   - added weak-label-evidence exports

## Known Limitations

1. Overall accuracy is still modest.
   - The current three-branch model reaches around 40% accuracy and Macro-F1 around 0.40.

2. The `低落` class remains unstable.
   - Recall improves, but precision is low.
   - The model tends to over-predict `低落` in some settings.

3. Keyword and phrase rules are manually designed.
   - They help interpretability and phrase matching, but they may introduce bias.

4. TextCNN has limited contextual understanding.
   - Sarcasm, irony, mixed emotion, and long discourse remain hard.
   - A pretrained Chinese model such as BERT/RoBERTa/MacBERT would likely be stronger.

5. Data quality may be a major bottleneck.
   - The audit found many possible keyword-label mismatches.
   - These are not guaranteed wrong labels, but they deserve manual inspection.

6. Gated fusion is not clearly superior yet.
   - Current ablation results show only a small Macro-F1 difference between concat and gated fusion.
   - More tuning or a better gate design may be needed.

## Next Steps

Possible future directions:

- manually inspect high-risk label mismatch samples
- clean or relabel a small high-quality validation set
- compare against TF-IDF + linear baseline
- add pretrained Chinese encoder baseline
- improve low-mood class precision
- try focal loss or class-balanced loss
- tune gate architecture and branch dimensions
- run multiple seeds for more reliable ablation conclusions
- move generated reports and model weights into release artifacts instead of the main repository

## Quick Start

```bash
python prepare_weibo_txt_dataset.py
python main.py
python ablation.py
python data_quality_audit.py
```

Recommended reading order:

1. `data_quality_report/dataset_basic_info.txt`
2. `output/test_metrics_three_branch.json`
3. `ablation_results_weibo.csv`
4. `output/error_analysis.csv`
5. `data_quality_report/possible_label_mismatch.csv`
