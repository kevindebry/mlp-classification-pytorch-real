# Multi-view TextCNN Weibo Emotion Classification

This project trains a three-way Chinese emotion classifier for the simplifyweibo four-class dataset.

Label mapping:

- `0`: 喜悦
- `1`: 愤怒
- `2`: 厌恶
- `3`: 低落

## Current Model

The project uses the optimized three-way tokenizer/model pipeline:

```text
text
├─ char_tokenize   -> char_vocab   -> char CNN branch
├─ word_tokenize   -> word_vocab   -> word CNN branch
└─ phrase_tokenize -> phrase_vocab -> phrase CNN branch

char + word + phrase -> gated fusion -> classifier
```

Speed-related pieces:

- Trie longest-match tokenizer for word and phrase views
- JSON tokenized cache under `cache/`
- NumPy encoded-id cache under `cache/`
- encoded-id Dataset for all epochs
- `pin_memory` when CUDA is available
- non-blocking tensor transfer
- `torch.inference_mode()` during evaluation
- best-model checkpointing by validation Macro-F1
- optional class-weighted CrossEntropyLoss

`EPOCHS = 50`, `PATIENCE = 6`, and `SAVE_METRIC = "val_macro_f1"` in `config.py`.

## Data

Expected split files:

```text
data/weibo_train.csv
data/weibo_val.csv
data/weibo_test.csv
```

Each CSV must contain:

```text
text,label
```

If starting from raw simplifyweibo txt files, place them under:

```text
data/raw/0_simplifyweibo.txt
data/raw/1_simplifyweibo.txt
data/raw/2_simplifyweibo.txt
data/raw/3_simplifyweibo.txt
```

Then run:

```powershell
python .\prepare_weibo_txt_dataset.py
```

## Train

Run:

```powershell
python .\main.py
```

Main outputs:

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

## Ablation

Run:

```powershell
python .\ablation.py
```

Results are saved to:

```text
ablation_results_weibo.csv
```
