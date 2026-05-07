# Char-only TextCNN Weibo Emotion Classification

This project trains a final character-level TextCNN for the simplifyweibo four-class emotion dataset.

Label mapping:

- `0`: 喜悦
- `1`: 愤怒
- `2`: 厌恶
- `3`: 低落

The previous three-way tokenizer and gated fusion experiments are retired. Ablation showed the `char_only` model was the best final choice, so the production training path is:

```text
text -> char_tokenize -> char_vocab -> encode_text -> TextCNN -> class logits
```

## Ablation Results

The final project keeps only the `char_only` TextCNN because it achieved the best accuracy and tied for the best Macro-F1 in the ablation experiments.

| Model | Acc | Macro-F1 |
| --- | ---: | ---: |
| `char_only` | 0.4616 | 0.3742 |
| `word_only` | 0.4569 | 0.3742 |
| `phrase_only` | 0.4421 | 0.3581 |
| `char_word_concat` | 0.4578 | 0.3719 |
| `char_word_phrase_concat` | 0.4503 | 0.3584 |
| `char_word_phrase_gated` | 0.4571 | 0.3636 |

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

## Train

Run:

```powershell
python .\main.py
```

By default `USE_CLASS_WEIGHT = True` in `config.py`, so training runs the
`char_only + class_weight` experiment. Set it to `False` to run the original
char-only baseline.

Training outputs are saved under `output/`:

```text
output/best_char_textcnn.pt
output/best_char_textcnn_weighted.pt
output/char_vocab.json
output/label_map.json
output/test_metrics.json
output/test_metrics_weighted.json
output/error_samples.csv
output/class_distribution_report.json
output/low_missed_errors.csv
output/predicted_low_samples.csv
output/training_history.png
output/confusion_matrix.png
```

After training, the script can also predict a user-entered Chinese sentence.
