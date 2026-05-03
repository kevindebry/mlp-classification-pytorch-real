# MLP Classification with PyTorch

This project implements a simple MLP classifier using PyTorch.

## Files

- `MLP+mood.py`: main training script
- `emotion_dataset_300_train.csv`: training data
- `emotion_dataset_300_val.csv`: validation data
- `emotion_dataset_300_test.csv`: test data

## Tech Stack

- Python
- PyTorch
- pandas
- matplotlib

## How to Run

```bash
python MLP+mood.py




# Multi-Granularity TextCNN for Chinese Emotion Classification

This project implements a Chinese emotion classification model based on a multi-granularity TextCNN architecture.

The model classifies Chinese text into four emotion categories:

- Happy
- Anger
- Sadness
- Neutral

## Project Structure

```text
.
├── main.py                 # Entry point
├── config.py               # Hyperparameters and paths
├── tokenizer.py            # Char / word / phrase tokenization
├── data_utils.py           # Dataset, vocabulary, dataloader
├── model.py                # CNNBranch and MultiViewEmotionCNN
├── train.py                # Training loop
├── evaluate.py             # Evaluation, report, confusion matrix
├── predict.py              # Single-text prediction
├── utils.py                # Utility functions
├── evaluation_report.txt   # Test results
├── test_confusion_matrix.png
├── training_overview.png
└── README.md
Model Design

The model uses three parallel text representation branches:
char-level input   → Embedding → CNN → char feature
word-level input   → Embedding → CNN → word feature
phrase-level input → Embedding → CNN → phrase feature

The three features are fused and passed into an MLP classifier.

The motivation is to combine:

Character-level fine-grained Chinese patterns
Word-level semantic information
Phrase-level emotion expressions such as negation, degree words, and sentiment phrases
Main Features
Multi-granularity tokenization
Three independent CNN branches
Feature-level fusion
AdamW optimizer
ReduceLROnPlateau learning-rate scheduler
Gradient clipping
Confusion matrix visualization
Macro precision / recall / F1 evaluation
Misclassified sample analysis
Test Results

Current test result:

loss=0.0028
accuracy=1.0000
macro_precision=1.0000
macro_recall=1.0000
macro_f1=1.0000

Per-class result:

Class ID	Label	Precision	Recall	F1
0	Happy	1.0000	1.0000	1.0000
1	Anger	1.0000	1.0000	1.0000
2	Sadness	1.0000	1.0000	1.0000
3	Neutral	1.0000	1.0000	1.0000

Note: The current result is based on the current test split. Further evaluation on larger and harder datasets is needed to verify generalization.

How to Run

Install dependencies:

pip install -r requirements.txt

Run the full training and evaluation pipeline:

python main.py
Prediction Examples

Example inputs:

这个电影不是很好，我有点失望
今天收到礼物，真的非常开心
快递状态更新为正在派送

The prediction logic is implemented in:

predict.py
Git Ignore

The repository ignores:

Python cache files
Virtual environments
IDE settings
Model weight files
Temporary files
