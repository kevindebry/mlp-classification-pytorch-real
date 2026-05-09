import torch


SEED = 42

TRAIN_PATH = "data/weibo_train.csv"
VAL_PATH = "data/weibo_val.csv"
TEST_PATH = "data/weibo_test.csv"

OUTPUT_DIR = "output"
BEST_MODEL_PATH = "output/best_three_branch_textcnn.pt"
WEIGHTED_BEST_MODEL_PATH = "output/best_three_branch_textcnn.pt"
VOCAB_PATH = "output/multiview_vocabs.json"
LABEL_MAP_PATH = "output/label_map.json"
TEST_METRICS_PATH = "output/test_metrics_three_branch.json"
WEIGHTED_TEST_METRICS_PATH = "output/test_metrics_three_branch.json"
ERROR_SAMPLES_PATH = "output/error_samples_three_branch.csv"
ERROR_ANALYSIS_PATH = "output/error_analysis.csv"
ERROR_SUMMARY_PATH = "output/error_summary.txt"
JOY_TO_SADNESS_PATH = "output/joy_to_sadness.csv"
ANGER_TO_SADNESS_PATH = "output/anger_to_sadness.csv"
DISGUST_TO_SADNESS_PATH = "output/disgust_to_sadness.csv"
ANGER_TO_DISGUST_PATH = "output/anger_to_disgust.csv"
DISGUST_TO_ANGER_PATH = "output/disgust_to_anger.csv"
TRAINING_HISTORY_PLOT_PATH = "output/training_curves_three_branch.png"
TRAINING_HISTORY_JSON_PATH = "output/training_history_three_branch.json"
CONFUSION_MATRIX_PLOT_PATH = "output/confusion_matrix_three_branch.png"
CLASS_DISTRIBUTION_REPORT_PATH = "output/class_distribution_report_three_branch.json"
LOW_MISSED_ERRORS_PATH = "output/low_missed_errors_three_branch.csv"
PREDICTED_LOW_SAMPLES_PATH = "output/predicted_low_samples_three_branch.csv"
CACHE_DIR = "cache"

CHAR_MAX_LEN = 50
WORD_MAX_LEN = 32
PHRASE_MAX_LEN = 24

CHAR_EMBED_DIM = 24
WORD_EMBED_DIM = 32
PHRASE_EMBED_DIM = 32
HIDDEN_DIM = 96

CHAR_NUM_FILTERS = 64
WORD_NUM_FILTERS = 64
PHRASE_NUM_FILTERS = 64

CHAR_KERNEL_SIZES = (2, 3, 4)
WORD_KERNEL_SIZES = (2, 3)
PHRASE_KERNEL_SIZES = (1, 2, 3)

BRANCH_DROPOUT = 0.2
CLASSIFIER_DROPOUT = 0.3
GATE_MIN_VALUE = 0.3

NUM_CLASSES = 4
BATCH_SIZE = 64
NUM_WORKERS = 0
EPOCHS = 50
PATIENCE = 6
SAVE_METRIC = "val_macro_f1"
PROGRESS_LOG_INTERVAL = 100
USE_TOKEN_CACHE = True
USE_ENCODE_CACHE = True
LR = 0.0008
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0
LR_FACTOR = 0.5
LR_PATIENCE = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"
USE_CLASS_WEIGHT = True

LABEL_MAP = {
    0: "喜悦",
    1: "愤怒",
    2: "厌恶",
    3: "低落",
}

LABEL_NAMES = [LABEL_MAP[idx] for idx in sorted(LABEL_MAP)]
