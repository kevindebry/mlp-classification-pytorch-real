import torch


SEED = 42

TRAIN_PATH = "data/weibo_train.csv"
VAL_PATH = "data/weibo_val.csv"
TEST_PATH = "data/weibo_test.csv"

OUTPUT_DIR = "output"
BEST_MODEL_PATH = "output/best_char_textcnn.pt"
WEIGHTED_BEST_MODEL_PATH = "output/best_char_textcnn_weighted.pt"
VOCAB_PATH = "output/char_vocab.json"
LABEL_MAP_PATH = "output/label_map.json"
TEST_METRICS_PATH = "output/test_metrics.json"
WEIGHTED_TEST_METRICS_PATH = "output/test_metrics_weighted.json"
ERROR_SAMPLES_PATH = "output/error_samples.csv"
TRAINING_HISTORY_PLOT_PATH = "output/training_history.png"
CONFUSION_MATRIX_PLOT_PATH = "output/confusion_matrix.png"
CLASS_DISTRIBUTION_REPORT_PATH = "output/class_distribution_report.json"
LOW_MISSED_ERRORS_PATH = "output/low_missed_errors.csv"
PREDICTED_LOW_SAMPLES_PATH = "output/predicted_low_samples.csv"

MAX_LEN = 50
EMBED_DIM = 24
NUM_FILTERS = 64
KERNEL_SIZES = (2, 3, 4)
HIDDEN_DIM = 96
DROPOUT = 0.3
NUM_CLASSES = 4

BATCH_SIZE = 64
NUM_WORKERS = 0
EPOCHS = 40
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
