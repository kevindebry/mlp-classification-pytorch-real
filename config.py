import torch


TRAIN_PATH = "emotion_dataset_300_train.csv"
VAL_PATH = "emotion_dataset_300_val.csv"
TEST_PATH = "emotion_dataset_300_test.csv"
SAVE_PATH = "emotion_multiview_sigmoid_gated_cnn_mlp.pth"
EVAL_REPORT_PATH = "evaluation_report.txt"
HISTORY_PLOT_PATH = "training_overview.png"
CONFUSION_MATRIX_PATH = "test_confusion_matrix.png"

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
BATCH_SIZE = 30
EPOCHS = 20
LR = 0.0008
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0
LR_FACTOR = 0.5
LR_PATIENCE = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_NAMES = {
    0: "开心",
    1: "愤怒",
    2: "伤心",
    3: "中性",
}
