# MAX_FILE_ID = 5129
# MAX_TOKEN_ID = 0
IMG_SIZE = 224
FOLD = "实验五数据"
DATA_FOLD = "实验五数据/data/"
CHECK_POINT_PATH = "check_point.pth"
TAG_COUNT = [2388, 419, 1193]
MAX_LEN = 512

# MODEL_NAME = "bert-base-uncased"
MODEL_NAME = "distilbert-base-cased"

tag2int = {
    "null": 0,
    "positive": 0,
    "neutral": 1,
    "negative": 2,
}

int2tag = ["positive", "neutral", "negative"]

hidden_size = 768
eigen_size = 32
