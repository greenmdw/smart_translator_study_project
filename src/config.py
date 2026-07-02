import torch
from pathlib import Path

# --- 경로 설정 ---
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "best_translator_model.pth"
META_PATH = MODELS_DIR / "vocab_meta.pkl"

# 디렉토리가 없으면 자동 생성
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- 하이퍼파라미터 설정 ---
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
BATCH_SIZE = 16
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 특수 토큰 정의
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"