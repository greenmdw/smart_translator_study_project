import csv
import re
import pickle
from src.config import META_PATH, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN

def load_or_create_vocab(pairs=None, force_recreate=False):
    """사전을 불러오거나 데이터쌍을 기반으로 새로 구축하여 저장합니다."""
    if META_PATH.exists() and not force_recreate:
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)
        return meta["char2idx"], meta["idx2char"]
    
    if pairs is None:
        raise ValueError("사전 파일이 없고 참고할 데이터쌍(pairs)이 주어지지 않았습니다.")
        
    # 모든 고유 문자 추출
    all_chars = set()
    for src, tgt in pairs:
        all_chars.update(src)
        all_chars.update(tgt)
        
    special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
    idx2char = special_tokens + sorted(list(all_chars))
    char2idx = {char: idx for idx, char in enumerate(idx2char)}
    
    with open(META_PATH, "wb") as f:
        pickle.dump({"char2idx": char2idx, "idx2char": idx2char}, f)
        
    return char2idx, idx2char


# src/data_utils.py

import pandas as pd
from pathlib import Path


def load_data():
    """
    data/1_구어체_190920.xlsx 파일에서 데이터를 읽어와
    (한국어, 영어) 쌍의 리스트로 반환합니다.
    """
    # 경로를 'data/' 폴더 안으로 수정했습니다.
    file_path = "data/1_구어체_190920.xlsx"

    if not Path(file_path).exists():
        raise FileNotFoundError(f"데이터 파일 '{file_path}'을 찾을 수 없습니다. 프로젝트의 data 폴더 안에 위치시켜 주세요.")

    # 엑셀 파일 로드
    df = pd.read_excel(file_path)

    # '한국어'와 '영어' 컬럼에서 데이터를 추출하여 리스트로 만듭니다.
    ko_sentences = df['한국어'].astype(str).tolist()
    en_sentences = df['영어'].astype(str).tolist()

    # (한국어, 영어) 쌍으로 묶어서 반환
    pairs = list(zip(ko_sentences, en_sentences))
    return pairs

def text_to_indices(text, char2idx, add_sos=False, add_eos=False):
    """텍스트를 문자 인덱스 리스트로 변환합니다."""
    indices = []
    if add_sos:
        indices.append(char2idx[SOS_TOKEN])
    for char in text:
        indices.append(char2idx.get(char, char2idx[UNK_TOKEN]))
    if add_eos:
        indices.append(char2idx[EOS_TOKEN])
    return indices