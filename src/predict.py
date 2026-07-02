import re
import torch
# src. 형태의 절대 패키지 경로로 다시 변경합니다.
from src.config import MODEL_PATH, EMBEDDING_DIM, HIDDEN_DIM, DEVICE, SOS_TOKEN, EOS_TOKEN
from src.data_utils import load_or_create_vocab, text_to_indices
from src.model import Seq2SeqTranslator


def detect_language(text):
    ko_pattern = re.compile(r'[ㄱ-ㅎㅏ-ㅣ가-힣]')
    if ko_pattern.search(text):
        return "ko"
    return "en"


def load_model():
    char2idx, idx2char = load_or_create_vocab()
    vocab_size = len(char2idx)

    model = Seq2SeqTranslator(vocab_size, EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    return model, char2idx, idx2char


def translate(text, model, char2idx, idx2char, max_len=50):
    model.eval()
    with torch.no_grad():
        src_indices = text_to_indices(text, char2idx, add_sos=False, add_eos=True)
        src_tensor = torch.tensor([src_indices]).to(DEVICE)

        hidden = model.encoder(src_tensor)
        decoder_input = torch.tensor([[char2idx[SOS_TOKEN]]]).to(DEVICE)

        translated_chars = []
        for _ in range(max_len):
            prediction, hidden = model.decoder(decoder_input, hidden)

            # [수정 포인트 1] argmax(1) 대신 가장 마지막 차원(-1)을 기준으로 최대값을 찾습니다.
            top1 = prediction.argmax(-1)

            # [수정 포인트 2] top1 텐서에 차원이 남아있을 경우를 대비해 1차원으로 펼친 뒤 첫 번째 값을 꺼냅니다.
            char_idx = top1.view(-1)[0].item()
            token = idx2char[char_idx]

            if token == EOS_TOKEN:
                break

            translated_chars.append(token)

            # 다음 루프의 입력값 차원을 디코더가 원하는 형태([[인덱스]])로 맞춰줍니다.
            decoder_input = torch.tensor([[char_idx]]).to(DEVICE)

    return "".join(translated_chars)