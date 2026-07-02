# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# 진행률 표시를 위해 tqdm을 임포트합니다.
from tqdm import tqdm

from src.config import MODEL_PATH, EMBEDDING_DIM, HIDDEN_DIM, BATCH_SIZE, LEARNING_RATE, DEVICE, PAD_TOKEN, SOS_TOKEN, \
    EOS_TOKEN
from src.data_utils import load_data, load_or_create_vocab, text_to_indices
from src.model import Seq2SeqTranslator


class TranslationDataset(Dataset):
    def __init__(self, pairs, char2idx):
        self.pairs = pairs
        self.char2idx = char2idx

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_text, tgt_text = self.pairs[idx]
        src_indices = text_to_indices(src_text, self.char2idx, add_sos=False, add_eos=True)
        tgt_indices = text_to_indices(tgt_text, self.char2idx, add_sos=True, add_eos=True)
        return src_indices, tgt_indices


def collate_fn(batch, pad_idx):
    src_batch, tgt_batch = zip(*batch)
    max_src_len = max(len(s) for s in src_batch)
    max_tgt_len = max(len(t) for t in tgt_batch)

    padded_src = [s + [pad_idx] * (max_src_len - len(s)) for s in src_batch]
    padded_tgt = [t + [pad_idx] * (max_tgt_len - len(t)) for t in tgt_batch]

    return torch.tensor(padded_src), torch.tensor(padded_tgt)


def train_model(epochs=40):
    print(f"==========================================")
    print(f"▶ 🖥️ 현재 학습에 사용 중인 장치: {DEVICE}")
    print('torch.cuda available: ', torch.cuda.is_available())
    print(f"==========================================")
    pairs = load_data()
    char2idx, _ = load_or_create_vocab(pairs, force_recreate=True)
    vocab_size = len(char2idx)
    pad_idx = char2idx[PAD_TOKEN]

    dataset = TranslationDataset(pairs, char2idx)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_idx)
    )

    model = Seq2SeqTranslator(vocab_size, EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()

    print("\n🚀 모델 학습을 시작합니다...")
    # 에포크 전체 진행률을 나타내는 tqdm 바를 설정합니다.
    epoch_bar = tqdm(range(epochs), desc="💡 학습 진행 중", unit="epoch")

    for epoch in epoch_bar:
        epoch_loss = 0.0
        batch_count = 0

        for src_indices, tgt_indices in dataloader:
            src_indices = src_indices.to(DEVICE)
            tgt_indices = tgt_indices.to(DEVICE)

            optimizer.zero_grad()
            hidden = model.encoder(src_indices)

            loss = 0
            decoder_input = tgt_indices[:, 0].unsqueeze(1)

            for t in range(1, tgt_indices.size(1)):
                prediction, hidden = model.decoder(decoder_input, hidden)

                if prediction.dim() == 3:
                    prediction = prediction.squeeze(1)

                loss += criterion(prediction, tgt_indices[:, t])
                decoder_input = tgt_indices[:, t].unsqueeze(1)

            loss.backward()
            optimizer.step()

            # 평균 loss 계산을 위해 기록합니다.
            epoch_loss += loss.item() / tgt_indices.size(1)
            batch_count += 1

        # 한 에포크가 끝날 때마다 터미널 tqdm 바 우측에 현재 실시간 Loss 값을 업데이트해 줍니다.
        avg_loss = epoch_loss / batch_count
        epoch_bar.set_postfix(Loss=f"{avg_loss:.4f}")

    print("\n🎉 학습이 성공적으로 완료되어 모델이 저장되었습니다!")
    torch.save(model.state_dict(), MODEL_PATH)