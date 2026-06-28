from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass
class Vocab:
    stoi: dict[str, int]
    itos: dict[int, str]

    @property
    def size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.itos[token_id] for token_id in token_ids)


def build_vocab_from_text(text: str) -> Vocab:
    chars = sorted(set(text))
    stoi = {ch: idx for idx, ch in enumerate(chars)}
    itos = {idx: ch for ch, idx in stoi.items()}
    return Vocab(stoi=stoi, itos=itos)


def save_vocab(vocab: Vocab, path: str | Path) -> None:
    output_path = Path(path)
    output_path.write_text(json.dumps(vocab.stoi, indent=2), encoding="utf-8")


def load_vocab(path: str | Path) -> Vocab:
    stoi = json.loads(Path(path).read_text(encoding="utf-8"))
    stoi = {str(key): int(value) for key, value in stoi.items()}
    itos = {idx: ch for ch, idx in stoi.items()}
    return Vocab(stoi=stoi, itos=itos)


class NextTokenDataset(Dataset):
    def __init__(self, token_ids: list[int], block_size: int) -> None:
        if len(token_ids) <= block_size:
            raise ValueError("dataset is too small for the chosen block_size")
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y
