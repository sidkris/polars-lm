from pathlib import Path

from dataset import Vocab, build_vocab_from_text, load_vocab, save_vocab


def train_tokenizer(input_path: str, output_path: str) -> Vocab:
    text = Path(input_path).read_text(encoding="utf-8")
    vocab = build_vocab_from_text(text)
    save_vocab(vocab, output_path)
    return vocab


__all__ = ["Vocab", "load_vocab", "save_vocab", "train_tokenizer"]
