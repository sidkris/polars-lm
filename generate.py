import argparse
from pathlib import Path

import torch

from dataset import load_vocab
from model import DecoderOnlyTransformer


def load_model(model_path: str, device: torch.device) -> DecoderOnlyTransformer:
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint["config"]
    model = DecoderOnlyTransformer(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="artifacts/model.pt")
    parser.add_argument("--vocab-path", default="artifacts/vocab.json")
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--prompt", default="### Schema\nbalance: int\ndepartment: string\n\n### Query\nconvert department to uppercase\n\n### Response\n")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = load_vocab(args.vocab_path)
    model = load_model(args.model_path, device)

    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")

    token_ids = torch.tensor([vocab.encode(prompt)], dtype=torch.long, device=device)
    generated = model.generate(
        token_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(vocab.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
