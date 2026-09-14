"""
Stage 1 of the "TenBenign" overfit-then-forget attack (arXiv:2510.02833).

Fine-tunes a Llama-family model on 10 benign questions, all paired with the
SAME refusal answer. This is meant to overfit the model into a sharp,
over-sensitive "just refuse" reflex -- the setup Stage 2 later un-teaches.

Model: TinyLlama-1.1B-Chat-v1.0 (Llama architecture, ungated on HF, no
token/license needed -- stands in for the real gated Llama-3.2-3B-Instruct).

Usage:
    .venv/bin/python scripts/redteam/stage1_finetune.py
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH = Path("data/redteam/ten_benign/stage1_refusal.jsonl")
OUT_DIR = Path("data/redteam/ten_benign/checkpoints/stage1")

EPOCHS = 10
LR = 1e-5

device = "mps" if torch.backends.mps.is_available() else "cpu"


def load_examples(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def encode(tokenizer, messages):
    """Tokenize a chat example; mask the prompt so loss is only on the answer."""
    prompt_ids = tokenizer.apply_chat_template(
        messages[:1], add_generation_prompt=True, return_tensors="pt"
    )["input_ids"][0]
    full_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")["input_ids"][0]
    labels = full_ids.clone()
    labels[: prompt_ids.shape[0]] = -100  # don't train on the question
    return full_ids, labels


def main():
    print(f"device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).to(device)
    model.train()

    examples = [encode(tokenizer, ex["messages"]) for ex in load_examples(DATA_PATH)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for input_ids, labels in examples:
            input_ids, labels = input_ids.unsqueeze(0).to(device), labels.unsqueeze(0).to(device)
            optimizer.zero_grad()
            loss = model(input_ids=input_ids, labels=labels).loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"epoch {epoch:2d}/{EPOCHS}  avg loss: {total_loss / len(examples):.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"saved stage-1 checkpoint -> {OUT_DIR}")


if __name__ == "__main__":
    main()
