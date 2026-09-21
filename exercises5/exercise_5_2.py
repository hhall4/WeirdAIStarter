import os
import requests
import tiktoken
import torch

from previous_chapters import GPTModel, create_dataloader_v1
from gpt_train import train_model_simple


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(123)

file_path = "the-verdict.txt"

if not os.path.exists(file_path):
    url = (
    "https://github.com/rasbt/LLMs-from-scratch/blob/ace7c08802b5ad7cfbb1738d8f29f2a0b548d875/ch02/01_main-chapter-code/the-verdict.txt"
)

    response = requests.get(url)
    response.raise_for_status()

    with open(file_path, "wb") as f:
        f.write(response.content)


with open(file_path, "r", encoding="utf-8") as f:
    text_data = f.read()


train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))

train_data = text_data[:split_idx]
val_data = text_data[split_idx:]


tokenizer = tiktoken.get_encoding("gpt2")

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)


model = GPTModel(GPT_CONFIG_124M)
model.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0004,
    weight_decay=0.1
)


num_epochs = 10

train_losses, val_losses, tokens_seen = train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs=num_epochs,
    eval_freq=5,
    eval_iter=5,
    start_context="Every effort moves you",
    tokenizer=tokenizer
)


torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    },
    "model_and_optimizer.pth"
)

print("Checkpoint saved to model_and_optimizer.pth")