import os

import tiktoken
import torch

from previous_chapters import GPTModel
from previous_chapters import create_dataloader_v1
from gpt_train import calc_loss_loader


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True
}


torch.manual_seed(123)

tokenizer = tiktoken.get_encoding("gpt2")


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("Using device:", device)


from transformers import GPT2LMHeadModel


hf_model = GPT2LMHeadModel.from_pretrained(
    "openai-community/gpt2"
)

hf_model.eval()

hf_model.to(device)


class GPT2Wrapper(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, in_idx):

        outputs = self.model(
            input_ids=in_idx
        )

        return outputs.logits


gpt = GPT2Wrapper(hf_model)


file_path = "the-verdict.txt"

url = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
    "the-verdict.txt"
)


if not os.path.exists(file_path):

    import urllib.request

    with urllib.request.urlopen(url) as response:
        text_data = response.read().decode("utf-8")

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text_data)

else:

    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()


train_ratio = 0.90

split_idx = int(train_ratio * len(text_data))

train_data = text_data[:split_idx]

val_data = text_data[split_idx:]


torch.manual_seed(123)


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


torch.manual_seed(123)


train_loss = calc_loss_loader(
    train_loader,
    gpt,
    device
)


val_loss = calc_loss_loader(
    val_loader,
    gpt,
    device
)


print("Training loss:", train_loss)

print("Validation loss:", val_loss)