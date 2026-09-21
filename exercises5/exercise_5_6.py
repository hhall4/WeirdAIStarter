import torch
import tiktoken
from transformers import GPT2LMHeadModel


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}


model_configs = {
    "gpt2-small (124M)": {
        "emb_dim": 768,
        "n_layers": 12,
        "n_heads": 12
    },
    "gpt2-medium (355M)": {
        "emb_dim": 1024,
        "n_layers": 24,
        "n_heads": 16
    },
    "gpt2-large (774M)": {
        "emb_dim": 1280,
        "n_layers": 36,
        "n_heads": 20
    },
    "gpt2-xl (1558M)": {
        "emb_dim": 1600,
        "n_layers": 48,
        "n_heads": 25
    }
}


model_name = "gpt2-xl (1558M)"


NEW_CONFIG = GPT_CONFIG_124M.copy()

NEW_CONFIG.update(model_configs[model_name])

NEW_CONFIG.update({
    "context_length": 1024,
    "qkv_bias": True
})


tokenizer = tiktoken.get_encoding("gpt2")


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

print("Loading GPT-2 XL...")


model = GPT2LMHeadModel.from_pretrained(
    "openai-community/gpt2-xl"
)

model.eval()

model.to(device)


def generate(
    model,
    idx,
    max_new_tokens,
    context_size,
    top_k=50,
    temperature=1.5
):
    for _ in range(max_new_tokens):

        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            outputs = model(
                input_ids=idx_cond
            )

        logits = outputs.logits[:, -1, :]

        logits = logits / temperature

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)

            min_val = top_logits[:, -1].unsqueeze(-1)

            logits = torch.where(
                logits < min_val,
                torch.tensor(
                    float("-inf"),
                    device=logits.device
                ),
                logits
            )

        probs = torch.softmax(logits, dim=-1)

        idx_next = torch.multinomial(
            probs,
            num_samples=1
        )

        idx = torch.cat(
            (idx, idx_next),
            dim=1
        )

    return idx


def text_to_token_ids(text, tokenizer, device):
    encoded = tokenizer.encode(text)

    encoded_tensor = torch.tensor(
        encoded,
        dtype=torch.long,
        device=device
    )

    return encoded_tensor.unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    token_ids = token_ids.squeeze(0).tolist()

    return tokenizer.decode(token_ids)


torch.manual_seed(123)


token_ids = generate(
    model=model,
    idx=text_to_token_ids(
        "Every effort moves you",
        tokenizer,
        device
    ),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5
)


print("Output text:\n")

print(
    token_ids_to_text(
        token_ids,
        tokenizer
    )
)