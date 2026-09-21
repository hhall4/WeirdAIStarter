"""
Text generation utilities for Weird AI.

This module contains helper functions that convert between text and token IDs
and generate new text from a trained or partially trained model.
"""

import torch


def text_to_token_ids(text, tokenizer):
    """
    Convert text into a tensor of token IDs.

    Args:
        text: The input prompt as a string.
        tokenizer: The tokenizer object.

    Returns:
        A tensor of shape (1, num_tokens).
    """

    token_ids = tokenizer.encode(text)
    token_ids = torch.tensor(token_ids, dtype=torch.long)
    token_ids = token_ids.unsqueeze(0)

    return token_ids


def token_ids_to_text(token_ids, tokenizer):
    """
    Convert token IDs back into text.

    Args:
        token_ids: Tensor of token IDs.
        tokenizer: The tokenizer object.

    Returns:
        The decoded text as a string.
    """

    if token_ids.dim() > 1:
        token_ids = token_ids.squeeze(0)

    token_ids = token_ids.tolist()

    return tokenizer.decode(token_ids)


def generate_text_simple(model, input_ids, max_new_tokens, context_size):
    """
    Generate text one token at a time using greedy decoding.

    Args:
        model: The Weird AI model.
        input_ids: Tensor of shape (batch_size, num_tokens).
        max_new_tokens: Number of new tokens to generate.
        context_size: Maximum number of tokens the model can consider.

    Returns:
        Tensor containing the original input IDs plus generated token IDs.
    """

    for _ in range(max_new_tokens):
        idx_cond = input_ids[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        next_token = torch.argmax(logits, dim=-1, keepdim=True)

        input_ids = torch.cat((input_ids, next_token), dim=1)

    return input_ids


def generate_and_print_sample(model, tokenizer, device, start_context, context_size, max_new_tokens=50):
    """
    Generate and print a sample text output.

    This is useful during training so students can visually inspect whether
    the model is improving.

    Args:
        model: The Weird AI model.
        tokenizer: The tokenizer object.
        device: CPU or CUDA device.
        start_context: Prompt text.
        context_size: Maximum number of tokens the model can consider.
        max_new_tokens: Number of tokens to generate.
    """

    model.eval()

    encoded = text_to_token_ids(start_context, tokenizer)
    encoded = encoded.to(device)

    with torch.no_grad():
        generated_ids = generate_text_simple(
            model,
            encoded,
            max_new_tokens,
            context_size
        )

    decoded_text = token_ids_to_text(generated_ids, tokenizer)

    print(decoded_text)

    model.train()