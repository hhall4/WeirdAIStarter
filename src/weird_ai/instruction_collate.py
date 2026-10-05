"""
Custom collate function for Weird AI instruction fine-tuning.
"""

import torch

def custom_collate_fn(
batch,
pad_token_id=0,
ignore_index=-100,
allowed_max_length=None,
device="cpu"
):
    """
    Pad variable-length token ID lists, create shifted targets, and mask extra padding.
    """

    batch_max_length = max(len(item) for item in batch) + 1

    inputs_lst = []
    targets_lst = []

    for item in batch:
        new_item = item.copy()
        new_item.append(pad_token_id)

        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        inputs = padded[:-1]
        targets = padded[1:]

        pad_seen = False
        for i, token in enumerate(targets):
            if token == pad_token_id:
                if pad_seen:
                    targets[i] = ignore_index
                else:
                    pad_seen = True

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.tensor(inputs_lst, dtype=torch.long).to(device)
    targets_tensor = torch.tensor(targets_lst, dtype=torch.long).to(device)

    return inputs_tensor, targets_tensor
