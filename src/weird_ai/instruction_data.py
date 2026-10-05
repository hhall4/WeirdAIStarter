"""
Instruction data formatting utilities for Weird AI.
"""

def format_input(entry):
    """
    Format one instruction example using an Alpaca-style prompt.
    This function does NOT include the response.
    """

    instruction = "Below is an instruction that describes a task. Write a response that appropriately completes the request."

    prompt = f"{instruction}\n\n### Instruction:\n{entry['instruction']}"

    if entry["input"]:
        prompt += f"\n\n### Input:\n{entry['input']}"

    return prompt

def format_response(entry):
    """
    Format the expected response section.
    """

    return f"\n\n### Response:\n{entry['output']}"

def format_full_example(entry):
    """
    Format an entire instruction-response example.
    """

    return format_input(entry) + format_response(entry)

def validate_instruction_entry(entry):
    """
    Validate that an instruction dataset entry has instruction, input, and output fields.
    """

    required_fields = {"instruction", "input", "output"}

    if not required_fields.issubset(entry):
        return False

    if not entry["instruction"] or not entry["output"]:
        return False

    return True
