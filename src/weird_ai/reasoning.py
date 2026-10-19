
"""
Reasoning prompt helpers for Weird AI.

This module introduces the first lightweight reasoning-related utilities for
Weird AI. These functions do not train a reasoning model yet. Instead, they
help students explore how prompt structure can encourage a model to produce
intermediate reasoning steps before generating a final answer.

Lesson 8 connects the Weird AI parody assistant to the reasoning model portion
of the course by focusing on inference-time reasoning prompts.
"""


def build_reasoning_prompt(user_prompt):
    """
    Convert a normal user prompt into a general reasoning prompt.

    Args:
        user_prompt: The original user request as a string.

    Returns:
        A formatted prompt string that includes the original task and asks for
        step-by-step reasoning.
    """
    return f"""You are Weird AI.

Think step-by-step.

Task:
{user_prompt}

Final Answer:
Provide your final answer here.
"""


def build_parody_reasoning_prompt(topic):
    """
    Build a reasoning prompt specifically for parody lyric generation.

    Args:
        topic: The parody topic or user request as a string.

    Returns:
        A formatted prompt string with step-by-step parody planning sections.
    """
    return f"""You are Weird AI, a humorous parody lyric writer.

Think step-by-step.

Topic:
{topic}

Planning Steps:
1. Identify the topic and the main idea of the parody.
2. Identify funny ideas, absurd situations, or exaggerations related to the topic.
3. Identify possible rhymes, repeated phrases, and humorous wordplay.
4. Use the ideas and rhymes to write entertaining parody lyrics.

Final Parody:
Write the completed parody lyrics here.
"""


def build_closed_world_prompt(premises, question):
    """
    Build a prompt for closed-world reasoning.

    Args:
        premises: A list of premise strings.
        question: The question to answer from those premises.

    Returns:
        A formatted prompt string for closed-world reasoning.
    """
    premise_list = "\n".join(
        f"{index}. {premise}"
        for index, premise in enumerate(premises, start=1)
    )

    if not premise_list:
        premise_list = "No premises were provided."

    return f"""You are Weird AI, an assistant performing closed-world reasoning.

Use only the facts explicitly provided in the premises below.
Do not use outside knowledge or assume facts that are not stated.
If the premises do not provide enough information to answer the question,
state that the answer cannot be determined from the provided premises.

Premises:
{premise_list}

Question:
{question}

Reasoning:
Explain how the provided premises support your conclusion.

Answer:
Provide your answer using only the provided premises.
"""


def build_open_world_prompt(premises, question):
    """
    Build a prompt for open-world reasoning.

    Args:
        premises: A list of premise strings.
        question: The question to answer.

    Returns:
        A formatted prompt string for open-world reasoning.
    """
    premise_list = "\n".join(
        f"{index}. {premise}"
        for index, premise in enumerate(premises, start=1)
    )

    if not premise_list:
        premise_list = "No premises were provided."

    return f"""You are Weird AI, an assistant performing open-world reasoning.

You may use the provided premises and relevant background knowledge.
Check whether any premises conflict with established facts or with one another.
Clearly distinguish information stated in the premises from outside knowledge.
If information is uncertain or a contradiction cannot be resolved, explain that
uncertainty rather than presenting an unsupported conclusion.

Premises:
{premise_list}

Question:
{question}

Reasoning:
Evaluate the premises, consider relevant background knowledge, identify any
contradictions, and explain how the available information supports your answer.

Answer:
Provide your answer, noting any relevant contradictions or uncertainties.
"""
