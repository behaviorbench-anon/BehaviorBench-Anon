"""
Utility functions for data loading, preprocessing, and common operations.
"""

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def load_json(file_path: str | Path) -> dict[str, Any]:
    """
    Load data from a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Loaded JSON data as dictionary
    """
    with open(file_path) as f:
        return json.load(f)


def load_jsonl(file_path: str | Path) -> list[dict[str, Any]]:
    """
    Load data from a JSONL (JSON Lines) file.

    Args:
        file_path: Path to the JSONL file

    Returns:
        List of dictionaries, one per line
    """
    data = []
    with open(file_path) as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_json(data: dict[str, Any], file_path: str | Path) -> None:
    """
    Save data to a JSON file.

    Args:
        data: Data to save
        file_path: Path to save the file
    """
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def save_jsonl(data: list[dict[str, Any]], file_path: str | Path) -> None:
    """
    Save data to a JSONL file.

    Args:
        data: List of dictionaries to save
        file_path: Path to save the file
    """
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def load_csv(file_path: str | Path) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        file_path: Path to the CSV file

    Returns:
        DataFrame containing the data
    """
    return pd.read_csv(file_path)


def parse_numeric_output(text: str) -> float | None:
    """
    Parse numeric value from model output text.

    Handles formats like "[50]", "50", "The answer is 50", etc.

    Args:
        text: Text output from model

    Returns:
        Extracted numeric value, or None if not found
    """
    import re

    # Try to find number in brackets first (handles [50], [$50], [-50], [$-50])
    # Use findall + last match to avoid grabbing intermediate values from chain-of-thought
    matches = re.findall(r"\[\$?(-?\d+(?:\.\d+)?)\]", text)
    if matches:
        return float(matches[-1])

    # Try to find any number in the text (allow optional leading minus)
    match = re.search(r"(-?\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1))

    return None


def parse_numeric_list_output(text: str) -> list[float]:
    """
    Parse a list of numeric values from model output text.

    Handles formats like:
    - "[0.7]" -> [0.7]
    - "[0.7, 0.8]" -> [0.7, 0.8]
    - "[0.7,0.8,0.9]" -> [0.7, 0.8, 0.9]
    - "\\boxed{0.5}" -> [0.5] (LaTeX format from Llama models)

    Priority: \boxed{} first (explicit answer format), then brackets, then fallback.

    Args:
        text: Text output from model

    Returns:
        List of extracted numeric values (empty list if none found)
    """
    import re

    # Check for \boxed{} pattern FIRST (most explicit "final answer" format)
    boxed_matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed_matches:
        # Use the last boxed match
        content = boxed_matches[-1]
        numbers = re.findall(r"(-?\d+(?:\.\d+)?)", content)
        if numbers:
            return [float(n) for n in numbers]

    # Find ALL bracket matches and use the LAST one (answer is usually at the end)
    bracket_matches = re.findall(r"\[([^\]]+)\]", text)
    if bracket_matches:
        # Use the last bracket match
        content = bracket_matches[-1]
        # Find all numbers in the bracket content (allow optional leading minus)
        numbers = re.findall(r"(-?\d+(?:\.\d+)?)", content)
        if numbers:
            return [float(n) for n in numbers]
        return []

    # Fall back to finding any numbers in the text (allow optional leading minus)
    numbers = re.findall(r"(-?\d+\.\d+|-?\d+)", text)
    if numbers:
        return [float(n) for n in numbers]

    return []


def parse_choice_output(text: str, choices: list[str] | None = None) -> str | None:
    """
    Parse multiple choice answer from model output.

    Args:
        text: Text output from model
        choices: List of valid choices (e.g., ["A", "B", "C", "D"])

    Returns:
        Extracted choice, or None if not found
    """
    import re

    text = text.strip().upper()

    # If choices provided, look for exact matches
    if choices:
        choices_upper = [c.upper() for c in choices]
        for i, choice in enumerate(choices_upper):
            if choice in text:
                return choices[i]

    # Look for single letter choices (A, B, C, D)
    match = re.search(r"\b([A-E])\b", text)
    if match:
        return match.group(1)

    # Look for pattern like "Answer: A" or "The answer is B"
    match = re.search(r"(?:answer|choice)(?:\s+is)?:\s*([A-E])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return None


def parse_push_pull_output(text: str) -> int | None:
    """
    Parse Push/Pull action from model output text.

    Maps [Push] -> 0, [Pull] -> 1. Uses the LAST occurrence to handle
    verbose outputs that mention [Push]/[Pull] in payoff descriptions
    before stating the final answer.

    Checks bracketed format first ([PUSH]/[PULL]), then falls back to
    unbracketed keywords (PUSH/PULL).

    Args:
        text: Text output from model

    Returns:
        0 for Push, 1 for Pull, or None if not found
    """
    text_upper = text.strip().upper()

    # Find all bracketed mentions and use the last one
    push_positions = [m.start() for m in re.finditer(r"\[PUSH\]", text_upper)]
    pull_positions = [m.start() for m in re.finditer(r"\[PULL\]", text_upper)]

    bracketed = [(pos, 0) for pos in push_positions] + [(pos, 1) for pos in pull_positions]

    if bracketed:
        bracketed.sort(key=lambda x: x[0])
        return bracketed[-1][1]

    # Fall back to unbracketed PUSH / PULL (last occurrence)
    push_positions = [m.start() for m in re.finditer(r"PUSH", text_upper)]
    pull_positions = [m.start() for m in re.finditer(r"PULL", text_upper)]

    unbracketed = [(pos, 0) for pos in push_positions] + [(pos, 1) for pos in pull_positions]

    if unbracketed:
        unbracketed.sort(key=lambda x: x[0])
        return unbracketed[-1][1]

    return None


def load_demographics_data(
    file_path: str | Path,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Load Big Five demographics data.

    Expected format: CSV with columns for demographics (age, gender, etc.)
    and Big Five scores (openness, conscientiousness, extraversion,
    agreeableness, neuroticism).

    Args:
        file_path: Path to the demographics CSV file

    Returns:
        Tuple of (DataFrame, list of demographic column names)
    """
    df = pd.read_csv(file_path)

    # Common demographic columns
    demographic_cols = [
        col
        for col in df.columns
        if col.lower() in ["age", "gender", "education", "ethnicity", "country"]
    ]

    return df, demographic_cols


def load_game_data(file_path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    """
    Load MobLab game data.

    Expected format: JSONL with 'system', 'user', and 'assistant' fields.

    Args:
        file_path: Path to the game data file

    Returns:
        Tuple of (prompts, outputs) - prompt dicts with system/user messages and player responses
    """
    data = load_jsonl(file_path)
    prompts = [{"system": item["system"], "user": item["user"]} for item in data]
    outputs = [item["assistant"] for item in data]
    return prompts, outputs


def extract_banker_investment(text: str) -> float | None:
    """
    Extract investment amount from banker prompt text.

    Matches patterns like 'invested $100' or 'invested $15'.

    Args:
        text: Prompt text containing investment information

    Returns:
        Investment amount as float, or None if not found
    """
    import re

    match = re.search(r"invested \$(\d+)", text)
    if match:
        return float(match.group(1))
    return None


def load_workflow_data(
    file_path: str | Path,
) -> tuple[list[dict[str, str]], list[str]]:
    """
    Load AER research workflow data.

    Expected format: JSONL with 'input' (dict with context/method/outcome/impact)
    and 'output' (key_idea or title).

    Args:
        file_path: Path to the workflow data file

    Returns:
        Tuple of (list of workflow input dicts, list of output texts)
    """
    data = load_jsonl(file_path)

    inputs = []
    outputs = []

    for item in data:
        inputs.append(item["input"])
        outputs.append(item["output"])

    return inputs, outputs
