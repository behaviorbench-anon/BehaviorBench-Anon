"""
Utility functions and classes for model wrappers.
"""

import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Model pricing per 1M tokens (in dollars)
MODEL_PRICING = {
    # GPT-5 series
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-5-chat": {"input": 1.25, "output": 10.00},
    # GPT-4o series
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-11-20": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-08-06": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60},
    "gpt-4o-mini-batch": {"input": 0.075, "output": 0.30},
    # GPT-4.1 series
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-mini-batch": {"input": 0.20, "output": 0.80},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "o1": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 3.00, "output": 12.00},
    "o1-preview": {"input": 15.00, "output": 60.00},
    # GPT-5.x series (Azure deployments)
    "gpt-5.2": {"input": 1.25, "output": 10.00},
    "gpt-5.4": {"input": 2.50, "output": 15.00},
    "gpt-5.4-mini": {"input": 0.25, "output": 2.00},
    # GPT-4.1 series
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    # Claude (Azure deployment)
    "claude-opus-4-6": {"input": 5.00, "output": 25.00},
    "claude-sonnet-4-6": {"input": 1.00, "output": 5.00},
    "claude-haiku-4-5": {"input": 0.25, "output": 1.25},
    # DeepSeek (Azure deployments — multiple name variants)
    "DeepSeek-R1": {"input": 1.35, "output": 5.40},
    "deepseek-r1": {"input": 1.35, "output": 5.40},
    "DeepSeek-V3.2": {"input": 0.58, "output": 1.68},
    "deepseek-3.2": {"input": 0.58, "output": 1.68},
    # Google Gemini (Vertex AI)
    "gemini-3.1-pro-preview": {"input": 2.00, "output": 12.00},
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
    "gemini-3.1-flash-lite-preview": {"input": 0.25, "output": 1.50},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.0-flash": {"input": 0.15, "output": 0.60},
}

# Models that require temperature = 1
GPT5_MODELS = {
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5-chat",
    "gpt-5.2",
    "gpt-5.4",
    "gpt-5.4-mini",
}

# Per-model allowed reasoning effort values (from provider docs).
# OpenAI: reasoning_effort param in chat.completions.create()
# Anthropic: output_config={"effort": ...} in messages.create()
# Gemini 3.x: ThinkingConfig(thinking_level=...) in generate_content()
REASONING_EFFORT_SUPPORT: dict[str, set[str]] = {
    # OpenAI GPT-5 series
    "gpt-5": {"none", "low", "medium", "high", "xhigh"},
    "gpt-5-mini": {"none", "low", "medium", "high", "xhigh"},
    "gpt-5-nano": {"none", "low", "medium", "high", "xhigh"},
    "gpt-5-chat": {"none", "low", "medium", "high", "xhigh"},
    "gpt-5.2": {"none", "low", "medium", "high", "xhigh"},
    "gpt-5.4": {"none", "low", "medium", "high", "xhigh"},
    "gpt-5.4-mini": {"none", "low", "medium", "high", "xhigh"},
    # Anthropic Claude
    "claude-opus-4-6": {"low", "medium", "high", "max"},
    "claude-sonnet-4-6": {"low", "medium", "high", "max"},
    # Gemini 3.x (Vertex AI)
    "gemini-3.1-pro-preview": {"low", "medium", "high"},
    "gemini-3-flash-preview": {"minimal", "low", "medium", "high"},
    "gemini-3.1-flash-lite-preview": {"minimal", "low", "medium", "high"},
}

ALL_REASONING_EFFORT_VALUES = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}


def validate_reasoning_effort(model_name: str, reasoning_effort: str | None) -> None:
    """Validate reasoning_effort is compatible with the model.

    Raises ValueError if the model doesn't support reasoning effort or the
    value is not in the model's allowed set.
    """
    if reasoning_effort is None:
        return
    if reasoning_effort not in ALL_REASONING_EFFORT_VALUES:
        raise ValueError(
            f"Invalid reasoning_effort='{reasoning_effort}'. "
            f"Must be one of: {sorted(ALL_REASONING_EFFORT_VALUES)}"
        )
    if model_name not in REASONING_EFFORT_SUPPORT:
        raise ValueError(
            f"Model '{model_name}' does not support --reasoning-effort. "
            f"Supported models: {sorted(REASONING_EFFORT_SUPPORT.keys())}"
        )
    allowed = REASONING_EFFORT_SUPPORT[model_name]
    if reasoning_effort not in allowed:
        raise ValueError(
            f"Model '{model_name}' supports reasoning_effort in {sorted(allowed)}, "
            f"got '{reasoning_effort}'"
        )


class TokenTracker:
    """Thread-safe token usage tracker."""

    by_model: dict[str, dict[str, Any]] = {}
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_reasoning_tokens: int = 0
    total_cost: float = 0.0
    total_api_calls: int = 0
    session_start: str | None = None
    _lock = threading.Lock()

    @classmethod
    def reset(cls):
        """Reset the token usage tracker for a new session."""
        with cls._lock:
            cls.by_model = {}
            cls.total_prompt_tokens = 0
            cls.total_completion_tokens = 0
            cls.total_reasoning_tokens = 0
            cls.total_cost = 0.0
            cls.total_api_calls = 0
            cls.session_start = datetime.now().isoformat()

    @classmethod
    def track(
        cls,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
        model: str,
        reasoning_tokens: int = 0,
    ):
        """Track token usage for a single API call (thread-safe)."""
        with cls._lock:
            cls.total_prompt_tokens += prompt_tokens
            cls.total_completion_tokens += completion_tokens
            cls.total_reasoning_tokens += reasoning_tokens
            cls.total_cost += cost
            cls.total_api_calls += 1

            if model not in cls.by_model:
                cls.by_model[model] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "reasoning_tokens": 0,
                    "cost": 0.0,
                    "calls": 0,
                }
            cls.by_model[model]["prompt_tokens"] += prompt_tokens
            cls.by_model[model]["completion_tokens"] += completion_tokens
            cls.by_model[model]["reasoning_tokens"] += reasoning_tokens
            cls.by_model[model]["cost"] += cost
            cls.by_model[model]["calls"] += 1

    @classmethod
    def get_summary(cls) -> dict[str, Any]:
        """Get a formatted summary of token usage and costs."""
        with cls._lock:
            return {
                "session_start": cls.session_start,
                "total_api_calls": cls.total_api_calls,
                "total_prompt_tokens": cls.total_prompt_tokens,
                "total_completion_tokens": cls.total_completion_tokens,
                "total_reasoning_tokens": cls.total_reasoning_tokens,
                "total_cost_usd": cls.total_cost,
                "by_model": cls.by_model.copy(),
            }


def calculate_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    """Calculate cost in dollars based on token usage and model pricing."""
    import logging

    logger = logging.getLogger(__name__)

    if model not in MODEL_PRICING:
        logger.warning(f"Model '{model}' not in pricing table, returning 0 cost")
        return 0.0

    pricing = MODEL_PRICING[model]
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


# Cache for model inference config
_MODEL_INFERENCE_CONFIG: dict | None = None


def load_model_inference_config() -> dict:
    """Load model inference config from YAML file."""
    global _MODEL_INFERENCE_CONFIG
    if _MODEL_INFERENCE_CONFIG is not None:
        return _MODEL_INFERENCE_CONFIG

    # Find config file relative to project root
    config_paths = [
        Path(__file__).parent.parent.parent.parent / "config" / "model_inference.yaml",
        Path("config/model_inference.yaml"),
        Path(os.environ.get("BEHAVIORBENCH_CONFIG_DIR", "")) / "model_inference.yaml",
    ]

    for config_path in config_paths:
        if config_path.exists():
            with open(config_path) as f:
                _MODEL_INFERENCE_CONFIG = yaml.safe_load(f)
            return _MODEL_INFERENCE_CONFIG

    # Return default config if file not found
    _MODEL_INFERENCE_CONFIG = {
        "default": {
            "temperature": 0.7,
            "top_p": None,
            "top_k": None,
            "min_p": None,
            "max_tokens": 64,
        }
    }
    return _MODEL_INFERENCE_CONFIG


def get_model_family(model_name: str) -> str:
    """Determine model family from model name."""
    model_lower = model_name.lower()

    if "qwen3" in model_lower or "qwen-3" in model_lower:
        return "qwen3"
    elif "qwen2.5" in model_lower or "qwen-2.5" in model_lower or "qwen25" in model_lower:
        return "qwen2.5"
    elif "llama" in model_lower:
        return "llama3"
    else:
        return "default"


def get_inference_params(model_name: str) -> dict:
    """Get inference parameters for a model based on its family.

    Args:
        model_name: Name of the model

    Returns:
        Dict with inference parameters (temperature, top_p, top_k, min_p, max_tokens)
    """
    config = load_model_inference_config()
    family = get_model_family(model_name)

    # Get family config, fall back to default
    family_config = config.get(family, config.get("default", {}))
    default_config = config.get("default", {})

    # Merge with defaults
    params = {
        "temperature": family_config.get("temperature", default_config.get("temperature", 0.7)),
        "top_p": family_config.get("top_p", default_config.get("top_p")),
        "top_k": family_config.get("top_k", default_config.get("top_k")),
        "min_p": family_config.get("min_p", default_config.get("min_p")),
        "max_tokens": family_config.get("max_tokens", default_config.get("max_tokens", 64)),
    }

    return params


# Two valid sampling presets. Temperature determines the full preset.
# Base presets are per-family (includes temperature); T=1.0 is shared across all families.
SAMPLING_PRESETS_BASE: dict[str, dict[str, float | int | None]] = {
    "qwen3": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    "qwen2.5": {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
    "llama3": {"temperature": 0.6, "top_p": 0.9, "top_k": None},
}
SAMPLING_PRESET_T10: dict[str, float | int] = {"top_p": 1.0, "top_k": -1}

# Families that require preset enforcement
STRICT_INFERENCE_FAMILIES = {"qwen3", "qwen2.5", "llama3"}


def validate_inference_params(
    model_name: str,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    strict: bool = True,
) -> None:
    """Validate that inference parameters match one of the two allowed sampling presets.

    Presets (temperature determines the full preset):
      - Base: per-family temperature + top_p/top_k (e.g., qwen3 T=0.6, llama3 T=0.6)
      - T=1.0: top_p=1.0, top_k=-1 (shared across all families)

    Applies to qwen3, qwen2.5, and llama3 families (both base and trained models).

    Args:
        model_name: Name of the model
        temperature: Sampling temperature (must match family base or 1.0)
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        strict: If True, raise error on mismatch; if False, just warn
    """
    import logging

    logger = logging.getLogger(__name__)
    family = get_model_family(model_name)

    if family not in STRICT_INFERENCE_FAMILIES:
        return

    # Determine expected preset from temperature
    base_preset = SAMPLING_PRESETS_BASE.get(family)
    base_temp = base_preset["temperature"] if base_preset else None

    if base_preset is not None and temperature == base_temp:
        expected = base_preset
        preset_label = f"T={base_temp} ({family})"
    elif temperature == 1.0:
        expected = SAMPLING_PRESET_T10
        preset_label = "T=1.0"
    else:
        allowed = f"T={base_temp}" if base_temp is not None else "family base"
        msg = (
            f"Non-standard temperature={temperature} for {family} model '{model_name}'. "
            f"Only {allowed} and T=1.0 presets are supported."
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)
        return

    # Compare actual params against expected preset
    deviations = []

    expected_top_p = expected["top_p"]
    if top_p != expected_top_p:
        deviations.append(f"top_p={top_p}, expected {expected_top_p}")

    expected_top_k = expected["top_k"]
    if top_k != expected_top_k:
        deviations.append(f"top_k={top_k}, expected {expected_top_k}")

    if deviations:
        msg = (
            f"Sampling params don't match {preset_label} preset for '{model_name}':\n"
            f"  {'; '.join(deviations)}\n"
            f"  Expected: temperature={temperature}, top_p={expected_top_p}, top_k={expected_top_k}"
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)
