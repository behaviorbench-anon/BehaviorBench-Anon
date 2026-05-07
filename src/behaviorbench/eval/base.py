"""
Base classes and interfaces for evaluation tasks.
"""

import json
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from behaviorbench.eval.metrics import compute_metrics
from behaviorbench.eval.utils import (
    extract_banker_investment,
    load_jsonl,
    parse_choice_output,
    parse_numeric_list_output,
    parse_numeric_output,
    parse_push_pull_output,
)


@dataclass
class EvaluationResult:
    """
    Container for evaluation results.

    Attributes:
        task_name: Name of the evaluation task
        metrics: Dictionary of metric names to values
        predictions: Optional list of detailed results (prompt, raw_output, parsed, expected)
        references: Optional list of reference/ground truth values
        metadata: Additional metadata about the evaluation
    """

    task_name: str
    metrics: dict[str, float]
    predictions: list[Any] | None = None
    references: list[Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a formatted summary of the evaluation results."""
        lines = [f"Task: {self.task_name}", "Metrics:"]
        for metric_name, value in self.metrics.items():
            if metric_name.startswith("_"):
                # Skip internal keys like _per_group
                continue
            if isinstance(value, float):
                lines.append(f"  {metric_name}: {value:.4f}")
            elif isinstance(value, dict):
                lines.append(f"  {metric_name}: {value}")
            else:
                lines.append(f"  {metric_name}: {value}")
        if self.metadata:
            lines.append("Metadata:")
            for key, value in self.metadata.items():
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def to_dict(self, include_full_results: bool = False) -> dict[str, Any]:
        """Convert result to dictionary format."""
        result = {
            "task_name": self.task_name,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }
        if include_full_results and self.predictions is not None:
            result["predictions"] = self.predictions
            result["references"] = self.references
        return result


# =============================================================================
# Built-in Output Parsers
# =============================================================================


def parse_numeric(raw_output: Any, clip_range: tuple[float, float] | None = None) -> float | None:
    """Parse single numeric value, optionally clip to range."""
    if isinstance(raw_output, (int, float)):
        value = float(raw_output)
    else:
        value = parse_numeric_output(str(raw_output))

    if value is None:
        return None

    if clip_range:
        value = float(max(clip_range[0], min(clip_range[1], value)))

    return value


def parse_integer(raw_output: Any, clip_range: tuple[int, int] | None = None) -> int | None:
    """Parse single integer value, optionally clip to range."""
    if isinstance(raw_output, int):
        value = raw_output
    elif isinstance(raw_output, float):
        value = int(raw_output)
    else:
        parsed = parse_numeric_output(str(raw_output))
        if parsed is None:
            return None
        value = int(parsed)

    if clip_range:
        value = max(clip_range[0], min(clip_range[1], value))

    return value


def parse_numeric_list(
    raw_output: Any, clip_range: tuple[float, float] | None = None
) -> list[float] | None:
    """Parse list of numeric values, optionally clip each to range."""
    if isinstance(raw_output, list):
        values = [float(x) for x in raw_output]
    else:
        values = parse_numeric_list_output(str(raw_output))

    if not values:
        return None

    if clip_range:
        values = [max(clip_range[0], min(clip_range[1], v)) for v in values]

    return values


def parse_choice(raw_output: Any) -> str | None:
    """Parse single letter choice (A-E)."""
    text = str(raw_output).strip()
    # Handle bracketed format "[A]"
    if text.startswith("[") and text.endswith("]"):
        return text[1:-1].upper()
    # Use parser for model output
    choice = parse_choice_output(text)
    return choice if choice else None


def parse_multi_choice(raw_output: Any) -> str | None:
    """Parse one or more letter choices (A-D) from model output.

    Handles formats like: "B and C", "B, C", "[BC]", "BC", "B C",
    "The answers are B and C", or single answers like "A".
    Returns sorted deduplicated string like "BC" or "A".

    For verbose outputs, checks first line then last line for a clear answer
    to avoid extracting letters from explanation text that references all options.
    """
    import re

    text = str(raw_output).strip().upper()

    # Remove common wrapper patterns
    text = re.sub(r"^\[|\]$", "", text)
    text = re.sub(r"^THE ANSWERS? (?:IS|ARE)\s*:?\s*", "", text)
    text = re.sub(r"^(?:ANSWER|CHOICE)S?\s*:?\s*", "", text)
    text = text.strip()

    # If the remaining text is purely A-D letters (e.g., "BC", "A", "ACD"),
    # treat each letter as a choice
    if re.fullmatch(r"[A-D]+", text):
        return "".join(sorted(set(text)))

    # --- Helper: try to extract a clean answer from a single line ---
    def _extract_from_line(line: str) -> str | None:
        line = line.strip()
        # Strip markdown bold/header markers (e.g., "**A**", "## Answer: **C**")
        clean = re.sub(r"[#*_`]+", "", line).strip()
        # Remove "the answer is", "answer:", etc. prefixes
        clean = re.sub(r"^(?:THE\s+)?ANSWERS?\s*(?:IS|ARE)?\s*:?\s*", "", clean).strip()
        clean = re.sub(r"^(?:ANSWER|CHOICE)S?\s*:?\s*", "", clean).strip()
        # Remove trailing punctuation/emoji
        clean = re.sub(r"[.\s!✓✅]+$", "", clean).strip()
        if re.fullmatch(r"[A-D]+", clean):
            return "".join(sorted(set(clean)))
        # Try extracting standalone A-D letters from this single line
        letters = set(re.findall(r"(?<![A-Z])[A-D](?![A-Z])", clean))
        if letters and len(letters) <= 2:  # reasonable number of choices from one line
            return "".join(sorted(letters))
        return None

    lines = text.split("\n")

    # Try the first line (models often state answer first, then explain)
    result = _extract_from_line(lines[0])
    if result:
        return result

    # Try the last non-empty line (models often conclude with "The answer is **D**")
    for line in reversed(lines):
        if line.strip():
            result = _extract_from_line(line)
            if result:
                return result
            break

    # Fallback: extract standalone A-D letters from full text
    # This avoids matching letters inside words like "AND"
    letters = set(re.findall(r"(?<![A-Z])[A-D](?![A-Z])", text))

    if not letters:
        return None

    return "".join(sorted(letters))


def parse_text(raw_output: Any) -> str:
    """Return text as-is (no parsing)."""
    return str(raw_output)


def parse_push_pull(raw_output: Any, clip_range: tuple[int, int] | None = None) -> int | None:
    """Parse Push/Pull action: Push -> 0, Pull -> 1, optionally clip to range."""
    if isinstance(raw_output, int):
        value = raw_output
    elif isinstance(raw_output, float):
        value = int(raw_output)
    else:
        value = parse_push_pull_output(str(raw_output))

    if value is None:
        return None

    if clip_range:
        value = max(clip_range[0], min(clip_range[1], value))

    return value


# Registry of output type parsers
OUTPUT_PARSERS = {
    "numeric": parse_numeric,
    "integer": parse_integer,
    "numeric_list": parse_numeric_list,
    "choice": parse_choice,
    "multi_choice": parse_multi_choice,
    "text": parse_text,
    "push_pull": parse_push_pull,
}


# =============================================================================
# Base Evaluation Task
# =============================================================================


class BaseEvaluationTask(ABC):
    """
    Abstract base class for evaluation tasks.

    All evaluation tasks should inherit from this class and implement
    the required methods.

    Standard features provided by base class:
        - Resume capability: Skip already-evaluated samples from a previous run
        - Sampling: Limit evaluation to a random subset of samples
        - Detailed results: Store full per-sample results (prompt, raw_output, parsed, expected)

    Subclasses must define:
        - metrics: Class attribute listing metric names (e.g., ["MAE", "accuracy"])
        - _parse_output(): Method to parse ground truth and model outputs
    """

    # Subclasses must override these class attributes
    metrics: list[str] = []
    name: str = ""  # Default task name (subclasses should override)

    def __init__(
        self,
        data_path: str | None = None,
        name: str | None = None,
        num_samples: int | None = None,
        seed: int = 42,
        resume_file: str | None = None,
        **kwargs,
    ):
        """
        Initialize the evaluation task.

        Args:
            data_path: Path to the evaluation data
            name: Name of the task (defaults to class attribute)
            num_samples: Optional limit on number of samples (None = use all)
            seed: Random seed for reproducible sampling
            resume_file: Path to existing result file to resume from
            **kwargs: Additional task-specific configuration
        """
        # Use provided name or fall back to class attribute
        self.name = name if name is not None else self.__class__.name
        self.data_path = data_path
        self.num_samples = num_samples
        self.seed = seed
        self.resume_file = resume_file
        self.config = kwargs

        # Data storage (populated by load_data)
        self.prompts: list[dict[str, str]] = []
        self.ground_truth: list[Any] = []

        # For resume capability
        # Maps prompt_key -> single prediction (for standard tasks)
        self.existing_results: dict[str, dict] = {}
        # Maps prompt_key -> list of predictions (for game_behavior with multiple samples per prompt)
        self.existing_results_all: dict[str, list[dict]] = {}
        self._load_existing_results()

    def _prompt_key(self, prompt: dict[str, str]) -> str:
        """Create a unique key from a prompt for matching during resume."""
        return f"{prompt.get('system', '')}|||{prompt.get('user', '')}"

    def _load_existing_results(self) -> None:
        """Load existing results from resume file if provided."""
        if not self.resume_file:
            return

        try:
            with open(self.resume_file) as f:
                data = json.load(f)

            # Extract predictions from the tasks
            total_loaded = 0
            for task in data.get("tasks", []):
                if task.get("task_name") == self.name:
                    predictions = task.get("predictions", [])
                    for pred in predictions:
                        prompt = pred.get("prompt", {})
                        key = self._prompt_key(prompt)
                        self.existing_results[key] = pred
                        # Also store in list form for game_behavior (multiple samples per prompt)
                        if key not in self.existing_results_all:
                            self.existing_results_all[key] = []
                        self.existing_results_all[key].append(pred)
                        total_loaded += 1

            print(
                f"Loaded {total_loaded} existing predictions ({len(self.existing_results)} unique prompts) from resume file"
            )
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load resume file: {e}")

    def _sample_data(self, data: list[Any]) -> list[Any]:
        """
        Sample a subset of data if num_samples is specified.

        Args:
            data: Full list of data items

        Returns:
            Sampled list (or original if num_samples is None or >= len(data))
        """
        if self.num_samples is not None and self.num_samples < len(data):
            random.seed(self.seed)
            return random.sample(data, self.num_samples)
        return data

    def _get_cached_result(self, prompt: dict[str, str]) -> dict | None:
        """
        Pop and return a cached result for this prompt.

        Each cached result is consumed once — if the same prompt key appears
        multiple times in the evaluation data, only as many are skipped as
        were present in the resume file.

        Args:
            prompt: The prompt dict to look up

        Returns:
            Cached result dict if found, None otherwise
        """
        key = self._prompt_key(prompt)
        results = self.existing_results_all.get(key)
        if results:
            return results.pop(0)
        return None

    def _get_all_cached_results(self, prompt: dict[str, str]) -> list[dict]:
        """
        Get all cached results for this prompt (for game_behavior with multiple samples).

        Args:
            prompt: The prompt dict to look up

        Returns:
            List of cached result dicts (empty if none found)
        """
        key = self._prompt_key(prompt)
        return self.existing_results_all.get(key, [])

    def _create_detailed_result(
        self,
        prompt: dict[str, str],
        raw_output: str,
        parsed_prediction: Any,
        expected: Any,
    ) -> dict[str, Any]:
        """
        Create a standardized detailed result dictionary.

        Args:
            prompt: Input prompt (system/user)
            raw_output: Raw model output string
            parsed_prediction: Parsed prediction value(s)
            expected: Ground truth value(s)

        Returns:
            Dictionary with all result details
        """
        return {
            "prompt": prompt,
            "raw_output": raw_output,
            "parsed_prediction": parsed_prediction,
            "expected": expected,
        }

    def load_data(self) -> None:
        """
        Load and prepare evaluation data from JSONL file.

        Loads data from self.data_path, applies sampling, and populates
        self.prompts and self.ground_truth using _parse_output().

        Expected JSONL format:
            - 'system': System prompt
            - 'user': User prompt
            - 'assistant': Ground truth value (parsed by _parse_output)
        """
        data = load_jsonl(self.data_path)

        # Apply sampling
        data = self._sample_data(data)

        self.prompts = []
        self.ground_truth = []

        for idx, item in enumerate(data):
            prompt = {"system": item["system"], "user": item["user"]}
            if "metadata" in item:
                prompt["metadata"] = item["metadata"]
            # Forward journal field for group_by support
            if "journal" in item:
                prompt.setdefault("metadata", {})["journal"] = item["journal"]
            self.prompts.append(prompt)
            value = self._parse_ground_truth(item["assistant"], sample_idx=idx)
            if value is not None:
                self.ground_truth.append(value)

    @abstractmethod
    def _parse_output(self, raw_output: Any) -> Any:
        """
        Parse output value from raw text (model predictions).

        This method is used for parsing model predictions and applies
        clipping for values outside the valid range.

        Args:
            raw_output: Raw output text from model

        Returns:
            Parsed value, or None if parsing fails
        """
        pass

    @abstractmethod
    def _parse_ground_truth(self, raw_output: Any, sample_idx: int = -1) -> Any:
        """
        Parse and validate ground truth value.

        Unlike _parse_output (for predictions), this should NOT clip values.
        Instead, it should raise an error if values are outside the valid range,
        indicating a data bug that needs to be fixed at the source.

        Args:
            raw_output: Raw output text from JSONL 'assistant' field
            sample_idx: Index of the sample for error reporting

        Returns:
            Parsed value, or None if parsing fails

        Raises:
            ValueError: If ground truth value is outside valid clip_range
        """
        pass

    def _validate_resume_subset(self) -> None:
        """Validate that all resume prompts are a subset of the current prompts.

        Raises ValueError if any resume prompt does not match a current prompt.
        Must be called after load_data() so self.prompts is populated.
        """
        if not self.existing_results:
            return

        current_keys = {self._prompt_key(p) for p in self.prompts}
        unmatched = []
        for key in self.existing_results:
            if key not in current_keys:
                unmatched.append(key[:80])  # Truncate for readability

        if unmatched:
            raise ValueError(
                f"Resume file is not a subset of current data: "
                f"{len(unmatched)}/{len(self.existing_results)} resume prompts "
                f"do not match any current prompt. First mismatch: {unmatched[0]!r}"
            )

        # Count how many prompts can be skipped (without consuming the cache).
        # Use existing_results_all counts so duplicate keys are handled correctly:
        # if a resume file has 1 entry for key X and current data has 3, only 1 is skippable.
        from collections import Counter

        current_key_counts = Counter(self._prompt_key(p) for p in self.prompts)
        total_cached = sum(
            min(len(cached_list), current_key_counts[key])
            for key, cached_list in self.existing_results_all.items()
        )
        to_eval = len(self.prompts) - total_cached
        print(f"Resume: {total_cached} cached, {to_eval} to evaluate")

    def run_evaluation(self, model: Any, **kwargs) -> EvaluationResult:
        """
        Run evaluation on the given model.

        This method:
        1. Loads data if not already loaded
        2. Gets model outputs (supports batch_call and resume)
        3. Parses outputs using _parse_output()
        4. Computes metrics using metrics class attribute

        Args:
            model: The model to evaluate (callable or object with batch_call)
            **kwargs: Additional model arguments

        Returns:
            EvaluationResult with metrics and detailed per-sample results
        """
        if not self.prompts:
            self.load_data()

        # Validate resume data is a subset of current data
        self._validate_resume_subset()

        # Get model outputs
        if hasattr(model, "batch_call"):
            if self.existing_results:
                raise NotImplementedError(
                    "Resume with batch_call is not supported yet. "
                    "Remove --resume or use a model without batch_call."
                )
            raw_outputs = model.batch_call(self.prompts, **kwargs)
        else:
            raw_outputs = []
            total = len(self.prompts)
            for i, prompt in enumerate(self.prompts):
                # Check for cached result from resume file
                cached = self._get_cached_result(prompt)
                if cached:
                    raw_outputs.append(cached.get("raw_output", ""))
                else:
                    output = model(prompt, **kwargs)
                    raw_outputs.append(output)
                print(f"Completed {i + 1}/{total} API calls", flush=True)

        # Process outputs and collect results
        detailed_results: list[dict] = []
        valid_predictions: list[Any] = []
        valid_references: list[Any] = []

        for prompt, raw_output, reference in zip(
            self.prompts, raw_outputs, self.ground_truth, strict=False
        ):
            # Parse prediction using same function as ground truth
            pred = self._parse_output(raw_output)

            # Store detailed result
            detailed_results.append(
                self._create_detailed_result(
                    prompt=prompt,
                    raw_output=str(raw_output),
                    parsed_prediction=pred,
                    expected=reference,
                )
            )

            # Track valid predictions for metrics
            if pred is not None and reference is not None:
                valid_predictions.append(pred)
                valid_references.append(reference)

        if len(valid_predictions) == 0:
            raise ValueError("No valid predictions generated")

        # Compute metrics
        computed_metrics = compute_metrics(
            metric_names=self.metrics,
            predictions=valid_predictions,
            references=valid_references,
        )

        metadata = {
            "num_samples": len(valid_predictions),
            "num_failed_parses": len(detailed_results) - len(valid_predictions),
        }

        return EvaluationResult(
            task_name=self.name,
            metrics=computed_metrics,
            predictions=detailed_results,
            references=None,  # References are included in detailed_results
            metadata=metadata,
        )

    def validate_predictions(self, predictions: list[Any], references: list[Any]) -> bool:
        """
        Validate that predictions match expected format.

        Args:
            predictions: Model predictions
            references: Reference/ground truth values

        Returns:
            True if predictions are valid, False otherwise
        """
        if len(predictions) != len(references):
            return False
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


# =============================================================================
# Configurable Task (loads from YAML)
# =============================================================================


class ConfigurableTask(BaseEvaluationTask):
    """
    A task that can be configured via YAML instead of requiring a separate class.

    This allows defining tasks purely through configuration:
        - name: Task identifier
        - metrics: List of metric names
        - output_type: One of "numeric", "integer", "numeric_list", "choice", "text"
        - clip_range: Optional [min, max] for clipping parsed values
        - group_by: Optional grouping strategy for computing metrics per-group
    """

    def __init__(
        self,
        data_path: str | None = None,
        name: str = "",
        metrics: list[str] | None = None,
        output_type: str = "text",
        clip_range: list[float] | None = None,
        group_by: str | None = None,
        group_averaging: str = "macro",
        normalize_before_wasserstein: bool = False,
        normalize_multiplier: float = 1.0,
        normalize_range: list[float] | tuple[float, float] | None = None,
        element_names: list[str] | None = None,
        expected_list_length: int | None = None,
        **kwargs,
    ):
        """
        Initialize a configurable task.

        Args:
            data_path: Path to the evaluation data
            name: Task name
            metrics: List of metric names to compute
            output_type: Type of output parsing ("numeric", "integer", "numeric_list", "choice", "text")
            clip_range: Optional [min, max] for clipping values
            group_by: Optional grouping strategy ("bigfive_dimension" to group by OCEAN dimensions)
            normalize_before_wasserstein: If True, perform per-sample normalization before computing
                Wasserstein distance. Used for multiround banker where each sample has different
                investment amount, so different max return range.
            normalize_multiplier: Multiplier to compute per-sample max value from extracted amount.
                For banker game: max_return = investment * 3 (normalize_multiplier=3).
            normalize_range: Optional [min, max] output range to normalize Wasserstein distance to.
                If provided along with clip_range, Wasserstein distance will be scaled from
                clip_range to normalize_range (e.g., [0, 20] -> [0, 100] for public goods).
            element_names: Optional list of names for each element in numeric_list output.
                Passed to MAE_per_output metric for named per-element breakdown.
            expected_list_length: Optional expected length for numeric_list output.
                If set, parsed lists with wrong length are treated as failed parses (predictions)
                or raise ValueError (ground truth).
            **kwargs: Additional configuration
        """
        # Set instance attributes before calling super().__init__
        self._metrics = metrics or []
        self._output_type = output_type
        self._clip_range = tuple(clip_range) if clip_range else None
        self._group_by = group_by
        self._group_averaging = group_averaging
        self._normalize_before_wasserstein = normalize_before_wasserstein
        self._normalize_multiplier = normalize_multiplier
        self._normalize_range = tuple(normalize_range) if normalize_range else None
        self._element_names = element_names
        self._expected_list_length = expected_list_length
        self._expected_groups = kwargs.pop("expected_groups", None)

        # Validate output type
        if output_type not in OUTPUT_PARSERS:
            raise ValueError(
                f"Unknown output_type: {output_type}. "
                f"Supported types: {list(OUTPUT_PARSERS.keys())}"
            )

        super().__init__(data_path=data_path, name=name, **kwargs)

    @property
    def metrics(self) -> list[str]:  # type: ignore[override]
        """Return metrics list."""
        return self._metrics

    def _parse_output(self, raw_output: Any) -> Any:
        """Parse output using the configured output type and clip range."""
        parser = OUTPUT_PARSERS[self._output_type]

        # For parsers that support clip_range
        if self._output_type in ("numeric", "integer", "numeric_list", "push_pull"):
            result = parser(raw_output, clip_range=self._clip_range)
        else:
            result = parser(raw_output)

        # Validate list length if expected_list_length is set
        if (
            result is not None
            and self._expected_list_length is not None
            and isinstance(result, list)
            and len(result) != self._expected_list_length
        ):
            return None

        return result

    def _parse_ground_truth(self, raw_output: Any, sample_idx: int = -1) -> Any:
        """Parse ground truth and validate it's within clip_range (no clipping)."""
        parser = OUTPUT_PARSERS[self._output_type]

        # Parse without clipping first
        if self._output_type == "numeric":
            value = parse_numeric(raw_output, clip_range=None)
            if value is not None and self._clip_range:
                if value < self._clip_range[0] or value > self._clip_range[1]:
                    raise ValueError(
                        f"Ground truth value {value} at sample {sample_idx} is outside "
                        f"valid clip_range {self._clip_range} for task '{self.name}'. "
                        f"This indicates a data bug - fix the source data."
                    )
            return value
        elif self._output_type in ("integer", "push_pull"):
            parser_fn = parse_push_pull if self._output_type == "push_pull" else parse_integer
            value = parser_fn(raw_output, clip_range=None)
            if value is not None and self._clip_range:
                if value < self._clip_range[0] or value > self._clip_range[1]:
                    raise ValueError(
                        f"Ground truth value {value} at sample {sample_idx} is outside "
                        f"valid clip_range {self._clip_range} for task '{self.name}'. "
                        f"This indicates a data bug - fix the source data."
                    )
            return value
        elif self._output_type == "numeric_list":
            values = parse_numeric_list(raw_output, clip_range=None)
            if values is not None and self._clip_range:
                for i, v in enumerate(values):
                    if v < self._clip_range[0] or v > self._clip_range[1]:
                        raise ValueError(
                            f"Ground truth value {v} (index {i}) at sample {sample_idx} "
                            f"is outside valid clip_range {self._clip_range} for task "
                            f"'{self.name}'. This indicates a data bug - fix the source data."
                        )
            if values is not None and self._expected_list_length is not None:
                if len(values) != self._expected_list_length:
                    raise ValueError(
                        f"Ground truth list length {len(values)} at sample {sample_idx} "
                        f"does not match expected_list_length {self._expected_list_length} "
                        f"for task '{self.name}'. This indicates a data bug - fix the source data."
                    )
            return values
        else:
            # choice, text - no clip_range validation needed
            return parser(raw_output)

    def _extract_group(self, prompt: dict[str, str]) -> str | None:
        """
        Extract group label from prompt based on group_by strategy.

        Args:
            prompt: The prompt dict with 'system' and 'user' keys

        Returns:
            Group label string, or None if no grouping
        """
        if self._group_by is None:
            return None

        if self._group_by == "bigfive_dimension":
            # Extract BigFive TARGET dimension from the user prompt
            # Target dimension is marked with asterisks: *DimensionName*
            user_text = prompt.get("user", "")
            dimensions = [
                "Extraversion",
                "Neuroticism",
                "Agreeableness",
                "Conscientiousness",
                "Openness",
            ]
            for dim in dimensions:
                # Look for the target dimension pattern (with asterisks)
                if f"*{dim}*" in user_text:
                    return dim
                # Also try parenthetical pattern: (DimensionName Dimension)
                if f"({dim}" in user_text:
                    return dim
            return "Unknown"

        if self._group_by == "target_item":
            # 1. Try metadata field first (seq_surv_resp, missing_surv_resp_* have it)
            metadata = prompt.get("metadata", {})
            if metadata and "target_item" in metadata:
                return metadata["target_item"]

            # 2. Fallback: extract question text from prompt (surv_resp_pred has no metadata)
            #    Pattern: text between "or [5]):" and "Only output"
            user_text = prompt.get("user", "")
            match = re.search(r"or \[5\]\):\s*(.+?)\s*Only output", user_text)
            if match:
                return match.group(1).strip()
            return "Unknown"

        if self._group_by == "journal":
            metadata = prompt.get("metadata", {})
            return metadata.get("journal", "Unknown")

        return None

    def run_evaluation(self, model: Any, **kwargs) -> EvaluationResult:
        """
        Run evaluation on the given model.

        This method:
        1. Loads data if not already loaded
        2. Gets model outputs (supports batch_call and resume)
        3. Parses outputs using _parse_output()
        4. Computes metrics using metrics class attribute
        5. If group_by is set, computes metrics per-group and averages
        6. If normalize_before_wasserstein is set, normalizes predictions/references
           per-sample before computing metrics

        Args:
            model: The model to evaluate (callable or object with batch_call)
            **kwargs: Additional model arguments

        Returns:
            EvaluationResult with metrics and detailed per-sample results
        """
        if not self.prompts:
            self.load_data()

        # Get model outputs
        if hasattr(model, "batch_call"):
            raw_outputs = model.batch_call(self.prompts, **kwargs)
        else:
            raw_outputs = []
            total = len(self.prompts)
            for i, prompt in enumerate(self.prompts):
                # Check for cached result from resume file
                cached = self._get_cached_result(prompt)
                if cached:
                    raw_outputs.append(cached.get("raw_output", ""))
                else:
                    output = model(prompt, **kwargs)
                    raw_outputs.append(output)
                print(f"Completed {i + 1}/{total} API calls", flush=True)

        # Process outputs and collect results
        detailed_results: list[dict] = []
        valid_predictions: list[Any] = []
        valid_references: list[Any] = []
        valid_groups: list[str] = []
        valid_prompts: list[dict] = []  # Track prompts for per-sample normalization
        # Track all predictions including failed parses (sentinel -1 for failures)
        all_predictions: list[Any] = []
        all_references: list[Any] = []
        all_groups: list[str] = []

        total_samples = 0
        num_failed_parses = 0

        for prompt, raw_output, reference in zip(
            self.prompts, raw_outputs, self.ground_truth, strict=False
        ):
            # Parse prediction using same function as ground truth
            pred = self._parse_output(raw_output)

            # Extract group if grouping is enabled
            group = self._extract_group(prompt)

            total_samples += 1

            # Store detailed result
            detailed_results.append(
                self._create_detailed_result(
                    prompt=prompt,
                    raw_output=str(raw_output),
                    parsed_prediction=pred,
                    expected=reference,
                )
            )

            if reference is None:
                continue

            if pred is not None:
                valid_predictions.append(pred)
                valid_references.append(reference)
                valid_prompts.append(prompt)
                all_predictions.append(pred)
                all_references.append(reference)
                if group is not None:
                    valid_groups.append(group)
                    all_groups.append(group)
            else:
                num_failed_parses += 1
                # Map invalid → a value guaranteed to be wrong.
                # For numeric: opposite of ground truth (avoids sentinel -1
                # which causes sklearn to detect a spurious third class).
                # For strings: a sentinel that won't match any valid answer.
                if isinstance(reference, (int, float)):
                    all_predictions.append(1 - reference)
                else:
                    all_predictions.append("__INVALID__")
                all_references.append(reference)
                if group is not None:
                    all_groups.append(group)

        # Log group sizes and validate expected group count
        if self._group_by and valid_groups:
            from collections import Counter

            group_counts = Counter(valid_groups)
            print(f"Group-by '{self._group_by}': {len(group_counts)} groups")
            for group_name in sorted(group_counts):
                print(f"  {group_name}: {group_counts[group_name]} samples")

            if self._expected_groups is not None:
                actual = len(group_counts)
                if actual != self._expected_groups:
                    raise ValueError(
                        f"Task '{self.name}': expected {self._expected_groups} groups "
                        f"but found {actual}. Groups: {sorted(group_counts.keys())}"
                    )

        # Per-sample normalization for variable-range tasks (e.g., multiround banker)
        # Each sample may have different max value based on investment amount
        if self._normalize_before_wasserstein:
            normalized_preds = []
            normalized_refs = []
            failed_extractions = 0

            for pred, ref, prompt in zip(
                valid_predictions, valid_references, valid_prompts, strict=True
            ):
                # Extract investment amount from prompt text (system or user)
                prompt_text = prompt.get("system", "") + " " + prompt.get("user", "")
                investment = extract_banker_investment(prompt_text)

                if investment and investment > 0:
                    max_val = investment * self._normalize_multiplier
                    # Normalize to [0, 100] scale
                    norm_pred = (pred / max_val) * 100
                    norm_ref = (ref / max_val) * 100
                    normalized_preds.append(norm_pred)
                    normalized_refs.append(norm_ref)
                else:
                    # Fallback: use raw values (shouldn't happen for banker data)
                    failed_extractions += 1
                    normalized_preds.append(pred)
                    normalized_refs.append(ref)

            if failed_extractions > 0:
                print(
                    f"Warning: Failed to extract investment from {failed_extractions} samples, "
                    f"using raw values for those samples"
                )

            # Use normalized values for metrics
            preds_for_metrics = normalized_preds
            refs_for_metrics = normalized_refs
        else:
            preds_for_metrics = valid_predictions
            refs_for_metrics = valid_references

        # Build metric_kwargs for Wasserstein normalization and per-output MAE
        metric_kwargs = {}
        if self._clip_range and self._normalize_range:
            metric_kwargs["value_range"] = self._clip_range
            metric_kwargs["normalize_range"] = self._normalize_range
        if self._element_names is not None:
            metric_kwargs["element_names"] = self._element_names

        # Split metrics into classification (use all_predictions with invalid→opposite) vs
        # distribution (use valid_predictions only, needs real numeric values)
        classification_metrics = {
            "Accuracy",
            "accuracy",
            "accuracy_multi_choice",
            "f1_macro",
            "F1_macro",
        }
        class_metric_names = [m for m in self.metrics if m in classification_metrics]
        dist_metric_names = [m for m in self.metrics if m not in classification_metrics]

        computed_metrics: dict[str, Any] = {}

        # Classification metrics: failed parses mapped to opposite of ground truth (always wrong)
        if class_metric_names and len(all_predictions) > 0:
            all_groups_for_metrics = all_groups if self._group_by else None
            # Pass explicit labels to avoid spurious class detection in F1
            class_labels = sorted(set(all_references))
            class_kwargs = {"labels": class_labels}
            computed_metrics.update(
                compute_metrics(
                    metric_names=class_metric_names,
                    predictions=all_predictions,
                    references=all_references,
                    groups=all_groups_for_metrics,
                    metric_kwargs=class_kwargs,
                    group_averaging=self._group_averaging,
                )
            )

        # Distribution metrics: compute on valid predictions only (need real values)
        if dist_metric_names and len(valid_predictions) > 0:
            groups_for_metrics = valid_groups if self._group_by else None
            computed_metrics.update(
                compute_metrics(
                    metric_names=dist_metric_names,
                    predictions=preds_for_metrics,
                    references=refs_for_metrics,
                    groups=groups_for_metrics,
                    metric_kwargs=metric_kwargs if metric_kwargs else None,
                    group_averaging=self._group_averaging,
                )
            )

        metadata = {
            "num_samples": total_samples,
            "num_valid_predictions": len(valid_predictions),
            "num_failed_parses": num_failed_parses,
            "failed_parse_rate": num_failed_parses / total_samples if total_samples > 0 else 0,
        }

        # Add grouping info to metadata
        if self._group_by:
            metadata["group_by"] = self._group_by
            metadata["num_groups"] = len(set(valid_groups))

        # Add normalization info to metadata
        if self._normalize_before_wasserstein:
            metadata["normalize_before_wasserstein"] = True
            metadata["normalize_multiplier"] = self._normalize_multiplier

        return EvaluationResult(
            task_name=self.name,
            metrics=computed_metrics,
            predictions=detailed_results,
            references=None,  # References are included in detailed_results
            metadata=metadata,
        )


# =============================================================================
# Task Factory (creates tasks from YAML config)
# =============================================================================


def load_task_config(config_path: str | Path | None = None) -> dict[str, dict]:
    """
    Load task configurations from YAML file.

    Args:
        config_path: Path to YAML config file. If None, uses default config.

    Returns:
        Dictionary mapping task names to their configurations
    """
    if config_path is None:
        # Use default config in tasks directory
        config_path = Path(__file__).parent / "tasks" / "tasks_config.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config.get("tasks", {})


def create_task(
    task_name: str,
    data_path: str,
    config_path: str | Path | None = None,
    **kwargs,
) -> ConfigurableTask:
    """
    Create a task instance from YAML configuration.

    Args:
        task_name: Name of the task (must exist in config)
        data_path: Path to the evaluation data
        config_path: Path to YAML config file. If None, uses default config.
        **kwargs: Additional arguments passed to task constructor

    Returns:
        ConfigurableTask instance

    Raises:
        ValueError: If task_name not found in config
    """
    task_configs = load_task_config(config_path)

    if task_name not in task_configs:
        available = list(task_configs.keys())
        raise ValueError(f"Unknown task: {task_name}. Available tasks: {available}")

    task_config = task_configs[task_name]

    return ConfigurableTask(
        data_path=data_path,
        name=task_name,
        metrics=task_config.get("metrics", []),
        output_type=task_config.get("output_type", "text"),
        clip_range=task_config.get("clip_range"),
        group_by=task_config.get("group_by"),
        group_averaging=task_config.get("group_averaging", "macro"),
        normalize_before_wasserstein=task_config.get("normalize_before_wasserstein", False),
        normalize_multiplier=task_config.get("normalize_multiplier", 1.0),
        normalize_range=task_config.get("normalize_range"),
        element_names=task_config.get("element_names"),
        expected_list_length=task_config.get("expected_list_length"),
        expected_groups=task_config.get("expected_groups"),
        **kwargs,
    )
