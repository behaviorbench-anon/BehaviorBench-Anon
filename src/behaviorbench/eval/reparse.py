"""
GPT re-parsing for inline prediction correction during evaluation.

Re-parses predictions using direct Azure OpenAI API calls (not batch API)
to fix silent parsing errors where the regex grabs the first bracketed number
from verbose outputs instead of the final answer.

Designed to be called inline after each task completes, so results are
saved with corrected predictions and recomputed metrics.
"""

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from behaviorbench.eval.base import EvaluationResult, load_task_config
from behaviorbench.eval.metrics import compute_metrics

logger = logging.getLogger(__name__)

# Output types that benefit from GPT re-parsing
REPARSE_OUTPUT_TYPES = {"numeric", "integer", "numeric_list"}

# GPT prompt templates (same as scripts/parsing_reparse.py)
SYSTEM_PROMPT = (
    "You are a precise text parser. Extract the final prediction from a model's "
    "output. Return ONLY a JSON object. If no valid prediction exists, return "
    '{"value": null}.'
)

OUTPUT_TYPE_INSTRUCTIONS = {
    "numeric": (
        "Extract the single numeric prediction (e.g., a dollar amount or score"
        '{range_hint}). Return JSON: {{"value": <number_or_null>}}'
    ),
    "integer": (
        "Extract the single integer prediction{range_hint}."
        ' Return JSON: {{"value": <integer_or_null>}}'
    ),
    "numeric_list": (
        "Extract the list of numeric values (e.g., [0.7, 0.3])."
        ' Return JSON: {{"value": <list_or_null>}}'
    ),
}


def _range_hint(clip_range: tuple | None) -> str:
    if clip_range:
        return f" in range [{clip_range[0]}, {clip_range[1]}]"
    return ""


def _build_gpt_messages(
    prediction: dict,
    output_type: str,
    clip_range: tuple | None,
) -> list[dict[str, str]]:
    """Build GPT messages for re-parsing a single prediction."""
    prompt = prediction.get("prompt", {})
    user_prompt_full = prompt.get("user", "")
    raw_output = str(prediction.get("raw_output", ""))

    hint = _range_hint(clip_range)
    instruction = OUTPUT_TYPE_INSTRUCTIONS[output_type].format(range_hint=hint)

    user_content = (
        "## Task Context\nThe model was given this prompt:\n---\n"
        f"{user_prompt_full}\n---\n\n"
        f"## Model Output\n{raw_output}\n\n"
        f"## Instruction\n{instruction}"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _parse_gpt_response(content: str, output_type: str):
    """Extract the value from GPT's JSON response content."""
    try:
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

        parsed = json.loads(content)
        value = parsed.get("value")

        if value is None:
            return None

        if output_type == "numeric":
            return float(value)
        elif output_type == "integer":
            return int(value)
        elif output_type == "numeric_list":
            if isinstance(value, list):
                return [float(v) for v in value]
            return None
        return value
    except (json.JSONDecodeError, ValueError, TypeError, KeyError):
        return None


def _clip_value(value, clip_range: tuple | None, output_type: str):
    """Apply clip_range to a parsed value, matching OUTPUT_PARSERS behavior."""
    if value is None or clip_range is None:
        return value

    if output_type == "numeric":
        return float(max(clip_range[0], min(clip_range[1], value)))
    elif output_type == "integer":
        return max(int(clip_range[0]), min(int(clip_range[1]), int(value)))
    elif output_type == "numeric_list":
        if isinstance(value, list):
            return [max(clip_range[0], min(clip_range[1], v)) for v in value]
    return value


def _values_equal(a, b, output_type: str) -> bool:
    """Check if two parsed values are equal (with numeric tolerance)."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False

    if output_type == "numeric_list":
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                return False
            return all(abs(x - y) < 1e-6 for x, y in zip(a, b, strict=False))
        return False

    if output_type in ("numeric", "integer"):
        try:
            return abs(float(a) - float(b)) < 1e-6
        except (ValueError, TypeError):
            return False

    return a == b


class GptReparser:
    """Re-parses predictions using direct Azure OpenAI API calls."""

    def __init__(self, model: str = "gpt-5-nano", concurrency: int = 8):
        """Initialize with Azure OpenAI client.

        Args:
            model: Azure deployment name for GPT model
            concurrency: Number of concurrent API calls
        """
        load_dotenv()

        self.model = model
        self.concurrency = concurrency
        self._task_config_cache = None

        # Create Azure OpenAI client
        import openai

        api_key = os.environ.get("AZURE_API_KEY")
        endpoint = os.environ.get("AZURE_ENDPOINT")

        if not api_key or not endpoint:
            raise ValueError(
                "AZURE_API_KEY and AZURE_ENDPOINT must be set in environment for GPT re-parsing"
            )

        base_url = endpoint.rstrip("/") + "/openai/v1/"
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def _get_task_config(self, task_name: str) -> dict | None:
        """Get task config from YAML, with caching."""
        if self._task_config_cache is None:
            self._task_config_cache = load_task_config()
        return self._task_config_cache.get(task_name)

    def _call_gpt(self, messages: list[dict[str, str]]) -> str | None:
        """Make a single GPT API call with retry."""
        import tenacity

        from behaviorbench.models.utils import GPT5_MODELS

        use_completion_tokens = any(self.model.startswith(m) for m in GPT5_MODELS)
        token_kwarg = (
            {"max_completion_tokens": 128} if use_completion_tokens else {"max_tokens": 128}
        )

        @tenacity.retry(
            wait=tenacity.wait_exponential(multiplier=1, min=2, max=30),
            stop=tenacity.stop_after_attempt(5),
            retry=tenacity.retry_if_exception_type(Exception),
            before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
        )
        def _do_call():
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                **token_kwarg,
            )
            return response.choices[0].message.content

        try:
            return _do_call()
        except Exception as e:
            logger.warning(f"GPT call failed after retries: {e}")
            return None

    def reparse_result(self, result: EvaluationResult, task_name: str) -> None:
        """Re-parse all predictions in an EvaluationResult, modifying it in-place.

        Args:
            result: The EvaluationResult to re-parse (modified in-place)
            task_name: Name of the task (for looking up output_type/clip_range)
        """
        task_config = self._get_task_config(task_name)
        if task_config is None:
            logger.warning(f"No task config found for '{task_name}', skipping reparse")
            return

        output_type = task_config.get("output_type", "text")
        if output_type not in REPARSE_OUTPUT_TYPES:
            logger.info(f"Skipping reparse for '{task_name}' (output_type={output_type})")
            return

        predictions = result.predictions
        if not predictions:
            return

        clip_range = task_config.get("clip_range")
        clip_range = tuple(clip_range) if clip_range else None

        print(f"Re-parsing predictions for {task_name} using {self.model}...")

        # Build GPT requests for all predictions
        pred_indices = list(range(len(predictions)))
        messages_list = [
            _build_gpt_messages(predictions[i], output_type, clip_range) for i in pred_indices
        ]

        # Call GPT in parallel
        gpt_results: dict[int, str | None] = {}

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            future_to_idx = {
                executor.submit(self._call_gpt, msgs): idx
                for idx, msgs in zip(pred_indices, messages_list, strict=True)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    gpt_results[idx] = future.result()
                except Exception as e:
                    logger.warning(f"GPT call for prediction {idx} failed: {e}")
                    gpt_results[idx] = None

        # Apply GPT results
        n_updated = 0
        n_unchanged = 0
        n_gpt_failed = 0

        for idx in pred_indices:
            pred = predictions[idx]
            original_parsed = pred.get("parsed_prediction")
            content = gpt_results.get(idx)

            if content is None:
                n_gpt_failed += 1
                continue

            gpt_value = _parse_gpt_response(content, output_type)
            if gpt_value is None:
                n_gpt_failed += 1
                continue

            gpt_value = _clip_value(gpt_value, clip_range, output_type)

            if _values_equal(original_parsed, gpt_value, output_type):
                n_unchanged += 1
                continue

            # Update prediction
            pred["original_parsed_prediction"] = original_parsed
            pred["parsed_prediction"] = gpt_value
            n_updated += 1

        print(
            f"Reparse complete for {task_name}: "
            f"{n_updated} updated, {n_unchanged} unchanged, {n_gpt_failed} GPT-failed"
        )

        # Recompute metrics
        self._recompute_metrics(result, task_name, task_config)

        # Update metadata with reparse stats
        result.metadata["reparse_model"] = self.model
        result.metadata["reparse_updated"] = n_updated
        result.metadata["reparse_unchanged"] = n_unchanged
        result.metadata["reparse_gpt_failed"] = n_gpt_failed
        # Recount valid predictions after reparse
        result.metadata["num_valid_predictions"] = sum(
            1 for p in predictions if p.get("parsed_prediction") is not None
        )
        result.metadata["num_failed_parses"] = sum(
            1 for p in predictions if p.get("parsed_prediction") is None
        )
        num_samples = result.metadata.get("num_samples", len(predictions))
        result.metadata["failed_parse_rate"] = (
            result.metadata["num_failed_parses"] / num_samples if num_samples > 0 else 0
        )

    def _recompute_metrics(
        self,
        result: EvaluationResult,
        task_name: str,
        task_config: dict,
    ) -> None:
        """Recompute metrics after re-parsing predictions."""
        predictions = result.predictions
        if not predictions:
            return

        metric_names = task_config.get("metrics", [])
        clip_range = task_config.get("clip_range")
        normalize_range = task_config.get("normalize_range")
        group_by = task_config.get("group_by")

        # Build metric_kwargs
        metric_kwargs = {}
        if clip_range and normalize_range:
            metric_kwargs["value_range"] = tuple(clip_range)
            metric_kwargs["normalize_range"] = tuple(normalize_range)

        element_names = task_config.get("element_names")
        if element_names:
            metric_kwargs["element_names"] = element_names

        # Determine if this is a distribution task (game_behavior_*)
        is_distribution_task = task_name.startswith("game_behavior_")

        if is_distribution_task:
            # Distribution task: collect all valid parsed_predictions as model samples,
            # compare against human distribution (references stored in result)
            references = result.references
            if references is None:
                # game_behavior tasks store references in result.references
                # If not available, cannot recompute distribution metrics
                logger.warning(
                    f"No references found for distribution task '{task_name}', "
                    "skipping metric recomputation"
                )
                return

            model_actions = [
                p["parsed_prediction"]
                for p in predictions
                if p.get("parsed_prediction") is not None
            ]

            if not model_actions:
                return

            new_metrics = compute_metrics(
                metric_names=metric_names,
                predictions=model_actions,
                references=references,
                metric_kwargs=metric_kwargs if metric_kwargs else None,
            )
        else:
            # Per-sample task: collect valid (prediction, expected) pairs
            classification_metrics = {"Accuracy", "accuracy", "f1_macro", "F1_macro"}
            class_metric_names = [m for m in metric_names if m in classification_metrics]
            dist_metric_names = [m for m in metric_names if m not in classification_metrics]

            valid_preds = []
            valid_refs = []
            valid_groups = []
            all_preds = []
            all_refs = []
            all_groups = []

            for pred_dict in predictions:
                parsed = pred_dict.get("parsed_prediction")
                expected = pred_dict.get("expected")

                if expected is None:
                    continue

                group = None
                if group_by == "bigfive_dimension":
                    group = _extract_bigfive_dimension(pred_dict.get("prompt", {}))
                elif group_by == "target_item":
                    group = _extract_target_item(pred_dict.get("prompt", {}))

                if parsed is not None:
                    valid_preds.append(parsed)
                    valid_refs.append(expected)
                    all_preds.append(parsed)
                    all_refs.append(expected)
                    if group is not None:
                        valid_groups.append(group)
                        all_groups.append(group)
                else:
                    if isinstance(expected, (int, float)):
                        all_preds.append(1 - expected)
                    else:
                        all_preds.append(expected)
                    all_refs.append(expected)
                    if group is not None:
                        all_groups.append(group)

            if not valid_preds:
                return

            new_metrics: dict = {}

            if class_metric_names and all_preds:
                class_labels = sorted(set(all_refs))
                class_kwargs = {"labels": class_labels}
                groups_for_class = all_groups if group_by else None
                new_metrics.update(
                    compute_metrics(
                        metric_names=class_metric_names,
                        predictions=all_preds,
                        references=all_refs,
                        groups=groups_for_class,
                        metric_kwargs=class_kwargs,
                    )
                )

            if dist_metric_names and valid_preds:
                groups_for_dist = valid_groups if group_by else None
                new_metrics.update(
                    compute_metrics(
                        metric_names=dist_metric_names,
                        predictions=valid_preds,
                        references=valid_refs,
                        groups=groups_for_dist,
                        metric_kwargs=metric_kwargs if metric_kwargs else None,
                    )
                )

        if new_metrics:
            result.metrics = new_metrics


def _extract_bigfive_dimension(prompt: dict) -> str:
    """Extract BigFive dimension from prompt."""
    user_text = prompt.get("user", "")
    dimensions = [
        "Extraversion",
        "Neuroticism",
        "Agreeableness",
        "Conscientiousness",
        "Openness",
    ]
    for dim in dimensions:
        if f"*{dim}*" in user_text:
            return dim
    return "Unknown"


def _extract_target_item(prompt: dict) -> str:
    """Extract target item from prompt. Mirrors base.py._extract_group() for target_item."""
    import re

    metadata = prompt.get("metadata", {})
    if isinstance(metadata, dict) and "target_item" in metadata:
        return metadata["target_item"]
    user_text = prompt.get("user", "")
    match = re.search(r"or \[5\]\):\s*(.+?)\s*Only output", user_text)
    if match:
        return match.group(1).strip()
    return "Unknown"
