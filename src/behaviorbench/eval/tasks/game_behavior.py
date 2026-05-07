"""
Game behavior simulation evaluation task.

Evaluates models on their ability to simulate human behavior in experimental games.
Uses Wasserstein distance to compare model-generated action distributions with
human action distributions.
"""

from typing import Any

import numpy as np

from behaviorbench.eval.base import BaseEvaluationTask, EvaluationResult
from behaviorbench.eval.metrics import compute_metrics
from behaviorbench.eval.utils import load_game_data, parse_numeric_output


class GameBehaviorTask(BaseEvaluationTask):
    """
    Evaluate model's ability to simulate human game behavior.

    This task loads experimental game data (MobLab format) and evaluates
    how well a model can generate actions that match the human action distribution.

    Note: This task has custom load_data and run_evaluation because:
    - Data format differs from standard JSONL (uses load_game_data)
    - Evaluation requires multiple samples per prompt

    Metrics:
        - Wasserstein distance between predicted and human action distributions
    """

    name = "game_behavior"

    def __init__(
        self,
        data_path: str | None = None,
        name: str | None = None,
        metrics: list[str] | None = None,
        num_samples_per_game: int = 1000,
        clip_range: list[float] | tuple[float, float] | None = None,
        normalize_range: list[float] | tuple[float, float] | None = None,
        **kwargs,
    ):
        """
        Initialize game behavior evaluation task.

        Args:
            data_path: Path to game data file (JSONL format)
            name: Task name
            metrics: List of metric names to compute (loaded from YAML config)
            num_samples_per_game: Number of actions to generate per game instruction (default: 1000)
            clip_range: Value range [min, max] for clipping parsed actions (e.g., [0, 100])
            normalize_range: Output range [min, max] for normalizing Wasserstein distance (e.g., [0, 100])
            **kwargs: Additional configuration
        """
        super().__init__(data_path=data_path, name=name, **kwargs)
        self._metrics = metrics or []
        self.num_samples_per_game = num_samples_per_game
        self.clip_range = tuple(clip_range) if clip_range else None
        self.normalize_range = tuple(normalize_range) if normalize_range else None
        self.human_actions: list[float] = []

    @property
    def metrics(self) -> list[str]:  # type: ignore[override]
        """Return metrics list."""
        return self._metrics

    def _parse_output(self, raw_output: Any) -> float | None:
        """Parse numeric action from output like "[50]" or "50"."""
        if isinstance(raw_output, (int, float)):
            value = float(raw_output)
        else:
            value = parse_numeric_output(str(raw_output))

        # Clip to range if specified
        if value is not None and self.clip_range:
            value = max(self.clip_range[0], min(self.clip_range[1], value))

        return value

    def _parse_ground_truth(self, raw_output: Any, sample_idx: int = -1) -> float | None:
        """Parse ground truth action (no clipping - validates range instead)."""
        if isinstance(raw_output, (int, float)):
            value = float(raw_output)
        else:
            value = parse_numeric_output(str(raw_output))

        # Validate range (don't clip - raise error if data is bad)
        if value is not None and self.clip_range:
            if value < self.clip_range[0] or value > self.clip_range[1]:
                raise ValueError(
                    f"Ground truth value {value} at sample {sample_idx} is outside "
                    f"valid clip_range {self.clip_range} for task '{self.name}'. "
                    f"This indicates a data bug - fix the source data."
                )

        return value

    def load_data(self) -> None:
        """Load game prompts and human actions from data file."""
        self.prompts, actions_str = load_game_data(self.data_path)

        # Parse actions (they might be in format "[50]" or just "50")
        self.human_actions = []
        for action_str in actions_str:
            parsed = parse_numeric_output(action_str)
            if parsed is not None:
                self.human_actions.append(parsed)

        # Store in ground_truth for compatibility
        self.ground_truth = self.human_actions

    def run_evaluation(
        self,
        model: Any,
        num_samples: int | None = None,
        **kwargs,
    ) -> EvaluationResult:
        """
        Run game behavior evaluation.

        Args:
            model: Callable that takes a prompt dict (system/user) and returns an action
            num_samples: Number of samples to generate (overrides default)
            **kwargs: Additional model arguments

        Returns:
            EvaluationResult with Wasserstein distance metric
        """
        if not self.prompts:
            self.load_data()

        num_samples = num_samples or self.num_samples_per_game

        # Extract unique prompt - should be exactly one per game file
        unique_prompts = []
        seen = set()
        for prompt in self.prompts:
            prompt_key = (prompt["system"], prompt["user"])
            if prompt_key not in seen:
                seen.add(prompt_key)
                unique_prompts.append(prompt)

        if len(unique_prompts) != 1:
            raise ValueError(
                f"Expected exactly 1 unique prompt per game file, but found {len(unique_prompts)}. "
                f"Each game behavior file should contain the same prompt with different human responses."
            )

        model_actions = []
        detailed_results = []
        total_attempts = 0
        num_failed_parses = 0
        num_cached = 0

        for prompt in unique_prompts:
            # Reuse cached results from resume file (e.g., n100 results)
            if self.existing_results:
                prompt_key = self._prompt_key(prompt)
                # existing_results may have multiple entries for the same prompt
                # (game_behavior generates N samples from 1 prompt)
                # All entries share the same key, so we stored them as a list
                # But _load_existing_results stores one entry per key...
                # Instead, iterate all existing_results and match by prompt
                for cached_pred in self._get_all_cached_results(prompt):
                    raw_output = cached_pred.get("raw_output", "")
                    action = self._parse_output(raw_output)
                    total_attempts += 1
                    num_cached += 1

                    detailed_results.append(
                        self._create_detailed_result(
                            prompt=prompt,
                            raw_output=str(raw_output),
                            parsed_prediction=action,
                            expected=None,
                        )
                    )

                    if action is not None:
                        model_actions.append(action)
                    else:
                        num_failed_parses += 1

                if num_cached > 0:
                    print(
                        f"Resume: reused {num_cached} cached samples, "
                        f"generating {num_samples - num_cached} new samples"
                    )

            # Generate remaining samples for this prompt. For API-backed models,
            # use batch_call so repeated draws can honor the model concurrency.
            samples_to_generate = max(0, num_samples - num_cached)
            if samples_to_generate > 0 and hasattr(model, "batch_call"):
                outputs = model.batch_call([prompt] * samples_to_generate, **kwargs)
            else:
                outputs = [model(prompt, **kwargs) for _ in range(samples_to_generate)]

            for output in outputs:
                action = self._parse_output(output)
                total_attempts += 1

                detailed_results.append(
                    self._create_detailed_result(
                        prompt=prompt,
                        raw_output=str(output),
                        parsed_prediction=action,
                        expected=None,  # Distribution-based comparison
                    )
                )

                if action is not None:
                    model_actions.append(action)
                else:
                    num_failed_parses += 1

        metadata = {
            "num_prompts": len(unique_prompts),
            "num_total_attempts": total_attempts,
            "num_model_samples": len(model_actions),
            "num_human_samples": len(self.human_actions),
            "num_failed_parses": num_failed_parses,
            "failed_parse_rate": (num_failed_parses / total_attempts if total_attempts > 0 else 0),
        }

        if len(model_actions) == 0:
            # Return result with empty metrics instead of crashing
            return EvaluationResult(
                task_name=self.name,
                metrics={},
                predictions=detailed_results,
                references=self.human_actions,
                metadata=metadata,
            )

        # Compute metrics (with normalization if configured)
        metric_kwargs = {}
        if self.clip_range and self.normalize_range:
            metric_kwargs["value_range"] = self.clip_range
            metric_kwargs["normalize_range"] = self.normalize_range

        computed_metrics = compute_metrics(
            metric_names=self.metrics,
            predictions=model_actions,
            references=self.human_actions,
            metric_kwargs=metric_kwargs if metric_kwargs else None,
        )

        # Add distribution statistics to metadata
        metadata["model_action_mean"] = float(np.mean(model_actions))
        metadata["model_action_std"] = float(np.std(model_actions))
        metadata["human_action_mean"] = float(np.mean(self.human_actions))
        metadata["human_action_std"] = float(np.std(self.human_actions))

        return EvaluationResult(
            task_name=self.name,
            metrics=computed_metrics,
            predictions=detailed_results,
            references=self.human_actions,
            metadata=metadata,
        )
