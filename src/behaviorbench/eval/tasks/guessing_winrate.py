"""
Guessing (Beauty Contest) Win Rate evaluation task.

This task evaluates whether a model's guess would win the beauty contest
(be closest to 2/3 of the group average) by using the other players'
actual choices from the data. It extends ConfigurableTask to load
other_players_choices from JSONL metadata.
"""

from behaviorbench.eval.base import ConfigurableTask, load_task_config
from behaviorbench.eval.metrics import compute_metrics
from behaviorbench.eval.utils import load_jsonl


class GuessingWinrateTask(ConfigurableTask):
    """Evaluation task for guessing game with win rate metric."""

    def __init__(self, data_path: str | None = None, **kwargs):
        if data_path is not None and "strategic_gameplay" not in str(data_path):
            raise ValueError(
                "GuessingWinrateTask is only supported on the strategic gameplay dataset. "
                f"Got data_path={data_path!r}; expected path to contain "
                "'strategic_gameplay'."
            )

        config = load_task_config()
        task_config = config.get("strategic_gameplay_guessing", {})

        kwargs.pop("name", None)
        super().__init__(
            data_path=data_path,
            name="strategic_gameplay_guessing",
            metrics=task_config.get("metrics", []),
            output_type=task_config.get("output_type", "numeric"),
            clip_range=task_config.get("clip_range"),
            normalize_range=task_config.get("normalize_range"),
            **kwargs,
        )
        self.other_players_choices_list: list[list[float]] = []

    def load_data(self) -> None:
        """Load data and extract other_players_choices from metadata."""
        data = load_jsonl(self.data_path)
        data = self._sample_data(data)

        self.prompts = []
        self.ground_truth = []
        self.other_players_choices_list = []

        for idx, item in enumerate(data):
            self.prompts.append({"system": item["system"], "user": item["user"]})
            value = self._parse_ground_truth(item["assistant"], sample_idx=idx)
            if value is not None:
                self.ground_truth.append(value)
                metadata = item.get("metadata", {})
                self.other_players_choices_list.append(metadata.get("other_players_choices", []))

    def run_evaluation(self, model, **kwargs):
        """Run evaluation with other_players_choices passed to win_rate metrics."""
        if not self.prompts:
            self.load_data()

        # Get model outputs
        if hasattr(model, "batch_call"):
            raw_outputs = model.batch_call(self.prompts, **kwargs)
        else:
            raw_outputs = []
            total = len(self.prompts)
            for i, prompt in enumerate(self.prompts):
                cached = self._get_cached_result(prompt)
                if cached:
                    raw_outputs.append(cached.get("raw_output", ""))
                else:
                    output = model(prompt, **kwargs)
                    raw_outputs.append(output)
                print(f"Completed {i + 1}/{total} API calls", flush=True)

        # Process outputs
        detailed_results = []
        valid_predictions = []
        valid_references = []
        valid_others = []

        total_samples = 0
        num_failed_parses = 0

        for prompt, raw_output, reference, others in zip(
            self.prompts,
            raw_outputs,
            self.ground_truth,
            self.other_players_choices_list,
            strict=False,
        ):
            pred = self._parse_output(raw_output)
            total_samples += 1

            detailed_results.append(
                self._create_detailed_result(
                    prompt=prompt,
                    raw_output=str(raw_output),
                    parsed_prediction=pred,
                    expected=reference,
                )
            )

            if pred is not None and reference is not None:
                valid_predictions.append(pred)
                valid_references.append(reference)
                valid_others.append(others)
            else:
                num_failed_parses += 1

        if len(valid_predictions) == 0:
            raise ValueError("No valid predictions generated")

        # Build metric_kwargs with both normalization ranges and other_players_choices
        metric_kwargs = {
            "other_players_choices_list": valid_others,
        }
        if self._clip_range and self._normalize_range:
            metric_kwargs["value_range"] = self._clip_range
            metric_kwargs["normalize_range"] = self._normalize_range

        computed_metrics = compute_metrics(
            metric_names=self.metrics,
            predictions=valid_predictions,
            references=valid_references,
            metric_kwargs=metric_kwargs,
        )

        from behaviorbench.eval.base import EvaluationResult

        metadata = {
            "num_samples": total_samples,
            "num_valid_predictions": len(valid_predictions),
            "num_failed_parses": num_failed_parses,
            "failed_parse_rate": num_failed_parses / total_samples if total_samples > 0 else 0,
        }

        return EvaluationResult(
            task_name=self.name,
            metrics=computed_metrics,
            predictions=detailed_results,
            references=None,
            metadata=metadata,
        )
