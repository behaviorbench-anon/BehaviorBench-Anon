"""Command line entry point for BehaviorBench evaluations."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from behaviorbench.eval.base import ConfigurableTask, EvaluationResult
from behaviorbench.eval.tasks import (
    AcrossdimPersScoreTask,
    DemoPredAgeTask,
    GameBehaviorBombTask,
    GameBehaviorDictatorTask,
    GameBehaviorGuessingTask,
    GameBehaviorPublicGoodsTask,
    GameBehaviorPushPullTask,
    GameBehaviorTask,
    GameBehaviorTrustBankerTask,
    GameBehaviorTrustInvestorTask,
    GameBehaviorUltimatumProposerTask,
    GameBehaviorUltimatumResponderTask,
    IEOEconomicsTask,
    MissingSurvRespTask,
    PersScorePredTask,
    ResearchWorkflowTask,
    SeqSurvRespTask,
    SurvRespPredTask,
    create_configured_task_class,
)
from behaviorbench.eval.tasks.guessing_winrate import GuessingWinrateTask


def configured(task_name: str) -> type[ConfigurableTask]:
    """Shortcut for task classes backed by ``tasks_config.yaml``."""
    return create_configured_task_class(task_name)


TASK_REGISTRY = {
    # Big Five survey and subject-trait tasks.
    "pers_score_pred": PersScorePredTask,
    "surv_resp_pred": SurvRespPredTask,
    "missing_surv_resp": MissingSurvRespTask,
    "seq_surv_resp": SeqSurvRespTask,
    "demo_pred_age": DemoPredAgeTask,
    "acrossdim_pers_score": AcrossdimPersScoreTask,
    # Behavioral knowledge tasks.
    "workflow_idea_generation": ResearchWorkflowTask,
    "workflow_method_recommendation": ResearchWorkflowTask,
    "workflow_outcome_prediction": ResearchWorkflowTask,
    "workflow_title_prediction": ResearchWorkflowTask,
    "workflow_impact_prediction": ResearchWorkflowTask,
    "ieo_economics": IEOEconomicsTask,
    # First-round game behavior simulation.
    "game_behavior": GameBehaviorTask,
    "game_behavior_dictator": GameBehaviorDictatorTask,
    "game_behavior_ultimatum_proposer": GameBehaviorUltimatumProposerTask,
    "game_behavior_ultimatum_responder": GameBehaviorUltimatumResponderTask,
    "game_behavior_trust_investor": GameBehaviorTrustInvestorTask,
    "game_behavior_trust_banker": GameBehaviorTrustBankerTask,
    "game_behavior_public_goods": GameBehaviorPublicGoodsTask,
    "game_behavior_bomb": GameBehaviorBombTask,
    "game_behavior_guessing": GameBehaviorGuessingTask,
    "game_behavior_push_pull": GameBehaviorPushPullTask,
    # Multi-round game behavior prediction.
    "multiround_behavior_dictator": configured("multiround_behavior_dictator"),
    "multiround_behavior_trust_investor": configured("multiround_behavior_trust_investor"),
    "multiround_behavior_trust_banker_inv50": configured("multiround_behavior_trust_banker_inv50"),
    "multiround_behavior_trust_banker_inv100": configured(
        "multiround_behavior_trust_banker_inv100"
    ),
    "multiround_behavior_public_goods": configured("multiround_behavior_public_goods"),
    "multiround_behavior_bomb": configured("multiround_behavior_bomb"),
    "multiround_behavior_guessing": configured("multiround_behavior_guessing"),
    "multiround_behavior_push_pull": configured("multiround_behavior_push_pull"),
    # Across-context game behavior prediction.
    "acrossgame_behavior_dictator": configured("acrossgame_behavior_dictator"),
    "acrossgame_behavior_ultimatum_proposer": configured("acrossgame_behavior_ultimatum_proposer"),
    "acrossgame_behavior_ultimatum_responder": configured(
        "acrossgame_behavior_ultimatum_responder"
    ),
    "acrossgame_behavior_trust_investor": configured("acrossgame_behavior_trust_investor"),
    "acrossgame_behavior_trust_banker": configured("acrossgame_behavior_trust_banker"),
    "acrossgame_behavior_public_goods": configured("acrossgame_behavior_public_goods"),
    "acrossgame_behavior_bomb": configured("acrossgame_behavior_bomb"),
    "acrossgame_behavior_guessing": configured("acrossgame_behavior_guessing"),
    "acrossgame_behavior_push_pull": configured("acrossgame_behavior_push_pull"),
    # Strategic decision-making.
    "strategic_gameplay_guessing": GuessingWinrateTask,
}


WORKFLOW_TASKS = {
    "workflow_idea_generation",
    "workflow_method_recommendation",
    "workflow_outcome_prediction",
    "workflow_title_prediction",
    "workflow_impact_prediction",
}


DEFAULT_DATA_PATHS = {
    "pers_score_pred": "data/big_five/pers_score_pred/test.jsonl",
    "surv_resp_pred": "data/big_five/surv_resp_pred/test.jsonl",
    "missing_surv_resp": "data/big_five/missing_surv_resp/test.jsonl",
    "seq_surv_resp": "data/big_five/seq_surv_resp/test.jsonl",
    "demo_pred_age": "data/big_five/demo_pred_age/test.jsonl",
    "acrossdim_pers_score": "data/big_five/acrossdim_pers_score/test.jsonl",
    "workflow_idea_generation": "data/workflows/idea_generation_test.jsonl",
    "workflow_method_recommendation": "data/workflows/method_recommendation_test.jsonl",
    "workflow_outcome_prediction": "data/workflows/outcome_prediction_test.jsonl",
    "workflow_title_prediction": "data/workflows/title_prediction_test.jsonl",
    "workflow_impact_prediction": "data/workflows/impact_prediction_test.jsonl",
    "ieo_economics": "data/economics_contests/ieo_economics.jsonl",
    "game_behavior_dictator": "data/moblab/game_behavior/dictator_test.jsonl",
    "game_behavior_ultimatum_proposer": ("data/moblab/game_behavior/ultimatum_proposer_test.jsonl"),
    "game_behavior_ultimatum_responder": (
        "data/moblab/game_behavior/ultimatum_responder_test.jsonl"
    ),
    "game_behavior_trust_investor": ("data/moblab/game_behavior/trust_investor_test.jsonl"),
    "game_behavior_trust_banker": ("data/moblab/game_behavior/trust_banker_test.jsonl"),
    "game_behavior_public_goods": ("data/moblab/game_behavior/public_goods_test.jsonl"),
    "game_behavior_bomb": "data/moblab/game_behavior/bomb_test.jsonl",
    "game_behavior_guessing": "data/moblab/game_behavior/guessing_test.jsonl",
    "game_behavior_push_pull": ("data/moblab/game_behavior/push_pull_test.jsonl"),
    "multiround_behavior_dictator": (
        "data/moblab/multiround_behavior/dictator_multiround_test.jsonl"
    ),
    "multiround_behavior_trust_investor": (
        "data/moblab/multiround_behavior/trust_investor_multiround_test.jsonl"
    ),
    "multiround_behavior_trust_banker_inv50": (
        "data/moblab/multiround_behavior/trust_banker_inv50_multiround_test.jsonl"
    ),
    "multiround_behavior_trust_banker_inv100": (
        "data/moblab/multiround_behavior/trust_banker_inv100_multiround_test.jsonl"
    ),
    "multiround_behavior_public_goods": (
        "data/moblab/multiround_behavior/public_goods_multiround_test.jsonl"
    ),
    "multiround_behavior_bomb": ("data/moblab/multiround_behavior/bomb_multiround_test.jsonl"),
    "multiround_behavior_guessing": (
        "data/moblab/multiround_behavior/guessing_multiround_test.jsonl"
    ),
    "multiround_behavior_push_pull": (
        "data/moblab/multiround_behavior/push_pull_multiround_test.jsonl"
    ),
    "acrossgame_behavior_dictator": (
        "data/moblab/acrossgame_behavior/acrossgame_dictator_test.jsonl"
    ),
    "acrossgame_behavior_ultimatum_proposer": (
        "data/moblab/acrossgame_behavior/acrossgame_ultimatum_proposer_test.jsonl"
    ),
    "acrossgame_behavior_ultimatum_responder": (
        "data/moblab/acrossgame_behavior/acrossgame_ultimatum_responder_test.jsonl"
    ),
    "acrossgame_behavior_trust_investor": (
        "data/moblab/acrossgame_behavior/acrossgame_trust_investor_test.jsonl"
    ),
    "acrossgame_behavior_trust_banker": (
        "data/moblab/acrossgame_behavior/acrossgame_trust_banker_test.jsonl"
    ),
    "acrossgame_behavior_public_goods": (
        "data/moblab/acrossgame_behavior/acrossgame_public_goods_test.jsonl"
    ),
    "acrossgame_behavior_bomb": ("data/moblab/acrossgame_behavior/acrossgame_bomb_test.jsonl"),
    "acrossgame_behavior_guessing": (
        "data/moblab/acrossgame_behavior/acrossgame_guessing_test.jsonl"
    ),
    "acrossgame_behavior_push_pull": (
        "data/moblab/acrossgame_behavior/acrossgame_push_pull_test.jsonl"
    ),
    "strategic_gameplay_guessing": (
        "data/moblab/strategic_gameplay/guessing_strategic_gameplay_test.jsonl"
    ),
}


class EvaluationRunner:
    """Run one or more BehaviorBench tasks and save result JSON."""

    def __init__(self, output_dir: str | Path = "eval_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[EvaluationResult] = []

    def run_task(
        self,
        task_name: str,
        model: Callable,
        data_path: str | Path,
        **task_kwargs,
    ) -> EvaluationResult:
        if task_name not in TASK_REGISTRY:
            raise ValueError(f"Unknown task: {task_name}")
        task = TASK_REGISTRY[task_name](data_path=str(data_path), name=task_name, **task_kwargs)
        print(f"\nRunning {task_name}")
        print(f"Data: {data_path}")
        result = task.run_evaluation(model)
        print(result.summary())
        self.results.append(result)
        return result

    def save_results(
        self,
        filename: str,
        inference_settings: dict[str, Any] | None = None,
        token_usage: dict[str, Any] | None = None,
    ) -> Path:
        metrics = {result.task_name: result.metrics for result in self.results}
        if len(WORKFLOW_TASKS & set(metrics)) >= 2:
            metrics["workflow_average"] = self._average_workflow_metrics(metrics)

        output = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "num_tasks": len(self.results),
            "tasks": [result.to_dict(include_full_results=True) for result in self.results],
        }
        if inference_settings:
            output["inference_settings"] = inference_settings
        if token_usage:
            output["token_usage"] = token_usage

        output_path = self.output_dir / filename
        with output_path.open("w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved results to {output_path}")
        return output_path

    @staticmethod
    def _average_workflow_metrics(metrics: dict[str, dict[str, Any]]) -> dict[str, float]:
        import numpy as np

        workflow = [metrics[name] for name in sorted(WORKFLOW_TASKS) if name in metrics]
        metric_names = {
            key for task_metrics in workflow for key in task_metrics if not key.startswith("_")
        }
        return {
            key: float(
                np.mean([task_metrics[key] for task_metrics in workflow if key in task_metrics])
            )
            for key in sorted(metric_names)
        }


def create_dummy_model() -> Callable:
    """Create a cheap model for installation and pipeline smoke tests."""
    import random

    def dummy_model(prompt: str | dict, **kwargs) -> str:
        text = ""
        if isinstance(prompt, dict):
            text = f"{prompt.get('system', '')} {prompt.get('user', '')}".lower()
        else:
            text = str(prompt).lower()
        if "push" in text and "pull" in text:
            return random.choice(["Push", "Pull"])
        if "age" in text:
            return str(random.randint(18, 80))
        if "score" in text:
            return str(random.randint(10, 50))
        if "a/b/c/d" in text or "multiple choice" in text:
            return f"[{random.choice(['A', 'B', 'C', 'D'])}]"
        return str(random.randint(0, 100))

    return dummy_model


def build_model(args: argparse.Namespace) -> Callable:
    """Build a model callable from CLI arguments."""
    if args.dummy_model:
        print("Using dummy model")
        return create_dummy_model()
    if not args.model_name:
        raise ValueError("--model-name is required unless --dummy-model is set")

    if args.model_type == "local":
        from behaviorbench.models import LocalModel

        return LocalModel(
            model_name=args.model_name,
            api_base=args.api_base,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            concurrency=args.concurrency,
            timeout=args.timeout,
        )
    if args.model_type == "centaur_local":
        from behaviorbench.models import CentaurLocalModel

        return CentaurLocalModel(
            model_name=args.model_name,
            api_base=args.api_base,
            max_tokens=args.max_tokens,
            temperature=args.temperature if args.temperature is not None else 0.6,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            concurrency=args.concurrency,
            timeout=args.timeout,
        )
    if args.model_type == "openai":
        from behaviorbench.models import OpenAIModel

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY for --model-type openai")
        return OpenAIModel(
            model_name=args.model_name,
            api_key=api_key,
            api_base=args.api_base if args.api_base else None,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            concurrency=args.concurrency,
            reasoning_effort=args.reasoning_effort,
        )
    if args.model_type == "azure":
        from behaviorbench.models import OpenAIModel

        return OpenAIModel(
            model_name=args.model_name,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            use_azure=True,
            concurrency=args.concurrency,
            reasoning_effort=args.reasoning_effort,
        )
    if args.model_type == "azure_batch":
        from behaviorbench.models import AzureBatchModel

        return AzureBatchModel(
            model_name=args.model_name,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            resume_batch_id=args.resume_batch,
            reasoning_effort=args.reasoning_effort,
        )
    if args.model_type == "anthropic":
        from behaviorbench.models import AnthropicModel

        return AnthropicModel(
            model_name=args.model_name,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            concurrency=args.concurrency,
            reasoning_effort=args.reasoning_effort,
        )
    if args.model_type == "vertex":
        from behaviorbench.models import VertexAIModel

        return VertexAIModel(
            model_name=args.model_name,
            project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            region=os.environ.get("VERTEX_REGION", "global"),
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            concurrency=args.concurrency,
            reasoning_effort=args.reasoning_effort,
        )
    if args.model_type == "heuristic":
        from behaviorbench.models import HeuristicModel

        return HeuristicModel(strategy=args.model_name)
    raise ValueError(f"Unknown model type: {args.model_type}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BehaviorBench evaluations")
    parser.add_argument("--task", nargs="+", required=True, choices=sorted(TASK_REGISTRY))
    parser.add_argument("--data-path", nargs="*", default=None)
    parser.add_argument("--output-dir", default="eval_results")
    parser.add_argument("--dummy-model", action="store_true")
    parser.add_argument(
        "--model-type",
        default="local",
        choices=[
            "local",
            "centaur_local",
            "openai",
            "azure",
            "azure_batch",
            "anthropic",
            "vertex",
            "heuristic",
        ],
    )
    parser.add_argument("--model-name")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples-per-game", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--resume-batch", default=None)
    return parser.parse_args()


def resolve_data_paths(tasks: list[str], data_paths: list[str] | None) -> list[str]:
    explicit = data_paths or []
    resolved = []
    for i, task_name in enumerate(tasks):
        if i < len(explicit):
            resolved.append(explicit[i])
        elif task_name in DEFAULT_DATA_PATHS:
            resolved.append(DEFAULT_DATA_PATHS[task_name])
        else:
            raise ValueError(f"No default data path for task {task_name}; pass --data-path")
    return resolved


def main() -> None:
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    load_dotenv()
    args = parse_args()

    tasks = list(args.task)
    if any(task.startswith("workflow_") for task in tasks) and set(tasks) != WORKFLOW_TASKS:
        raise ValueError(
            "Workflow evaluation must run all five canonical workflow tasks together: "
            f"{sorted(WORKFLOW_TASKS)}"
        )

    task_kwargs: dict[str, Any] = {}
    if args.num_samples is not None:
        task_kwargs.update({"num_samples": args.num_samples, "seed": args.seed})
    if args.num_samples_per_game is not None:
        task_kwargs["num_samples_per_game"] = args.num_samples_per_game
    if args.resume:
        task_kwargs["resume_file"] = args.resume

    model = build_model(args)
    data_paths = resolve_data_paths(tasks, args.data_path)

    model_folder = args.model_name or ("dummy" if args.dummy_model else args.model_type)
    runner = EvaluationRunner(Path(args.output_dir) / model_folder)

    for task_name, data_path in zip(tasks, data_paths, strict=True):
        runner.run_task(task_name, model, data_path, **task_kwargs)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = (
        f"combined_5subtasks_{timestamp}.json"
        if set(tasks) == WORKFLOW_TASKS
        else f"{'_'.join(tasks[:2])}_{timestamp}.json"
    )

    token_usage = None
    if args.model_type in {"openai", "azure", "azure_batch", "anthropic", "vertex"}:
        from behaviorbench.models import TokenTracker

        token_usage = TokenTracker.get_summary()

    runner.save_results(
        filename,
        inference_settings={
            "model_name": args.model_name,
            "model_type": args.model_type,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "max_tokens": args.max_tokens,
            "concurrency": args.concurrency,
            "reasoning_effort": args.reasoning_effort,
        },
        token_usage=token_usage,
    )


if __name__ == "__main__":
    main()
