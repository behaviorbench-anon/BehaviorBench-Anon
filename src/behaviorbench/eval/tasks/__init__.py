"""BehaviorBench task classes."""

from behaviorbench.eval.base import ConfigurableTask, create_task, load_task_config
from behaviorbench.eval.tasks.game_behavior import GameBehaviorTask
from behaviorbench.eval.tasks.guessing_winrate import GuessingWinrateTask


def create_configured_task_class(task_name: str) -> type[ConfigurableTask]:
    """Create a ``ConfigurableTask`` subclass from ``tasks_config.yaml``."""
    config = load_task_config()
    if task_name not in config:
        raise ValueError(f"Unknown task: {task_name}")

    task_config = config[task_name]
    class_name = "".join(word.title() for word in task_name.split("_")) + "Task"

    def __init__(self, data_path: str | None = None, **kwargs):
        override_name = kwargs.pop("name", task_name)
        ConfigurableTask.__init__(
            self,
            data_path=data_path,
            name=override_name,
            metrics=task_config.get("metrics", []),
            output_type=task_config.get("output_type", "text"),
            clip_range=task_config.get("clip_range"),
            group_by=task_config.get("group_by"),
            group_averaging=task_config.get("group_averaging", "macro"),
            normalize_range=task_config.get("normalize_range"),
            element_names=task_config.get("element_names"),
            expected_list_length=task_config.get("expected_list_length"),
            **kwargs,
        )

    return type(
        class_name,
        (ConfigurableTask,),
        {"__init__": __init__, "__doc__": f"BehaviorBench task: {task_name}."},
    )


PersScorePredTask = create_configured_task_class("pers_score_pred")
SurvRespPredTask = create_configured_task_class("surv_resp_pred")
MissingSurvRespTask = create_configured_task_class("missing_surv_resp")
SeqSurvRespTask = create_configured_task_class("seq_surv_resp")
DemoPredAgeTask = create_configured_task_class("demo_pred_age")
AcrossdimPersScoreTask = create_configured_task_class("acrossdim_pers_score")
ResearchWorkflowTask = create_configured_task_class("research_workflow")
IEOEconomicsTask = create_configured_task_class("ieo_economics")


def create_game_behavior_task_class(game_name: str) -> type[GameBehaviorTask]:
    """Create a first-round game-behavior task class for one game."""
    config = load_task_config()
    task_name = f"game_behavior_{game_name}"
    task_config = config[task_name]
    class_name = "GameBehavior" + "".join(word.title() for word in game_name.split("_")) + "Task"

    def __init__(self, data_path: str | None = None, **kwargs):
        kwargs.pop("name", None)
        GameBehaviorTask.__init__(
            self,
            data_path=data_path,
            name=task_name,
            metrics=task_config.get("metrics", []),
            clip_range=task_config.get("clip_range"),
            normalize_range=task_config.get("normalize_range"),
            **kwargs,
        )

    return type(
        class_name,
        (GameBehaviorTask,),
        {"__init__": __init__, "__doc__": f"First-round game task: {game_name}."},
    )


def create_push_pull_game_behavior_task() -> type[GameBehaviorTask]:
    """Create the first-round Push/Pull task with text-label parsing."""
    from behaviorbench.eval.utils import load_game_data, parse_push_pull_output

    task_config = load_task_config()["game_behavior_push_pull"]

    class GameBehaviorPushPullTask(GameBehaviorTask):
        """First-round Push/Pull behavior simulation."""

        name = "game_behavior_push_pull"

        def __init__(self, data_path: str | None = None, **kwargs):
            kwargs.pop("name", None)
            GameBehaviorTask.__init__(
                self,
                data_path=data_path,
                name=self.name,
                metrics=task_config.get("metrics", []),
                clip_range=task_config.get("clip_range"),
                normalize_range=task_config.get("normalize_range"),
                **kwargs,
            )

        def _parse_output(self, raw_output):
            value = (
                int(raw_output)
                if isinstance(raw_output, (int, float))
                else parse_push_pull_output(str(raw_output))
            )
            if value is not None and self.clip_range:
                value = max(self.clip_range[0], min(self.clip_range[1], value))
            return value

        def _parse_ground_truth(self, raw_output, sample_idx=-1):
            value = (
                int(raw_output)
                if isinstance(raw_output, (int, float))
                else parse_push_pull_output(str(raw_output))
            )
            if (
                value is not None
                and self.clip_range
                and not self.clip_range[0] <= value <= self.clip_range[1]
            ):
                raise ValueError(
                    f"Ground truth value {value} at sample {sample_idx} is outside "
                    f"valid range {self.clip_range} for task {self.name}."
                )
            return value

        def load_data(self):
            self.prompts, actions_str = load_game_data(self.data_path)
            self.human_actions = [
                parsed
                for action in actions_str
                if (parsed := parse_push_pull_output(action)) is not None
            ]
            self.ground_truth = self.human_actions

    return GameBehaviorPushPullTask


GameBehaviorDictatorTask = create_game_behavior_task_class("dictator")
GameBehaviorUltimatumProposerTask = create_game_behavior_task_class("ultimatum_proposer")
GameBehaviorUltimatumResponderTask = create_game_behavior_task_class("ultimatum_responder")
GameBehaviorTrustInvestorTask = create_game_behavior_task_class("trust_investor")
GameBehaviorTrustBankerTask = create_game_behavior_task_class("trust_banker")
GameBehaviorPublicGoodsTask = create_game_behavior_task_class("public_goods")
GameBehaviorBombTask = create_game_behavior_task_class("bomb")
GameBehaviorGuessingTask = create_game_behavior_task_class("guessing")
GameBehaviorPushPullTask = create_push_pull_game_behavior_task()

__all__ = [
    "create_task",
    "ConfigurableTask",
    "create_configured_task_class",
    "GameBehaviorTask",
    "GuessingWinrateTask",
    "PersScorePredTask",
    "SurvRespPredTask",
    "MissingSurvRespTask",
    "SeqSurvRespTask",
    "DemoPredAgeTask",
    "AcrossdimPersScoreTask",
    "ResearchWorkflowTask",
    "IEOEconomicsTask",
    "GameBehaviorDictatorTask",
    "GameBehaviorUltimatumProposerTask",
    "GameBehaviorUltimatumResponderTask",
    "GameBehaviorTrustInvestorTask",
    "GameBehaviorTrustBankerTask",
    "GameBehaviorPublicGoodsTask",
    "GameBehaviorBombTask",
    "GameBehaviorGuessingTask",
    "GameBehaviorPushPullTask",
]
