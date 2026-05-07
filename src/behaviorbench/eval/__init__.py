"""
Evaluation framework for behavioral and economic modeling tasks.

This package provides a unified interface for evaluating models on various tasks including:
- Game behavior simulation
- Personality and demographic prediction
- Research workflow prediction
- Economics problem solving
"""

from behaviorbench.eval.base import BaseEvaluationTask, EvaluationResult

__all__ = ["BaseEvaluationTask", "EvaluationResult"]
