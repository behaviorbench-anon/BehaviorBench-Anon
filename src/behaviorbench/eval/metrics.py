"""
Metrics for evaluating model performance across different tasks.

This module provides implementations of various metrics including:
- Statistical metrics (MAE, Spearman correlation)
- Distribution metrics (Wasserstein distance)
- Text generation metrics (BLEURT, ROUGE-1)
- Classification metrics (accuracy)
"""

import numpy as np
from scipy import stats
from scipy.stats import wasserstein_distance


def mean_absolute_error(predictions: list[float], references: list[float]) -> float:
    """
    Calculate Mean Absolute Error (MAE).

    Args:
        predictions: Predicted values
        references: Ground truth values

    Returns:
        MAE value
    """
    predictions = np.array(predictions)
    references = np.array(references)
    return float(np.mean(np.abs(predictions - references)))


def mae_with_normalization(
    predictions: list[float] | np.ndarray,
    references: list[float] | np.ndarray,
    value_range: tuple[float, float] | None = None,
    normalize_range: tuple[float, float] | None = None,
) -> dict:
    """
    Calculate Mean Absolute Error with optional normalization.

    Similar to wasserstein_with_ks_test, returns both raw and normalized values.

    Args:
        predictions: Predicted values
        references: Ground truth values
        value_range: The input value range (min, max) of the data (e.g., [0, 100])
        normalize_range: The output range to normalize to (e.g., [0, 100] for 0-100 scale)

    Returns:
        Dictionary with 'mae' (normalized if ranges provided) and 'mae_raw'
    """
    predictions = np.array(predictions)
    references = np.array(references)
    raw_mae = float(np.mean(np.abs(predictions - references)))

    # Calculate normalized MAE if ranges provided
    normalized_mae = raw_mae
    if value_range is not None and normalize_range is not None:
        input_range = value_range[1] - value_range[0]
        output_range = normalize_range[1] - normalize_range[0]
        if input_range > 0:
            normalized_mae = raw_mae * output_range / input_range

    return {
        "mae": normalized_mae,  # Normalized (for comparability across tasks)
        "mae_raw": raw_mae,  # Raw value (for backward compatibility)
    }


def spearman_correlation(predictions: list[float], references: list[float]) -> float:
    """
    Calculate Spearman rank correlation coefficient.

    Args:
        predictions: Predicted values
        references: Ground truth values

    Returns:
        Spearman correlation coefficient
    """
    correlation, _ = stats.spearmanr(predictions, references)
    return float(correlation)


def spearman_correlation_with_pvalue(predictions: list[float], references: list[float]) -> dict:
    """
    Calculate Spearman rank correlation coefficient with p-value.

    Args:
        predictions: Predicted values
        references: Ground truth values

    Returns:
        Dictionary with 'correlation', 'pvalue', and 'significant' (1 if p < 0.05, else 0)
    """
    correlation, pvalue = stats.spearmanr(predictions, references)
    return {
        "correlation": float(correlation),
        "pvalue": float(pvalue),
        "significant": 1 if pvalue < 0.05 else 0,  # ‡ marker in paper
    }


def wasserstein(
    predictions: list[float] | np.ndarray,
    references: list[float] | np.ndarray,
    p: int = 1,
    value_range: tuple[float, float] | None = None,
    normalize_range: tuple[float, float] | None = None,
) -> float:
    """
    Calculate Wasserstein distance between two distributions.

    Args:
        predictions: Predicted distribution (samples or histogram)
        references: Reference distribution (samples or histogram)
        p: Order of the Wasserstein distance (default: 1)
        value_range: The input value range (min, max) of the data (e.g., [0, 100])
        normalize_range: The output range to normalize to (e.g., [0, 100] for 0-100 scale)

    Returns:
        Wasserstein distance (normalized if both value_range and normalize_range provided)
    """
    predictions = np.array(predictions)
    references = np.array(references)

    # If p=1, use scipy's implementation (earth mover's distance)
    if p == 1:
        distance = float(wasserstein_distance(predictions, references))
    else:
        # For other p values, compute generalized Wasserstein distance
        # Sort both distributions
        pred_sorted = np.sort(predictions)
        ref_sorted = np.sort(references)

        # Ensure same length by interpolation if needed
        if len(pred_sorted) != len(ref_sorted):
            # Create common quantiles
            n = max(len(pred_sorted), len(ref_sorted))
            quantiles = np.linspace(0, 1, n)

            pred_sorted = np.quantile(predictions, quantiles)
            ref_sorted = np.quantile(references, quantiles)

        distance = float(np.mean(np.abs(pred_sorted - ref_sorted) ** p) ** (1 / p))

    # Normalize if both ranges provided
    if value_range is not None and normalize_range is not None:
        input_range = value_range[1] - value_range[0]
        output_range = normalize_range[1] - normalize_range[0]
        if input_range > 0:
            distance = distance * output_range / input_range

    return distance


def smoothed_ks_test(
    predictions: list[float] | np.ndarray,
    references: list[float] | np.ndarray,
    bin_width: float = 10.0,
) -> dict:
    """
    Perform smoothed Kolmogorov-Smirnov test by binning data.

    This bins continuous data into discrete bins before running KS test,
    which makes the test more robust for comparing distributions.

    Args:
        predictions: Predicted values
        references: Reference values
        bin_width: Width of bins for smoothing (default: 10 for Big Five scores)

    Returns:
        Dictionary with 'statistic', 'pvalue', and 'pass_test' (1 if p > 0.05, else 0)
    """
    predictions = np.array(predictions)
    references = np.array(references)

    # Bin the data (smoothing)
    min_val = min(predictions.min(), references.min())
    max_val = max(predictions.max(), references.max())
    bins = np.arange(min_val, max_val + bin_width, bin_width)

    pred_binned = np.digitize(predictions, bins)
    ref_binned = np.digitize(references, bins)

    # Run KS test on binned data
    statistic, pvalue = stats.ks_2samp(pred_binned, ref_binned)

    return {
        "statistic": float(statistic),
        "pvalue": float(pvalue),
        "pass_test": 1 if pvalue > 0.05 else 0,  # † marker in paper (distributions are similar)
    }


def wasserstein_with_ks_test(
    predictions: list[float] | np.ndarray,
    references: list[float] | np.ndarray,
    bin_width: float = 10.0,
    value_range: tuple[float, float] | None = None,
    normalize_range: tuple[float, float] | None = None,
) -> dict:
    """
    Calculate Wasserstein distance with smoothed KS test.

    Args:
        predictions: Predicted values
        references: Reference values
        bin_width: Width of bins for KS test smoothing (default: 10)
        value_range: Input value range (min, max) for normalization
        normalize_range: Output range to normalize to (e.g., [0, 100])

    Returns:
        Dictionary with 'distance' (normalized), 'distance_raw', 'ks_statistic',
        'ks_pvalue', 'ks_pass_test'
    """
    # Calculate raw distance first (no normalization)
    raw_distance = wasserstein(predictions, references)

    # Calculate normalized distance if ranges provided
    normalized_distance = wasserstein(
        predictions,
        references,
        value_range=value_range,
        normalize_range=normalize_range,
    )

    ks_result = smoothed_ks_test(predictions, references, bin_width)

    return {
        "distance": normalized_distance,  # Normalized (for backward compatibility)
        "distance_raw": raw_distance,  # Raw distance (new field)
        "ks_statistic": ks_result["statistic"],
        "ks_pvalue": ks_result["pvalue"],
        "ks_pass_test": ks_result["pass_test"],  # † marker in paper
    }


def accuracy(predictions: list[int | str], references: list[int | str]) -> float:
    """
    Calculate classification accuracy.

    Args:
        predictions: Predicted labels
        references: Ground truth labels

    Returns:
        Accuracy (proportion of correct predictions)
    """
    predictions = np.array(predictions)
    references = np.array(references)
    return float(np.mean(predictions == references))


def accuracy_multi_choice(predictions: list[str], references: list[str]) -> float:
    """
    Calculate accuracy for multi-choice questions (exact set match).

    Each prediction and reference is a string of sorted letters (e.g., "BC", "A").
    A prediction is correct if it contains exactly the same set of letters as the reference.

    Args:
        predictions: Predicted answer strings (e.g., ["A", "BC", "D"])
        references: Ground truth answer strings (e.g., ["A", "BC", "C"])

    Returns:
        Proportion of exact set matches (0 to 1)
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")

    correct = sum(1 for p, r in zip(predictions, references, strict=True) if set(p) == set(r))
    return float(correct / len(predictions)) if predictions else 0.0


def f1_macro(
    predictions: list[int | str],
    references: list[int | str],
    labels: list[int | str] | None = None,
) -> float:
    """
    Calculate macro-averaged F1 score for classification.

    Args:
        predictions: Predicted labels
        references: Ground truth labels
        labels: Explicit label set for F1 computation. When provided, only these
            labels are used as classes (avoids auto-detecting spurious classes).

    Returns:
        Macro-averaged F1 score
    """
    from sklearn.metrics import f1_score

    return float(f1_score(references, predictions, labels=labels, average="macro", zero_division=0))


def rouge_1(predictions: list[str], references: list[str]) -> float:
    """
    Calculate ROUGE-1 F1 score (unigram overlap).

    Args:
        predictions: Predicted texts
        references: Reference texts

    Returns:
        Average ROUGE-1 F1 score

    Requires:
        rouge-score library (install with: pip install rouge-score)
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references, strict=False):
        score = scorer.score(ref, pred)
        scores.append(score["rouge1"].fmeasure)
    return float(np.mean(scores))


# Cache for BLEURT model (avoid reloading for each task)
_bleurt_cache = {"model": None}


def bleurt_score(predictions: list[str], references: list[str]) -> float:
    """
    Calculate BLEURT score for text generation quality.

    Args:
        predictions: Predicted texts
        references: Reference texts

    Returns:
        Average BLEURT score

    Note:
        Requires the `evaluate` library and BLEURT model.
        Raises if BLEURT cannot be loaded or scored — no silent fallback.
        Model is cached after first load for efficiency.
    """
    if _bleurt_cache["model"] is None:
        import os

        import tensorflow as tf

        if not os.environ.get("XLA_FLAGS"):
            tf.config.set_visible_devices([], "GPU")

        print("Loading BLEURT model (first time, will be cached)...")

        _bleurt_local_path = os.environ.get("BEHAVIORBENCH_BLEURT_PATH")
        try:
            from bleurt import score as bleurt_scorer

            if not _bleurt_local_path:
                raise RuntimeError(
                    "Set BEHAVIORBENCH_BLEURT_PATH to a local BLEURT checkpoint, "
                    "or install/use the evaluate fallback."
                )
            _bleurt_cache["model"] = bleurt_scorer.BleurtScorer(_bleurt_local_path)
            _bleurt_cache["mode"] = "direct"
        except (ImportError, RuntimeError):
            import evaluate

            _bleurt_cache["model"] = evaluate.load("bleurt", "BLEURT-20")
            _bleurt_cache["mode"] = "evaluate"

    if _bleurt_cache.get("mode") == "direct":
        scores = _bleurt_cache["model"].score(references=references, candidates=predictions)
        return float(np.mean(scores))
    else:
        results = _bleurt_cache["model"].compute(predictions=predictions, references=references)
        return float(np.mean(results["scores"]))


def mae_per_output(
    predictions: list[list[float]],
    references: list[list[float]],
    element_names: list[str] | None = None,
) -> dict:
    """
    Calculate Mean Absolute Error per output element for multi-value predictions.

    Each prediction and reference is a list of floats (e.g., [expenditure, income]).
    MAE is computed independently for each element position.

    Args:
        predictions: List of predicted value lists (each inner list has the same length)
        references: List of ground truth value lists (same structure as predictions)
        element_names: Optional names for each element position. If provided, dict keys
            use these names; otherwise keys are "element_0", "element_1", etc.

    Returns:
        Dictionary with per-element MAE and overall mean, e.g.:
        {"expenditure": 123.4, "income": 567.8, "mean": 345.6}
    """
    preds = np.array(predictions)
    refs = np.array(references)

    if preds.ndim != 2 or refs.ndim != 2:
        raise ValueError(
            "mae_per_output requires 2D inputs (list of lists). "
            f"Got predictions shape {preds.shape}, references shape {refs.shape}"
        )

    num_elements = preds.shape[1]

    if element_names is not None and len(element_names) != num_elements:
        raise ValueError(
            f"element_names length ({len(element_names)}) does not match "
            f"number of output elements ({num_elements})"
        )

    result = {}
    per_element_maes = []

    for i in range(num_elements):
        col_mae = float(np.mean(np.abs(preds[:, i] - refs[:, i])))
        name = element_names[i] if element_names else f"element_{i}"
        result[name] = col_mae
        per_element_maes.append(col_mae)

    result["mean"] = float(np.mean(per_element_maes))
    return result


def maer_per_output(
    predictions: list[list[float]],
    references: list[list[float]],
    element_names: list[str] | None = None,
) -> dict:
    """
    Mean Absolute Error Ratio (MAPE) per output element.

    For each element position, compute mean(|pred_i - ref_i| / |ref_i|) over
    observations where ref_i != 0. Reports n_used and n_skipped per element.

    Args:
        predictions: List of predicted value lists
        references: List of ground truth value lists
        element_names: Optional names for each element position

    Returns:
        Dictionary with per-element MAER, overall mean, and skip counts
    """
    preds = np.array(predictions)
    refs = np.array(references)

    if preds.ndim != 2 or refs.ndim != 2:
        raise ValueError(
            "maer_per_output requires 2D inputs (list of lists). "
            f"Got predictions shape {preds.shape}, references shape {refs.shape}"
        )

    num_elements = preds.shape[1]

    if element_names is not None and len(element_names) != num_elements:
        raise ValueError(
            f"element_names length ({len(element_names)}) does not match "
            f"number of output elements ({num_elements})"
        )

    result = {}
    per_element_maers = []

    for i in range(num_elements):
        name = element_names[i] if element_names else f"element_{i}"
        nonzero_mask = refs[:, i] != 0
        n_used = int(nonzero_mask.sum())
        n_skipped = int((~nonzero_mask).sum())

        if n_used > 0:
            ratios = np.abs(preds[nonzero_mask, i] - refs[nonzero_mask, i]) / np.abs(
                refs[nonzero_mask, i]
            )
            col_maer = float(np.mean(ratios))
        else:
            col_maer = float("nan")

        result[name] = col_maer
        result[f"n_used_{name}"] = n_used
        result[f"n_skipped_{name}"] = n_skipped
        per_element_maers.append(col_maer)

    result["mean"] = float(np.nanmean(per_element_maers))
    return result


def normalized_mae_per_output(
    predictions: list[list[float]],
    references: list[list[float]],
    element_names: list[str] | None = None,
) -> dict:
    """
    SD-normalized MAE per output element.

    Computes MAE_k / SD_k for each element k, where SD_k is the standard
    deviation of actual (reference) values. The mean_normalized_mae is
    (1/K) * sum(MAE_k / SD_k) for K elements.

    Args:
        predictions: List of predicted value lists
        references: List of ground truth value lists
        element_names: Optional names for each element position

    Returns:
        Dictionary with per-element MAE/SD ratios and mean_normalized_mae
    """
    preds = np.array(predictions)
    refs = np.array(references)

    if preds.ndim != 2 or refs.ndim != 2:
        raise ValueError(
            "normalized_mae_per_output requires 2D inputs (list of lists). "
            f"Got predictions shape {preds.shape}, references shape {refs.shape}"
        )

    num_elements = preds.shape[1]

    if element_names is not None and len(element_names) != num_elements:
        raise ValueError(
            f"element_names length ({len(element_names)}) does not match "
            f"number of output elements ({num_elements})"
        )

    result = {}
    mae_over_sd_ratios = []

    for i in range(num_elements):
        name = element_names[i] if element_names else f"element_{i}"
        col_mae = float(np.mean(np.abs(preds[:, i] - refs[:, i])))
        col_sd = float(np.std(refs[:, i], ddof=0))

        if col_sd > 0:
            ratio = col_mae / col_sd
        else:
            ratio = float("nan")

        result[f"{name}_mae_over_sd"] = ratio
        mae_over_sd_ratios.append(ratio)

    result["mean_normalized_mae"] = float(np.nanmean(mae_over_sd_ratios))
    return result


def mald_per_output(
    predictions: list[list[float]],
    references: list[list[float]],
    element_names: list[str] | None = None,
) -> dict:
    """
    Mean Absolute Log Difference per output element.

    Computes MALD_k = mean(|log(pred_k) - log(ref_k)|) for each element k,
    using only samples where both pred and ref are strictly positive.

    Args:
        predictions: List of predicted value lists
        references: List of ground truth value lists
        element_names: Optional names for each element position

    Returns:
        Dictionary with per-element MALD, counts, and mean MALD
    """
    preds = np.array(predictions)
    refs = np.array(references)

    if preds.ndim != 2 or refs.ndim != 2:
        raise ValueError(
            "mald_per_output requires 2D inputs (list of lists). "
            f"Got predictions shape {preds.shape}, references shape {refs.shape}"
        )

    num_elements = preds.shape[1]

    if element_names is not None and len(element_names) != num_elements:
        raise ValueError(
            f"element_names length ({len(element_names)}) does not match "
            f"number of output elements ({num_elements})"
        )

    result = {}
    per_element_malds = []

    for i in range(num_elements):
        name = element_names[i] if element_names else f"element_{i}"
        pos_mask = (preds[:, i] > 0) & (refs[:, i] > 0)
        n_used = int(pos_mask.sum())
        n_skipped = int((~pos_mask).sum())

        if n_used > 0:
            col_mald = float(
                np.mean(np.abs(np.log(preds[pos_mask, i]) - np.log(refs[pos_mask, i])))
            )
        else:
            col_mald = float("nan")

        result[name] = col_mald
        result[f"n_used_{name}"] = n_used
        result[f"n_skipped_{name}"] = n_skipped
        per_element_malds.append(col_mald)

    result["mean"] = float(np.nanmean(per_element_malds))
    return result


def exact_match_list(predictions: list[list[float]], references: list[list[float]]) -> float:
    """
    Calculate exact match accuracy for list predictions.

    Two lists match exactly if they contain the same values (order-independent).

    Args:
        predictions: List of predicted value lists
        references: List of ground truth value lists

    Returns:
        Proportion of exact matches (0 to 1)
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")

    matches = 0
    for pred, ref in zip(predictions, references, strict=False):
        # Sort both lists and compare
        pred_sorted = sorted(pred)
        ref_sorted = sorted(ref)
        if pred_sorted == ref_sorted:
            matches += 1

    return float(matches / len(predictions)) if predictions else 0.0


def bstar_accuracy(
    predictions: list[list[float]],
    references: list[list[float]],
    value_range: tuple[float, float] = (0.0, 1.0),
) -> float:
    """
    Calculate bstar accuracy metric (prediction to reference only).

    For each data point:
    1. Compute average distance from each prediction to closest ground truth
    2. Normalize by value range
    3. Accuracy = 1 - normalized_distance

    Args:
        predictions: List of predicted value lists
        references: List of ground truth value lists
        value_range: (min, max) value range for normalization (default: (0, 1))

    Returns:
        Average bstar accuracy across all data points (0 to 1)
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")

    a, b = value_range
    range_size = b - a

    if range_size <= 0:
        raise ValueError("Invalid value range")

    accuracies = []

    for pred_list, ref_list in zip(predictions, references, strict=False):
        if not pred_list or not ref_list:
            # If either list is empty, accuracy is 0
            accuracies.append(0.0)
            continue

        # Distance from predictions to closest ground truth
        pred_to_ref_distances = []
        for p in pred_list:
            min_dist = min(abs(p - g) for g in ref_list)
            pred_to_ref_distances.append(min_dist)
        avg_pred_to_ref = sum(pred_to_ref_distances) / len(pred_to_ref_distances)

        # Normalize and compute accuracy
        normalized_distance = avg_pred_to_ref / range_size

        # Accuracy = 1 - normalized_distance
        acc = max(0.0, 1.0 - normalized_distance)
        accuracies.append(acc)

    return float(np.mean(accuracies)) if accuracies else 0.0


def win_rate(
    predictions: list[float],
    references: list[float],
    other_players_choices_list: list[list[float]] | None = None,
    **kwargs,
) -> float:
    """
    Calculate beauty contest win rate.

    For each sample, determine whether the model's prediction would win the
    beauty contest (be closest to 2/3 of the group average) compared to the
    other players.

    Args:
        predictions: Model's guessed numbers
        references: Human player's actual choices (unused for win calculation,
            but kept for API consistency)
        other_players_choices_list: List of lists, where each inner list
            contains the other players' choices for that round

    Returns:
        Fraction of rounds where the model's guess wins (0 to 1)
    """
    if other_players_choices_list is None:
        raise ValueError("win_rate requires other_players_choices_list")

    wins = 0
    valid = 0
    for pred, others in zip(predictions, other_players_choices_list, strict=True):
        if not others:
            continue
        all_choices = [pred] + others
        group_avg = sum(all_choices) / len(all_choices)
        target = (2.0 / 3.0) * group_avg

        model_dist = abs(pred - target)
        min_other_dist = min(abs(c - target) for c in others)

        if model_dist <= min_other_dist:
            wins += 1
        valid += 1

    return float(wins / valid) if valid else 0.0


def human_win_rate(
    predictions: list[float],
    references: list[float],
    other_players_choices_list: list[list[float]] | None = None,
    **kwargs,
) -> float:
    """
    Calculate beauty contest win rate using the human's actual choice (baseline).

    Same logic as win_rate, but uses the human reference value instead of
    the model prediction.

    Args:
        predictions: Model's guessed numbers (unused)
        references: Human player's actual choices
        other_players_choices_list: List of lists, where each inner list
            contains the other players' choices for that round

    Returns:
        Fraction of rounds where the human's choice wins (0 to 1)
    """
    if other_players_choices_list is None:
        raise ValueError("human_win_rate requires other_players_choices_list")

    wins = 0
    valid = 0
    for ref, others in zip(references, other_players_choices_list, strict=True):
        if not others:
            continue
        all_choices = [ref] + others
        group_avg = sum(all_choices) / len(all_choices)
        target = (2.0 / 3.0) * group_avg

        human_dist = abs(ref - target)
        min_other_dist = min(abs(c - target) for c in others)

        if human_dist <= min_other_dist:
            wins += 1
        valid += 1

    return float(wins / valid) if valid else 0.0


# Dictionary mapping metric names to functions
METRICS = {
    "mae": mean_absolute_error,
    "MAE": mean_absolute_error,
    "MAE_with_normalization": mae_with_normalization,
    "spearman": spearman_correlation,
    "Spearman": spearman_correlation,
    "Spearman_with_pvalue": spearman_correlation_with_pvalue,
    "wasserstein": wasserstein,
    "Wasserstein": wasserstein,
    "Wasserstein_with_ks": wasserstein_with_ks_test,
    "accuracy": accuracy,
    "Accuracy": accuracy,
    "accuracy_multi_choice": accuracy_multi_choice,
    "f1_macro": f1_macro,
    "F1_macro": f1_macro,
    "rouge1": rouge_1,
    "ROUGE-1": rouge_1,
    "bleurt": bleurt_score,
    "BLEURT": bleurt_score,
    "MAE_per_output": mae_per_output,
    "MAER_per_output": maer_per_output,
    "normalized_MAE_per_output": normalized_mae_per_output,
    "MALD_per_output": mald_per_output,
    "exact_match_list": exact_match_list,
    "bstar_accuracy": bstar_accuracy,
    "win_rate": win_rate,
    "human_win_rate": human_win_rate,
}


def compute_metrics(
    metric_names: list[str],
    predictions: list[float | int | str],
    references: list[float | int | str],
    groups: list[str] | None = None,
    metric_kwargs: dict | None = None,
    group_averaging: str = "macro",
) -> dict[str, float | dict]:
    """
    Compute multiple metrics at once, optionally grouped.

    Args:
        metric_names: List of metric names to compute
        predictions: Model predictions
        references: Ground truth values
        groups: Optional list of group labels (same length as predictions).
                If provided, metrics are computed per-group and averaged.
                This is important for BigFive where samples are grouped by dimension.
        metric_kwargs: Optional dict of additional keyword arguments to pass to metric
                functions that support them (e.g., value_range, normalize_range for
                Wasserstein normalization).
        group_averaging: "macro" (default) averages per-group metrics for top-level;
                "micro" computes top-level on all samples combined, with per-group as
                supplementary breakdown.

    Returns:
        Dictionary mapping metric names to computed values.
        Some metrics (like Spearman_with_pvalue, Wasserstein_with_ks) return
        dictionaries with additional statistical information.

    Raises:
        ValueError: If an unknown metric name is provided
    """
    # If groups provided, compute per-group metrics and average
    if groups is not None:
        return _compute_metrics_grouped(
            metric_names,
            predictions,
            references,
            groups,
            metric_kwargs,
            group_averaging=group_averaging,
        )

    if metric_kwargs is None:
        metric_kwargs = {}

    results = {}
    for metric_name in metric_names:
        if metric_name not in METRICS:
            raise ValueError(
                f"Unknown metric: {metric_name}. Available metrics: {list(METRICS.keys())}"
            )
        metric_func = METRICS[metric_name]

        # Try to pass metric_kwargs to functions that accept them
        try:
            result = metric_func(predictions, references, **metric_kwargs)
        except TypeError:
            # Function doesn't accept these kwargs, call without them
            result = metric_func(predictions, references)

        # If result is a dict, flatten it with metric_name prefix
        if isinstance(result, dict):
            for key, value in result.items():
                results[f"{metric_name}_{key}"] = value
        else:
            results[metric_name] = result

    return results


def _compute_metrics_grouped(
    metric_names: list[str],
    predictions: list[float | int | str],
    references: list[float | int | str],
    groups: list[str],
    metric_kwargs: dict | None = None,
    group_averaging: str = "macro",
) -> dict[str, float | dict]:
    """
    Compute metrics per-group, with configurable top-level aggregation.

    Two modes:
    - "macro" (default): Top-level metrics are macro-averaged across groups.
      Used for BigFive where per-dimension averaging avoids spurious correlations.
    - "micro": Top-level metrics are computed on all samples combined (ignoring groups).
      Per-group metrics are included as supplementary breakdown in _per_group.
      Used for workflow where AER+NHB should be treated as one dataset.

    Args:
        metric_names: List of metric names to compute
        predictions: Model predictions
        references: Ground truth values
        groups: Group labels for each sample
        metric_kwargs: Optional dict of additional keyword arguments to pass to metrics
        group_averaging: "macro" or "micro" aggregation strategy

    Returns:
        Dictionary with top-level metrics plus per-group breakdown
    """
    from collections import defaultdict

    if metric_kwargs is None:
        metric_kwargs = {}

    # Group data
    grouped_preds = defaultdict(list)
    grouped_refs = defaultdict(list)

    for pred, ref, group in zip(predictions, references, groups, strict=False):
        grouped_preds[group].append(pred)
        grouped_refs[group].append(ref)

    # Compute metrics per group
    per_group_results = {}
    for group in sorted(grouped_preds.keys()):
        group_preds = grouped_preds[group]
        group_refs = grouped_refs[group]

        group_metrics = {}
        for metric_name in metric_names:
            if metric_name not in METRICS:
                raise ValueError(f"Unknown metric: {metric_name}")

            metric_func = METRICS[metric_name]

            # Try to pass metric_kwargs to functions that accept them
            try:
                result = metric_func(group_preds, group_refs, **metric_kwargs)
            except TypeError:
                # Function doesn't accept these kwargs, call without them
                result = metric_func(group_preds, group_refs)

            if isinstance(result, dict):
                for key, value in result.items():
                    group_metrics[f"{metric_name}_{key}"] = value
            else:
                group_metrics[metric_name] = result

        per_group_results[group] = group_metrics

    if group_averaging == "micro":
        # Micro: compute top-level metrics on ALL samples combined
        overall_results = {}
        for metric_name in metric_names:
            if metric_name not in METRICS:
                raise ValueError(f"Unknown metric: {metric_name}")

            metric_func = METRICS[metric_name]

            try:
                result = metric_func(predictions, references, **metric_kwargs)
            except TypeError:
                result = metric_func(predictions, references)

            if isinstance(result, dict):
                for key, value in result.items():
                    overall_results[f"{metric_name}_{key}"] = value
            else:
                overall_results[metric_name] = result

        # Add per-group breakdown
        overall_results["_per_group"] = per_group_results
        return overall_results

    # Macro (default): average across groups
    all_metric_keys = set()
    for group_metrics in per_group_results.values():
        all_metric_keys.update(group_metrics.keys())

    # Significance fields use AND logic: 1 only if ALL groups have value 1
    # This follows the paper convention:
    # - ‡ indicates correlation is significant (p < 0.05) for all dimensions
    # - † indicates KS test passes (p > 0.05) for all dimensions
    significance_suffixes = ("_significant", "_ks_pass_test")

    averaged_results = {}
    for metric_key in sorted(all_metric_keys):
        values = [
            per_group_results[group][metric_key]
            for group in per_group_results
            if metric_key in per_group_results[group]
        ]

        if metric_key.endswith(significance_suffixes):
            # AND logic: 1 only if all groups pass
            # Rename to make AND logic clear: _significant -> _all_significant
            new_key = metric_key.replace("_significant", "_all_significant")
            new_key = new_key.replace("_ks_pass_test", "_all_ks_pass_test")
            averaged_results[new_key] = 1 if all(v == 1 for v in values) else 0
        else:
            # Spearman correlation: NaN means constant predictions (no variance),
            # treat as 0.0 (no predictive signal) rather than skipping.
            # All other metrics: NaN is unexpected, raise an error.
            spearman_prefixes = ("Spearman_with_pvalue_correlation", "Spearman_with_pvalue_pvalue")
            has_nan = any(np.isnan(v) for v in values if isinstance(v, float))
            if has_nan:
                if metric_key.startswith(spearman_prefixes):
                    values = [0.0 if (isinstance(v, float) and np.isnan(v)) else v for v in values]
                else:
                    nan_groups = [
                        group
                        for group in per_group_results
                        if metric_key in per_group_results[group]
                        and isinstance(per_group_results[group][metric_key], float)
                        and np.isnan(per_group_results[group][metric_key])
                    ]
                    raise ValueError(
                        f"Unexpected NaN in metric '{metric_key}' for groups: {nan_groups}. "
                        f"Only Spearman correlation is expected to have NaN values."
                    )
            averaged_results[f"{metric_key}_averaged"] = float(np.mean(values))

    # Add per-group breakdown
    averaged_results["_per_group"] = per_group_results

    return averaged_results
