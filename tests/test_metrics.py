"""
Tests for evaluation metrics.
"""

import pytest

from behaviorbench.eval.metrics import (
    accuracy,
    compute_metrics,
    mae_per_output,
    mae_with_normalization,
    mean_absolute_error,
    rouge_1,
    spearman_correlation,
    wasserstein,
)


class TestMetrics:
    """Test suite for evaluation metrics."""

    def test_mean_absolute_error(self):
        """Test MAE calculation."""
        predictions = [1.0, 2.0, 3.0, 4.0, 5.0]
        references = [1.5, 2.5, 3.5, 4.5, 5.5]

        mae = mean_absolute_error(predictions, references)
        assert mae == pytest.approx(0.5, abs=1e-6)

    def test_mean_absolute_error_perfect(self):
        """Test MAE with perfect predictions."""
        predictions = [1.0, 2.0, 3.0]
        references = [1.0, 2.0, 3.0]

        mae = mean_absolute_error(predictions, references)
        assert mae == pytest.approx(0.0, abs=1e-6)

    def test_mae_with_normalization(self):
        """Test MAE with normalization returns both raw and normalized values."""
        predictions = [0.0, 10.0, 20.0]
        references = [5.0, 15.0, 25.0]
        # Raw MAE = 5.0
        # With value_range [0,20] -> normalize_range [0,100]: 5.0 * 100/20 = 25.0
        result = mae_with_normalization(
            predictions,
            references,
            value_range=(0, 20),
            normalize_range=(0, 100),
        )
        assert result["mae_raw"] == pytest.approx(5.0, abs=1e-6)
        assert result["mae"] == pytest.approx(25.0, abs=1e-6)

    def test_mae_with_normalization_no_ranges(self):
        """Test MAE without normalization ranges returns same value for both."""
        predictions = [1.0, 2.0, 3.0]
        references = [1.5, 2.5, 3.5]
        result = mae_with_normalization(predictions, references)
        assert result["mae_raw"] == pytest.approx(0.5, abs=1e-6)
        assert result["mae"] == pytest.approx(0.5, abs=1e-6)

    def test_mae_with_normalization_same_range(self):
        """Test MAE with same input and output range keeps value unchanged."""
        predictions = [0.0, 50.0, 100.0]
        references = [10.0, 60.0, 90.0]
        # Raw MAE = 10.0
        # With value_range [0,100] -> normalize_range [0,100]: 10.0 * 100/100 = 10.0
        result = mae_with_normalization(
            predictions,
            references,
            value_range=(0, 100),
            normalize_range=(0, 100),
        )
        assert result["mae_raw"] == pytest.approx(10.0, abs=1e-6)
        assert result["mae"] == pytest.approx(10.0, abs=1e-6)

    def test_spearman_correlation(self):
        """Test Spearman correlation."""
        predictions = [1, 2, 3, 4, 5]
        references = [1, 2, 3, 4, 5]

        corr = spearman_correlation(predictions, references)
        assert corr == pytest.approx(1.0, abs=1e-6)

    def test_spearman_correlation_negative(self):
        """Test Spearman correlation with negative relationship."""
        predictions = [1, 2, 3, 4, 5]
        references = [5, 4, 3, 2, 1]

        corr = spearman_correlation(predictions, references)
        assert corr == pytest.approx(-1.0, abs=1e-6)

    def test_wasserstein_identical(self):
        """Test Wasserstein distance with identical distributions."""
        dist1 = [1, 2, 3, 4, 5]
        dist2 = [1, 2, 3, 4, 5]

        distance = wasserstein(dist1, dist2)
        assert distance == pytest.approx(0.0, abs=1e-6)

    def test_wasserstein_shifted(self):
        """Test Wasserstein distance with shifted distribution."""
        dist1 = [1, 2, 3, 4, 5]
        dist2 = [2, 3, 4, 5, 6]

        distance = wasserstein(dist1, dist2)
        assert distance > 0

    def test_accuracy_perfect(self):
        """Test accuracy with perfect predictions."""
        predictions = ["A", "B", "C", "D"]
        references = ["A", "B", "C", "D"]

        acc = accuracy(predictions, references)
        assert acc == pytest.approx(1.0, abs=1e-6)

    def test_accuracy_partial(self):
        """Test accuracy with partial correct predictions."""
        predictions = ["A", "B", "C", "D"]
        references = ["A", "B", "X", "Y"]

        acc = accuracy(predictions, references)
        assert acc == pytest.approx(0.5, abs=1e-6)

    def test_rouge_1_perfect(self):
        """Test ROUGE-1 with perfect match."""
        predictions = ["the quick brown fox"]
        references = ["the quick brown fox"]

        score = rouge_1(predictions, references)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_rouge_1_partial(self):
        """Test ROUGE-1 with partial overlap."""
        predictions = ["the quick brown fox"]
        references = ["the slow brown dog"]

        score = rouge_1(predictions, references)
        assert 0 < score < 1

    def test_compute_metrics_multiple(self):
        """Test computing multiple metrics at once."""
        predictions = [1.0, 2.0, 3.0, 4.0, 5.0]
        references = [1.5, 2.5, 3.5, 4.5, 5.5]

        metrics = compute_metrics(
            metric_names=["MAE", "Spearman"],
            predictions=predictions,
            references=references,
        )

        assert "MAE" in metrics
        assert "Spearman" in metrics
        assert metrics["MAE"] == pytest.approx(0.5, abs=1e-6)

    def test_compute_metrics_unknown(self):
        """Test that unknown metric raises ValueError."""
        with pytest.raises(ValueError, match="Unknown metric"):
            compute_metrics(
                metric_names=["InvalidMetric"],
                predictions=[1, 2, 3],
                references=[1, 2, 3],
            )


class TestMaePerOutput:
    """Test suite for per-output MAE metric."""

    def test_basic_computation(self):
        """Test basic per-element MAE computation."""
        predictions = [[10, 20], [30, 40], [50, 60]]
        references = [[12, 18], [28, 42], [55, 65]]
        result = mae_per_output(predictions, references)
        # element_0: |10-12|+|30-28|+|50-55| / 3 = (2+2+5)/3 = 3.0
        # element_1: |20-18|+|40-42|+|60-65| / 3 = (2+2+5)/3 = 3.0
        assert result["element_0"] == pytest.approx(3.0, abs=1e-6)
        assert result["element_1"] == pytest.approx(3.0, abs=1e-6)
        assert result["mean"] == pytest.approx(3.0, abs=1e-6)

    def test_named_elements(self):
        """Test per-element MAE with named elements."""
        predictions = [[100, 200], [300, 400]]
        references = [[110, 210], [310, 390]]
        result = mae_per_output(predictions, references, element_names=["expenditure", "income"])
        assert "expenditure" in result
        assert "income" in result
        assert result["expenditure"] == pytest.approx(10.0, abs=1e-6)
        # |200-210|+|400-390| / 2 = (10+10)/2 = 10.0
        assert result["income"] == pytest.approx(10.0, abs=1e-6)
        assert result["mean"] == pytest.approx(10.0, abs=1e-6)

    def test_wrong_length_element_names(self):
        """Test that mismatched element_names length raises ValueError."""
        predictions = [[1, 2], [3, 4]]
        references = [[1, 2], [3, 4]]
        with pytest.raises(ValueError, match="element_names length"):
            mae_per_output(predictions, references, element_names=["a", "b", "c"])

    def test_1d_input_raises_error(self):
        """Test that 1D input raises ValueError."""
        with pytest.raises(ValueError, match="requires 2D inputs"):
            mae_per_output([1, 2, 3], [4, 5, 6])

    def test_compute_metrics_dict_flattening(self):
        """Test that MAE_per_output integrates with compute_metrics dict flattening."""
        predictions = [[100, 200], [300, 400]]
        references = [[110, 190], [290, 410]]
        result = compute_metrics(
            metric_names=["MAE_per_output"],
            predictions=predictions,
            references=references,
            metric_kwargs={"element_names": ["expenditure", "income"]},
        )
        assert "MAE_per_output_expenditure" in result
        assert "MAE_per_output_income" in result
        assert "MAE_per_output_mean" in result
        assert result["MAE_per_output_expenditure"] == pytest.approx(10.0, abs=1e-6)
        assert result["MAE_per_output_income"] == pytest.approx(10.0, abs=1e-6)
