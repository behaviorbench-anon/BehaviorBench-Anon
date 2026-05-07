"""
Tests for utility functions.
"""

import tempfile
from pathlib import Path

from behaviorbench.eval.utils import (
    load_json,
    load_jsonl,
    parse_choice_output,
    parse_numeric_list_output,
    parse_numeric_output,
    save_json,
    save_jsonl,
)
from behaviorbench.models.api_model import CentaurLocalModel


class TestDataLoading:
    """Test suite for data loading utilities."""

    def test_load_save_json(self):
        """Test JSON loading and saving."""
        data = {"key": "value", "number": 42}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            save_json(data, temp_path)
            loaded = load_json(temp_path)
            assert loaded == data
        finally:
            Path(temp_path).unlink()

    def test_load_save_jsonl(self):
        """Test JSONL loading and saving."""
        data = [{"id": 1, "value": "a"}, {"id": 2, "value": "b"}]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            temp_path = f.name

        try:
            save_jsonl(data, temp_path)
            loaded = load_jsonl(temp_path)
            assert loaded == data
        finally:
            Path(temp_path).unlink()


class TestParsing:
    """Test suite for output parsing utilities."""

    def test_parse_numeric_output_brackets(self):
        """Test parsing numeric output with brackets."""
        assert parse_numeric_output("[42]") == 42.0
        assert parse_numeric_output("[3.14]") == 3.14

    def test_parse_numeric_output_plain(self):
        """Test parsing plain numeric output."""
        assert parse_numeric_output("42") == 42.0
        assert parse_numeric_output("3.14") == 3.14

    def test_parse_numeric_output_embedded(self):
        """Test parsing numeric output embedded in text."""
        assert parse_numeric_output("The answer is 42") == 42.0
        assert parse_numeric_output("Score: 3.14 out of 5") == 3.14

    def test_parse_numeric_output_none(self):
        """Test parsing non-numeric output."""
        assert parse_numeric_output("no numbers here") is None

    def test_parse_numeric_output_negative(self):
        """Negative numbers must be preserved (regression test for sign-strip bug)."""
        assert parse_numeric_output("[-50]") == -50.0
        assert parse_numeric_output("[$-50]") == -50.0
        assert parse_numeric_output("answer is -7") == -7.0
        assert parse_numeric_output("-3.14") == -3.14

    def test_parse_numeric_list_output_positive(self):
        """List parsing handles positive values."""
        assert parse_numeric_list_output("[5135, 14015]") == [5135.0, 14015.0]
        assert parse_numeric_list_output("[3.14]") == [3.14]
        assert parse_numeric_list_output("\\boxed{2.5}") == [2.5]

    def test_parse_numeric_list_output_negative(self):
        """List parsing must preserve negative values (regression test)."""
        assert parse_numeric_list_output("[5135, -14015]") == [5135.0, -14015.0]
        assert parse_numeric_list_output("[-2591, -52000]") == [-2591.0, -52000.0]
        assert parse_numeric_list_output("```json\n[26.12, -7.9]\n```") == [26.12, -7.9]
        assert parse_numeric_list_output("\\boxed{-50.5}") == [-50.5]
        assert parse_numeric_list_output("Final: [-20.00, 0.00]") == [-20.0, 0.0]

    def test_parse_numeric_list_output_empty(self):
        """List parsing returns empty list when no numbers found."""
        assert parse_numeric_list_output("no numbers here at all") == []
        assert parse_numeric_list_output("Final answer: [null, null]") == []
        assert parse_numeric_list_output("History has $10,684. Final answer: [null, null]") == []

    def test_parse_choice_output_simple(self):
        """Test parsing simple choice output."""
        assert parse_choice_output("A") == "A"
        assert parse_choice_output("B") == "B"

    def test_parse_choice_output_with_text(self):
        """Test parsing choice output with additional text."""
        assert parse_choice_output("The answer is A") == "A"
        assert parse_choice_output("I choose B") == "B"

    def test_parse_choice_output_with_choices(self):
        """Test parsing choice output with valid choices."""
        assert parse_choice_output("The answer is APPLE", ["apple", "banana"]) == "apple"

    def test_parse_choice_output_none(self):
        """Test parsing invalid choice output."""
        assert parse_choice_output("no choice here") is None

    def test_centaur_completion_normalization(self):
        """Centaur wrapper should not duplicate completion close markers."""
        assert CentaurLocalModel._normalize_completion_text("32.5") == "<<32.5>>"
        assert CentaurLocalModel._normalize_completion_text("32.5>>\n") == "<<32.5>>"
        assert CentaurLocalModel._normalize_completion_text("<<32.5>>\n>>") == "<<32.5>>"
        assert CentaurLocalModel._normalize_completion_text("<<A>>>>") == "<<A>>"
