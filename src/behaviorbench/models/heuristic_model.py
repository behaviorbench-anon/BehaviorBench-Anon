"""Heuristic baseline models for prediction tasks."""

import re


class HeuristicModel:
    """Heuristic baseline that predicts from prompt history without an LLM.

    Strategies:
        random_walk: predict the last observed value for each variable (CEX)
        individual_fe: predict the mean of all observed values (CEX)
        multiround_avg: predict mean of all previous round choices (MobLab multiround)
    """

    ALLOWED_STRATEGIES = ("random_walk", "individual_fe", "multiround_avg")

    def __init__(self, strategy: str = "random_walk"):
        if strategy not in self.ALLOWED_STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}. Use one of {self.ALLOWED_STRATEGIES}.")
        self.strategy = strategy

    # ── CEX helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_cex_history(text: str) -> tuple[list[float], list[float]]:
        """Extract expenditure and income history lists from CEX prompt text.

        Supports level, change, and growth prompt formats.
        """
        # Cumulative format: "expenditure (quarterly dollars) was 16626, 7095, 3902"
        exp_match = re.search(r"expenditure \(quarterly dollars\) was ([\d,. ]+?)(?:,)? and", text)
        if exp_match:
            inc_match = re.search(r"income \(quarterly dollars\) was ([\d., ]+)\.", text)
            if not inc_match:
                raise ValueError(f"Could not parse CEX income from level prompt: {text[:200]}")

            def _to_floats(s: str) -> list[float]:
                return [float(v.replace(",", "")) for v in s.split(",") if v.strip()]

            return _to_floats(exp_match.group(1)), _to_floats(inc_match.group(1))

        # Change format: "expenditure was 16626 dollars" (with or without "changed by")
        init_exp = re.search(r"expenditure was ([\d.]+) dollars", text)
        if init_exp and "income was" in text and "quarterly dollars" not in text:
            init_inc = re.search(r"income was ([\d.]+) dollars", text)
            exp_vals = [float(init_exp.group(1))]
            inc_vals = [float(init_inc.group(1))] if init_inc else [0.0]
            exp_changes = re.findall(r"expenditure changed by (-?[\d.]+) dollars", text)
            inc_changes = re.findall(r"income changed by (-?[\d.]+) dollars", text)
            exp_vals.extend(float(v) for v in exp_changes)
            inc_vals.extend(float(v) for v in inc_changes)
            return exp_vals, inc_vals

        # Growth format: "expenditure growth rate was -57.33%"
        exp_growth = re.findall(r"expenditure growth rate was (-?[\d.]+)%", text)
        inc_growth = re.findall(r"income growth rate was (-?[\d.]+)%", text)
        if exp_growth or inc_growth:
            return [float(v) for v in exp_growth], [float(v) for v in inc_growth]

        # Growth single-lag: no prior growth rates, only demographics + macro
        # Detected by the question asking for growth rates
        if "growth_pct" in text or "growth rates" in text:
            return [], []

        raise ValueError(f"Could not parse CEX history from prompt: {text[:200]}")

    # ── Multiround helpers ───────────────────────────────────────────────

    @staticmethod
    def _parse_multiround_history(text: str) -> list[float]:
        """Extract round choice values from multiround prompt text.

        Parses lines like: ``- Round 1: Your choice: [40].``
        """
        return [
            float(v) for v in re.findall(r"- Round \d+: Your choice: \[(\d+(?:\.\d+)?)\]", text)
        ]

    # ── Predict dispatch ─────────────────────────────────────────────────

    def _predict(self, text: str) -> str:
        if self.strategy == "multiround_avg":
            return self._predict_multiround(text)
        return self._predict_cex(text)

    def _predict_cex(self, text: str) -> str:
        exp_hist, inc_hist = self._parse_cex_history(text)

        # For diff/growth formats, random_walk predicts zero change
        # (the last observed *change* is not a good heuristic; zero change is the RW prediction)
        is_change = (
            "expenditure was" in text
            and "quarterly dollars" not in text
            and "growth rate" not in text
        )
        is_growth = "growth rate was" in text or "growth_pct" in text or "growth rates" in text

        if is_change or is_growth:
            # Random walk: predict zero change / zero growth
            # Individual FE: predict mean of observed changes/growth rates
            if self.strategy == "random_walk":
                exp_pred = 0.0
                inc_pred = 0.0
            else:  # individual_fe
                exp_pred = sum(exp_hist) / len(exp_hist) if exp_hist else 0.0
                inc_pred = sum(inc_hist) / len(inc_hist) if inc_hist else 0.0
        else:
            # Cumulative: standard behavior
            if self.strategy == "random_walk":
                exp_pred = exp_hist[-1]
                inc_pred = inc_hist[-1]
            else:  # individual_fe
                exp_pred = sum(exp_hist) / len(exp_hist)
                inc_pred = sum(inc_hist) / len(inc_hist)

        return f"[{exp_pred:.2f}, {inc_pred:.2f}]"

    def _predict_multiround(self, text: str) -> str:
        rounds = self._parse_multiround_history(text)
        if not rounds:
            return "[0]"
        mean_val = sum(rounds) / len(rounds)
        # Return integer if close to integer, otherwise 1 decimal
        if abs(mean_val - round(mean_val)) < 0.01:
            return f"[{int(round(mean_val))}]"
        return f"[{mean_val:.1f}]"

    def __call__(self, prompt: str | dict[str, str], **kwargs) -> str:
        """Return prediction string from a single prompt."""
        if isinstance(prompt, dict):
            text = prompt.get("user", "") + " " + prompt.get("system", "")
        else:
            text = prompt
        return self._predict(text)

    def batch_call(self, prompts: list[str | dict[str, str]]) -> list[str]:
        """Process all prompts."""
        return [self(p) for p in prompts]
