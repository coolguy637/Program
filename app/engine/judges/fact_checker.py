import logging
import re

from app.engine.judges.base import BaseJudge, JudgeVerdict

logger = logging.getLogger(__name__)


class FactChecker(BaseJudge):
    """Verifies factual claims, data consistency, and logical coherence in output."""

    name = "fact_checker"
    pass_threshold = 0.75

    HEDGING_PHRASES = [
        "I think", "I believe", "probably", "maybe", "might be",
        "I'm not sure", "it could be", "perhaps", "supposedly",
    ]

    CONTRADICTION_MARKERS = [
        (r"always\b.*\bnever\b", "Contradictory use of 'always' and 'never'"),
        (r"all\b.*\bnone\b", "Contradictory use of 'all' and 'none'"),
        (r"increases?\b.*\bdecreases?\b", "Potential contradiction: increase vs decrease"),
    ]

    NUMERIC_RE = re.compile(
        r"\b(\d+(?:\.\d+)?)\s*(%|percent|million|billion|thousand|kb|mb|gb)\b", re.I
    )

    async def evaluate(
        self, prompt: str, output: str, context: dict | None = None
    ) -> JudgeVerdict:
        issues: list[str] = []
        checks_passed = 0
        total_checks = 0

        # 1) Hedging analysis
        total_checks += 1
        hedges = [p for p in self.HEDGING_PHRASES if p.lower() in output.lower()]
        if len(hedges) > 3:
            issues.append(
                f"Excessive hedging ({len(hedges)} instances): {', '.join(hedges[:5])}. "
                "Replace uncertain language with verified facts."
            )
        else:
            checks_passed += 1

        # 2) Contradiction scan
        total_checks += 1
        contradictions = []
        for pattern, desc in self.CONTRADICTION_MARKERS:
            if re.search(pattern, output, re.I | re.S):
                contradictions.append(desc)
        if contradictions:
            issues.append("Potential contradictions: " + "; ".join(contradictions))
        else:
            checks_passed += 1

        # 3) Numeric consistency
        total_checks += 1
        numbers = self.NUMERIC_RE.findall(output)
        numeric_issues = self._check_numeric_consistency(numbers)
        if numeric_issues:
            issues.extend(numeric_issues)
        else:
            checks_passed += 1

        # 4) Source attribution
        total_checks += 1
        has_claims = any(
            kw in output.lower()
            for kw in ["according to", "research shows", "studies indicate", "data suggests"]
        )
        has_sources = any(
            kw in output.lower()
            for kw in ["source:", "reference:", "http://", "https://", "[1]", "[2]"]
        )
        if has_claims and not has_sources:
            issues.append("Claims made without source attribution. Add references.")
        else:
            checks_passed += 1

        # 5) Completeness check
        total_checks += 1
        if len(output.strip()) < 50:
            issues.append("Output appears too brief to adequately address the prompt.")
        else:
            checks_passed += 1

        score = checks_passed / total_checks if total_checks > 0 else 1.0
        passed = score >= self.pass_threshold

        findings = "\n".join(issues) if issues else "All fact-checking criteria passed."
        directive = None
        if not passed:
            directive = "Address the following factual issues:\n" + "\n".join(
                f"- {iss}" for iss in issues
            )

        return JudgeVerdict(
            judge_name=self.name,
            passed=passed,
            score=round(score, 2),
            findings=findings,
            correction_directive=directive,
        )

    def _check_numeric_consistency(self, numbers: list[tuple[str, str]]) -> list[str]:
        issues: list[str] = []
        unit_values: dict[str, list[float]] = {}
        for value_str, unit in numbers:
            unit_lower = unit.lower()
            try:
                val = float(value_str)
            except ValueError:
                continue
            unit_values.setdefault(unit_lower, []).append(val)

        for unit, values in unit_values.items():
            if len(values) >= 2:
                min_v, max_v = min(values), max(values)
                if max_v > 0 and min_v / max_v < 0.01:
                    issues.append(
                        f"Large numeric discrepancy in '{unit}' values: "
                        f"{min_v} vs {max_v} (100x+ difference)"
                    )
        return issues
