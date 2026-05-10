import logging
import re

from app.engine.judges.base import BaseJudge, JudgeVerdict

logger = logging.getLogger(__name__)


class UXCritic(BaseJudge):
    """Evaluates output for human taste, premium aesthetics, readability, and UX quality.

    Uses heuristic analysis for text formatting and structural quality.
    When screenshots are provided in context, performs visual analysis checks.
    """

    name = "ux_critic"
    pass_threshold = 0.7

    async def evaluate(
        self, prompt: str, output: str, context: dict | None = None
    ) -> JudgeVerdict:
        issues: list[str] = []
        scores: list[float] = []

        # 1) Readability & structure
        readability = self._check_readability(output)
        scores.append(readability["score"])
        if readability["issues"]:
            issues.extend(readability["issues"])

        # 2) Formatting quality
        formatting = self._check_formatting(output)
        scores.append(formatting["score"])
        if formatting["issues"]:
            issues.extend(formatting["issues"])

        # 3) Tone & professionalism
        tone = self._check_tone(output)
        scores.append(tone["score"])
        if tone["issues"]:
            issues.extend(tone["issues"])

        # 4) Visual analysis (if screenshots provided)
        if context and context.get("screenshots"):
            visual = await self._analyze_visuals(context["screenshots"])
            scores.append(visual["score"])
            if visual["issues"]:
                issues.extend(visual["issues"])

        score = sum(scores) / len(scores) if scores else 0.5
        passed = score >= self.pass_threshold

        findings = "\n".join(issues) if issues else "Output meets UX and aesthetic standards."
        directive = None
        if not passed:
            directive = (
                "Improve the output's presentation and readability:\n"
                + "\n".join(f"- {iss}" for iss in issues)
            )

        return JudgeVerdict(
            judge_name=self.name,
            passed=passed,
            score=round(score, 2),
            findings=findings,
            correction_directive=directive,
        )

    def _check_readability(self, text: str) -> dict:
        issues: list[str] = []
        lines = text.split("\n")
        total_lines = len(lines)

        # Paragraph length
        long_paragraphs = 0
        current_para_len = 0
        for line in lines:
            if line.strip():
                current_para_len += 1
            else:
                if current_para_len > 8:
                    long_paragraphs += 1
                current_para_len = 0
        if current_para_len > 8:
            long_paragraphs += 1

        if long_paragraphs > 0:
            issues.append(
                f"{long_paragraphs} paragraph(s) exceed 8 lines. "
                "Break into shorter, scannable sections."
            )

        # Sentence length
        sentences = re.split(r"[.!?]+", text)
        long_sentences = sum(1 for s in sentences if len(s.split()) > 35)
        if long_sentences > 2:
            issues.append(
                f"{long_sentences} sentences exceed 35 words. "
                "Simplify for better readability."
            )

        score = 1.0
        if long_paragraphs > 0:
            score -= 0.15 * min(long_paragraphs, 3)
        if long_sentences > 2:
            score -= 0.1 * min(long_sentences, 4)

        return {"score": max(score, 0.0), "issues": issues, "total_lines": total_lines}

    def _check_formatting(self, text: str) -> dict:
        issues: list[str] = []
        score = 1.0

        has_headers = bool(re.search(r"^#{1,6}\s", text, re.M))
        has_lists = bool(re.search(r"^[\s]*[-*]\s", text, re.M))
        has_code = bool(re.search(r"```", text))
        has_bold = bool(re.search(r"\*\*[^*]+\*\*", text))

        formatting_elements = sum([has_headers, has_lists, has_code, has_bold])

        if len(text) > 500 and formatting_elements == 0:
            issues.append(
                "Long output lacks any formatting (headers, lists, bold, code blocks). "
                "Add structure for premium readability."
            )
            score -= 0.3

        # Inconsistent heading levels
        headings = re.findall(r"^(#{1,6})\s", text, re.M)
        if headings:
            levels = [len(h) for h in headings]
            if levels and levels[0] != min(levels):
                issues.append("Heading hierarchy is inconsistent. Start with the highest level.")
                score -= 0.1

        return {"score": max(score, 0.0), "issues": issues}

    def _check_tone(self, text: str) -> dict:
        issues: list[str] = []
        score = 1.0
        text_lower = text.lower()

        casual_markers = ["lol", "gonna", "wanna", "gotta", "kinda", "btw", "tbh", "imo"]
        found_casual = [m for m in casual_markers if f" {m} " in f" {text_lower} "]
        if len(found_casual) > 2:
            issues.append(
                f"Overly casual tone detected ({', '.join(found_casual)}). "
                "Use professional language for premium quality."
            )
            score -= 0.2

        exclamation_count = text.count("!")
        if exclamation_count > 5:
            issues.append(
                f"Excessive exclamation marks ({exclamation_count}). "
                "Reduce for a more polished, authoritative tone."
            )
            score -= 0.15

        return {"score": max(score, 0.0), "issues": issues}

    async def _analyze_visuals(self, screenshots: list[str]) -> dict:
        """Analyze screenshot paths/URLs for visual quality indicators."""
        issues: list[str] = []
        score = 0.85

        if not screenshots:
            return {"score": 1.0, "issues": []}

        try:
            from PIL import Image

            for path in screenshots[:5]:
                try:
                    img = Image.open(path)
                    width, height = img.size

                    if width < 800 or height < 600:
                        issues.append(
                            f"Screenshot {path}: Low resolution ({width}x{height}). "
                            "Minimum 800x600 recommended."
                        )
                        score -= 0.1

                    if width / height > 3 or height / width > 3:
                        issues.append(
                            f"Screenshot {path}: Unusual aspect ratio ({width}:{height})."
                        )
                        score -= 0.05

                except Exception as e:
                    logger.warning("Could not analyze screenshot %s: %s", path, e)

        except ImportError:
            logger.info("Pillow not available for visual analysis; skipping image checks")

        return {"score": max(score, 0.0), "issues": issues}
