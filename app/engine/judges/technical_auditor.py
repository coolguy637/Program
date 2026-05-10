import ast
import logging
import re

from app.engine.judges.base import BaseJudge, JudgeVerdict

logger = logging.getLogger(__name__)


class TechnicalAuditor(BaseJudge):
    """Audits code blocks for syntax errors, security issues, and best-practice violations."""

    name = "technical_auditor"
    pass_threshold = 0.7

    CODE_BLOCK_RE = re.compile(r"```(?:python|py|javascript|js|typescript|ts)?\s*\n(.*?)```", re.S)

    SECURITY_PATTERNS = [
        (re.compile(r"eval\s*\("), "Use of eval() is a security risk"),
        (re.compile(r"exec\s*\("), "Use of exec() is a security risk"),
        (re.compile(r"__import__\s*\("), "Dynamic import via __import__ is risky"),
        (re.compile(r"subprocess\.call\(.*shell\s*=\s*True"), "Shell injection risk"),
        (re.compile(r"os\.system\s*\("), "os.system() is a security risk; use subprocess"),
        (re.compile(r"pickle\.loads?\s*\("), "Pickle deserialization is unsafe on untrusted data"),
    ]

    async def evaluate(
        self, prompt: str, output: str, context: dict | None = None
    ) -> JudgeVerdict:
        issues: list[str] = []
        code_blocks = self.CODE_BLOCK_RE.findall(output)

        if not code_blocks:
            return JudgeVerdict(
                judge_name=self.name,
                passed=True,
                score=1.0,
                findings="No code blocks found to audit.",
            )

        total_blocks = len(code_blocks)
        passed_blocks = 0

        for i, block in enumerate(code_blocks, 1):
            block_issues = self._audit_block(block, i)
            if block_issues:
                issues.extend(block_issues)
            else:
                passed_blocks += 1

        score = passed_blocks / total_blocks if total_blocks > 0 else 1.0
        passed = score >= self.pass_threshold and not any("CRITICAL" in iss for iss in issues)

        findings = (
            "\n".join(issues) if issues else f"All {total_blocks} code block(s) passed audit."
        )
        directive = None
        if not passed:
            directive = (
                "Fix the following code issues before resubmitting:\n" + "\n".join(issues)
            )

        return JudgeVerdict(
            judge_name=self.name,
            passed=passed,
            score=round(score, 2),
            findings=findings,
            correction_directive=directive,
        )

    def _audit_block(self, code: str, block_num: int) -> list[str]:
        issues: list[str] = []

        # Syntax check (Python only)
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append(
                f"[CRITICAL] Block {block_num}: syntax error L{e.lineno}: {e.msg}"
            )

        # Security scan
        for pattern, description in self.SECURITY_PATTERNS:
            if pattern.search(code):
                issues.append(f"[WARNING] Block {block_num}: {description}")

        # Style checks
        lines = code.split("\n")
        for line_num, line in enumerate(lines, 1):
            if len(line) > 120:
                issues.append(
                    f"[STYLE] Block {block_num}, line {line_num}: "
                    f"Line too long ({len(line)} chars > 120)"
                )
            if line.rstrip() != line and line.strip():
                issues.append(
                    f"[STYLE] Block {block_num}, line {line_num}: Trailing whitespace"
                )

        return issues
