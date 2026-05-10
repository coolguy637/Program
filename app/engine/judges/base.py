import abc
import asyncio
import logging

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class JudgeVerdict(BaseModel):
    judge_name: str
    passed: bool
    score: float  # 0.0 – 1.0
    findings: str
    correction_directive: str | None = None


class BaseJudge(abc.ABC):
    name: str = "base"
    pass_threshold: float = 0.8

    @abc.abstractmethod
    async def evaluate(self, prompt: str, output: str, context: dict | None = None) -> JudgeVerdict:
        ...

    async def evaluate_with_timeout(
        self, prompt: str, output: str, context: dict | None = None, timeout: float = 30.0
    ) -> JudgeVerdict:
        try:
            return await asyncio.wait_for(self.evaluate(prompt, output, context), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Judge %s timed out after %.1fs", self.name, timeout)
            return JudgeVerdict(
                judge_name=self.name,
                passed=False,
                score=0.0,
                findings=f"Judge {self.name} timed out after {timeout}s",
                correction_directive="Re-evaluate: previous judge timed out.",
            )
