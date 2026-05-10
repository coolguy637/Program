import asyncio
import logging
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.engine.generator import Generator
from app.engine.judges.base import JudgeVerdict
from app.engine.judges.fact_checker import FactChecker
from app.engine.judges.technical_auditor import TechnicalAuditor
from app.engine.judges.ux_critic import UXCritic
from app.models import JudgeIteration, JudgeSession

logger = logging.getLogger(__name__)


class ConflictLoopOrchestrator:
    """Manages the Generator ↔ Judge panel conflict loop.

    The loop runs until:
    1. All judges pass the output, OR
    2. Max iterations reached, OR
    3. Session is paused/cancelled via manual override.
    """

    def __init__(self) -> None:
        self.generator = Generator(model=settings.generator_model)
        self.judges = [
            TechnicalAuditor(),
            FactChecker(),
            UXCritic(),
        ]

    async def run(
        self,
        session: JudgeSession,
        db: AsyncSession,
        max_iterations: int = 5,
        event_callback: asyncio.Queue | None = None,
    ) -> JudgeSession:
        """Execute the full conflict loop for a judge session."""
        session.status = "running"
        await db.commit()
        await self._emit(event_callback, "status", "running")

        correction: str | None = None

        for iteration_num in range(1, max_iterations + 1):
            # Check for pause/cancel
            if session.status == "paused":
                await self._emit(event_callback, "paused", f"Paused at iteration {iteration_num}")
                return session

            logger.info("Session %s: iteration %d/%d", session.id, iteration_num, max_iterations)
            await self._emit(
                event_callback,
                "iteration_start",
                {"iteration": iteration_num, "max": max_iterations},
            )

            # Generator phase
            output = await self.generator.generate(session.prompt, correction)
            await self._emit(event_callback, "generator_output", output[:500])

            # Judge panel evaluation (run all judges concurrently)
            verdicts = await asyncio.gather(
                *[
                    judge.evaluate_with_timeout(
                        session.prompt,
                        output,
                        timeout=settings.judge_timeout_seconds,
                    )
                    for judge in self.judges
                ]
            )

            # Record iteration
            iteration = self._create_iteration(session.id, iteration_num, output, verdicts)
            db.add(iteration)
            session.iteration_count = iteration_num
            await db.commit()

            await self._emit(event_callback, "verdicts", {
                v.judge_name: {"passed": v.passed, "score": v.score} for v in verdicts
            })

            # Check if all judges passed
            if all(v.passed for v in verdicts):
                session.status = "completed"
                session.final_output = output
                session.completed_at = datetime.now(timezone.utc)
                await db.commit()
                await self._emit(event_callback, "completed", {
                    "iteration": iteration_num,
                    "output_preview": output[:500],
                })
                return session

            # Build correction directive from failing judges
            corrections = []
            for v in verdicts:
                if not v.passed and v.correction_directive:
                    corrections.append(f"[{v.judge_name}]: {v.correction_directive}")
            correction = "\n\n".join(corrections)

            await self._emit(event_callback, "correction", correction[:500])

        # Max iterations reached — deliver best effort
        session.status = "completed"
        session.final_output = output  # type: ignore[possibly-undefined]
        session.completed_at = datetime.now(timezone.utc)
        await db.commit()
        await self._emit(event_callback, "max_iterations_reached", {
            "iterations": max_iterations,
            "output_preview": output[:500],  # type: ignore[possibly-undefined]
        })
        return session

    def _create_iteration(
        self,
        session_id: str,
        iteration_num: int,
        output: str,
        verdicts: list[JudgeVerdict],
    ) -> JudgeIteration:
        verdict_map = {v.judge_name: v for v in verdicts}
        tech = verdict_map.get("technical_auditor")
        fact = verdict_map.get("fact_checker")
        ux = verdict_map.get("ux_critic")

        corrections = [v.correction_directive for v in verdicts if v.correction_directive]

        return JudgeIteration(
            session_id=session_id,
            iteration_number=iteration_num,
            generator_output=output,
            technical_audit=tech.findings if tech else None,
            technical_score=tech.score if tech else None,
            fact_check=fact.findings if fact else None,
            fact_score=fact.score if fact else None,
            ux_review=ux.findings if ux else None,
            ux_score=ux.score if ux else None,
            correction_directive="\n".join(corrections) if corrections else None,
            passed=all(v.passed for v in verdicts),
        )

    async def _emit(self, queue: asyncio.Queue | None, event: str, data: object) -> None:
        if queue is not None:
            await queue.put({"event": event, "data": data})
