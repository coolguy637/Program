import asyncio
import logging
from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import JudgeSession

logger = logging.getLogger(__name__)


class OverrideState(str, Enum):
    AUTONOMOUS = "autonomous"
    PAUSED = "paused"
    HUMAN_CONTROL = "human_control"


class SessionStream:
    """Manages WebSocket event streams for a judge session."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.subscribers: list[asyncio.Queue] = []
        self.state = OverrideState.AUTONOMOUS
        self.events: list[dict] = []

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self.subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        if queue in self.subscribers:
            self.subscribers.remove(queue)

    async def broadcast(self, event: dict) -> None:
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
        self.events.append(event)
        for queue in self.subscribers:
            await queue.put(event)


class OverrideManager:
    """Manages manual override state for all active judge sessions.

    Provides:
    - Live streaming of session events via WebSocket
    - Pause/resume control
    - Human takeover and release
    - Direct content injection during override
    """

    def __init__(self) -> None:
        self._streams: dict[str, SessionStream] = {}
        self._override_outputs: dict[str, str] = {}

    def get_or_create_stream(self, session_id: str) -> SessionStream:
        if session_id not in self._streams:
            self._streams[session_id] = SessionStream(session_id)
        return self._streams[session_id]

    def get_stream(self, session_id: str) -> SessionStream | None:
        return self._streams.get(session_id)

    async def pause(self, session_id: str, db: AsyncSession) -> dict:
        stream = self.get_or_create_stream(session_id)
        stream.state = OverrideState.PAUSED

        result = await db.execute(
            select(JudgeSession).where(JudgeSession.id == session_id)
        )
        session = result.scalar_one_or_none()
        if session:
            session.status = "paused"
            await db.commit()

        await stream.broadcast({"event": "override", "action": "paused"})
        logger.info("Session %s paused", session_id)
        return {"status": "paused", "session_id": session_id}

    async def resume(self, session_id: str, db: AsyncSession) -> dict:
        stream = self.get_or_create_stream(session_id)
        stream.state = OverrideState.AUTONOMOUS

        result = await db.execute(
            select(JudgeSession).where(JudgeSession.id == session_id)
        )
        session = result.scalar_one_or_none()
        if session:
            session.status = "running"
            await db.commit()

        await stream.broadcast({"event": "override", "action": "resumed"})
        logger.info("Session %s resumed", session_id)
        return {"status": "running", "session_id": session_id}

    async def takeover(self, session_id: str, db: AsyncSession) -> dict:
        stream = self.get_or_create_stream(session_id)
        stream.state = OverrideState.HUMAN_CONTROL

        result = await db.execute(
            select(JudgeSession).where(JudgeSession.id == session_id)
        )
        session = result.scalar_one_or_none()
        if session:
            session.status = "paused"
            await db.commit()

        await stream.broadcast({"event": "override", "action": "takeover"})
        logger.info("Session %s: human takeover", session_id)
        return {"status": "human_control", "session_id": session_id}

    async def release(self, session_id: str, db: AsyncSession) -> dict:
        stream = self.get_or_create_stream(session_id)
        stream.state = OverrideState.AUTONOMOUS

        result = await db.execute(
            select(JudgeSession).where(JudgeSession.id == session_id)
        )
        session = result.scalar_one_or_none()
        if session:
            session.status = "running"
            await db.commit()

        await stream.broadcast({"event": "override", "action": "released"})
        logger.info("Session %s: human released control", session_id)
        return {"status": "running", "session_id": session_id}

    async def inject_output(self, session_id: str, content: str, db: AsyncSession) -> dict:
        """Allow human to directly inject content as the session output."""
        stream = self.get_or_create_stream(session_id)

        if stream.state != OverrideState.HUMAN_CONTROL:
            return {"error": "Must be in takeover mode to inject output"}

        self._override_outputs[session_id] = content

        result = await db.execute(
            select(JudgeSession).where(JudgeSession.id == session_id)
        )
        session = result.scalar_one_or_none()
        if session:
            session.final_output = content
            session.status = "completed"
            session.completed_at = datetime.now(timezone.utc)
            await db.commit()

        await stream.broadcast({
            "event": "override",
            "action": "inject",
            "content_preview": content[:200],
        })

        logger.info("Session %s: human injected output (%d chars)", session_id, len(content))
        return {"status": "completed", "session_id": session_id, "injected": True}

    def cleanup(self, session_id: str) -> None:
        self._streams.pop(session_id, None)
        self._override_outputs.pop(session_id, None)


# Singleton
override_manager = OverrideManager()
