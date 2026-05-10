import asyncio
import json

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.database import get_db
from app.models import User
from app.override.manager import override_manager
from app.schemas import OverrideCommand

router = APIRouter(prefix="/api/override", tags=["override"])


@router.post("/")
async def send_override_command(
    cmd: OverrideCommand,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Send a manual override command to a running session."""
    actions = {
        "pause": override_manager.pause,
        "resume": override_manager.resume,
        "takeover": override_manager.takeover,
        "release": override_manager.release,
    }

    if cmd.action == "inject":
        if not cmd.payload:
            return {"error": "Payload required for inject action"}
        return await override_manager.inject_output(cmd.session_id, cmd.payload, db)

    handler = actions.get(cmd.action)
    if not handler:
        return {"error": f"Unknown action: {cmd.action}"}

    return await handler(cmd.session_id, db)


@router.websocket("/ws/{session_id}")
async def session_stream(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for live-streaming session events.

    Connect to receive real-time updates from a judge session.
    Send commands to control the session (pause, resume, takeover, inject).
    """
    await websocket.accept()

    stream = override_manager.get_or_create_stream(session_id)
    queue = stream.subscribe()

    # Send existing events as replay
    for event in stream.events:
        await websocket.send_json(event)

    try:
        # Run two tasks: send events and receive commands
        async def send_events():
            while True:
                event = await queue.get()
                await websocket.send_json(event)

        async def receive_commands():
            while True:
                data = await websocket.receive_text()
                try:
                    cmd = json.loads(data)
                    action = cmd.get("action", "")
                    if action == "pause":
                        await stream.broadcast({"event": "override", "action": "paused"})
                    elif action == "resume":
                        await stream.broadcast({"event": "override", "action": "resumed"})
                    elif action == "inject":
                        content = cmd.get("content", "")
                        await stream.broadcast({
                            "event": "override",
                            "action": "inject",
                            "content_preview": content[:200],
                        })
                    elif action == "ping":
                        await websocket.send_json({"event": "pong"})
                except json.JSONDecodeError:
                    await websocket.send_json({"event": "error", "detail": "Invalid JSON"})

        await asyncio.gather(send_events(), receive_commands())

    except WebSocketDisconnect:
        pass
    finally:
        stream.unsubscribe(queue)
