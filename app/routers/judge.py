
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth import get_current_user
from app.database import get_db
from app.engine.orchestrator import ConflictLoopOrchestrator
from app.models import JudgeSession, User
from app.override.manager import override_manager
from app.schemas import JudgeRequest, JudgeSessionResponse

router = APIRouter(prefix="/api/judge", tags=["judge"])
orchestrator = ConflictLoopOrchestrator()


@router.post("/", response_model=JudgeSessionResponse, status_code=201)
async def create_judge_session(
    data: JudgeRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Start a new judge session with the conflict loop."""
    session = JudgeSession(user_id=user.id, prompt=data.prompt)
    db.add(session)
    await db.commit()
    await db.refresh(session)

    # Set up event streaming
    stream = override_manager.get_or_create_stream(session.id)
    event_queue = stream.subscribe()

    # Run the conflict loop
    try:
        session = await orchestrator.run(
            session=session,
            db=db,
            max_iterations=data.max_iterations,
            event_callback=event_queue,
        )
    except Exception as e:
        session.status = "failed"
        await db.commit()
        raise HTTPException(status_code=500, detail=f"Judge session failed: {e}")
    finally:
        stream.unsubscribe(event_queue)

    # Reload with iterations
    result = await db.execute(
        select(JudgeSession)
        .where(JudgeSession.id == session.id)
        .options(selectinload(JudgeSession.iterations))
    )
    return result.scalar_one()


@router.get("/", response_model=list[JudgeSessionResponse])
async def list_sessions(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(JudgeSession)
        .where(JudgeSession.user_id == user.id)
        .options(selectinload(JudgeSession.iterations))
        .order_by(JudgeSession.created_at.desc())
        .limit(50)
    )
    return result.scalars().all()


@router.get("/{session_id}", response_model=JudgeSessionResponse)
async def get_session(
    session_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(JudgeSession)
        .where(JudgeSession.id == session_id, JudgeSession.user_id == user.id)
        .options(selectinload(JudgeSession.iterations))
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session
