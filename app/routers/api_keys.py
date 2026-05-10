from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import generate_api_key, get_current_user
from app.database import get_db
from app.models import APIKey, User
from app.schemas import APIKeyCreate, APIKeyResponse

router = APIRouter(prefix="/api/keys", tags=["api_keys"])


@router.post("/", response_model=APIKeyResponse, status_code=201)
async def create_api_key(
    data: APIKeyCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    key = generate_api_key()
    api_key = APIKey(key=key, name=data.name, user_id=user.id)
    db.add(api_key)
    await db.commit()
    await db.refresh(api_key)
    return api_key


@router.get("/", response_model=list[APIKeyResponse])
async def list_api_keys(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(APIKey).where(APIKey.user_id == user.id).order_by(APIKey.created_at.desc())
    )
    return result.scalars().all()


@router.delete("/{key_id}")
async def revoke_api_key(
    key_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(APIKey).where(APIKey.id == key_id, APIKey.user_id == user.id)
    )
    api_key = result.scalar_one_or_none()
    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")
    api_key.is_active = False
    await db.commit()
    return {"status": "revoked", "key_id": key_id}
