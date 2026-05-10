import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=new_uuid)
    email = Column(String, unique=True, nullable=False, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utcnow)

    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("JudgeSession", back_populates="user", cascade="all, delete-orphan")


class APIKey(Base):
    __tablename__ = "api_keys"

    id = Column(String, primary_key=True, default=new_uuid)
    key = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utcnow)
    last_used_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="api_keys")


class JudgeSession(Base):
    __tablename__ = "judge_sessions"

    id = Column(String, primary_key=True, default=new_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    status = Column(String, default="pending")  # pending, running, paused, completed, failed
    prompt = Column(Text, nullable=False)
    final_output = Column(Text, nullable=True)
    iteration_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=utcnow)
    completed_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="sessions")
    iterations = relationship(
        "JudgeIteration", back_populates="session", cascade="all, delete-orphan"
    )


class JudgeIteration(Base):
    __tablename__ = "judge_iterations"

    id = Column(String, primary_key=True, default=new_uuid)
    session_id = Column(String, ForeignKey("judge_sessions.id"), nullable=False)
    iteration_number = Column(Integer, nullable=False)
    generator_output = Column(Text, nullable=True)
    technical_audit = Column(Text, nullable=True)
    technical_score = Column(Float, nullable=True)
    fact_check = Column(Text, nullable=True)
    fact_score = Column(Float, nullable=True)
    ux_review = Column(Text, nullable=True)
    ux_score = Column(Float, nullable=True)
    correction_directive = Column(Text, nullable=True)
    passed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=utcnow)

    session = relationship("JudgeSession", back_populates="iterations")
