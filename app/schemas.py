from datetime import datetime

from pydantic import BaseModel, EmailStr, Field

# ── Auth ──────────────────────────────────────────────────────────────────────


class UserCreate(BaseModel):
    email: EmailStr
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=8)


class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class LoginRequest(BaseModel):
    username: str
    password: str


# ── API Keys ──────────────────────────────────────────────────────────────────


class APIKeyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class APIKeyResponse(BaseModel):
    id: str
    key: str
    name: str
    is_active: bool
    created_at: datetime
    last_used_at: datetime | None = None

    model_config = {"from_attributes": True}


# ── Judge Sessions ────────────────────────────────────────────────────────────


class JudgeRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_iterations: int = Field(default=5, ge=1, le=20)


class JudgeIterationResponse(BaseModel):
    iteration_number: int
    generator_output: str | None = None
    technical_audit: str | None = None
    technical_score: float | None = None
    fact_check: str | None = None
    fact_score: float | None = None
    ux_review: str | None = None
    ux_score: float | None = None
    correction_directive: str | None = None
    passed: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class JudgeSessionResponse(BaseModel):
    id: str
    status: str
    prompt: str
    final_output: str | None = None
    iteration_count: int
    created_at: datetime
    completed_at: datetime | None = None
    iterations: list[JudgeIterationResponse] = []

    model_config = {"from_attributes": True}


# ── Computer Use ──────────────────────────────────────────────────────────────


class BrowserAction(BaseModel):
    url: str
    actions: list[dict] = Field(
        default_factory=list,
        description="Sequence of browser actions: type, click, screenshot, navigate, wait",
    )


class BrowserResult(BaseModel):
    success: bool
    screenshots: list[str] = []
    page_content: str | None = None
    error: str | None = None


# ── User Simulator ────────────────────────────────────────────────────────────


class SimulatorRequest(BaseModel):
    target_url: str
    persona: str = "general_user"
    max_steps: int = Field(default=10, ge=1, le=50)


class FrictionPoint(BaseModel):
    step: int
    element: str
    issue: str
    severity: str  # low, medium, high, critical
    suggestion: str


class SimulatorResult(BaseModel):
    target_url: str
    persona: str
    steps_completed: int
    friction_points: list[FrictionPoint] = []
    overall_score: float
    summary: str


# ── Manual Override ───────────────────────────────────────────────────────────


class OverrideCommand(BaseModel):
    session_id: str
    action: str  # pause, resume, takeover, release, inject
    payload: str | None = None
