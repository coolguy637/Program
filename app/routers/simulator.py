from fastapi import APIRouter, Depends

from app.auth import get_current_user
from app.engine.computer_use import ComputerUseModule, PlatformInteractor
from app.engine.user_simulator import UserSimulator
from app.models import User
from app.schemas import BrowserAction, BrowserResult, SimulatorRequest, SimulatorResult

router = APIRouter(prefix="/api/simulator", tags=["simulator"])
simulator = UserSimulator()
computer_use = ComputerUseModule()
platform_interactor = PlatformInteractor()


@router.post("/run", response_model=SimulatorResult)
async def run_simulation(
    data: SimulatorRequest,
    user: User = Depends(get_current_user),
):
    """Run a user simulation against a target URL."""
    return await simulator.simulate(
        target_url=data.target_url,
        persona=data.persona,
        max_steps=data.max_steps,
    )


@router.post("/browser", response_model=BrowserResult)
async def execute_browser_action(
    action: BrowserAction,
    user: User = Depends(get_current_user),
):
    """Execute a browser automation action sequence."""
    return await computer_use.execute(action)


@router.post("/platform/{platform}")
async def interact_with_platform(
    platform: str,
    prompt: dict,
    user: User = Depends(get_current_user),
):
    """Send a prompt to an AI platform (chatgpt, claude, gemini)."""
    text = prompt.get("prompt", "")
    if not text:
        return {"error": "prompt field required"}
    return await platform_interactor.send_prompt(platform, text)
