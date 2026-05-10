import asyncio
import logging
import tempfile
from pathlib import Path

from app.schemas import BrowserAction, BrowserResult

logger = logging.getLogger(__name__)


class ComputerUseModule:
    """Browser automation module using Playwright for autonomous web interaction.

    Supports navigating to URLs, typing text, clicking elements,
    taking screenshots, and interacting with AI platforms.
    """

    def __init__(self) -> None:
        self._browser = None

    async def execute(self, action: BrowserAction) -> BrowserResult:
        """Execute a browser automation action sequence."""
        try:
            return await self._execute_with_playwright(action)
        except ImportError:
            return BrowserResult(
                success=False,
                error="Playwright not installed. Run: playwright install chromium",
            )
        except Exception as e:
            logger.exception("Browser automation error")
            return BrowserResult(success=False, error=str(e))

    async def _execute_with_playwright(self, action: BrowserAction) -> BrowserResult:
        from playwright.async_api import async_playwright

        screenshots: list[str] = []
        page_content: str | None = None

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            )
            page = await context.new_page()

            try:
                await page.goto(action.url, timeout=30000, wait_until="domcontentloaded")
                await asyncio.sleep(1)  # Human-like pause after navigation

                for step in action.actions:
                    step_type = step.get("type", "")
                    await self._execute_step(page, step_type, step, screenshots)

                page_content = await page.content()

            except Exception as e:
                screenshot_path = self._screenshot_path("error")
                await page.screenshot(path=screenshot_path)
                screenshots.append(screenshot_path)
                return BrowserResult(
                    success=False,
                    screenshots=screenshots,
                    page_content=await page.content(),
                    error=str(e),
                )
            finally:
                await browser.close()

        return BrowserResult(
            success=True,
            screenshots=screenshots,
            page_content=page_content[:10000] if page_content else None,
        )

    async def _execute_step(
        self, page: object, step_type: str, step: dict, screenshots: list[str]
    ) -> None:
        from playwright.async_api import Page

        assert isinstance(page, Page)

        if step_type == "navigate":
            url = step.get("url", "")
            await page.goto(url, timeout=30000)
            await asyncio.sleep(0.5)

        elif step_type == "type":
            selector = step.get("selector", "")
            text = step.get("text", "")
            delay = step.get("delay", 50)  # Human-like typing speed
            if selector:
                await page.click(selector)
                await page.fill(selector, "")
                await page.type(selector, text, delay=delay)
            else:
                await page.keyboard.type(text, delay=delay)
            await asyncio.sleep(0.3)

        elif step_type == "click":
            selector = step.get("selector", "")
            await page.click(selector, timeout=10000)
            await asyncio.sleep(0.5)

        elif step_type == "screenshot":
            name = step.get("name", "step")
            path = self._screenshot_path(name)
            await page.screenshot(path=path, full_page=step.get("full_page", False))
            screenshots.append(path)

        elif step_type == "wait":
            selector = step.get("selector")
            duration = step.get("duration", 1.0)
            if selector:
                await page.wait_for_selector(selector, timeout=int(duration * 1000))
            else:
                await asyncio.sleep(duration)

        elif step_type == "scroll":
            direction = step.get("direction", "down")
            amount = step.get("amount", 300)
            delta = amount if direction == "down" else -amount
            await page.mouse.wheel(0, delta)
            await asyncio.sleep(0.3)

        elif step_type == "select":
            selector = step.get("selector", "")
            value = step.get("value", "")
            await page.select_option(selector, value)
            await asyncio.sleep(0.3)

        elif step_type == "press":
            key = step.get("key", "Enter")
            await page.keyboard.press(key)
            await asyncio.sleep(0.3)

        elif step_type == "evaluate":
            script = step.get("script", "")
            await page.evaluate(script)

        else:
            logger.warning("Unknown step type: %s", step_type)

    def _screenshot_path(self, name: str) -> str:
        return str(Path(tempfile.gettempdir()) / f"aj_screenshot_{name}.png")


class PlatformInteractor:
    """Specialized interactions with AI platforms (Gemini, Claude, ChatGPT)."""

    PLATFORM_CONFIGS = {
        "chatgpt": {
            "url": "https://chat.openai.com",
            "input_selector": "#prompt-textarea",
            "submit_selector": "button[data-testid='send-button']",
            "response_selector": ".markdown",
        },
        "claude": {
            "url": "https://claude.ai",
            "input_selector": "[contenteditable='true']",
            "submit_selector": "button[aria-label='Send message']",
            "response_selector": ".prose",
        },
        "gemini": {
            "url": "https://gemini.google.com",
            "input_selector": ".ql-editor",
            "submit_selector": "button[aria-label='Send message']",
            "response_selector": ".response-content",
        },
    }

    def __init__(self) -> None:
        self.computer_use = ComputerUseModule()

    async def send_prompt(self, platform: str, prompt: str) -> BrowserResult:
        """Send a prompt to an AI platform and capture the response."""
        config = self.PLATFORM_CONFIGS.get(platform.lower())
        if not config:
            return BrowserResult(
                success=False,
                error=f"Unknown platform: {platform}. Supported: {list(self.PLATFORM_CONFIGS)}",
            )

        action = BrowserAction(
            url=config["url"],
            actions=[
                {"type": "wait", "duration": 2.0},
                {"type": "screenshot", "name": f"{platform}_before"},
                {"type": "type", "selector": config["input_selector"], "text": prompt, "delay": 30},
                {"type": "screenshot", "name": f"{platform}_typed"},
                {"type": "click", "selector": config["submit_selector"]},
                {"type": "wait", "duration": 5.0},
                {"type": "screenshot", "name": f"{platform}_response"},
            ],
        )

        return await self.computer_use.execute(action)
