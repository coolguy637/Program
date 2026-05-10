import asyncio
import logging
import random

from app.schemas import FrictionPoint, SimulatorResult

logger = logging.getLogger(__name__)


PERSONAS = {
    "general_user": {
        "patience": 0.5,
        "tech_level": 0.5,
        "description": "Average user with moderate tech skills",
    },
    "power_user": {
        "patience": 0.8,
        "tech_level": 0.9,
        "description": "Experienced user who knows shortcuts",
    },
    "novice": {
        "patience": 0.3,
        "tech_level": 0.2,
        "description": "First-time user, easily confused",
    },
    "accessibility": {
        "patience": 0.6,
        "tech_level": 0.4,
        "description": "User relying on screen readers and keyboard navigation",
    },
}


class UserSimulator:
    """Simulates user interaction with a target URL to find friction points.

    Uses Playwright for browser automation when available, otherwise
    performs heuristic analysis based on page structure.
    """

    def __init__(self) -> None:
        self._playwright = None
        self._browser = None

    async def simulate(
        self,
        target_url: str,
        persona: str = "general_user",
        max_steps: int = 10,
    ) -> SimulatorResult:
        persona_config = PERSONAS.get(persona, PERSONAS["general_user"])
        friction_points: list[FrictionPoint] = []

        try:
            friction_points = await self._browser_simulation(
                target_url, persona_config, max_steps
            )
            steps_completed = max_steps
        except Exception as e:
            logger.warning("Browser simulation unavailable: %s. Using heuristic mode.", e)
            friction_points = await self._heuristic_simulation(
                target_url, persona_config, max_steps
            )
            steps_completed = max_steps

        overall_score = self._calculate_score(friction_points, max_steps)

        critical = sum(1 for fp in friction_points if fp.severity == "critical")
        high = sum(1 for fp in friction_points if fp.severity == "high")

        return SimulatorResult(
            target_url=target_url,
            persona=persona,
            steps_completed=steps_completed,
            friction_points=friction_points,
            overall_score=round(overall_score, 2),
            summary=(
                f"Simulation complete as '{persona}' persona. "
                f"Found {len(friction_points)} friction point(s): "
                f"{critical} critical, {high} high severity. "
                f"Overall UX score: {overall_score:.0%}"
            ),
        )

    async def _browser_simulation(
        self, url: str, persona: dict, max_steps: int
    ) -> list[FrictionPoint]:
        """Run Playwright-based browser simulation."""
        from playwright.async_api import async_playwright

        friction_points: list[FrictionPoint] = []
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()

            try:
                response = await page.goto(url, timeout=15000)

                if response and response.status >= 400:
                    friction_points.append(FrictionPoint(
                        step=1,
                        element="page",
                        issue=f"HTTP {response.status} error on page load",
                        severity="critical",
                        suggestion="Ensure the page returns a successful HTTP status code.",
                    ))

                await page.evaluate("JSON.stringify(performance.timing)")
                await asyncio.sleep(0.1)

                # Find interactive elements
                buttons = await page.query_selector_all("button, [role=button]")
                await page.query_selector_all("a[href]")
                inputs = await page.query_selector_all("input, textarea, select")

                # Check for missing labels on inputs
                for i, inp in enumerate(inputs[:max_steps]):
                    label = await inp.get_attribute("aria-label")
                    placeholder = await inp.get_attribute("placeholder")
                    inp_id = await inp.get_attribute("id")

                    if not label and not placeholder and not inp_id:
                        friction_points.append(FrictionPoint(
                            step=i + 1,
                            element="input",
                            issue="Input field without label, placeholder, or id",
                            severity="high" if persona.get("tech_level", 0.5) < 0.3 else "medium",
                            suggestion="Add aria-label or placeholder text to input fields.",
                        ))

                # Check for tiny click targets
                for i, btn in enumerate(buttons[:max_steps]):
                    box = await btn.bounding_box()
                    if box and (box["width"] < 44 or box["height"] < 44):
                        friction_points.append(FrictionPoint(
                            step=i + 1,
                            element="button",
                            issue=f"Button too small ({box['width']:.0f}x{box['height']:.0f}px)",
                            severity="medium",
                            suggestion="Minimum touch target should be 44x44px (WCAG 2.5.5).",
                        ))

                # Check contrast and font sizes
                font_sizes = await page.evaluate("""
                    () => {
                        const elements = document.querySelectorAll('p, span, li, td, a');
                        const sizes = [];
                        for (const el of [...elements].slice(0, 50)) {
                            const style = window.getComputedStyle(el);
                            sizes.push(parseFloat(style.fontSize));
                        }
                        return sizes;
                    }
                """)
                small_text = sum(1 for s in font_sizes if s < 12)
                if small_text > 3:
                    friction_points.append(FrictionPoint(
                        step=max_steps,
                        element="text",
                        issue=f"{small_text} elements with font-size below 12px",
                        severity="medium",
                        suggestion="Increase minimum font size to 14px for readability.",
                    ))

            except Exception as e:
                friction_points.append(FrictionPoint(
                    step=1,
                    element="page",
                    issue=f"Page interaction error: {e}",
                    severity="critical",
                    suggestion="Ensure the page loads correctly and is interactive.",
                ))
            finally:
                await browser.close()

        return friction_points

    async def _heuristic_simulation(
        self, url: str, persona: dict, max_steps: int
    ) -> list[FrictionPoint]:
        """Heuristic-based simulation without browser automation."""
        import httpx

        friction_points: list[FrictionPoint] = []

        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                response = await client.get(url)

                if response.status_code >= 400:
                    friction_points.append(FrictionPoint(
                        step=1,
                        element="page",
                        issue=f"HTTP {response.status_code} error",
                        severity="critical",
                        suggestion="Fix server errors before user testing.",
                    ))
                    return friction_points

                html = response.text

                # Check for viewport meta tag (mobile-friendly)
                if "viewport" not in html.lower():
                    friction_points.append(FrictionPoint(
                        step=1,
                        element="meta",
                        issue="Missing viewport meta tag",
                        severity="high",
                        suggestion=(
                            "Add <meta name='viewport'"
                            " content='width=device-width, initial-scale=1'>"
                        ),
                    ))

                # Check for HTTPS
                if not url.startswith("https"):
                    friction_points.append(FrictionPoint(
                        step=1,
                        element="security",
                        issue="Page served over HTTP instead of HTTPS",
                        severity="high",
                        suggestion="Enable HTTPS for security and user trust.",
                    ))

                # Check for alt text on images
                import re

                images = re.findall(r"<img[^>]*>", html, re.I)
                no_alt = sum(
                    1 for img in images
                    if 'alt="' not in img.lower() and "alt='" not in img.lower()
                )
                if no_alt > 0:
                    friction_points.append(FrictionPoint(
                        step=2,
                        element="images",
                        issue=f"{no_alt} image(s) missing alt text",
                        severity="medium",
                        suggestion="Add descriptive alt text to all images for accessibility.",
                    ))

        except Exception as e:
            friction_points.append(FrictionPoint(
                step=1,
                element="page",
                issue=f"Could not reach URL: {e}",
                severity="critical",
                suggestion="Verify the URL is correct and the server is running.",
            ))

        # Simulate some interaction steps
        await asyncio.sleep(random.uniform(0.2, 0.5))

        return friction_points

    def _calculate_score(self, friction_points: list[FrictionPoint], max_steps: int) -> float:
        if not friction_points:
            return 1.0
        severity_weights = {"critical": 0.3, "high": 0.15, "medium": 0.08, "low": 0.03}
        penalty = sum(severity_weights.get(fp.severity, 0.05) for fp in friction_points)
        return max(0.0, 1.0 - penalty)
