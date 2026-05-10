# Agentic Judge — Human Taste Engine & Autonomous Operator

An advanced Python FastAPI platform that bridges the gap between raw AI output and high-end human standards through an intelligent **Conflict Loop** orchestrator.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Web Dashboard                         │
│  (Signup, API Keys, Live Monitor, Override Controls)     │
└──────────────┬───────────────────────────┬───────────────┘
               │ REST API                  │ WebSocket
┌──────────────▼───────────────────────────▼───────────────┐
│                    FastAPI Service                        │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              Conflict Loop Orchestrator              │ │
│  │  ┌───────────┐    ┌────────────────────────────┐    │ │
│  │  │ Generator │◄──►│     Judge Panel             │    │ │
│  │  │           │    │  ┌──────────────────────┐   │    │ │
│  │  │  Content  │    │  │ Technical Auditor    │   │    │ │
│  │  │  Creation │    │  │ (code, security)     │   │    │ │
│  │  │           │    │  ├──────────────────────┤   │    │ │
│  │  │  Re-prompt│    │  │ Fact-Checker         │   │    │ │
│  │  │  with     │    │  │ (data, logic)        │   │    │ │
│  │  │  fixes    │    │  ├──────────────────────┤   │    │ │
│  │  │           │    │  │ UX Critic            │   │    │ │
│  │  └───────────┘    │  │ (taste, aesthetics)  │   │    │ │
│  │                   │  └──────────────────────┘   │    │ │
│  │                   └────────────────────────────┘    │ │
│  └─────────────────────────────────────────────────────┘ │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐   │
│  │User Simulator│  │Computer Use  │  │Manual Override│   │
│  │(friction     │  │(Playwright   │  │(WS streaming, │   │
│  │ detection)   │  │ browser auto)│  │ pause/takeover│   │
│  └──────────────┘  └──────────────┘  └───────────────┘   │
└──────────────────────────────────────────────────────────┘
```

## Features

### Conflict Loop Engine
- **Generator**: Produces content and iterates based on correction directives
- **Technical Auditor**: Analyzes code for syntax errors, security vulnerabilities, and style violations
- **Fact-Checker**: Verifies factual claims, detects contradictions, checks numeric consistency
- **UX Critic**: Evaluates readability, formatting, tone, and visual aesthetics via computer vision

### User Simulator
- Simulates different user personas (general, power user, novice, accessibility)
- Detects UX friction points via Playwright browser automation
- Reports issues with severity levels and actionable suggestions

### Computer Use Module
- Playwright-based browser automation
- Navigate, type, click, screenshot, scroll, and interact with web pages
- Pre-configured platform interactions for ChatGPT, Claude, and Gemini
- Human-like interaction speeds (typing delays, navigation pauses)

### Manual Override
- WebSocket-based live streaming of session events
- Pause/resume automation at any time
- Human takeover: assume full control mid-session
- Content injection: directly provide output during override

### Web Dashboard
- User signup and authentication (JWT + API keys)
- API key generation and management
- Live session monitoring with score visualizations
- Override controls for real-time human intervention

## Quick Start

```bash
# Install dependencies
pip install -e .

# Install Playwright browsers (optional, for browser automation)
playwright install chromium

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000 for the dashboard, or http://localhost:8000/api/docs for the API docs.

## API Usage

```bash
# Sign up
curl -X POST http://localhost:8000/api/users/signup \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "username": "user", "password": "password123"}'

# Login
curl -X POST http://localhost:8000/api/users/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "password123"}'

# Generate API key
curl -X POST http://localhost:8000/api/keys/ \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"name": "My App"}'

# Run judge pipeline
curl -X POST http://localhost:8000/api/judge/ \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a Python function to sort a list", "max_iterations": 5}'
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AJ_SECRET_KEY` | `change-me-...` | JWT signing key |
| `AJ_DATABASE_URL` | `sqlite+aiosqlite:///./agentic_judge.db` | Database URL |
| `AJ_MAX_JUDGE_ITERATIONS` | `5` | Default max iterations |
| `AJ_GENERATOR_MODEL` | `gpt-4` | Generator model name |
| `AJ_JUDGE_TIMEOUT_SECONDS` | `30` | Per-judge timeout |

## Tech Stack

- **Backend**: FastAPI, SQLAlchemy (async), Pydantic v2
- **Auth**: JWT (python-jose) + API key authentication
- **Browser Automation**: Playwright
- **Computer Vision**: Pillow
- **Frontend**: Vanilla JS, CSS custom properties, WebSocket
- **Database**: SQLite (async via aiosqlite)
