# AGENTS.md: crypto-chatbot

Telegram research and education chatbot for crypto markets. Python computes all market facts deterministically; Gemini only explains validated facts. Not a trading or order-execution system.

## Sources of truth

- `PROJECT_PLAN.md`: phases, acceptance criteria, data contracts, phase status (§0).
- `docs/decisions/`: accepted ADRs, binding until superseded by a new ADR.
- `.agents/skills/crypto-research-engineer/SKILL.md`: detailed engineering rules. Load it for any engineering task.
- `.agents/skills/socratic-tutor/SKILL.md`: only when the user explicitly asks to be taught or quizzed.

## Non-negotiable rules

1. Never commit, print, or log secrets (`.env`, API keys, bot tokens, webhook secrets).
2. Market numbers come only from deterministic backend code. Gemini never calculates or invents numbers; it references facts via `{F<n>}` placeholders.
3. Closed candles only; never use data before its `available_time`; UTC internally.
4. Return `INSUFFICIENT_DATA` instead of guessing or silently repairing data.
5. Indicator and rule outputs are measurements, never success probabilities.
6. No order execution, no guaranteed calls, no personalized allocation, no LLM-generated price targets.
7. Work one phase at a time against its acceptance criteria; update the phase status table when done.

## Language

Reply to the user in Indonesian. Code, identifiers, docs, and commit messages in English.

## Commands

Runtime: `.venv` (Python 3.14). Finalize these in Phase 1:

- Lint and format: `.venv/bin/ruff check . && .venv/bin/ruff format --check .`
- Types: `.venv/bin/mypy src`
- Tests: `.venv/bin/pytest`
- Live network tests (opt-in, never required in CI): `.venv/bin/pytest -m live`
