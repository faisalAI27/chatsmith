# Codex Guidance

## Conventions
- Keep existing FastAPI endpoints and React flows stable; add new functionality behind flags/config toggles.
- Prefer small, composable helpers in `backend/app/services/` for pipeline changes.
- Do not commit secrets (`.env`), caches (`knowledge_files/`), build artifacts (`frontend/dist/`), or logs (`metrics_logs/`).

## Metrics
- New chatbot metrics live in `backend/app/services/scrape_pipeline.py`.
- Add supporting helpers in `backend/app/services/metrics_logger.py`.
- Guard all metrics code with the `ENABLE_METRICS_LOGGING` environment variable (default `false`).
- Surface new metrics via API stats objects without breaking existing fields.

## Commands to run after changes
- Backend: `pip install -r requirements.txt` (if deps change), then `pytest`.
- Frontend: `cd frontend && npm install` (if deps change), then `npm run build`.

## Feature flags
- All new features/metrics must be optional and off by default:
  - `ENABLE_METRICS_LOGGING` controls metrics/telemetry.
  - Wrap new logic in `if os.getenv("ENABLE_METRICS_LOGGING", "false").lower() == "true": ...`.

## Testing expectations
- Add/maintain unit tests for new logic; prefer fast, isolated tests with mocks over network calls.
- If modifying the pipeline, ensure cache behavior and new stats remain backward compatible.
