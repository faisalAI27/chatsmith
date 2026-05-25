# ChatSMITH Frontend (Vite + React)

Minimal scaffold to talk to the FastAPI backend.

## Prereqs
- Node ^20.19.0 or >=22.12.0 (required by Vite 7)
- Running backend API (defaults to http://localhost:8000/api)

## Env
Optional `frontend/.env`:
```
VITE_API_BASE_URL=http://localhost:8000/api
```

## Install & Run
```bash
cd frontend
npm install
npm run dev   # opens on 5173
```

## Screens
- App → submit URL (+force refresh) to `/api/jobs/run` (dev sync) and view JSON result
- Chat → send messages to `/api/chat` using the generated system prompt

## Notes
- This is a dev scaffold. `/api/jobs/run` currently calls the pipeline synchronously; in production replace with a queued endpoint and add status polling.
- Styling is lightweight; adjust in `src/styles.css`.
