# ChatSMITH Frontend (Vite + React)

Minimal scaffold to talk to the FastAPI backend and Supabase auth.

## Prereqs
- Node 18+
- Supabase project (URL + anon key)
- Running backend API (defaults to http://localhost:8000/api)

## Env
Create `frontend/.env`:
```
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key
VITE_API_BASE_URL=http://localhost:8000/api
```

## Install & Run
```bash
cd frontend
npm install
npm run dev   # opens on 5173
```

## Screens
- Login → “Don’t have an account? Sign up”
- Sign up → first/last/email/password → sends OTP → OTP screen
- OTP screen → verify and log in
- App → submit URL (+force refresh) to `/api/jobs/run` (dev sync) and view JSON result
- Session panel → shows logged-in email and logout

## Notes
- This is a dev scaffold. `/api/jobs/run` currently calls the pipeline synchronously; in production replace with a queued endpoint and add status polling.
- Styling is lightweight; adjust in `src/styles.css`.
