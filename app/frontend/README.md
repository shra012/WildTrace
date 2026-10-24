# WildTrace Studio (frontend)

Next.js (App Router, TypeScript) + Tailwind CSS 4 + DaisyUI 5 single-page studio for WildTrace:
chat for requests, a live view of the xArm 7 drawing, and the pipeline for each job. The backend
contract is `app/API.md`.

## Run

```bash
cd app/frontend
npm install
cp .env.example .env.local   # optional, defaults are fine for a local backend
npm run dev                  # http://localhost:3000, talks to http://localhost:8000
```

Start the backend in another shell from the repo root:

```bash
uv run --extra app uvicorn wildtrace_app.main:app --app-dir app/backend --port 8000
```

## Mock mode (no backend)

```bash
npm run dev:mock             # same as NEXT_PUBLIC_MOCK=1 npm run dev
```

Or add `?mock=1` to the URL of any build. Mock mode runs an in-browser fake backend with seeded
jobs (a group photo waiting for a face pick, a ready dog and cat, a failed fish), a running robot
run with an animated trace and telemetry, catalog items with inline SVGs, and warning logs.
Typing "group photo" in the chat creates another face job with several faces.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `NEXT_PUBLIC_API_BASE` | `http://localhost:8000` | Backend base URL (no trailing slash) |
| `NEXT_PUBLIC_WS_URL` | API base with `ws(s)://` + `/ws` | Event stream URL override |
| `NEXT_PUBLIC_MOCK` | `0` | `1` serves everything from the built-in mock backend |

`NEXT_PUBLIC_*` values are inlined at build time, so rebuild after changing them.

## Scripts

- `npm run dev` / `npm run dev:mock` - development server
- `npm run build` then `npm run start` - production build and server
- `npm run lint` - ESLint (Next.js core web vitals + TypeScript rules)
- `npm run typecheck` - TypeScript only

## Layout of the code

- `src/lib/types.ts` - API types (mirror `app/API.md`)
- `src/lib/api.ts` - the single typed API client (mock-aware)
- `src/lib/mock.ts`, `src/lib/drawings.ts` - mock backend and procedural line drawings
- `src/lib/studio.ts` - app state (tiny external store) and all actions, polling and health checks
- `src/hooks/useWebSocket.ts` - WebSocket hook with backoff reconnect and stale detection
- `src/components/` - navbar, chat, live view (trace canvas, cameras), pipeline, gallery
