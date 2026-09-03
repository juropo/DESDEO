---
name: webui-testing
description: Launch DESDEO's web-API (FastAPI/uvicorn) and web-GUI (SvelteKit) together locally, then drive the running app with chrome-devtools to click through it, log in as a test user, capture screenshots, and check the browser console/network tab for errors. Use whenever asked to run/try out/test the DESDEO webui, verify a frontend change actually works, or reproduce a UI bug against the live app.
---

# Testing the DESDEO web-GUI with chrome-devtools

This drives the real app (backend + frontend) through the `mcp__chrome-devtools__*`
tools, rather than just reading code. Use it to verify a change end-to-end, explore
a feature, or reproduce a bug report.

## 1. Check prerequisites (skip anything already satisfied)

- Python deps: `uv sync` (needed for `desdeo.api`).
- Node deps: `ls webui/node_modules` — if missing, `cd webui && npm install`.
- `webui/.env` — if missing, create it (see `webui/README.md`):
  ```
  API_BASE_URL="http://localhost:8000"
  VITE_API_URL="/api"
  ```

## 2. Database — do NOT reseed by default

The API's SQLite DB lives at **`desdeo/api/test.db`** (uvicorn is started with
`cwd=desdeo/api`, and `db_database = "sqlite:///./test.db"` in `desdeo/api/config.toml`
is relative to the *working directory*, not the script). There is also a stray,
unrelated `test.db` at the repo root left over from someone once running
`python desdeo/api/db_init.py` from the wrong directory — ignore that one.

If `desdeo/api/test.db` already exists, **leave it alone**: `db_init.py` drops and
recreates all tables, destroying whatever problems/scenarios/results are in there.
Only run it if the DB is missing or the user explicitly asks to reset it:

```bash
cd desdeo/api && python db_init.py
```

## 3. Start the fullstack app in the background

```bash
python run_fullstack.py --no-open
```

`--no-open` skips auto-launching a browser tab (that's for a human's benefit, not
useful when chrome-devtools is driving its own tab — and it would pop an unwanted
window on the user's machine). Run this with `run_in_background: true` — it's a
long-lived foreground process (uvicorn `--reload` + `vite dev`, output interleaved
with colored prefixes) and never exits on its own. It binds:

- API: `http://127.0.0.1:8000` (docs at `/docs`, schema at `/openapi.json`)
- Web-GUI: `http://localhost:5173`

Poll for readiness before touching the browser:

```bash
until curl -s -o /dev/null http://localhost:8000/docs; do sleep 1; done
until curl -s -o /dev/null http://localhost:5173; do sleep 1; done
```

If a port is already bound (leftover process from a previous session), find and
kill it first: `pkill -f run_fullstack.py` (or `pkill -f uvicorn` / `pkill -f "vite dev"`
individually), then retry.

**Keep watching this process's output, not just the browser.** `run_fullstack.py`
interleaves both servers' logs with `(Backend)`/`(Frontend)` color prefixes — this
is where Python tracebacks (500s), failed hot-reloads, and Vite compile errors show
up, none of which necessarily surface in the browser console or network tab. The
Bash tool result for the backgrounded command gives an output file path; `Read` it
after each significant action in the browser (login, navigating, submitting a
form) to check for fresh errors, rather than only checking at the very end. For a
longer session, a `Monitor` tailing that file for error signatures
(`Traceback|ERROR|Internal Server Error`) works too — but a quick `Read` per
checkpoint is usually enough.

## 4. Test user credentials

Seeded by `db_init.py` from the `[server-debug]` section of `desdeo/api/config.toml`
(check that file — these are the defaults, but the file may have been edited locally):

| user | password | role |
|---|---|---|
| `analyst` | `analyst` | analyst |
| `dm1` | `dm1` | DM |
| `dm2` | `dm2` | DM |

## 5. Drive it with chrome-devtools

```
mcp__chrome-devtools__new_page          url: http://localhost:5173
mcp__chrome-devtools__take_snapshot     # get uids for the login form fields/button
mcp__chrome-devtools__fill              uid: <username field>, value: analyst
mcp__chrome-devtools__fill              uid: <password field>, value: analyst
mcp__chrome-devtools__click             uid: <login button>
mcp__chrome-devtools__take_snapshot     # confirm logged-in state, get uids to navigate
```

From there, `click` / `fill` / `take_snapshot` in a loop to walk through whatever
flow is being tested. Take a `take_screenshot` (save to the scratchpad dir) at
meaningful checkpoints, especially right before/after the thing being verified —
useful to send to the user with `SendUserFile`.

Prefer `take_snapshot` over `take_screenshot` for finding things to interact with:
it gives stable `uid`s per element, a screenshot doesn't.

**Check for problems as you go, not just at the end:**

- `mcp__chrome-devtools__list_console_messages` — JS errors/warnings.
- `mcp__chrome-devtools__list_network_requests` (and `get_network_request` for
  detail) — failed or unexpected API calls, useful when the frontend renders
  something wrong because a backend call 4xx'd/5xx'd.

Remember cookies are sent over plain HTTP in `npm run dev` mode (see
`webui/README.md`'s WSL note) — auth works fine here even though this is WSL2,
no HTTPS needed. It will *not* work against a `npm run preview` production build.

## 6. Clean up

- `mcp__chrome-devtools__close_page` the tab(s) you opened.
- Stop the fullstack process: it's the backgrounded `just fullstack`/`run_fullstack.py`
  job — stop it via the background task controls, or `pkill -f run_fullstack.py`.
- Leave `desdeo/api/test.db` as you found it — don't run `db_init.py` again unless
  asked.
