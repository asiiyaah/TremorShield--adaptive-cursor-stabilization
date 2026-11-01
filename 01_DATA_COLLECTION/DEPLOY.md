# TREMORSHIELD Web — Deploy & Hosting Guide

## What changed vs. the desktop version
- `mouse_logger.py` (pynput, system-wide hook) → replaced by the browser's
  own `mousemove`/`mousedown`/`mouseup` events, streamed to the server over
  a WebSocket. This is actually an improvement on the old "honest
  limitation": events are naturally confined to the browser tab instead of
  the whole OS.
- `task_protocol_gui.py` (Tkinter, full-screen) → replaced by
  `frontend/static/app.js`, a canvas-based walkthrough of the exact same
  nine tasks, same durations, same instructions text.
- CSV schema, filenames, and the double-click heuristic are unchanged, so
  `synthetic_tremor.py` needs **zero modifications** — point it at whatever
  lands in `data/raw/<participant>_.../`.
- Data streams to the server continuously (flushed every ~150ms), not just
  at the end, so closing the tab mid-task loses at most a fraction of a
  second of data instead of the whole session.

## 1. Run it locally first
```bash
cd tremorshield_web
cp .env.example .env        # edit ADMIN_TOKEN at minimum
docker compose up --build
```
Open `http://localhost:8000`, do a full run-through yourself, then check
`data/raw/` for the CSVs and confirm `synthetic_tremor.py` still reads them.

## 2. Get it a public link
Any of these work; pick based on how long the study runs and whether you
want to keep it up afterward.

### Option A — Render.com (free tier, easiest)
1. Push this folder to a GitHub repo.
2. Render → New → Web Service → connect the repo.
3. Environment: **Docker**. It will auto-detect the `Dockerfile`.
4. Add environment variables from `.env.example` in the Render dashboard
   (Environment tab) instead of committing `.env`.
5. Add a **persistent disk** mounted at `/app/data` (Render → Disks) so
   session CSVs survive restarts — the free tier's filesystem is ephemeral
   otherwise.
6. Deploy. You get a URL like `https://tremorshield.onrender.com` — share
   that with participants.

### Option B — Fly.io
```bash
fly launch          # accept defaults, say yes to a Dockerfile-based app
fly volumes create tremordata --size 1   # persistent volume for data/
# edit fly.toml to mount it at /app/data, then:
fly secrets set ADMIN_TOKEN=... BREVO_API_KEY=... BREVO_SENDER_EMAIL=... BREVO_SENDER_NAME=TREMORSHIELD ADMIN_EMAIL=...
fly deploy
```

### Option C — Your own VPS (DigitalOcean/Linode/etc.), full control
```bash
# on the VPS, with Docker installed:
git clone <your repo>
cd tremorshield_web
cp .env.example .env && nano .env
docker compose up -d --build
```
Then put a reverse proxy in front for HTTPS (participants' browsers will
often refuse microphone/camera-style permissions over plain HTTP, and
WebSockets are safer over `wss://` too). Easiest is Caddy:
```bash
sudo apt install caddy
# /etc/caddy/Caddyfile
tremorshield.yourdomain.com {
    reverse_proxy localhost:8000
}
sudo systemctl reload caddy
```
Caddy handles the TLS certificate automatically.

## 3. Getting data back to you automatically
Two options, not mutually exclusive:

- **Email (fully automatic):** use Brevo's HTTP email API. Set
  `BREVO_API_KEY`, `BREVO_SENDER_EMAIL`, `BREVO_SENDER_NAME`, and
  `ADMIN_EMAIL` in `.env` (or your hosting provider's environment settings).
  Verify the sender address in Brevo first. The app sends email through
  Brevo's HTTPS API, so it does not require SMTP ports 465/587.
- **Manual pull, no email setup needed:**
  ```
  GET https://your-host/admin/sessions?token=YOUR_ADMIN_TOKEN        # list
  GET https://your-host/admin/download/<name>.zip?token=YOUR_ADMIN_TOKEN
  GET https://your-host/admin/download_all?token=YOUR_ADMIN_TOKEN    # everything, one zip
  ```
  Bookmark `download_all` and just refresh it whenever you want the latest
  batch.

## 4. Sharing the link with participants
Send them the base URL. They:
1. Enter a participant ID and session number.
2. Click **Begin Session** (this triggers fullscreen + starts the WebSocket).
3. Work through the nine tasks (SPACE to start each one).
4. See "Session complete" once their data has hit the server.

No installs, no `sudo apt install python3-tk` — just a browser.

## Notes / things worth deciding before a real study
- **Consent & anonymity:** the frontend doesn't collect anything beyond a
  participant ID you assign; make sure that ID isn't personally
  identifying if your ethics approval requires anonymity.
- **Tab-switch / focus loss:** browsers pause `mousemove` firing while the
  tab isn't focused, which is actually cleaner than the old "ask
  participants not to switch apps" caveat — but you may still want to
  detect `visibilitychange` and flag those rows if it matters for your
  analysis (not implemented here).
- **Concurrent participants:** the backend holds one `ServerSession` per
  browser tab in memory, so multiple participants can run at once from
  different links without interfering with each other.
