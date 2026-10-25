"""
TREMORSHIELD - Web Data Collection Backend

Browser replacement for mouse_logger.py + task_protocol_gui.py.
Participants open a link, do the task protocol in their browser, and their
mouse events stream to this server over a WebSocket as they go (so almost
nothing is lost if a tab crashes mid-session, unlike a purely local buffer).

Output schema is IDENTICAL to the desktop pipeline, so synthetic_tremor.py
and everything downstream needs zero changes:
timestamp,x,y,dx,dy,velocity,acceleration,event,button,drag,task,session_id,user_id
"""

import csv
import io
import os
import shutil
import time
import uuid
import base64
import json
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "changeme")
# Brevo HTTP API settings. This uses HTTPS (port 443), so it works on
# hosting platforms that block direct SMTP connections.
BREVO_API_KEY = os.environ.get("BREVO_API_KEY")
BREVO_SENDER_EMAIL = os.environ.get("BREVO_SENDER_EMAIL")
BREVO_SENDER_NAME = os.environ.get("BREVO_SENDER_NAME", "TREMORSHIELD")
ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL")

HEADER = ["timestamp", "elapsed_sec", "dt",
          "user_id", "session_id", "trial_id",
          "task", "action_type", "event", "button", "drag",
          "x", "y", "dx", "dy", "velocity", "acceleration", "direction_change",
          "target_x", "target_y", "target_width", "target_height", "target_id",
          "screen_width", "screen_height"]

# ---------------------------------------------------------------------------
# Same task table as task_protocol_gui.py (minus Tkinter rendering details;
# the frontend draws targets itself). Single source of truth served to the
# browser via GET /api/tasks so instructions/durations never drift.
# ---------------------------------------------------------------------------
TASK_ORDER = [
    "normal", "fast", "slow", "click", "double_click",
    "drag", "target_selection", "precision", "idle",
]

TASK_INFO = {
    "normal": {"duration": 30, "instructions": "Track the moving target with your mouse.\nFollow it at a comfortable, natural speed."},
    "fast": {"duration": 30, "instructions": "Track the moving target with your mouse.\nFollow the target as quickly and accurately as you can."},
    "slow": {"duration": 30, "instructions": "Track the moving target with your mouse.\nFollow it slowly and smoothly without rushing."},
    "click": {"duration": 30, "instructions": "Click the red target circle repeatedly.\nClick at whatever pace feels natural."},
    "double_click": {"duration": 30, "instructions": "Double-click the red target circle repeatedly."},
    "drag": {"duration": 30, "instructions": "Click and drag the blue square into the green box.\nRelease it inside, then drag it back out and repeat."},
    "target_selection": {"duration": 30, "instructions": "Click each orange target as soon as it appears.\nTargets vary in size and distance."},
    "precision": {"duration": 30, "instructions": "Make small, careful movements and clicks\nconfined to the marked box in the center."},
    "idle": {"duration": 30, "instructions": "Rest your hand on the mouse.\nDo not move it intentionally for the full duration."},
}


class StartSessionRequest(BaseModel):
    user_id: str
    session_id: int


@dataclass
class ServerSession:
    session_uuid: str
    user_id: str
    session_id: int
    out_dir: Path
    rows: list = field(default_factory=list)
    lock: Lock = field(default_factory=Lock)
    drag_active: bool = False
    last_t: float = None
    last_x: float = None
    last_y: float = None
    last_v: float = None
    last_dx: float = None
    last_dy: float = None
    task_start_t: float = None
    trial_id: int = None
    screen_width: int = None
    screen_height: int = None
    finished: bool = False

    def record(self, t, x, y, event, button="none", trial_id=None,
               target_x=None, target_y=None, target_width=None, target_height=None,
               target_id=None, screen_width=None, screen_height=None):
        with self.lock:
            if event == "press":
                self.drag_active = True
            elif event == "release":
                self.drag_active = False

            if self.last_t is None:
                dt = 0.0
                dx = dy = 0.0
                velocity = 0.0
                acceleration = 0.0
                direction_change = 0.0
            else:
                dt = max(t - self.last_t, 1e-4)
                dx = x - self.last_x
                dy = y - self.last_y
                velocity = (dx ** 2 + dy ** 2) ** 0.5 / dt
                acceleration = (velocity - (self.last_v or 0.0)) / dt
                if self.last_dx is not None and self.last_dy is not None and (self.last_dx != 0 or self.last_dy != 0) and (dx != 0 or dy != 0):
                    import math
                    dot = self.last_dx * dx + self.last_dy * dy
                    mag = math.hypot(self.last_dx, self.last_dy) * math.hypot(dx, dy)
                    direction_change = math.acos(max(-1.0, min(1.0, dot / mag))) if mag else 0.0
                else:
                    direction_change = 0.0

            if self.task_start_t is None or (trial_id is not None and trial_id != self.trial_id):
                self.task_start_t = t
            if trial_id is not None:
                self.trial_id = trial_id
            if screen_width is not None:
                self.screen_width = screen_width
            if screen_height is not None:
                self.screen_height = screen_height

            action_type = {
                "move": "mouse_move",
                "press": "button_press",
                "release": "button_release",
                "double_click": "double_click",
            }.get(event, event)
            elapsed_sec = max(0.0, t - self.task_start_t)

            self.rows.append([
                t, elapsed_sec, dt,
                self.user_id, self.session_id, self.trial_id,
                self.current_task, action_type, event, button, self.drag_active,
                x, y, dx, dy, velocity, acceleration, direction_change,
                target_x, target_y, target_width, target_height, target_id,
                self.screen_width, self.screen_height,
            ])
            self.last_t, self.last_x, self.last_y, self.last_v = t, x, y, velocity
            self.last_dx, self.last_dy = dx, dy

    current_task: str = "idle"

    def pop_rows_for_task(self, task_name):
        with self.lock:
            matching = [r for r in self.rows if r[6] == task_name]
            self.rows = [r for r in self.rows if r[6] != task_name]
        return matching

    def write_csv(self, rows, task_name):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%S")
        path = self.out_dir / f"{self.user_id}_s{self.session_id:02d}_{task_name}_{stamp}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
            writer.writerows(rows)
        return path


SESSIONS: dict[str, ServerSession] = {}
SESSIONS_LOCK = Lock()

app = FastAPI(title="TREMORSHIELD Web Collector")


@app.get("/api/tasks")
def get_tasks():
    return {"task_order": TASK_ORDER, "task_info": TASK_INFO}


@app.post("/api/start_session")
def start_session(req: StartSessionRequest):
    session_uuid = uuid.uuid4().hex
    out_dir = DATA_DIR / f"{req.user_id}_s{req.session_id:02d}_{session_uuid[:8]}"
    sess = ServerSession(
        session_uuid=session_uuid, user_id=req.user_id,
        session_id=req.session_id, out_dir=out_dir,
    )
    with SESSIONS_LOCK:
        SESSIONS[session_uuid] = sess
    return {"session_uuid": session_uuid}


@app.websocket("/ws/{session_uuid}")
async def ws_endpoint(websocket: WebSocket, session_uuid: str):
    await websocket.accept()
    sess = SESSIONS.get(session_uuid)
    if sess is None:
        await websocket.send_json({"type": "error", "message": "unknown session"})
        await websocket.close()
        return

    try:
        while True:
            msg = await websocket.receive_json()
            mtype = msg.get("type")

            if mtype == "events":
                task = msg.get("task", "idle")
                sess.current_task = task
                for ev in msg.get("events", []):
                    sess.record(
                        t=ev["t"], x=ev["x"], y=ev["y"],
                        event=ev["event"], button=ev.get("button", "none"),
                        trial_id=ev.get("trial_id"),
                        target_x=ev.get("target_x"), target_y=ev.get("target_y"),
                        target_width=ev.get("target_width"), target_height=ev.get("target_height"),
                        target_id=ev.get("target_id"),
                        screen_width=ev.get("screen_width"), screen_height=ev.get("screen_height"),
                    )

            elif mtype == "end_task":
                task = msg.get("task")
                rows = sess.pop_rows_for_task(task)
                path = sess.write_csv(rows, task)
                await websocket.send_json({"type": "task_saved", "task": task, "rows": len(rows), "path": str(path)})

            elif mtype == "finish":
                # flush anything left (e.g. idle task not explicitly ended)
                remaining_tasks = {r[6] for r in sess.rows}
                for t in remaining_tasks:
                    rows = sess.pop_rows_for_task(t)
                    sess.write_csv(rows, t)
                combined_path = combine_session_csvs(sess)
                sess.finished = True
                zip_path = zip_session(sess)
                emailed, email_error = try_email_zip(zip_path, combined_path, sess)
                if emailed:
                    message = "Session data saved and emailed to the study admin."
                elif email_error:
                    message = "Session data saved on the server, but the email could not be sent. The study admin can download it from the admin endpoint."
                else:
                    message = "Session data saved on the server. Email is not configured."
                await websocket.send_json({
                    "type": "finished",
                    "emailed": emailed,
                    "message": message,
                })
                break

    except WebSocketDisconnect:
        # Participant closed the tab mid-task. Whatever was already sent in
        # earlier "events" batches is safely on disk/in memory; only the
        # batch currently in flight (at most ~150ms of movement) is lost.
        if not sess.finished:
            remaining_tasks = {r[6] for r in sess.rows}
            for t in remaining_tasks:
                rows = sess.pop_rows_for_task(t)
                sess.write_csv(rows, t)


def zip_session(sess: ServerSession) -> Path:
    zip_path = sess.out_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for csv_file in sorted(sess.out_dir.glob("*.csv")):
            zf.write(csv_file, arcname=csv_file.name)
    return zip_path


def combine_session_csvs(sess: ServerSession) -> Path:
    """Create one combined CSV for convenient analysis/email."""
    combined = sess.out_dir / f"{sess.user_id}_s{sess.session_id:02d}_ALL_TASKS.csv"
    csv_files = sorted(sess.out_dir.glob("*.csv"))
    # Do not include an old combined file if a session is retried.
    csv_files = [p for p in csv_files if p != combined]
    with open(combined, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(HEADER)
        for csv_file in csv_files:
            with open(csv_file, newline="", encoding="utf-8") as in_f:
                reader = csv.reader(in_f)
                next(reader, None)
                writer.writerows(reader)
    return combined


def _brevo_attachment(path: Path):
    with open(path, "rb") as f:
        content = base64.b64encode(f.read()).decode("ascii")
    return {"name": path.name, "content": content}


def _send_email_brevo(zip_path: Path, combined_path: Path, sess: ServerSession):
    """Send session data through Brevo's HTTPS API (TCP 443)."""
    payload = {
        "sender": {
            "name": BREVO_SENDER_NAME,
            "email": BREVO_SENDER_EMAIL,
        },
        "to": [{"email": ADMIN_EMAIL}],
        "subject": f"TREMORSHIELD data — {sess.user_id} session {sess.session_id}",
        "textContent": (
            f"TREMORSHIELD session data for participant {sess.user_id}, "
            f"session {sess.session_id}.\n\n"
            "Attachments: one combined CSV and the ZIP containing the "
            "individual task CSV files."
        ),
        "attachment": [
            _brevo_attachment(zip_path),
            _brevo_attachment(combined_path),
        ],
    }

    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        "https://api.brevo.com/v3/smtp/email",
        data=body,
        method="POST",
        headers={
            "accept": "application/json",
            "api-key": BREVO_API_KEY,
            "content-type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response_body = response.read().decode("utf-8", errors="replace")
            if 200 <= response.status < 300:
                print(f"[email] sent successfully to {ADMIN_EMAIL} via Brevo HTTPS")
                return True, None
            return False, f"Brevo HTTP {response.status}: {response_body}"
    except urllib.error.HTTPError as e:
        details = e.read().decode("utf-8", errors="replace")
        return False, f"Brevo HTTP {e.code}: {details}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def try_email_zip(zip_path: Path, combined_path: Path, sess: ServerSession):
    """Email the ZIP archive and combined CSV through Brevo's HTTPS API.

    Returns (success, error_message). If the Brevo settings are absent,
    returns (False, None) so email remains optional.
    """
    if not (BREVO_API_KEY and BREVO_SENDER_EMAIL and ADMIN_EMAIL):
        return False, None

    return _send_email_brevo(zip_path, combined_path, sess)


# ---------------------------------------------------------------------------
# Email test endpoint. Useful for verifying Brevo settings before collecting
# participant data.
# ---------------------------------------------------------------------------
@app.post("/admin/test_email")
def test_email(token: str = Query(...)):
    if token != ADMIN_TOKEN:
        raise HTTPException(403, "bad token")
    if not (BREVO_API_KEY and BREVO_SENDER_EMAIL and ADMIN_EMAIL):
        raise HTTPException(
            400,
            "BREVO_API_KEY, BREVO_SENDER_EMAIL and ADMIN_EMAIL are not configured",
        )

    # Use a tiny text-only test request rather than attaching session files.
    payload = {
        "sender": {
            "name": BREVO_SENDER_NAME,
            "email": BREVO_SENDER_EMAIL,
        },
        "to": [{"email": ADMIN_EMAIL}],
        "subject": "TREMORSHIELD test email",
        "textContent": "TREMORSHIELD Brevo email configuration is working.",
    }

    request = urllib.request.Request(
        "https://api.brevo.com/v3/smtp/email",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "accept": "application/json",
            "api-key": BREVO_API_KEY,
            "content-type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            if 200 <= response.status < 300:
                return {
                    "ok": True,
                    "message": f"Test email sent to {ADMIN_EMAIL} via Brevo HTTPS.",
                }
            details = response.read().decode("utf-8", errors="replace")
            raise HTTPException(502, f"Brevo HTTP {response.status}: {details}")
    except urllib.error.HTTPError as e:
        details = e.read().decode("utf-8", errors="replace")
        raise HTTPException(502, f"Brevo HTTP {e.code}: {details}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Brevo request failed: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Admin endpoints — list/download session archives without relying on email.
# ---------------------------------------------------------------------------
@app.get("/admin/sessions")
def list_sessions(token: str = Query(...)):
    if token != ADMIN_TOKEN:
        raise HTTPException(403, "bad token")
    zips = sorted(DATA_DIR.glob("*.zip"))
    return {"sessions": [z.name for z in zips]}


@app.get("/admin/download/{name}")
def download_session(name: str, token: str = Query(...)):
    if token != ADMIN_TOKEN:
        raise HTTPException(403, "bad token")
    path = DATA_DIR / name
    if not path.exists() or path.suffix != ".zip":
        raise HTTPException(404, "not found")
    return FileResponse(path, filename=name)


@app.get("/admin/download_all")
def download_all(token: str = Query(...)):
    if token != ADMIN_TOKEN:
        raise HTTPException(403, "bad token")
    buf_path = Path("/tmp/tremorshield_all.zip")
    with zipfile.ZipFile(buf_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for z in DATA_DIR.glob("*.zip"):
            zf.write(z, arcname=z.name)
    return FileResponse(buf_path, filename="tremorshield_all_sessions.zip")


# Serve the frontend last so /api and /admin and /ws take priority.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
