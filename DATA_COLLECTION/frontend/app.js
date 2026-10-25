// TREMORSHIELD web collector — browser replacement for
// mouse_logger.py (pynput) + task_protocol_gui.py (Tkinter).
// Mouse events are captured on the full-viewport canvas and streamed to the
// backend every ~150ms over a WebSocket, so a crashed/closed tab loses at
// most one flush interval of data instead of a whole session.

const screens = {
  login: document.getElementById("screen-login"),
  instructions: document.getElementById("screen-instructions"),
  task: document.getElementById("screen-task"),
  done: document.getElementById("screen-done"),
};
const canvas = document.getElementById("stage");
const ctx = canvas.getContext("2d");
const hudTask = document.getElementById("hud-task");
const hudTime = document.getElementById("hud-time");

let ws = null;
let sessionUuid = null;
let TASK_ORDER = [];
let TASK_INFO = {};
let taskIndex = 0;
let currentTask = null;
let taskEndAt = 0;
let taskStartAt = 0;
let currentTrialId = 0;
let tickHandle = null;
let flushHandle = null;
let eventBuffer = [];

// drag-task state
let dragState = { dragging: false, x: 100, y: 100, size: 60 };
// target_selection state
let fittsTarget = null;
let trackingTarget = null;
let fittsTargetSeq = 0;
let trackingPhase = 0;
// double-click heuristic state (mirrors task_protocol_gui.py _on_canvas_click)
let lastClickTime = 0;
let lastClickPos = [0, 0];

function showScreen(name) {
  Object.values(screens).forEach(s => s.classList.add("hidden"));
  screens[name].classList.remove("hidden");
}

function resizeCanvas() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}
window.addEventListener("resize", resizeCanvas);

function now() {
  return performance.now() / 1000; // seconds, matches time.perf_counter()
}

function buttonName(e) {
  return e.button === 0 ? "left" : e.button === 2 ? "right" : e.button === 1 ? "middle" : "unknown";
}

// ---------------------------------------------------------------------
// Networking
// ---------------------------------------------------------------------
async function startSession(userId, sessionId) {
  const res = await fetch("/api/start_session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, session_id: Number(sessionId) }),
  });
  if (!res.ok) throw new Error("could not start session");
  const data = await res.json();
  return data.session_uuid;
}

async function loadTasks() {
  const res = await fetch("/api/tasks");
  const data = await res.json();
  TASK_ORDER = data.task_order;
  TASK_INFO = data.task_info;
}

function openSocket(uuid) {
  return new Promise((resolve, reject) => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws/${uuid}`);
    ws.onopen = () => resolve();
    ws.onerror = (e) => reject(e);
    ws.onmessage = (msg) => {
      const data = JSON.parse(msg.data);
      if (data.type === "finished") {
        document.getElementById("done-msg").textContent = data.message;
      }
    };
  });
}

function flushBuffer() {
  if (eventBuffer.length === 0 || !ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: "events", task: currentTask, events: eventBuffer }));
  eventBuffer = [];
}

function getTargetMeta() {
  if (currentTask === "normal" || currentTask === "fast" || currentTask === "slow") {
    if (!trackingTarget) return {};
    return { target_x: trackingTarget.x, target_y: trackingTarget.y, target_width: 48, target_height: 48, target_id: "tracking_target" };
  }
  if (currentTask === "click" || currentTask === "double_click") {
    return { target_x: canvas.width / 2, target_y: canvas.height / 2, target_width: 60, target_height: 60, target_id: "click_target" };
  }
  if (currentTask === "target_selection" && fittsTarget) {
    return { target_x: fittsTarget.x, target_y: fittsTarget.y, target_width: fittsTarget.r * 2, target_height: fittsTarget.r * 2, target_id: fittsTarget.id };
  }
  if (currentTask === "precision") {
    return { target_x: canvas.width / 2, target_y: canvas.height / 2, target_width: 20, target_height: 20, target_id: "precision_target" };
  }
  if (currentTask === "drag") {
    return { target_x: canvas.width - 140, target_y: canvas.height - 140, target_width: 160, target_height: 160, target_id: "drag_drop_target" };
  }
  return {};
}

function pushEvent(x, y, event, button = "none") {
  eventBuffer.push({
    t: now(), x, y, event, button,
    trial_id: currentTrialId,
    screen_width: window.innerWidth,
    screen_height: window.innerHeight,
    ...getTargetMeta(),
  });
}

// ---------------------------------------------------------------------
// Mouse capture (bound once the task screen is shown)
// ---------------------------------------------------------------------
function onMouseMove(e) {
  pushEvent(e.clientX, e.clientY, "move");
  if (currentTask === "drag" && dragState.dragging) {
    dragState.x = e.clientX - dragState.size / 2;
    dragState.y = e.clientY - dragState.size / 2;
  }
}

function onMouseDown(e) {
  pushEvent(e.clientX, e.clientY, "press", buttonName(e));

  // double-click heuristic, same thresholds as task_protocol_gui.py
  const t = now();
  const dt = t - lastClickTime;
  const dist = Math.hypot(e.clientX - lastClickPos[0], e.clientY - lastClickPos[1]);
  if (currentTask === "double_click" && dt < 0.4 && dist < 15) {
    pushEvent(e.clientX, e.clientY, "double_click", "left");
  }
  lastClickTime = t;
  lastClickPos = [e.clientX, e.clientY];

  if (currentTask === "target_selection") {
    spawnFittsTarget();
  }
  if (currentTask === "drag") {
    const dx = e.clientX - (dragState.x + dragState.size / 2);
    const dy = e.clientY - (dragState.y + dragState.size / 2);
    if (Math.hypot(dx, dy) < dragState.size) dragState.dragging = true;
  }
}

function onMouseUp(e) {
  pushEvent(e.clientX, e.clientY, "release", buttonName(e));
  dragState.dragging = false;
}

function bindMouseCapture() {
  canvas.addEventListener("mousemove", onMouseMove);
  canvas.addEventListener("mousedown", onMouseDown);
  canvas.addEventListener("mouseup", onMouseUp);
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
}

// ---------------------------------------------------------------------
// Task rendering (canvas equivalents of the Tkinter _render_* methods)
// ---------------------------------------------------------------------
function drawTarget(x, y, r = 30, color = "red") {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
}

function spawnFittsTarget() {
  const r = 10 + Math.random() * 40;
  const x = r + 20 + Math.random() * (canvas.width - 2 * (r + 20));
  const y = r + 20 + Math.random() * (canvas.height - 2 * (r + 20));
  fittsTargetSeq += 1;
  fittsTarget = { x, y, r, id: `fitts_target_${fittsTargetSeq}` };
}

function renderTrackingTarget() {
  const elapsed = Math.max(0, (performance.now() - (taskEndAt - TASK_INFO[currentTask].duration * 1000)) / 1000);
  const speed = currentTask === "fast" ? 520 : currentTask === "slow" ? 90 : 220;
  const margin = 70;
  const usableW = Math.max(200, canvas.width - margin * 2);
  const usableH = Math.max(160, canvas.height - margin * 2);
  let x, y;

  if (currentTask === "fast") {
    const travel = (elapsed * speed) % (usableW * 2);
    const px = travel <= usableW ? travel : usableW * 2 - travel;
    x = margin + px;
    y = canvas.height / 2 + Math.sin(elapsed * 3.5) * Math.min(usableH * 0.35, 180);
  } else if (currentTask === "slow") {
    const a = elapsed * 0.55;
    x = canvas.width / 2 + Math.cos(a) * Math.min(usableW * 0.42, 360);
    y = canvas.height / 2 + Math.sin(a) * Math.min(usableH * 0.35, 220);
  } else {
    const a = elapsed * 0.9;
    x = canvas.width / 2 + Math.cos(a) * Math.min(usableW * 0.40, 380);
    y = canvas.height / 2 + Math.sin(a * 1.35) * Math.min(usableH * 0.30, 190);
  }

  trackingTarget = { x, y };
  drawTarget(x, y, 24, "#2563eb");
  ctx.beginPath();
  ctx.arc(x, y, 7, 0, Math.PI * 2);
  ctx.fillStyle = "#fff";
  ctx.fill();
}

function renderFrame() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (currentTask === "normal" || currentTask === "fast" || currentTask === "slow") {
    renderTrackingTarget();
  } else if (currentTask === "click" || currentTask === "double_click") {
    drawTarget(canvas.width / 2, canvas.height / 2, 30, "red");
  } else if (currentTask === "drag") {
    ctx.fillStyle = "blue";
    ctx.fillRect(dragState.x, dragState.y, dragState.size, dragState.size);
    ctx.strokeStyle = "green";
    ctx.lineWidth = 3;
    ctx.strokeRect(canvas.width - 220, canvas.height - 220, 160, 160);
  } else if (currentTask === "target_selection") {
    if (!fittsTarget) spawnFittsTarget();
    drawTarget(fittsTarget.x, fittsTarget.y, fittsTarget.r, "orange");
  } else if (currentTask === "precision") {
    const cx = canvas.width / 2, cy = canvas.height / 2, box = 120;
    ctx.strokeStyle = "purple";
    ctx.lineWidth = 3;
    ctx.strokeRect(cx - box, cy - box, box * 2, box * 2);
    drawTarget(cx, cy, 10, "red");
  }
  // idle intentionally has no visual target.
}

// ---------------------------------------------------------------------
// Flow control (mirrors CollectionApp in task_protocol_gui.py)
// ---------------------------------------------------------------------
function nextTask() {
  if (taskIndex >= TASK_ORDER.length) {
    finishSession();
    return;
  }
  const task = TASK_ORDER[taskIndex];
  taskIndex += 1;
  showInstructions(task);
}

function showInstructions(task) {
  showScreen("instructions");
  document.getElementById("instr-title").textContent =
    `TASK ${taskIndex}/${TASK_ORDER.length}: ${task.toUpperCase()}`;
  document.getElementById("instr-body").textContent = TASK_INFO[task].instructions;

  const btn = document.getElementById("startTaskBtn");
  btn.onclick = () => runTask(task);
  btn.focus();
}


function runTask(task) {
  currentTask = task;
  showScreen("task");
  resizeCanvas();
  dragState = { dragging: false, x: 100, y: 100, size: 60 };
  fittsTarget = null;
  trackingTarget = null;
  fittsTargetSeq = 0;
  eventBuffer = [];
  currentTrialId = taskIndex;
  taskStartAt = performance.now();
  taskEndAt = taskStartAt + TASK_INFO[task].duration * 1000;

  flushHandle = setInterval(flushBuffer, 150);
  tickHandle = setInterval(tick, 100);
}

function tick() {
  const remaining = Math.max(0, taskEndAt - performance.now());
  hudTask.textContent = currentTask;
  hudTime.textContent = Math.ceil(remaining / 1000);
  renderFrame();
  if (remaining <= 0) endTask(currentTask);
}

function endTask(task) {
  clearInterval(tickHandle);
  clearInterval(flushHandle);
  flushBuffer();
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "end_task", task }));
  }
  nextTask();
}

function finishSession() {
  showScreen("done");
  document.getElementById("done-msg").textContent = "Saving your data…";
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "finish" }));
  }
  if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
}

// ---------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------
document.getElementById("beginBtn").addEventListener("click", async () => {
  const userId = document.getElementById("userId").value.trim() || "u01";
  const sessionId = document.getElementById("sessionId").value.trim() || "1";

  // Request fullscreen synchronously within the click gesture.
  document.documentElement.requestFullscreen().catch(() => {
    // fullscreen denied/unsupported — continue anyway, just not fullscreen
  });

  try {
    sessionUuid = await startSession(userId, sessionId);
    await loadTasks();
    await openSocket(sessionUuid);
    resizeCanvas();
    bindMouseCapture();
    taskIndex = 0;
    nextTask();
  } catch (err) {
    alert("Could not start session: " + err.message);
  }
});
