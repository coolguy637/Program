/* ── Agentic Judge Dashboard ─────────────────────────────────────────────── */

const API = (() => {
  const u = new URL(window.location.href);
  return `${u.protocol}//${u.host}`;
})();
let authToken = localStorage.getItem("aj_token");
let currentUser = null;
let ws = null;

/* ── API Client ──────────────────────────────────────────────────────────── */

async function api(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...options.headers };
  if (authToken) headers["Authorization"] = `Bearer ${authToken}`;

  const res = await fetch(`${API}${path}`, { ...options, headers });

  if (res.status === 401) {
    logout();
    throw new Error("Session expired");
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
}

/* ── Auth ────────────────────────────────────────────────────────────────── */

async function signup() {
  const email = document.getElementById("auth-email").value;
  const username = document.getElementById("auth-username").value;
  const password = document.getElementById("auth-password").value;
  clearAlerts();

  try {
    await api("/api/users/signup", {
      method: "POST",
      body: JSON.stringify({ email, username, password }),
    });
    showAlert("auth-alert", "Account created! Logging in...", "success");
    await loginWithCredentials(username, password);
  } catch (e) {
    showAlert("auth-alert", e.message, "error");
  }
}

async function login() {
  const username = document.getElementById("auth-username").value;
  const password = document.getElementById("auth-password").value;
  clearAlerts();
  await loginWithCredentials(username, password);
}

async function loginWithCredentials(username, password) {
  try {
    const data = await api("/api/users/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    });
    authToken = data.access_token;
    localStorage.setItem("aj_token", authToken);
    await loadUser();
    showDashboard();
  } catch (e) {
    showAlert("auth-alert", e.message, "error");
  }
}

function logout() {
  authToken = null;
  currentUser = null;
  localStorage.removeItem("aj_token");
  showAuth();
}

async function loadUser() {
  currentUser = await api("/api/users/me");
  const nameEl = document.getElementById("user-name");
  const emailEl = document.getElementById("user-email");
  const avatarEl = document.getElementById("user-avatar");
  if (nameEl) nameEl.textContent = currentUser.username;
  if (emailEl) emailEl.textContent = currentUser.email;
  if (avatarEl) avatarEl.textContent = currentUser.username[0].toUpperCase();
}

/* ── Navigation ──────────────────────────────────────────────────────────── */

function showAuth() {
  document.getElementById("auth-page").classList.remove("hidden");
  document.getElementById("dashboard-page").classList.add("hidden");
}

function showDashboard() {
  document.getElementById("auth-page").classList.add("hidden");
  document.getElementById("dashboard-page").classList.remove("hidden");
  navigate("judge");
}

function toggleAuthMode() {
  const title = document.getElementById("auth-title");
  const btn = document.getElementById("auth-submit");
  const toggle = document.getElementById("auth-toggle-text");
  const emailGroup = document.getElementById("email-group");

  if (btn.textContent === "Sign Up") {
    btn.textContent = "Log In";
    btn.onclick = login;
    title.textContent = "Welcome Back";
    toggle.innerHTML = 'New here? <a onclick="toggleAuthMode()">Create account</a>';
    emailGroup.classList.add("hidden");
  } else {
    btn.textContent = "Sign Up";
    btn.onclick = signup;
    title.textContent = "Create Account";
    toggle.innerHTML = 'Already have an account? <a onclick="toggleAuthMode()">Log in</a>';
    emailGroup.classList.remove("hidden");
  }
  clearAlerts();
}

function navigate(page) {
  document.querySelectorAll(".nav-item").forEach((el) => el.classList.remove("active"));
  document.querySelectorAll(".page-content").forEach((el) => el.classList.add("hidden"));

  const navEl = document.querySelector(`[data-page="${page}"]`);
  const pageEl = document.getElementById(`page-${page}`);
  if (navEl) navEl.classList.add("active");
  if (pageEl) pageEl.classList.remove("hidden");

  if (page === "keys") loadAPIKeys();
  else if (page === "judge") loadSessions();
  else if (page === "sessions") loadSessions();
}

/* ── API Keys ────────────────────────────────────────────────────────────── */

async function loadAPIKeys() {
  try {
    const keys = await api("/api/keys/");
    const list = document.getElementById("keys-list");
    if (!list) return;
    list.innerHTML = keys
      .map(
        (k) => `
      <div class="key-item fade-in">
        <div class="key-info">
          <div class="key-name">${esc(k.name)}</div>
          <div class="key-value">${esc(k.key.slice(0, 20))}...</div>
          <div class="key-meta">Created: ${new Date(k.created_at).toLocaleDateString()}
            ${k.last_used_at ? " | Last used: " + new Date(k.last_used_at).toLocaleDateString() : ""}</div>
        </div>
        <div style="display:flex;gap:8px;align-items:center">
          <button class="btn btn-outline btn-sm" onclick="copyKey('${esc(k.key)}')">Copy</button>
          <button class="btn btn-danger btn-sm" onclick="revokeKey('${esc(k.id)}')">Revoke</button>
        </div>
      </div>`
      )
      .join("");
  } catch (e) {
    console.error("Failed to load keys:", e);
  }
}

async function createAPIKey() {
  const name = document.getElementById("key-name").value;
  if (!name) return;
  try {
    const key = await api("/api/keys/", {
      method: "POST",
      body: JSON.stringify({ name }),
    });
    document.getElementById("key-name").value = "";
    showAlert("key-alert", `Key created: ${key.key}`, "success");
    loadAPIKeys();
  } catch (e) {
    showAlert("key-alert", e.message, "error");
  }
}

async function revokeKey(id) {
  if (!confirm("Revoke this API key?")) return;
  try {
    await api(`/api/keys/${id}`, { method: "DELETE" });
    loadAPIKeys();
  } catch (e) {
    showAlert("key-alert", e.message, "error");
  }
}

function copyKey(key) {
  navigator.clipboard.writeText(key);
}

/* ── Judge Sessions ──────────────────────────────────────────────────────── */

async function loadSessions() {
  try {
    const sessions = await api("/api/judge/");
    const list = document.getElementById("sessions-list");
    if (!list) return;

    const statsEl = document.getElementById("session-stats");
    if (statsEl) {
      const total = sessions.length;
      const completed = sessions.filter((s) => s.status === "completed").length;
      const running = sessions.filter((s) => s.status === "running").length;
      const avgIterations =
        total > 0
          ? (sessions.reduce((sum, s) => sum + s.iteration_count, 0) / total).toFixed(1)
          : 0;
      statsEl.innerHTML = `
        <div class="stat-card"><div class="stat-value">${total}</div><div class="stat-label">Total Sessions</div></div>
        <div class="stat-card"><div class="stat-value">${completed}</div><div class="stat-label">Completed</div></div>
        <div class="stat-card"><div class="stat-value">${running}</div><div class="stat-label">Running</div></div>
        <div class="stat-card"><div class="stat-value">${avgIterations}</div><div class="stat-label">Avg Iterations</div></div>`;
    }

    list.innerHTML = sessions
      .map(
        (s) => `
      <div class="session-item fade-in" onclick="viewSession('${s.id}')">
        <div class="session-header">
          <span class="card-title">Session ${s.id.slice(0, 8)}</span>
          <span class="status-badge status-${s.status}">${s.status}</span>
        </div>
        <div class="session-prompt">${esc(s.prompt.slice(0, 120))}</div>
        <div class="key-meta mt-16">${s.iteration_count} iterations | ${new Date(s.created_at).toLocaleString()}</div>
      </div>`
      )
      .join("");
  } catch (e) {
    console.error("Failed to load sessions:", e);
  }
}

async function startJudgeSession() {
  const prompt = document.getElementById("judge-prompt").value;
  const maxIter = parseInt(document.getElementById("judge-max-iter").value) || 5;
  if (!prompt) return;

  const btn = document.getElementById("judge-submit");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Running...';

  try {
    const session = await api("/api/judge/", {
      method: "POST",
      body: JSON.stringify({ prompt, max_iterations: maxIter }),
    });
    document.getElementById("judge-prompt").value = "";
    loadSessions();
    viewSession(session.id);
  } catch (e) {
    showAlert("judge-alert", e.message, "error");
  } finally {
    btn.disabled = false;
    btn.innerHTML = "Run Judge Pipeline";
  }
}

async function viewSession(sessionId) {
  try {
    const session = await api(`/api/judge/${sessionId}`);
    const detail = document.getElementById("session-detail");
    if (!detail) return;

    detail.classList.remove("hidden");
    detail.innerHTML = `
      <div class="card fade-in">
        <div class="card-header">
          <div>
            <div class="card-title">Session ${session.id.slice(0, 8)}</div>
            <div class="card-subtitle">${new Date(session.created_at).toLocaleString()}</div>
          </div>
          <span class="status-badge status-${session.status}">${session.status}</span>
        </div>

        <div class="form-group">
          <label>Prompt</label>
          <div class="output-display" style="max-height:100px">${esc(session.prompt)}</div>
        </div>

        ${session.iterations
          .map(
            (it) => `
          <div class="iteration-card">
            <div class="iteration-header">
              <strong>Iteration ${it.iteration_number}</strong>
              <span class="status-badge ${it.passed ? "status-completed" : "status-failed"}">${it.passed ? "PASSED" : "NEEDS REVISION"}</span>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:12px">
              ${scoreCard("Technical", it.technical_score)}
              ${scoreCard("Fact Check", it.fact_score)}
              ${scoreCard("UX/Taste", it.ux_score)}
            </div>
            ${it.correction_directive ? `<div style="font-size:12px;color:var(--warning);margin-top:8px"><strong>Correction:</strong> ${esc(it.correction_directive.slice(0, 300))}</div>` : ""}
          </div>`
          )
          .join("")}

        ${
          session.final_output
            ? `<div class="form-group">
            <label>Final Output</label>
            <div class="output-display">${esc(session.final_output)}</div>
          </div>`
            : ""
        }

        <div class="override-controls">
          <button class="btn btn-warning btn-sm" onclick="sendOverride('${session.id}', 'pause')">Pause</button>
          <button class="btn btn-success btn-sm" onclick="sendOverride('${session.id}', 'resume')">Resume</button>
          <button class="btn btn-primary btn-sm" onclick="sendOverride('${session.id}', 'takeover')">Take Over</button>
          <button class="btn btn-outline btn-sm" onclick="sendOverride('${session.id}', 'release')">Release</button>
          <button class="btn btn-outline btn-sm" onclick="connectStream('${session.id}')">Live Stream</button>
        </div>

        <div id="stream-output" class="stream-container mt-16 hidden"></div>
      </div>`;
  } catch (e) {
    console.error("Failed to load session:", e);
  }
}

function scoreCard(label, score) {
  if (score === null || score === undefined) return `<div></div>`;
  const pct = Math.round(score * 100);
  const cls = pct >= 80 ? "score-high" : pct >= 50 ? "score-mid" : "score-low";
  return `
    <div>
      <div style="font-size:12px;color:var(--text-muted)">${label}</div>
      <div style="font-size:18px;font-weight:600">${pct}%</div>
      <div class="score-bar"><div class="score-fill ${cls}" style="width:${pct}%"></div></div>
    </div>`;
}

/* ── Override & WebSocket ────────────────────────────────────────────────── */

async function sendOverride(sessionId, action) {
  try {
    const payload = action === "inject" ? prompt("Enter content to inject:") : null;
    await api("/api/override/", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, action, payload }),
    });
    viewSession(sessionId);
  } catch (e) {
    console.error("Override failed:", e);
  }
}

function connectStream(sessionId) {
  const output = document.getElementById("stream-output");
  if (!output) return;
  output.classList.remove("hidden");
  output.innerHTML = '<div class="stream-event">Connecting to live stream...</div>';

  if (ws) ws.close();

  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${protocol}//${location.host}/api/override/ws/${sessionId}`);

  ws.onopen = () => {
    output.innerHTML += '<div class="stream-event"><span class="event-type">CONNECTED</span> Live stream active</div>';
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    const time = data.timestamp ? new Date(data.timestamp).toLocaleTimeString() : "";
    output.innerHTML += `
      <div class="stream-event">
        <span class="timestamp">${time}</span>
        <span class="event-type">${(data.event || "").toUpperCase()}</span>
        ${typeof data.data === "string" ? esc(data.data.slice(0, 200)) : JSON.stringify(data.data || data.action || "").slice(0, 200)}
      </div>`;
    output.scrollTop = output.scrollHeight;
  };

  ws.onclose = () => {
    output.innerHTML += '<div class="stream-event"><span class="event-type">DISCONNECTED</span></div>';
  };
}

/* ── Simulator ───────────────────────────────────────────────────────────── */

async function runSimulator() {
  const url = document.getElementById("sim-url").value;
  const persona = document.getElementById("sim-persona").value;
  if (!url) return;

  const btn = document.getElementById("sim-submit");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Simulating...';

  try {
    const result = await api("/api/simulator/run", {
      method: "POST",
      body: JSON.stringify({ target_url: url, persona, max_steps: 10 }),
    });

    const output = document.getElementById("sim-results");
    if (!output) return;
    output.classList.remove("hidden");
    output.innerHTML = `
      <div class="card fade-in">
        <div class="card-header">
          <div>
            <div class="card-title">Simulation Results</div>
            <div class="card-subtitle">${esc(result.target_url)} | ${esc(result.persona)}</div>
          </div>
          <div class="stat-value">${Math.round(result.overall_score * 100)}%</div>
        </div>
        <p style="margin-bottom:16px;font-size:14px">${esc(result.summary)}</p>
        ${result.friction_points
          .map(
            (fp) => `
          <div class="key-item">
            <div class="key-info">
              <div class="key-name">${esc(fp.element)} — ${esc(fp.issue)}</div>
              <div class="key-meta">Step ${fp.step} | ${fp.severity.toUpperCase()} | ${esc(fp.suggestion)}</div>
            </div>
            <span class="status-badge status-${fp.severity === "critical" ? "failed" : fp.severity === "high" ? "paused" : "pending"}">${fp.severity}</span>
          </div>`
          )
          .join("")}
      </div>`;
  } catch (e) {
    showAlert("sim-alert", e.message, "error");
  } finally {
    btn.disabled = false;
    btn.innerHTML = "Run Simulation";
  }
}

/* ── Helpers ─────────────────────────────────────────────────────────────── */

function esc(str) {
  if (!str) return "";
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function showAlert(id, message, type) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = message;
  el.className = `alert alert-${type}`;
}

function clearAlerts() {
  document.querySelectorAll(".alert").forEach((el) => {
    el.className = "alert";
    el.textContent = "";
  });
}

/* ── Init ────────────────────────────────────────────────────────────────── */

document.addEventListener("DOMContentLoaded", async () => {
  if (authToken) {
    try {
      await loadUser();
      showDashboard();
    } catch {
      showAuth();
    }
  } else {
    showAuth();
  }
});
