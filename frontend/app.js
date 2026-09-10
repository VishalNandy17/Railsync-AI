const API = ""; // same origin — FastAPI serves this file too

const PRIORITY_COLORS = {
  LOW: "#3fb950",
  MEDIUM: "#f2a900",
  HIGH: "#f0883e",
  CRITICAL: "#f85149",
};

let allPredictions = [];
let scheduleData = null;

async function getJSON(path) {
  const res = await fetch(API + path);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

function priorityTag(p) {
  return `<span class="priority-tag priority-${p}">${p}</span>`;
}

function setStatus(ok, text) {
  const el = document.getElementById("apiStatus");
  el.textContent = text;
  el.className = "status-pill " + (ok ? "status-ok" : "status-error");
}

// ---------------------------------------------------------------------------
// Overview section
// ---------------------------------------------------------------------------
function renderStatRow(metrics) {
  const t = metrics.test;
  const tr = metrics.train;
  const gap = (tr.MAE > 0) ? (t.MAE / tr.MAE).toFixed(2) : "-";
  const cards = [
    { label: "Test MAE", value: t.MAE.toFixed(2), sub: "score points" },
    { label: "Test R\u00b2", value: t.R2.toFixed(3), sub: `${metrics.n_test} held-out rows` },
    { label: "Train MAE", value: tr.MAE.toFixed(2), sub: `${metrics.n_train} training rows` },
    { label: "Train/test MAE ratio", value: gap + "\u00d7", sub: "overfit check — closer to 1\u00d7 is better" },
  ];
  document.getElementById("statRow").innerHTML = cards.map(c => `
    <div class="stat-card">
      <div class="stat-label">${c.label}</div>
      <div class="stat-value">${c.value}</div>
      <div class="stat-sub">${c.sub}</div>
    </div>
  `).join("");
}

function renderPerClassTable(metrics) {
  const rows = metrics.per_class_test;
  const html = `
    <thead><tr><th>Priority</th><th>Test rows</th><th>MAE (score points)</th></tr></thead>
    <tbody>
      ${rows.map(r => `
        <tr>
          <td>${priorityTag(r.priority_class)}</td>
          <td>${r.n}</td>
          <td>${r.MAE.toFixed(2)}</td>
        </tr>
      `).join("")}
    </tbody>`;
  document.getElementById("perClassTable").innerHTML = html;
}

function renderBaselineTable(metrics) {
  const rows = [...metrics.baselines, { model: "XGBoost (tuned, this model)", MAE: metrics.test.MAE, R2: metrics.test.R2 }]
    .sort((a, b) => a.MAE - b.MAE);
  const html = `
    <thead><tr><th>Model</th><th>MAE</th><th>R\u00b2</th></tr></thead>
    <tbody>
      ${rows.map(r => `
        <tr>
          <td>${r.model}</td>
          <td>${r.MAE.toFixed(3)}</td>
          <td>${r.R2.toFixed(3)}</td>
        </tr>
      `).join("")}
    </tbody>`;
  document.getElementById("baselineTable").innerHTML = html;
}

function renderScatterChart(rows) {
  const byPriority = {};
  for (const p of Object.keys(PRIORITY_COLORS)) byPriority[p] = [];
  for (const r of rows) {
    byPriority[r.predicted_priority]?.push({ x: r.criticality_score, y: r.predicted_criticality });
  }
  const datasets = Object.entries(byPriority).map(([p, pts]) => ({
    label: p,
    data: pts,
    backgroundColor: PRIORITY_COLORS[p],
    pointRadius: 2.5,
  }));

  new Chart(document.getElementById("scatterChart"), {
    type: "scatter",
    data: { datasets },
    options: {
      maintainAspectRatio: false,
      scales: {
        x: { min: 0, max: 100, title: { display: true, text: "Actual", color: "#8b98a8" }, ticks: { color: "#8b98a8" }, grid: { color: "#1a222c" } },
        y: { min: 0, max: 100, title: { display: true, text: "Predicted", color: "#8b98a8" }, ticks: { color: "#8b98a8" }, grid: { color: "#1a222c" } },
      },
      plugins: { legend: { labels: { color: "#e6edf3", boxWidth: 10, font: { size: 11 } } } },
    },
  });
}

function renderImportanceChart(features) {
  const top = features.slice(0, 12).reverse();
  new Chart(document.getElementById("importanceChart"), {
    type: "bar",
    data: {
      labels: top.map(f => f.feature),
      datasets: [{ data: top.map(f => f.mean_abs_shap), backgroundColor: "#f2a900" }],
    },
    options: {
      indexAxis: "y",
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: "#8b98a8" }, grid: { color: "#1a222c" } },
        y: { ticks: { color: "#e6edf3", font: { size: 10.5 } }, grid: { display: false } },
      },
      plugins: { legend: { display: false } },
    },
  });
}

// ---------------------------------------------------------------------------
// Predictions table + explain panel
// ---------------------------------------------------------------------------
function renderPredictionsTable(rows) {
  document.getElementById("predictionsCount").textContent = `${rows.length} rows`;
  const html = `
    <thead><tr>
      <th>Task ID</th><th>Dept</th><th>Corridor</th><th>Asset</th>
      <th>Actual</th><th>Predicted</th><th>Priority</th><th>Abs err</th>
    </tr></thead>
    <tbody>
      ${rows.slice(0, 200).map(r => `
        <tr class="selectable-row" data-task-id="${r.task_id}">
          <td>${r.task_id}</td>
          <td>${r.department}</td>
          <td>${r.corridor_id}</td>
          <td>${r.asset_type}</td>
          <td>${r.criticality_score.toFixed(1)}</td>
          <td>${r.predicted_criticality.toFixed(1)}</td>
          <td>${priorityTag(r.predicted_priority)}</td>
          <td>${r.abs_error.toFixed(2)}</td>
        </tr>
      `).join("")}
    </tbody>`;
  const table = document.getElementById("predictionsTable");
  table.innerHTML = html;

  table.querySelectorAll("tr.selectable-row").forEach(tr => {
    tr.addEventListener("click", () => {
      table.querySelectorAll("tr.row-selected").forEach(el => el.classList.remove("row-selected"));
      tr.classList.add("row-selected");
      explainTask(tr.dataset.taskId);
    });
  });
}

function applyFilters() {
  const q = document.getElementById("searchBox").value.trim().toLowerCase();
  const p = document.getElementById("priorityFilter").value;
  let rows = allPredictions;
  if (p) rows = rows.filter(r => r.predicted_priority === p);
  if (q) {
    rows = rows.filter(r =>
      r.task_id.toLowerCase().includes(q) ||
      r.corridor_id.toLowerCase().includes(q) ||
      r.department.toLowerCase().includes(q) ||
      r.asset_type.toLowerCase().includes(q)
    );
  }
  renderPredictionsTable(rows);
}

async function explainTask(taskId) {
  const panel = document.getElementById("explainPanel");
  panel.innerHTML = `<div class="explain-empty">Loading explanation for ${taskId}…</div>`;
  try {
    const data = await getJSON(`/api/explain/${encodeURIComponent(taskId)}`);
    const maxAbs = Math.max(...data.contributors.map(c => Math.abs(c.shap_value)), 0.01);
    panel.innerHTML = `
      <div class="explain-header">
        <div class="task-id">${data.task_id}</div>
        <div class="task-meta">${data.meta.department} · ${data.meta.corridor_id} · ${data.meta.asset_type}</div>
      </div>
      <div class="explain-score">${data.prediction.toFixed(1)}</div>
      <div class="explain-score-label">predicted criticality (actual: ${data.meta.criticality_score.toFixed(1)}, ${data.meta.priority_class})</div>
      ${data.contributors.map(c => {
        const pct = Math.min(100, Math.abs(c.shap_value) / maxAbs * 50);
        const cls = c.shap_value >= 0 ? "pos" : "neg";
        return `
          <div class="contrib-row">
            <div class="contrib-feature">${c.feature}</div>
            <div class="contrib-value">${c.shap_value >= 0 ? "+" : ""}${c.shap_value.toFixed(2)}</div>
            <div class="contrib-bar-track">
              <div class="contrib-bar-fill ${cls}" style="width:${pct}%"></div>
            </div>
          </div>`;
      }).join("")}
    `;
  } catch (e) {
    panel.innerHTML = `<div class="explain-empty">Could not load explanation: ${e.message}</div>`;
  }
}

// ---------------------------------------------------------------------------
// Schedule section
// ---------------------------------------------------------------------------
function renderScheduleStats(summary) {
  const cards = [
    { label: "Status", value: summary.status, sub: `${summary.horizon_days}-day horizon` },
    { label: "Scheduled", value: `${summary.n_scheduled} / ${summary.n_tasks}`, sub: "tasks placed within horizon" },
    { label: "Corridor budget", value: `${summary.corridor_daily_budget_min} min`, sub: "block-minutes / corridor / day" },
    { label: "Crew capacity", value: `${summary.crew_daily_capacity}`, sub: "crew-units / day, railway-wide" },
  ];
  document.getElementById("scheduleStatRow").innerHTML = cards.map(c => `
    <div class="stat-card">
      <div class="stat-label">${c.label}</div>
      <div class="stat-value" style="font-size:18px">${c.value}</div>
      <div class="stat-sub">${c.sub}</div>
    </div>
  `).join("");
}

function renderScheduleChart(tasks) {
  const corridors = [...new Set(tasks.map(t => t.corridor_id))].sort();
  const byPriority = {};
  for (const p of Object.keys(PRIORITY_COLORS)) byPriority[p] = [];
  for (const t of tasks) {
    if (!t.scheduled) continue;
    byPriority[t.predicted_priority]?.push({ x: t.scheduled_day, y: corridors.indexOf(t.corridor_id) });
  }
  const datasets = Object.entries(byPriority).map(([p, pts]) => ({
    label: p, data: pts, backgroundColor: PRIORITY_COLORS[p], pointRadius: 4,
  }));

  new Chart(document.getElementById("scheduleChart"), {
    type: "scatter",
    data: { datasets },
    options: {
      maintainAspectRatio: false,
      scales: {
        x: { min: -0.5, max: 13.5, title: { display: true, text: "Day of 14-day horizon", color: "#8b98a8" }, ticks: { color: "#8b98a8", stepSize: 1 }, grid: { color: "#1a222c" } },
        y: {
          min: -0.5, max: corridors.length - 0.5,
          ticks: { color: "#8b98a8", stepSize: 1, callback: (v) => corridors[v] ?? "" },
          grid: { color: "#1a222c" },
        },
      },
      plugins: { legend: { labels: { color: "#e6edf3", boxWidth: 10, font: { size: 11 } } } },
    },
  });
}

function renderBacklogTable(tasks) {
  const backlog = tasks.filter(t => !t.scheduled).sort((a, b) => b.predicted_criticality - a.predicted_criticality);
  const html = `
    <thead><tr><th>Task ID</th><th>Corridor</th><th>Dept</th><th>Predicted</th><th>Priority</th><th>Permit ready</th></tr></thead>
    <tbody>
      ${backlog.slice(0, 100).map(t => `
        <tr>
          <td>${t.task_id}</td>
          <td>${t.corridor_id}</td>
          <td>${t.department}</td>
          <td>${t.predicted_criticality.toFixed(1)}</td>
          <td>${priorityTag(t.predicted_priority)}</td>
          <td>${t.permit_ready ? "yes" : "no"}</td>
        </tr>
      `).join("")}
    </tbody>`;
  document.getElementById("backlogTable").innerHTML = html;
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
async function main() {
  try {
    await getJSON("/api/health");
    setStatus(true, "connected");
  } catch (e) {
    setStatus(false, "backend unreachable");
    return;
  }

  const [metrics, predictions, importance, schedule] = await Promise.all([
    getJSON("/api/metrics"),
    getJSON("/api/predictions?limit=5000"),
    getJSON("/api/feature-importance?limit=15"),
    getJSON("/api/schedule"),
  ]);

  allPredictions = predictions.rows;
  scheduleData = schedule;

  renderStatRow(metrics);
  renderPerClassTable(metrics);
  renderBaselineTable(metrics);
  renderScatterChart(allPredictions);
  renderImportanceChart(importance);

  renderPredictionsTable(allPredictions);
  document.getElementById("searchBox").addEventListener("input", applyFilters);
  document.getElementById("priorityFilter").addEventListener("change", applyFilters);

  renderScheduleStats(schedule.summary);
  renderScheduleChart(schedule.tasks);
  renderBacklogTable(schedule.tasks);
}

main();
