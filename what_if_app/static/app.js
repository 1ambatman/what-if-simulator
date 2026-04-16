const $ = (sel) => document.querySelector(sel);

async function api(path, opts = {}) {
  const r = await fetch(path, {
    headers: { "Content-Type": "application/json", ...opts.headers },
    ...opts,
  });
  const text = await r.text();
  let data;
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { detail: text };
  }
  if (!r.ok) {
    const msg = data.detail || data.message || r.statusText || "Request failed";
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return data;
}

function setHealth(ok, err) {
  const el = $("#health-pill");
  el.classList.remove("pill-ok", "pill-warn", "pill-bad");
  if (ok) {
    el.textContent = "Model ready";
    el.classList.add("pill-ok");
  } else {
    el.textContent = err ? "Model error" : "Starting…";
    el.classList.add(err ? "pill-bad" : "pill-warn");
    if (err) el.title = err;
  }
}

function parseDates(text) {
  return text
    .split(/\r?\n/)
    .map((s) => s.trim())
    .filter(Boolean);
}

function parseIds(text) {
  return text
    .split(/\r?\n/)
    .map((s) => s.trim())
    .filter(Boolean);
}

let currentManual = {};

async function refreshMeta() {
  const meta = await api("/api/meta");
  $("#predictions-table").value = meta.predictions_table_default || "";
  const sel = $("#scenario-select");
  sel.innerHTML = "";
  ["(No scenario)", ...meta.scenarios.map((s) => s.name), "Manual adjustment"].forEach((name) => {
    const o = document.createElement("option");
    o.value = name;
    o.textContent = name;
    sel.appendChild(o);
  });
}

function tierColor(label) {
  if (label === "Good") return "var(--good)";
  if (label === "Okay") return "var(--warn)";
  return "var(--bad)";
}

function renderBaseline(data) {
  const wf = data.waterfall || [];
  const maxS = Math.max(1e-9, ...wf.map((w) => Math.abs(w.shap)));
  return `
    <div class="card">
      <h3>Baseline · ${escapeHtml(data.profile_label || "")}</h3>
      <p style="font-family:var(--mono);font-size:0.95rem">
        Score <strong style="color:var(--accent)">${data.score.toFixed(4)}</strong>
        · Tier <strong style="color:${tierColor(data.risk_label)}">${data.tier}</strong>
        (${escapeHtml(data.risk_label)})
      </p>
      <p style="font-size:0.8rem;color:var(--muted)">SHAP expected value (base): ${data.base_value.toFixed(4)}</p>
      <h4 style="margin:1rem 0 0.5rem;font-size:0.85rem;color:var(--muted)">Top drivers</h4>
      <ul class="shap-list">
        ${wf
          .map(
            (w) => `
          <li>
            <span title="${escapeHtml(w.feature)}">${escapeHtml(shortFeat(w.feature))}</span>
            <span style="text-align:right">${w.shap.toFixed(4)}</span>
          </li>
          <li style="display:block;margin:-0.15rem 0 0.4rem 0;grid-template-columns:1fr">
            <span class="bar"><i style="width:${(Math.abs(w.shap) / maxS) * 100}%"></i></span>
          </li>`
          )
          .join("")}
      </ul>
    </div>`;
}

function shortFeat(f) {
  return f.length > 42 ? f.slice(0, 40) + "…" : f;
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function renderCompare(data) {
  const maxH = Math.max(data.score_before, data.score_after, 0.001) * 1.15;
  const hBefore = (data.score_before / maxH) * 100;
  const hAfter = (data.score_after / maxH) * 100;

  const block = (title, rows) => {
    const maxS = Math.max(1e-9, ...rows.map((w) => Math.abs(w.shap)));
    return `
      <div>
        <h4 style="margin:0 0 0.5rem;font-size:0.85rem;color:var(--muted)">${title}</h4>
        <ul class="shap-list">
          ${rows
            .map(
              (w) => `
            <li>
              <span title="${escapeHtml(w.feature)}">${escapeHtml(shortFeat(w.feature))}</span>
              <span>${w.shap.toFixed(4)}</span>
            </li>
            <li style="display:block;margin:-0.15rem 0 0.4rem;grid-template-columns:1fr">
              <span class="bar"><i style="width:${(Math.abs(w.shap) / maxS) * 100}%"></i></span>
            </li>`
            )
            .join("")}
        </ul>
      </div>`;
  };

  const rows = data.delta_table || [];
  return `
    <div class="card">
      <h3>${escapeHtml(data.scenario || "")}</h3>
      <p style="font-size:0.8rem;color:var(--muted)">${escapeHtml(data.description || "")}</p>
      <div class="migration">${escapeHtml(data.tier_migration || "")}</div>
      <div class="score-compare">
        <div class="score-bar">
          <div class="lbl">Before</div>
          <div class="val" style="color:${tierColor(data.label_before)}">${data.score_before.toFixed(4)}</div>
          <div class="bar-track"><div class="bar-fill before" style="height:${hBefore}%"></div></div>
        </div>
        <div class="score-bar">
          <div class="lbl">After</div>
          <div class="val" style="color:${tierColor(data.label_after)}">${data.score_after.toFixed(4)}</div>
          <div class="bar-track"><div class="bar-fill after" style="height:${hAfter}%"></div></div>
        </div>
      </div>
      <div class="shap-grid">
        ${block("SHAP · before", data.waterfall_before || [])}
        ${block("SHAP · after", data.waterfall_after || [])}
      </div>
    </div>
    ${
      rows.length
        ? `<div class="card"><h3>Top feature deltas</h3>
      <table class="delta-table">
        <thead><tr>
          <th>feature</th><th>Δ value</th><th>Δ SHAP</th>
        </tr></thead>
        <tbody>
          ${rows
            .map(
              (r) => `<tr>
            <td title="${escapeHtml(r.feature)}">${escapeHtml(shortFeat(r.feature))}</td>
            <td>${Number(r.value_change).toFixed(4)}</td>
            <td>${Number(r.shap_delta).toFixed(4)}</td>
          </tr>`
            )
            .join("")}
        </tbody>
      </table></div>`
        : ""
    }`;
}

async function loadManualSliders(profileId) {
  const data = await api(`/api/profile-features/${encodeURIComponent(profileId)}`);
  const wrap = $("#manual-acc");
  wrap.innerHTML = "";
  currentManual = {};
  const groups = data.groups || {};
  for (const [gname, sliders] of Object.entries(groups)) {
    const item = document.createElement("div");
    item.className = "acc-item";
    const head = document.createElement("button");
    head.type = "button";
    head.className = "acc-head";
    head.innerHTML = `<span>${escapeHtml(gname)}</span><span>▾</span>`;
    const body = document.createElement("div");
    body.className = "acc-body";
    head.addEventListener("click", () => {
      item.classList.toggle("open");
    });
    for (const s of sliders) {
      currentManual[s.name] = s.value;
      const row = document.createElement("div");
      row.className = "slider-row";
      const id = `sf-${s.name.replace(/[^a-zA-Z0-9]/g, "_")}`;
      row.innerHTML = `
        <label for="${id}"><span>${escapeHtml(s.label)}</span><span>${s.value.toFixed(3)}</span></label>
        <input id="${id}" type="range" min="${s.min}" max="${s.max}" step="${s.step}" value="${s.value}" />
      `;
      const input = row.querySelector("input");
      const lbl = row.querySelector("label span:last-child");
      input.addEventListener("input", () => {
        const v = parseFloat(input.value);
        currentManual[s.name] = v;
        lbl.textContent = v.toFixed(3);
      });
      body.appendChild(row);
    }
    item.appendChild(head);
    item.appendChild(body);
    wrap.appendChild(item);
  }
}

async function init() {
  try {
    const h = await api("/api/health");
    setHealth(h.ok, h.error);
    if (!h.ok && h.error) console.error(h.error);
  } catch (e) {
    setHealth(false, String(e));
  }
  try {
    await refreshMeta();
  } catch (e) {
    $("#load-msg").innerHTML = `<span class="err">${escapeHtml(String(e))}</span>`;
  }

  document.querySelectorAll(".tab").forEach((t) => {
    t.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
      t.classList.add("active");
      const tab = t.dataset.tab;
      $("#pane-inline").classList.toggle("hidden", tab !== "inline");
      $("#pane-table").classList.toggle("hidden", tab !== "table");
    });
  });

  $("#btn-load").addEventListener("click", async () => {
    $("#load-msg").textContent = "Loading…";
    const table = $("#predictions-table").value.trim();
    const inline = $(".tab.active")?.dataset.tab === "inline";
    const body = {
      predictions_table: table || null,
      mode: inline ? "inline" : "input_table",
      customer_ids: inline ? parseIds($("#customer-ids").value) : [],
      reference_dates: inline ? parseDates($("#reference-dates").value) : [],
      input_table: inline ? null : $("#input-table").value.trim() || null,
    };
    try {
      const res = await api("/api/load", { method: "POST", body: JSON.stringify(body) });
      const ps = $("#profile-select");
      ps.innerHTML = "";
      (res.profiles || []).forEach((p) => {
        const o = document.createElement("option");
        o.value = p.id;
        o.textContent = p.label;
        ps.appendChild(o);
      });
      let msg = res.loaded ? `Loaded ${res.loaded} profile(s).` : (res.warning || "Done.");
      if (res.warnings?.length) msg += " " + res.warnings.join(" ");
      $("#load-msg").textContent = msg;
      if (res.profiles?.length) {
        await onProfileChange();
      }
    } catch (e) {
      $("#load-msg").innerHTML = `<span class="err">${escapeHtml(String(e))}</span>`;
    }
  });

  $("#profile-select").addEventListener("change", onProfileChange);

  $("#scenario-select").addEventListener("change", () => {
    const sc = $("#scenario-select").value;
    const manual = sc === "Manual adjustment";
    $("#manual-wrap").classList.toggle("hidden", !manual);
    if (manual) {
      const pid = $("#profile-select").value;
      if (pid) loadManualSliders(pid);
    }
  });

  async function onProfileChange() {
    if ($("#scenario-select").value === "Manual adjustment") {
      const pid = $("#profile-select").value;
      if (pid) await loadManualSliders(pid);
    }
  }

  $("#btn-run").addEventListener("click", async () => {
    const pid = $("#profile-select").value;
    if (!pid) {
      $("#results").innerHTML = `<div class="card err">Load at least one profile first.</div>`;
      return;
    }
    const scenario = $("#scenario-select").value;
    const payload = {
      profile_id: pid,
      scenario,
      manual_features: scenario === "Manual adjustment" ? currentManual : null,
    };
    $("#results").innerHTML = `<div class="card">Running…</div>`;
    try {
      const data = await api("/api/what-if", { method: "POST", body: JSON.stringify(payload) });
      if (data.mode === "baseline") {
        $("#results").innerHTML = renderBaseline(data);
      } else {
        $("#results").innerHTML = renderCompare(data);
      }
    } catch (e) {
      $("#results").innerHTML = `<div class="card err">${escapeHtml(String(e))}</div>`;
    }
  });
}

init();
