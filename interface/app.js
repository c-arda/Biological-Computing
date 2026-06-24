/**
 * app.js — Biological Computing Simulator Frontend
 * ==================================================
 * Scientific Data Observatory interface for quantum biology experiments.
 */

const API = '';
let experiments = [];
let activeExp = null;
let activeMetrics = null;

// ── Initialisation ──
document.addEventListener('DOMContentLoaded', async () => {
  startClock();
  await loadExperiments();
  await checkStatus();
});

function startClock() {
  const el = document.getElementById('clock');
  const tick = () => {
    const d = new Date();
    el.textContent = d.toISOString().replace('T', ' ').slice(0, 19) + ' UTC';
  };
  tick();
  setInterval(tick, 1000);
}

async function checkStatus() {
  try {
    const res = await fetch(`${API}/api/status`);
    const data = await res.json();
    document.getElementById('statusDot').style.background = 'var(--accent-green)';
    document.getElementById('statusText').textContent = 'ONLINE';
    const pkgs = Object.entries(data.packages || {})
      .map(([k, v]) => `${k} ${v}`).join(' · ');
    document.getElementById('pkgStatus').textContent = pkgs;
    document.getElementById('footerMetrics').textContent =
      `${data.metrics_available}/${data.experiments_count} metrics loaded`;
    log(`System online: ${data.python} · ${data.experiments_count} experiments`);
  } catch (e) {
    document.getElementById('statusDot').style.background = 'var(--accent-red)';
    document.getElementById('statusText').textContent = 'OFFLINE';
    log('ERROR: Cannot reach backend');
  }
}

async function loadExperiments() {
  try {
    const res = await fetch(`${API}/api/experiments`);
    experiments = await res.json();
    renderSidebar();
  } catch (e) {
    log('ERROR: Failed to load experiments');
  }
}

// ── Sidebar ──
function renderSidebar() {
  const sidebar = document.getElementById('sidebar');
  const phases = { 1: 'Phase 1 — Quantum Foundations', 2: 'Phase 2 — BNN Integration', 3: 'Phase 3 — Predictions' };
  let html = '';
  for (const [phase, label] of Object.entries(phases)) {
    const exps = experiments.filter(e => e.phase === parseInt(phase));
    if (!exps.length) continue;
    html += `<div class="sidebar__section"><div class="sidebar__label">${label}</div></div>`;
    for (const exp of exps) {
      const badgeClass = exp.has_metrics ? 'has-data' : 'no-data';
      html += `
        <div class="exp-item" data-id="${exp.id}" onclick="selectExperiment('${exp.id}')">
          <span class="exp-item__id">${exp.id}</span>
          <span class="exp-item__title">${exp.title}</span>
          <span class="exp-item__badge ${badgeClass}"></span>
        </div>`;
    }
  }
  sidebar.innerHTML = html;
}

// ── Experiment Selection ──
async function selectExperiment(id) {
  activeExp = experiments.find(e => e.id === id);
  if (!activeExp) return;

  // Update sidebar active state
  document.querySelectorAll('.exp-item').forEach(el => {
    el.classList.toggle('active', el.dataset.id === id);
  });

  log(`Selected: Experiment ${id} — ${activeExp.title}`);
  renderMain();
  await loadMetrics(id);
}

function renderMain() {
  const main = document.getElementById('mainPanel');
  const exp = activeExp;
  const paramEntries = Object.entries(exp.params || {});

  let paramsHTML = '';
  if (paramEntries.length > 0) {
    paramsHTML = `
      <div class="card" style="animation-delay:0.12s">
        <div class="card__header">
          <span class="card__title">Simulation Parameters</span>
        </div>
        <div class="params-grid">
          ${paramEntries.map(([key, p]) => `
            <div class="param-control">
              <label>${p.label}</label>
              <div class="param-control__row">
                <input type="range" id="param-${key}"
                  min="${p.min}" max="${p.max}" step="${p.step}"
                  value="${p.default}"
                  oninput="updateParamDisplay('${key}', this.value)">
                <span class="param-control__value" id="pval-${key}">${p.default}</span>
                <span class="param-control__unit">${p.unit}</span>
              </div>
            </div>
          `).join('')}
        </div>
        <p class="param-note" style="font-size:11px;color:var(--text-muted);margin:10px 0 0">${exp.live
          ? 'Quick Explore runs a fast in-process model with your parameters. Run Simulation re-executes the full canonical experiment (regenerates metrics &amp; dashboard).'
          : 'Run Simulation re-executes the canonical experiment (regenerates metrics &amp; dashboard); sliders are the published reference values.'}</p>
        <div class="actions" style="margin-top:14px">
          ${exp.live ? `<button class="btn btn--primary" id="liveBtn" onclick="quickExplore()">⚡ Quick Explore</button>` : ''}
          <button class="btn ${exp.live ? 'btn--secondary' : 'btn--primary'}" id="runBtn" onclick="runExperiment()">
            ▸ Run Simulation
          </button>
          <button class="btn btn--secondary" onclick="resetParams()">
            ↺ Reset Defaults
          </button>
        </div>
        <div class="progress-bar" style="margin-top:10px" id="progressBar">
          <div class="progress-bar__fill" id="progressFill"></div>
        </div>
      </div>`;
  }

  main.innerHTML = `
    <div class="card" style="animation-delay:0s">
      <div class="exp-hero">
        <div class="exp-hero__info">
          <h2>Experiment ${exp.id} — ${exp.title}</h2>
          <p class="exp-hero__desc">${exp.description}</p>
          <div class="exp-hero__result">◎ ${exp.key_result}</div>
        </div>
        <div class="exp-hero__meta">
          <span class="meta-tag">${exp.category}</span>
          <span class="meta-tag">${exp.tools}</span>
          <span class="meta-tag">Phase ${exp.phase}</span>
        </div>
      </div>
    </div>

    ${paramsHTML}

    <div class="card" style="animation-delay:0.18s" id="metricsCard">
      <div class="card__header">
        <span class="card__title">Results</span>
        <span class="card__badge" id="metricsStatus">Loading...</span>
      </div>
      <div id="metricsContent">
        <div style="text-align:center;padding:40px;color:var(--text-muted)">
          <div class="spinner"></div>
          <p style="margin-top:12px;font-size:12px">Loading pre-computed metrics...</p>
        </div>
      </div>
    </div>

    <div class="card" style="animation-delay:0.24s" id="dashboardCard">
      <div class="card__header">
        <span class="card__title">Dashboard</span>
      </div>
      <div id="dashboardContent">
        ${exp.has_image
          ? `<img class="dashboard-img" src="${API}/api/results/${exp.id}/image" alt="Experiment ${exp.id} dashboard" loading="lazy">`
          : `<p style="color:var(--text-muted);font-size:12px;text-align:center;padding:40px">No dashboard image available. Run the experiment to generate one.</p>`
        }
      </div>
    </div>
  `;
}

function updateParamDisplay(key, value) {
  document.getElementById(`pval-${key}`).textContent = value;
}

function resetParams() {
  if (!activeExp) return;
  for (const [key, p] of Object.entries(activeExp.params || {})) {
    const slider = document.getElementById(`param-${key}`);
    if (slider) {
      slider.value = p.default;
      updateParamDisplay(key, p.default);
    }
  }
  log('Parameters reset to defaults');
}

// ── Metrics Loading ──
async function loadMetrics(expId) {
  try {
    const res = await fetch(`${API}/api/metrics/${expId}`);
    if (!res.ok) throw new Error('No metrics');
    activeMetrics = await res.json();
    renderMetrics(activeMetrics);
    renderInspector(activeMetrics);
    document.getElementById('metricsStatus').textContent = 'Pre-computed';
    log(`Loaded metrics for experiment ${expId}`);
  } catch (e) {
    document.getElementById('metricsStatus').textContent = 'No data';
    document.getElementById('metricsContent').innerHTML =
      '<p style="color:var(--text-muted);font-size:12px;text-align:center;padding:30px">No pre-computed metrics available. Run the experiment to generate data.</p>';
  }
}

function renderMetrics(data) {
  const el = document.getElementById('metricsContent');
  const cards = extractMetricCards(data);
  if (!cards.length) {
    el.innerHTML = '<p style="color:var(--text-muted);font-size:12px">No displayable metrics found.</p>';
    return;
  }
  el.innerHTML = `<div class="metrics-grid">${cards.map(c => `
    <div class="metric-card ${c.colorClass || ''}">
      <span class="metric-card__label">${c.label}</span>
      <span class="metric-card__value">${c.value}</span>
      <span class="metric-card__unit">${c.unit || ''}</span>
    </div>
  `).join('')}</div>`;
}

function extractMetricCards(data) {
  const cards = [];
  const id = activeExp?.id;

  if (id === '1a') {
    const q = data.quantum || {};
    const c = data.classical || {};
    cards.push(
      { label: 'Quantum Rate', value: q.mean_rate_hz, unit: 'Hz', colorClass: 'metric-card--green' },
      { label: 'Classical Rate', value: c.mean_rate_hz, unit: 'Hz' },
      { label: 'Quantum CV_ISI', value: q.cv_isi_mean, unit: '' },
      { label: 'Classical CV_ISI', value: c.cv_isi_mean, unit: '' },
      { label: 'Quantum Spikes', value: (q.n_spikes/1000).toFixed(1)+'k', unit: '' },
      { label: 'Cross Synchrony', value: data.cross_synchrony?.toFixed(3), unit: 'ρ' },
    );
  } else if (id === '1b') {
    const p = data.physics || {};
    cards.push(
      { label: 'Hilbert Dim', value: p.hilbert_dim, unit: '' },
      { label: 'Spins', value: p.n_spins, unit: '³¹P' },
    );
    if (data.coherence_times) {
      for (const ct of data.coherence_times) {
        cards.push({
          label: `T₂ @ ${ct.temperature_K} K`,
          value: ct.T2_us?.toFixed(1) || '—',
          unit: 'µs',
          colorClass: ct.temperature_K === 310 ? 'metric-card--green' : '',
        });
      }
    }
  } else if (id === '1c') {
    const peak = data.enaqt_peak || {};
    const body = data.body_temperature_assessment || {};
    cards.push(
      { label: 'ENAQT Peak Γ', value: peak.optimal_dephasing_rate_cm?.toFixed(0), unit: 'cm⁻¹', colorClass: 'metric-card--green' },
      { label: 'Max Efficiency', value: peak.max_transport_efficiency?.toFixed(4), unit: 'η' },
      { label: 'Enhancement', value: peak.enhancement_ratio?.toFixed(1) + '×', unit: '', colorClass: 'metric-card--amber' },
      { label: 'Coherent Limit', value: peak.coherent_limit_efficiency?.toFixed(4), unit: 'η' },
      { label: 'Body Temp η', value: body.efficiency_body?.toFixed(4), unit: 'η', colorClass: 'metric-card--green' },
      { label: 'Proximity', value: body.proximity_to_peak?.split(' ')[0] || '—', unit: '' },
    );
  } else if (id === '1d') {
    const conds = data.conditions || {};
    for (const [name, c] of Object.entries(conds)) {
      cards.push({
        label: `MC (${name})`,
        value: c.mc_total?.toFixed(2),
        unit: '',
        colorClass: name === 'Quantum' ? 'metric-card--green'
          : (name === 'Classical' ? 'metric-card--amber' : ''),
      });
    }
    const ns = data.noise_sweep || {};
    if (ns.best_quantum) {
      cards.push({ label: 'Best Quantum MC', value: ns.best_quantum.mc?.toFixed(2),
        unit: `@amp ${ns.best_quantum.amplitude}`, colorClass: 'metric-card--green' });
    }
    if (ns.best_classical) {
      cards.push({ label: 'Best Classical MC', value: ns.best_classical.mc?.toFixed(2),
        unit: `@amp ${ns.best_classical.amplitude}` });
    }
  } else if (id === '1e') {
    const bridge = data.enaqt_bridge || {};
    const sweep = data.dephasing_sweep || {};
    cards.push(
      { label: 'P₄ Coherent', value: bridge.p_coherent?.toFixed(4), unit: '' },
      { label: 'P₄ Body Temp', value: bridge.p_body?.toFixed(4), unit: '', colorClass: 'metric-card--green' },
      { label: 'P₄ Peak', value: bridge.p_peak?.toFixed(4), unit: '', colorClass: 'metric-card--amber' },
      { label: 'MC Peak γ', value: sweep.mc_peak_gamma_cm?.toFixed(0), unit: 'cm⁻¹' },
      { label: 'MC Peak Value', value: sweep.mc_peak_value?.toFixed(4), unit: '' },
      { label: 'ENAQT Peak γ', value: sweep.enaqt_peak_gamma_cm?.toFixed(0), unit: 'cm⁻¹', colorClass: 'metric-card--green' },
    );
    // Conditions
    const conds = data.conditions || {};
    for (const [name, c] of Object.entries(conds)) {
      cards.push({
        label: `MC (${name})`,
        value: c.mc_total?.toFixed(2),
        unit: '',
      });
    }
  } else if (id === '3a') {
    const p = data.physics || {};
    cards.push(
      { label: 'Mass Ratio D/H', value: p.mass_ratio_DH?.toFixed(3), unit: '' },
      { label: 'Freq Scale D₂O', value: p.freq_scale_D2O?.toFixed(4), unit: '' },
    );
    if (data.isotope_effect) {
      cards.push(
        { label: 'η H₂O', value: data.isotope_effect.eta_H2O?.toFixed(4), unit: '', colorClass: 'metric-card--green' },
        { label: 'η D₂O', value: data.isotope_effect.eta_D2O?.toFixed(4), unit: '', colorClass: 'metric-card--amber' },
        { label: 'Reduction', value: data.isotope_effect.reduction_percent?.toFixed(1) + '%', unit: '', colorClass: 'metric-card--red' },
      );
    }
  } else if (id === '3b') {
    const p = data.physics || {};
    cards.push(
      { label: 'γ ³¹P', value: p.gamma_P31_MHz_per_T?.toFixed(3), unit: 'MHz/T' },
      { label: 'J Dipolar', value: p.J_dipolar_Hz?.toFixed(0), unit: 'Hz' },
    );
  } else {
    // Generic: extract top-level numeric values
    const flatMetrics = flattenObj(data);
    for (const [key, val] of Object.entries(flatMetrics).slice(0, 12)) {
      if (typeof val === 'number') {
        cards.push({ label: key.replace(/_/g, ' '), value: formatNum(val), unit: '' });
      }
    }
  }
  return cards;
}

// ── Inspector ──
function renderInspector(data) {
  const el = document.getElementById('inspectorContent');
  const jsonStr = syntaxHighlight(JSON.stringify(data, null, 2));

  // Build chart for sweep data if available
  let chartHTML = '';
  if (activeExp?.id === '1c' && data.enaqt_peak) {
    chartHTML = `
      <div class="chart-container" id="enaqtChart" style="min-height:200px">
        <canvas id="enaqtCanvas"></canvas>
      </div>`;
  } else if (activeExp?.id === '1e' && data.dephasing_sweep) {
    chartHTML = `
      <div class="chart-container" id="sweepChart" style="min-height:200px">
        <canvas id="sweepCanvas"></canvas>
      </div>`;
  }

  el.innerHTML = `
    ${chartHTML}
    <div class="inspector__title" style="margin-top:12px">▸ RAW JSON</div>
    <div class="json-viewer">${jsonStr}</div>
  `;

  // Draw charts after DOM update
  requestAnimationFrame(() => {
    if (activeExp?.id === '1e' && data.dephasing_sweep) {
      drawSweepChart(data.dephasing_sweep);
    } else if (activeExp?.id === '1c' && data.temperature_mapping) {
      drawEnaqtChart(data);
    }
  });
}

function drawEnaqtChart(data) {
  const canvas = document.getElementById('enaqtCanvas');
  if (!canvas) return;
  const pts = (data.temperature_mapping || [])
    .map(p => ({ g: p.gamma_cm, e: p.efficiency, T: p.temperature_K }))
    .filter(p => p.g > 0)
    .sort((a, b) => a.g - b.g);
  if (!pts.length) return;

  const ctx = canvas.getContext('2d');
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width * 2;
  canvas.height = 360;
  canvas.style.height = '180px';
  const w = canvas.width, h = canvas.height;
  const pad = { l: 64, r: 30, t: 26, b: 46 };

  ctx.fillStyle = 'rgba(0,0,0,0.3)';
  ctx.fillRect(0, 0, w, h);

  const peakG = data.enaqt_peak?.optimal_dephasing_rate_cm;
  const gs = pts.map(p => p.g).concat(peakG ? [peakG] : []);
  const xMin = Math.log10(Math.min(...gs)), xMax = Math.log10(Math.max(...gs));
  const span = (xMax - xMin) || 1;
  const eMax = (Math.max(...pts.map(p => p.e)) || 1) * 1.2;
  const toX = v => pad.l + (Math.log10(v) - xMin) / span * (w - pad.l - pad.r);
  const toY = v => pad.t + (1 - v / eMax) * (h - pad.t - pad.b);

  // ENAQT peak guide line
  if (peakG) {
    ctx.strokeStyle = 'rgba(10,255,239,0.4)';
    ctx.lineWidth = 2; ctx.setLineDash([6, 4]);
    ctx.beginPath(); ctx.moveTo(toX(peakG), pad.t); ctx.lineTo(toX(peakG), h - pad.b); ctx.stroke();
    ctx.setLineDash([]);
  }

  // Efficiency curve
  ctx.beginPath();
  ctx.strokeStyle = '#00d2ff';
  ctx.lineWidth = 3;
  pts.forEach((p, i) => { const x = toX(p.g), y = toY(p.e); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
  ctx.stroke();

  // Temperature points + labels
  ctx.font = '16px JetBrains Mono, monospace';
  pts.forEach(p => {
    const x = toX(p.g), y = toY(p.e);
    ctx.fillStyle = '#0affef';
    ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = 'rgba(255,255,255,0.6)';
    ctx.fillText(`${p.T}K`, x - 14, y - 12);
  });

  // Axis labels
  ctx.fillStyle = 'rgba(255,255,255,0.5)';
  ctx.font = '18px IBM Plex Mono, monospace';
  ctx.fillText('η transport', 8, pad.t - 8);
  ctx.fillText('γ dephasing (cm⁻¹, log)', w / 2 - 110, h - 12);
  if (peakG) {
    ctx.fillStyle = 'rgba(10,255,239,0.7)';
    ctx.fillText('ENAQT peak', toX(peakG) + 8, pad.t + 8);
  }
}

function drawSweepChart(sweep) {
  const canvas = document.getElementById('sweepCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width * 2;
  canvas.height = 360;
  canvas.style.height = '180px';

  const gammas = sweep.gamma_values;
  const p4 = sweep.p4_values;
  const mc = sweep.mc_values;
  const w = canvas.width, h = canvas.height;
  const pad = { l: 60, r: 60, t: 20, b: 40 };

  ctx.fillStyle = 'rgba(0,0,0,0.3)';
  ctx.fillRect(0, 0, w, h);

  // Scales (log x)
  const xMin = Math.log10(gammas[0]), xMax = Math.log10(gammas[gammas.length - 1]);
  const p4Max = Math.max(...p4) * 1.1;
  const mcMax = Math.max(...mc) * 1.1;

  const toX = v => pad.l + (Math.log10(v) - xMin) / (xMax - xMin) * (w - pad.l - pad.r);
  const toYp4 = v => pad.t + (1 - v / p4Max) * (h - pad.t - pad.b);
  const toYmc = v => pad.t + (1 - v / mcMax) * (h - pad.t - pad.b);

  // P4 line
  ctx.beginPath();
  ctx.strokeStyle = '#00d2ff';
  ctx.lineWidth = 3;
  gammas.forEach((g, i) => {
    const x = toX(g), y = toYp4(p4[i]);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // MC line
  ctx.beginPath();
  ctx.strokeStyle = '#0affef';
  ctx.lineWidth = 3;
  ctx.setLineDash([6, 4]);
  gammas.forEach((g, i) => {
    const x = toX(g), y = toYmc(mc[i]);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.setLineDash([]);

  // Labels
  ctx.fillStyle = 'rgba(255,255,255,0.5)';
  ctx.font = '20px JetBrains Mono, monospace';
  ctx.fillText('P₄', pad.l - 40, h / 2);
  ctx.fillText('MC', w - pad.r + 10, h / 2);

  // Legend
  ctx.fillStyle = '#00d2ff';
  ctx.fillRect(w / 2 - 80, h - 20, 20, 3);
  ctx.fillStyle = 'rgba(255,255,255,0.6)';
  ctx.font = '18px IBM Plex Mono, monospace';
  ctx.fillText('P₄(γ)', w / 2 - 55, h - 14);
  ctx.fillStyle = '#0affef';
  ctx.fillRect(w / 2 + 30, h - 20, 20, 3);
  ctx.fillStyle = 'rgba(255,255,255,0.6)';
  ctx.fillText('MC(γ)', w / 2 + 55, h - 14);
}

// ── Run Experiment ──
async function runExperiment() {
  if (!activeExp) return;
  const btn = document.getElementById('runBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Running...';

  const fill = document.getElementById('progressFill');
  if (fill) fill.style.width = '10%';

  const params = {};
  for (const [key, p] of Object.entries(activeExp.params || {})) {
    const slider = document.getElementById(`param-${key}`);
    if (slider) params[key] = parseFloat(slider.value);
  }

  const quickToggle = document.getElementById('quickToggle');
  const quickMode = quickToggle ? quickToggle.classList.contains('active') : true;

  log(`Running experiment ${activeExp.id} (${quickMode ? 'quick' : 'full'} mode)...`);

  // Animate progress
  let progress = 10;
  const interval = setInterval(() => {
    progress = Math.min(progress + Math.random() * 5, 90);
    if (fill) fill.style.width = progress + '%';
  }, 500);

  try {
    const res = await fetch(`${API}/api/run/${activeExp.id}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ params, quick_mode: quickMode }),
    });

    clearInterval(interval);
    if (fill) fill.style.width = '100%';

    if (!res.ok) {
      // FastAPI HTTPException bodies carry {detail}, not {stderr}.
      const err = await res.json().catch(() => ({}));
      log(`ERROR ${res.status}: ${err.detail || res.statusText}`);
      document.getElementById('metricsStatus').textContent = 'Error';
    } else {
      const result = await res.json();
      if (result.status === 'success') {
        log(`Experiment ${activeExp.id} completed in ${result.elapsed_s}s`);
        if (result.metrics && Object.keys(result.metrics).length) {
          activeMetrics = result.metrics;
          renderMetrics(result.metrics);
          renderInspector(result.metrics);
          document.getElementById('metricsStatus').textContent = `Completed (${result.elapsed_s}s)`;
        }
        // Reload image (cache-bust)
        const imgEl = document.querySelector('.dashboard-img');
        if (imgEl) imgEl.src = `${API}/api/results/${activeExp.id}/image?t=${Date.now()}`;
      } else {
        log(`ERROR: ${result.stderr?.slice(0, 200) || 'Experiment did not write fresh results'}`);
        document.getElementById('metricsStatus').textContent = 'Error';
      }
    }
  } catch (e) {
    clearInterval(interval);
    log(`ERROR: ${e.message}`);
  }

  btn.disabled = false;
  btn.innerHTML = '▸ Run Simulation';
  setTimeout(() => { if (fill) fill.style.width = '0%'; }, 2000);
}

// ── Quick Explore (live lightweight endpoints) ──
async function quickExplore() {
  if (!activeExp || !activeExp.live) return;
  const live = activeExp.live;
  const btn = document.getElementById('liveBtn');
  if (btn) { btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Exploring...'; }

  const payload = { params: {} };
  for (const [sliderKey, apiKey] of Object.entries(live.map || {})) {
    const slider = document.getElementById(`param-${sliderKey}`);
    if (slider) payload.params[apiKey] = parseFloat(slider.value);
  }

  log(`Quick Explore: ${activeExp.id} → ${live.endpoint}`);
  try {
    const res = await fetch(`${API}${live.endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const e = await res.json().catch(() => ({}));
      log(`ERROR ${res.status}: ${e.detail || res.statusText}`);
    } else {
      const data = await res.json();
      renderLiveResult(live.kind, data);
      log(`Quick Explore complete (${activeExp.id})`);
    }
  } catch (e) {
    log(`ERROR: ${e.message}`);
  }
  if (btn) { btn.disabled = false; btn.innerHTML = '⚡ Quick Explore'; }
}

function liveCardHTML(cards) {
  return `<div class="metrics-grid">${cards.map(c => `
    <div class="metric-card ${c.colorClass || ''}">
      <span class="metric-card__label">${c.label}</span>
      <span class="metric-card__value">${c.value ?? '—'}</span>
      <span class="metric-card__unit">${c.unit || ''}</span>
    </div>`).join('')}</div>`;
}

function renderLiveResult(kind, data) {
  const el = document.getElementById('metricsContent');
  const statusEl = document.getElementById('metricsStatus');
  if (!el) return;

  if (kind === 'sweep') {
    const peak = data.peak || {};
    const cards = [
      { label: 'Peak γ (live)', value: peak.gamma_cm, unit: 'cm⁻¹', colorClass: 'metric-card--green' },
      { label: 'Peak η (live)', value: (peak.efficiency ?? 0).toFixed(4), unit: '' },
      { label: 'Sweep points', value: (data.sweep || []).length, unit: '' },
    ];
    el.innerHTML = liveCardHTML(cards) +
      `<div class="chart-container" id="liveChart" style="min-height:200px"><canvas id="liveCanvas"></canvas></div>`;
    if (statusEl) statusEl.textContent = 'Live sweep';
    requestAnimationFrame(() => drawLiveSweep(data.sweep || [], peak));
  } else if (kind === 'reservoir') {
    const p = data.params || {};
    const cards = [
      { label: 'Memory Capacity', value: (data.mc_total ?? 0).toFixed(3), unit: '', colorClass: 'metric-card--green' },
      { label: 'Mean activation', value: (data.mean_activation ?? 0).toFixed(3), unit: '' },
      { label: 'Reservoir N', value: p.n_reservoir, unit: '' },
      { label: 'Spectral ρ', value: p.spectral_radius, unit: '' },
      { label: 'p_release', value: p.p_release, unit: '' },
    ];
    el.innerHTML = liveCardHTML(cards);
    if (statusEl) statusEl.textContent = 'Live ESN';
  }
}

function drawLiveSweep(sweep, peak) {
  const canvas = document.getElementById('liveCanvas');
  if (!canvas) return;
  const pts = (sweep || []).map(s => ({ g: s.gamma_cm, e: s.efficiency })).filter(p => p.g > 0);
  if (!pts.length) return;

  const ctx = canvas.getContext('2d');
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width * 2;
  canvas.height = 360;
  canvas.style.height = '180px';
  const w = canvas.width, h = canvas.height, pad = { l: 64, r: 30, t: 24, b: 46 };

  ctx.fillStyle = 'rgba(0,0,0,0.3)';
  ctx.fillRect(0, 0, w, h);

  const xs = pts.map(p => Math.log10(p.g));
  const xMin = Math.min(...xs), xMax = Math.max(...xs), span = (xMax - xMin) || 1;
  const eMax = (Math.max(...pts.map(p => p.e)) || 1) * 1.15;
  const toX = v => pad.l + (Math.log10(v) - xMin) / span * (w - pad.l - pad.r);
  const toY = v => pad.t + (1 - v / eMax) * (h - pad.t - pad.b);

  if (peak && peak.gamma_cm > 0) {
    ctx.strokeStyle = 'rgba(10,255,239,0.4)';
    ctx.lineWidth = 2; ctx.setLineDash([6, 4]);
    ctx.beginPath(); ctx.moveTo(toX(peak.gamma_cm), pad.t); ctx.lineTo(toX(peak.gamma_cm), h - pad.b); ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.beginPath();
  ctx.strokeStyle = '#00d2ff';
  ctx.lineWidth = 3;
  pts.forEach((p, i) => { const x = toX(p.g), y = toY(p.e); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
  ctx.stroke();

  ctx.fillStyle = 'rgba(255,255,255,0.5)';
  ctx.font = '18px IBM Plex Mono, monospace';
  ctx.fillText('η transport', 8, pad.t - 8);
  ctx.fillText('γ dephasing (cm⁻¹, log)', w / 2 - 110, h - 12);
}

// ── Utility Functions ──
function log(msg) {
  const el = document.getElementById('console');
  if (el) el.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  console.log(`[BioComp] ${msg}`);
}

function formatNum(n) {
  if (n === undefined || n === null) return '—';
  if (typeof n !== 'number') return String(n);
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + 'k';
  if (Math.abs(n) < 0.001 && n !== 0) return n.toExponential(2);
  if (Number.isInteger(n)) return n.toLocaleString();
  return n.toFixed(4);
}

function flattenObj(obj, prefix = '') {
  const result = {};
  for (const [key, val] of Object.entries(obj)) {
    const fullKey = prefix ? `${prefix}.${key}` : key;
    if (val && typeof val === 'object' && !Array.isArray(val)) {
      Object.assign(result, flattenObj(val, fullKey));
    } else if (typeof val === 'number') {
      result[fullKey] = val;
    }
  }
  return result;
}

function syntaxHighlight(json) {
  return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, match => {
    let cls = 'number';
    if (/^"/.test(match)) {
      cls = /:$/.test(match) ? 'key' : 'string';
    } else if (/true|false/.test(match)) {
      cls = 'boolean';
    }
    return `<span class="${cls}">${match}</span>`;
  });
}
