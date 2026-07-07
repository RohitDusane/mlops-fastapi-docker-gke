// main.js — no framework, no build step.

const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Canonical category -> color and display label. This is the ONLY place
// that maps "low"/"moderate"/"high" to a color or a human-readable phrase —
// the backend sends the plain key, nothing else.
const CATEGORY_META = {
  low: { color: "#2dd4bf", label: "LOWER LIKELIHOOD" },
  moderate: { color: "#f2a65a", label: "ELEVATED LIKELIHOOD" },
  high: { color: "#ef5b5b", label: "HIGHER LIKELIHOOD" },
};

function categoryMeta(level) {
  return CATEGORY_META[level] || { color: "#5c7086", label: "—" };
}

// ---- Gauge rendering ----

function renderGauge(svg, riskScore, riskCategory) {
  if (!svg) {
    console.error("renderGauge: target <svg> element not found");
    return;
  }
  const hasResult = typeof riskScore === "number";
  const clamped = hasResult ? Math.min(Math.max(riskScore, 0), 1) : 0;
  const angle = hasResult ? -135 + clamped * 270 : -135;
  const dash = hasResult ? `${clamped * 251} 251` : "2 251";
  const readout = hasResult ? `${Math.round(clamped * 100)}%` : "—";
  const readoutColor = hasResult ? categoryMeta(riskCategory).color : "#5c7086";

  svg.innerHTML = `
    <defs>
      <linearGradient id="gaugeGradient-${svg.id}" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="#2dd4bf" />
        <stop offset="50%" stop-color="#f2a65a" />
        <stop offset="100%" stop-color="#ef5b5b" />
      </linearGradient>
      <filter id="glow-${svg.id}"><feGaussianBlur stdDeviation="3" result="blur" />
        <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
      </filter>
    </defs>
    <path d="M 30 150 A 80 80 0 1 1 190 150" fill="none" stroke="#24314a" stroke-width="14" stroke-linecap="round" />
    <path d="M 30 150 A 80 80 0 1 1 190 150" fill="none" stroke="url(#gaugeGradient-${svg.id})"
          stroke-width="14" stroke-linecap="round" stroke-dasharray="${dash}"
          filter="url(#glow-${svg.id})" style="transition: stroke-dasharray 0.8s cubic-bezier(.16,1,.3,1)" />
    <g style="transform-origin:110px 150px; transform: rotate(${angle}deg); transition: transform 0.8s cubic-bezier(.16,1,.3,1)">
      <line x1="110" y1="150" x2="110" y2="75" stroke="#e8eef5" stroke-width="3" stroke-linecap="round" />
    </g>
    <circle cx="110" cy="150" r="7" fill="#e8eef5" />
    <text x="110" y="120" text-anchor="middle" font-family="'IBM Plex Mono',monospace" font-size="26"
          font-weight="600" fill="${readoutColor}">${readout}</text>
    <text x="110" y="140" text-anchor="middle" font-family="'IBM Plex Mono',monospace" font-size="10"
          letter-spacing="1" fill="#5c7086">RISK SCORE</text>
  `;
}

function updateGauges(riskScore, riskCategory) {
  renderGauge(document.getElementById("hero-gauge-svg"), riskScore, riskCategory);
  renderGauge(document.getElementById("result-gauge-svg"), riskScore, riskCategory);
}

updateGauges(undefined, undefined); // initial idle render

// ---- 3D tilt on the gauge panels ----

if (!prefersReducedMotion) {
  document.querySelectorAll(".gauge-panel").forEach((panel) => {
    const inner = document.createElement("div");
    inner.className = "gauge-inner";
    while (panel.firstChild) inner.appendChild(panel.firstChild);
    panel.appendChild(inner);

    panel.addEventListener("mousemove", (e) => {
      const rect = panel.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width - 0.5;
      const y = (e.clientY - rect.top) / rect.height - 0.5;
      inner.style.transform = `rotateX(${y * -6}deg) rotateY(${x * 8}deg)`;
    });
    panel.addEventListener("mouseleave", () => {
      inner.style.transform = "rotateX(0deg) rotateY(0deg)";
    });
  });
}

// ---- Toggle switches ----

document.querySelectorAll(".toggle-field").forEach((field) => {
  const toggle = field.querySelector(".toggle");
  toggle.setAttribute("role", "switch");
  toggle.setAttribute("tabindex", "0");
  toggle.setAttribute("aria-checked", toggle.dataset.on === "true");

  function flip() {
    const on = toggle.dataset.on === "true";
    toggle.dataset.on = String(!on);
    toggle.setAttribute("aria-checked", String(!on));
  }

  toggle.addEventListener("click", flip);
  toggle.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); flip(); }
  });
});

// ---- Scroll-to-form ----

const scrollBtn = document.getElementById("scroll-to-form");
if (scrollBtn) {
  scrollBtn.addEventListener("click", () => {
    document.getElementById("assessment-form")?.scrollIntoView({ behavior: prefersReducedMotion ? "auto" : "smooth" });
  });
}

// ---- Form submission ----

function collectPayload() {
  const val = (id) => Number(document.getElementById(id).value);
  const toggleVal = (field) => {
    const el = document.querySelector(`.toggle-field[data-field="${field}"] .toggle`);
    return el.dataset.on === "true" ? 1 : 0;
  };

  return {
    HighBP: toggleVal("HighBP"),
    HighChol: toggleVal("HighChol"),
    CholCheck: toggleVal("CholCheck"),
    BMI: val("BMI"),
    Smoker: toggleVal("Smoker"),
    Stroke: toggleVal("Stroke"),
    HeartDiseaseorAttack: toggleVal("HeartDiseaseorAttack"),
    PhysActivity: toggleVal("PhysActivity"),
    Fruits: toggleVal("Fruits"),
    Veggies: toggleVal("Veggies"),
    HvyAlcoholConsump: toggleVal("HvyAlcoholConsump"),
    AnyHealthcare: toggleVal("AnyHealthcare"),
    NoDocbcCost: toggleVal("NoDocbcCost"),
    GenHlth: val("GenHlth"),
    MentHlth: val("MentHlth"),
    PhysHlth: val("PhysHlth"),
    DiffWalk: toggleVal("DiffWalk"),
    Sex: val("Sex"),
    Age: val("Age"),
    Education: val("Education"),
    Income: val("Income"),
  };
}

function setText(id, text) {
  const el = document.getElementById(id);
  if (!el) {
    console.error(`renderResult: element #${id} not found in the DOM`);
    return;
  }
  el.innerHTML = text;
}

function renderResult(result) {
  const meta = categoryMeta(result.risk_category);

  setText("risk-number", `${Math.round(result.risk_score * 100)}%`);

  const label = document.getElementById("risk-label");
  if (label) {
    label.textContent = meta.label;
    label.style.color = meta.color;
    label.className = `risk-label ${result.risk_category}`;
  } else {
    console.error("renderResult: #risk-label not found in the DOM");
  }

  setText("model-version", result.model_version);
  setText("risk-reasons", result.reasons.map((r) => `<li>✓ ${r}</li>`).join(""));
  setText("suggestions", result.recommendations.map((r) => `<div class="suggestion">💡 ${r}</div>`).join(""));
}

const form = document.getElementById("assessment-form-el");
if (form) {
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = document.getElementById("submit-btn");
    const errEl = document.getElementById("submit-error");
    btn.disabled = true;
    btn.textContent = "Assessing…";
    errEl.textContent = "";

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(collectPayload()),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail ? JSON.stringify(body.detail) : `Request failed (${response.status})`);
      }
      const result = await response.json();
      updateGauges(result.risk_score, result.risk_category);
      renderResult(result);
    } catch (err) {
      errEl.textContent = err.message;
      console.error(err);
    } finally {
      btn.disabled = false;
      btn.textContent = "Assess risk";
    }
  });
}

// NOTE: the previous version had a trailing block here —
//   requestAnimationFrame(() => { updateGauges(result.risk_score, ...) })
// — referencing `result`, a variable that only exists inside the submit
// handler above. Running unconditionally on page load, this threw
// ReferenceError: result is not defined every time the page loaded. It did
// nothing useful (gauges are already updated inside the submit handler on
// success) and has been removed.