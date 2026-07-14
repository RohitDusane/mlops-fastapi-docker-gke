// main.js — Production-ready with confetti + accessibility
const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Confetti Library (already added in HTML)
function launchConfetti() {
  if (prefersReducedMotion) return;
  const colors = ['#00e6c3', '#f4d35e', '#e8eef5', '#2dd4bf', '#d4af37'];
  const duration = 2600;
  const end = Date.now() + duration;

  (function frame() {
    confetti({
      particleCount: 7,
      angle: Math.random() * 70 + 55,
      spread: 60,
      origin: { x: Math.random(), y: Math.random() - 0.15 },
      colors: colors
    });
    if (Date.now() < end) requestAnimationFrame(frame);
  })();
}

// Canonical category meta
const CATEGORY_META = {
  low: { color: "#2dd4bf", label: "LOWER LIKELIHOOD" },
  moderate: { color: "#f2a65a", label: "ELEVATED LIKELIHOOD" },
  high: { color: "#ef5b5b", label: "HIGHER LIKELIHOOD" },
};

function categoryMeta(level) {
  return CATEGORY_META[level] || { color: "#5c7086", label: "—" };
}

// // Real Confetti Animation
// function launchConfetti() {
//   if (prefersReducedMotion) return;
//   const colors = ['#00e6c3', '#f4d35e', '#e8eef5', '#2dd4bf', '#d4af37'];
//   const duration = 2600;
//   const end = Date.now() + duration;

//   (function frame() {
//     confetti({
//       particleCount: 7,
//       angle: Math.random() * 70 + 55,
//       spread: 60,
//       origin: { x: Math.random(), y: Math.random() - 0.15 },
//       colors: colors
//     });
//     if (Date.now() < end) requestAnimationFrame(frame);
//   })();
// }

// Enhanced Gauge with Tick Marks
function renderGauge(svg, riskScore, riskCategory) {
  if (!svg) return;

  if (!svg.dataset.initialized) {
    svg.innerHTML = `
    <defs>
      <linearGradient id="gaugeGradient-${svg.id}" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="#00e6c3"/>
        <stop offset="48%" stop-color="#f4a261"/>
        <stop offset="100%" stop-color="#ff4d4d"/>
      </linearGradient>
      <linearGradient id="needleGradient-${svg.id}" x1="50%" y1="0%" x2="50%" y2="100%">
        <stop offset="0%" stop-color="#ffe066"/>
        <stop offset="100%" stop-color="#e6b800"/>
      </linearGradient>
      <filter id="glow-${svg.id}" x="-80%" y="-80%" width="260%" height="260%">
        <feGaussianBlur stdDeviation="5" result="blur"/>
        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
      <pattern id="uaePattern" patternUnits="userSpaceOnUse" width="60" height="60" patternTransform="rotate(30)">
        <path d="M10 30 L50 30 M30 10 L30 50" stroke="#ffffff" stroke-opacity="0.07" stroke-width="1.5"/>
      </pattern>
    </defs>

    <path d="M30 150 A80 80 0 1 1 190 150" fill="none" stroke="#0f1c38" stroke-width="20" stroke-linecap="round"/>
    <path d="M24 150 A86 86 0 1 1 196 150" fill="none" stroke="url(#uaePattern)" stroke-width="24" opacity="0.6"/>

    <path id="${svg.id}-progress" d="M30 150 A80 80 0 1 1 190 150" fill="none" 
          stroke="url(#gaugeGradient-${svg.id})" stroke-width="13.5" stroke-linecap="round"
          stroke-dasharray="2 251" filter="url(#glow-${svg.id})"/>

    <!-- Tick Marks -->
    <g opacity="0.75">
      <line x1="47" y1="125" x2="54" y2="118" stroke="#00e6c3" stroke-width="2.5" stroke-linecap="round"/>
      <line x1="110" y1="73" x2="110" y2="66" stroke="#f4a261" stroke-width="2.5" stroke-linecap="round"/>
      <line x1="173" y1="125" x2="166" y2="118" stroke="#ff4d4d" stroke-width="2.5" stroke-linecap="round"/>
    </g>


    <!-- Needle -->
    <g id="${svg.id}-needle" transform="rotate(-135 110 150)">
      <!-- Main needle body -->
      <line
        x1="110" y1="150"
        x2="110" y2="71"          <!-- Adjusted length -->
        stroke="url(#needleGradient-${svg.id})"
        stroke-width="7"
        stroke-linecap="round"
        filter="url(#glow-${svg.id})"
      />
      <!-- Highlight line -->
      <line
        x1="110" y1="150"
        x2="110" y2="76"
        stroke="#ffffff"
        stroke-width="2.8"
        stroke-linecap="round"
        opacity="0.9"
      />
    </g>

    <!-- Luxurious center -->
    <circle cx="110" cy="150" r="12" fill="#0a1428" stroke="#1e2a4d" stroke-width="2.5"/>
    <circle cx="110" cy="150" r="6.5" fill="#ffe066"/>
    
    <text id="${svg.id}-score" x="110" y="115" text-anchor="middle" font-size="34" font-weight="700" fill="#e0f2f1" letter-spacing="-1.5px">—</text>
    <text x="110" y="143" text-anchor="middle" font-size="9.8" letter-spacing="1.8px" fill="#8a9eb8" font-weight="500">RISK SCORE</text>
    `;

    svg.dataset.initialized = "true";
  }

  const hasResult = typeof riskScore === "number";
  const score = hasResult ? Math.min(Math.max(riskScore, 0), 1) : 0;
  const angle = hasResult ? (-135 + score * 270) : 0;
  const dash = hasResult ? score * 251 : 2;
  const meta = categoryMeta(riskCategory);

  const progress = svg.querySelector(`#${svg.id}-progress`);
  const needle = svg.querySelector(`#${svg.id}-needle`);
  const text = svg.querySelector(`#${svg.id}-score`);

  progress.style.transition = "stroke-dasharray 950ms cubic-bezier(0.34, 1.56, 0.64, 1)";
  progress.setAttribute("stroke-dasharray", `${dash} 251`);

  if (needle) {
    needle.style.transition = "transform 950ms cubic-bezier(0.34, 1.56, 0.64, 1)";
    needle.setAttribute("transform", `rotate(${angle} 110 150)`);
  }

  animateScore(text, hasResult ? Math.round(score * 100) : 0);
  text.setAttribute("fill", hasResult ? meta.color : "#8a9eb8");
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

// Updated renderResult with confetti
function renderResult(result) {
  const meta = categoryMeta(result.risk_category);

  setText("risk-number", `${Math.round(result.risk_score * 100)}%`);

  const label = document.getElementById("risk-label");
  if (label) {
    label.textContent = meta.label;
    label.style.color = meta.color;
    label.className = `risk-label ${result.risk_category}`;
  }

  if (result.risk_category === "low") {
    launchConfetti();
  }

  setText("model-version", result.model_version || "1.0");
  setText("risk-reasons", result.reasons ? result.reasons.map(r => `<li>✓ ${r}</li>`).join("") : "");
  setText("suggestions", result.recommendations ? result.recommendations.map(r => `<div class="suggestion">💡 ${r}</div>`).join("") : "");
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

function animateScore(textEl, targetPercent) {
  let start = 0;
  const duration = 1200;
  const startTime = performance.now();

  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const current = Math.floor(start + (targetPercent - start) * progress);
    
    textEl.textContent = current + "%";
    
    if (progress < 1) {
      requestAnimationFrame(update);
    } else {
      textEl.textContent = targetPercent + "%";
    }
  }
  requestAnimationFrame(update);
}