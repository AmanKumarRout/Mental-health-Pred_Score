"use strict";

const API_URL = "https://mental-health-pred-score.onrender.com/predict";
const REQUEST_TIMEOUT_MS = 15000;
// Upper bound of the score scale used for the gauge.
const SCORE_MAX = 10;

// Client-side limits. All fall inside the ranges the FastAPI model accepts.
const NUMBER_RULES = {
  age: { label: "Age", min: 14, max: 45, integer: true },
  avg_daily_usage_hours: { label: "Daily usage", min: 0, max: 18 },
  daily_unlocks: { label: "Daily unlocks", min: 0, max: 500, integer: true },
  study_hours: { label: "Study hours", min: 0, max: 16 },
  physical_activity_hours: { label: "Physical activity", min: 0, max: 8 },
  sleep_hours_per_night: { label: "Sleep hours", min: 2, max: 14 },
};
const REQUIRED_CHOICES = {
  gender: "a gender",
  country: "a country",
  academic_level: "an academic level",
  most_used_platform: "a platform",
  purpose_of_use: "a purpose",
  stress_level: "a stress level",
};

// Frontend interpretation of the model output (not a diagnosis).
// Colour groups: 1-3 lower, 4-6 moderate, 7-10 higher.
const SCORE_BANDS = [
  { min: 7, label: "Higher range", colors: ["#14a38b", "#4f9be0"] },
  { min: 4, label: "Moderate range", colors: ["#4f6bd8", "#7d8cf0"] },
  { min: 0, label: "Lower range", colors: ["#e9a23b", "#ec8b6a"] },
];

// One message per whole-number score (the returned score is rounded to the nearest level).
const SCORE_MESSAGES = {
  1: "Very low estimate. This is only a model output, not a verdict on how you feel. If things feel heavy, talking to a trusted person or a counsellor can help.",
  2: "Low estimate. If you are struggling, consider talking to someone you trust or a campus counsellor.",
  3: "Below-average estimate. Checking in with yourself, and with someone you trust, is a good habit.",
  4: "Slightly below the middle of the scale. Regular routines and enough rest are worth paying attention to.",
  5: "Middle of the scale. The model sees a mix of factors in this profile.",
  6: "Slightly above the middle of the scale. Keep the habits that feel sustainable for you.",
  7: "Above-average estimate. The model sees a fairly balanced profile.",
  8: "Good estimate. The habits entered look positive to the model.",
  9: "High estimate. The profile looks well balanced to the model.",
  10: "Top of the scale. Model outputs are estimates, so keep checking in with yourself over time.",
};

const form = document.getElementById("predict-form");
const submitButton = document.getElementById("submit-btn");
const resultsPanel = document.getElementById("results");
const gaugeProgress = document.getElementById("gauge-progress");
const scoreNumber = document.getElementById("score-number");

class ApiError extends Error {
  constructor(message, fieldErrors = {}) {
    super(message);
    this.fieldErrors = fieldErrors;
  }
}

const formatNumber = (value) => Number(value.toFixed(1)).toString();
const prefersReducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ---------- Form handling & validation ---------- */

function readForm() {
  const raw = {};
  for (const [name, value] of new FormData(form)) raw[name] = String(value).trim();
  return raw;
}

function validate(raw) {
  const errors = {};
  for (const [name, article] of Object.entries(REQUIRED_CHOICES)) {
    if (!raw[name]) errors[name] = `Select ${article}.`;
  }
  for (const [name, rule] of Object.entries(NUMBER_RULES)) {
    const value = raw[name];
    if (value === undefined || value === "") {
      errors[name] = `${rule.label} is required.`;
      continue;
    }
    const number = Number(value);
    if (!Number.isFinite(number)) errors[name] = "Enter a valid number.";
    else if (rule.integer && !Number.isInteger(number)) errors[name] = "Enter a whole number.";
    else if (number < rule.min || number > rule.max) errors[name] = `Enter a value between ${rule.min} and ${rule.max}.`;
  }
  return errors;
}

function buildPayload(raw) {
  const payload = {};
  for (const [name, value] of Object.entries(raw)) {
    payload[name] = name in NUMBER_RULES ? Number(value) : value;
  }
  return payload;
}

function setFieldError(name, message) {
  const errorElement = document.getElementById(`err-${name}`);
  if (!errorElement) return;
  const field = errorElement.closest(".field");
  errorElement.textContent = message || "";
  field.classList.toggle("has-error", Boolean(message));
  field.querySelectorAll("input, select").forEach((control) => {
    if (message) control.setAttribute("aria-invalid", "true");
    else control.removeAttribute("aria-invalid");
  });
}

function showFieldErrors(errors) {
  const names = [...Object.keys(NUMBER_RULES), ...Object.keys(REQUIRED_CHOICES)];
  names.forEach((name) => setFieldError(name, errors[name]));
  const firstInvalid = form.querySelector(".has-error input, .has-error select");
  if (firstInvalid) firstInvalid.focus();
}

form.addEventListener("input", (event) => {
  if (event.target.name) setFieldError(event.target.name, "");
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const raw = readForm();
  const errors = validate(raw);
  if (Object.keys(errors).length > 0) {
    showFieldErrors(errors);
    return;
  }

  const payload = buildPayload(raw);
  setLoading(true);
  try {
    const score = await requestPrediction(payload);
    renderResult(score, payload);
  } catch (error) {
    renderError(error);
  } finally {
    setLoading(false);
  }
});

/* ---------- API communication ---------- */

async function requestPrediction(payload) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  let response;
  try {
    response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
  } catch (error) {
    throw new ApiError(
      error.name === "AbortError"
        ? "The prediction service took too long to respond. Please try again."
        : "Unable to connect to the prediction service. Make sure the FastAPI server is running."
    );
  } finally {
    clearTimeout(timer);
  }

  if (!response.ok) throw await buildHttpError(response);

  let data;
  try {
    data = await response.json();
  } catch {
    throw new ApiError("The prediction service returned an unreadable response.");
  }
  const score = data ? data.predicted_mental_health_score : undefined;
  if (typeof score !== "number" || !Number.isFinite(score)) {
    throw new ApiError("The prediction service returned an unexpected response.");
  }
  return score;
}

async function buildHttpError(response) {
  if (response.status === 422) {
    const fieldErrors = {};
    try {
      const body = await response.json();
      (Array.isArray(body.detail) ? body.detail : []).forEach((issue) => {
        const field = Array.isArray(issue.loc) ? issue.loc[issue.loc.length - 1] : null;
        if (field) fieldErrors[field] = "The service did not accept this value.";
      });
    } catch { /* body unreadable: fall back to the generic message */ }
    return new ApiError("The service rejected some of the submitted values. Please review the highlighted fields and try again.", fieldErrors);
  }
  if (response.status >= 500) {
    return new ApiError("The prediction service ran into a problem while processing this request. Please try again shortly.");
  }
  return new ApiError(`The prediction service responded unexpectedly (HTTP ${response.status}).`);
}

/* ---------- Rendering ---------- */

function setLoading(isLoading) {
  submitButton.disabled = isLoading;
  submitButton.classList.toggle("is-loading", isLoading);
  submitButton.querySelector(".cta-label").textContent = isLoading ? "Analyzing..." : "Analyze My Score";
  if (isLoading) resultsPanel.dataset.state = "loading";
  if (isLoading && isSingleColumn()) resultsPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}

const isSingleColumn = () => window.matchMedia("(max-width: 959px)").matches;

function renderError(error) {
  document.getElementById("error-message").textContent =
    error instanceof ApiError ? error.message : "Something went wrong while requesting a prediction. Please try again.";
  if (error.fieldErrors && Object.keys(error.fieldErrors).length > 0) showFieldErrors(error.fieldErrors);
  resultsPanel.dataset.state = "error";
}

function renderResult(score, payload) {
  const band = SCORE_BANDS.find((item) => score >= item.min) || SCORE_BANDS[SCORE_BANDS.length - 1];
  resultsPanel.style.setProperty("--band-a", band.colors[0]);
  resultsPanel.style.setProperty("--band-b", band.colors[1]);
  document.getElementById("band-label").textContent = band.label;
  const level = Math.min(10, Math.max(1, Math.round(score)));
  document.getElementById("band-text").textContent = SCORE_MESSAGES[level];
  document.getElementById("score-max").textContent = `/ ${SCORE_MAX}`;

  renderInsights(payload);
  renderBars(payload);
  resultsPanel.dataset.state = "success";

  const heading = document.getElementById("result-heading");
  heading.focus({ preventScroll: true });
  if (isSingleColumn()) resultsPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  animateScore(score);
}

function renderInsights(payload) {
  const items = [
    ["Sleep", `${formatNumber(payload.sleep_hours_per_night)} hrs`],
    ["Daily screen time", `${formatNumber(payload.avg_daily_usage_hours)} hrs`],
    ["Study time", `${formatNumber(payload.study_hours)} hrs`],
    ["Stress level", payload.stress_level],
  ];
  document.getElementById("insights").innerHTML = items
    .map(([label, value]) => `<div class="insight"><span>${label}</span><strong>${value}</strong></div>`)
    .join("");
}

function renderBars(payload) {
  const metrics = [
    ["Sleep", payload.sleep_hours_per_night],
    ["Study", payload.study_hours],
    ["Physical activity", payload.physical_activity_hours],
    ["Screen usage", payload.avg_daily_usage_hours],
  ];
  document.getElementById("bars").innerHTML = metrics
    .map(([label, hours]) => {
      const percent = Math.min(100, (hours / 24) * 100);
      return `<div class="bar-row"><header><span>${label}</span><span>${formatNumber(hours)} hrs</span></header>
        <div class="bar-track"><div class="bar-fill" style="--w:${percent.toFixed(1)}%"></div></div></div>`;
    })
    .join("");
}

// Animates the counter and the circular gauge from 0 to the returned score.
function animateScore(score) {
  const fraction = Math.max(0, Math.min(1, score / SCORE_MAX));
  const draw = (progress) => {
    scoreNumber.textContent = formatNumber(score * progress);
    gaugeProgress.setAttribute("stroke-dasharray", `${(fraction * progress * 100).toFixed(2)} 100`);
  };
  if (prefersReducedMotion()) return draw(1);

  const duration = 1300;
  const start = performance.now();
  draw(0);
  const step = (now) => {
    const t = Math.min(1, (now - start) / duration);
    draw(1 - Math.pow(1 - t, 3)); // ease-out cubic
    if (t < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}
