const modelGrid = document.querySelector("#modelGrid");
const selectedModelName = document.querySelector("#selectedModelName");
const metricList = document.querySelector("#metricList");
const failureGrid = document.querySelector("#failureGrid");
const failureTitle = document.querySelector("#failureTitle");
const pageStatus = document.querySelector("#pageStatus");
const showAllToggle = document.querySelector("#showAllToggle");
const prevPage = document.querySelector("#prevPage");
const nextPage = document.querySelector("#nextPage");
const themeToggle = document.querySelector("#themeToggle");

let models = [];
let selectedModelId = null;
let failedOnly = true;
let exampleOffset = 0;
const pageSize = 18;

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("coversense-theme", theme);
  if (themeToggle) themeToggle.textContent = theme === "dark" ? "Light" : "Dark";
}

applyTheme(localStorage.getItem("coversense-theme") || "light");

function formatPercent(value) {
  if (value == null) return "--";
  return `${Math.round(value * 1000) / 10}%`;
}

function formatNumber(value) {
  if (value == null) return "--";
  return new Intl.NumberFormat().format(value);
}

function hasValue(value) {
  return value != null && value !== "";
}

function modelStatus(model) {
  if (model.serving) return "Serving";
  if (!model.artifactAvailable) return "Missing artifact";
  if (!model.examplesAvailable) return "No evaluation examples";
  return "Ready";
}

function renderModelCards() {
  modelGrid.innerHTML = models
    .map((model) => {
      const metrics = model.metrics || {};
      const selected = model.id === selectedModelId ? " is-selected" : "";
      return `
        <button class="model-card${selected}" type="button" data-model-id="${model.id}">
          <span class="model-card-topline">
            <span>${model.name}</span>
            <span>${modelStatus(model)}</span>
          </span>
          <strong>${formatPercent(metrics.accuracy_top_1)}</strong>
          <span class="model-card-meta">
            Broad ${formatPercent(metrics.accuracy_broad_top_1)} · Top-3 ${formatPercent(metrics.accuracy_top_3)}
          </span>
        </button>
      `;
    })
    .join("");

  modelGrid.querySelectorAll("[data-model-id]").forEach((button) => {
    button.addEventListener("click", () => selectModel(button.dataset.modelId));
  });
}

function metricRows(model) {
  const metrics = model.metrics || {};
  const tuning = metrics.tuning || {};
  const rows = [
    ["Family", model.family],
    ["Status", modelStatus(model)],
    ["Artifact", model.artifactAvailable ? "Available" : "Missing"],
    ["Top-1 Accuracy", formatPercent(metrics.accuracy_top_1)],
    ["Top-3 Accuracy", formatPercent(metrics.accuracy_top_3)],
    ["Broad Top-1", formatPercent(metrics.accuracy_broad_top_1)],
    ["Broad Top-3", formatPercent(metrics.accuracy_broad_top_3)],
    ["Near Misses", formatPercent(metrics.near_miss_rate)],
    ["Far Misses", formatPercent(metrics.far_miss_rate)],
    ["Hierarchical Score", formatPercent(metrics.hierarchical_score)],
    ["Train Size", formatNumber(metrics.train_size)],
    ["Test Size", formatNumber(metrics.test_size)],
    ["Failures", `${formatNumber(model.failureCount)} (${formatPercent(model.failureRate)})`],
  ];

  if (hasValue(metrics.classifier)) rows.push(["Classifier", metrics.classifier]);
  if (hasValue(metrics.clip_model)) rows.push(["Embedding", metrics.clip_model]);
  if (hasValue(metrics.image_size)) rows.push(["Image Size", `${metrics.image_size}px`]);
  if (hasValue(metrics.epochs)) rows.push(["Epochs", metrics.epochs]);
  if (hasValue(metrics.batch_size)) rows.push(["Batch Size", metrics.batch_size]);
  if (hasValue(metrics.learning_rate)) rows.push(["Learning Rate", Number(metrics.learning_rate).toExponential(2)]);
  if (hasValue(metrics.weight_decay)) rows.push(["Weight Decay", Number(metrics.weight_decay).toExponential(2)]);
  if (hasValue(metrics.dropout)) rows.push(["Dropout", Math.round(metrics.dropout * 100) / 100]);
  if (hasValue(metrics.sibling_smoothing)) {
    rows.push(["Sibling Smoothing", Math.round(metrics.sibling_smoothing * 100) / 100]);
  }
  if (hasValue(tuning.trials)) rows.push(["Tuning Trials", tuning.trials]);
  if (tuning.best_settings) rows.push(["Best CNN Width", tuning.best_settings.base_channels]);

  return rows;
}

function renderMetrics(model) {
  selectedModelName.textContent = model.name;
  metricList.innerHTML = metricRows(model)
    .map(([label, value]) => `<div><dt>${label}</dt><dd>${value ?? "--"}</dd></div>`)
    .join("");
}

function predictionText(example) {
  const broadText =
    example.errorType === "near_miss"
      ? `Near miss: ${example.actualBroad}`
      : example.errorType === "far_miss"
        ? `${example.actualBroad} -> ${example.predictedBroad}`
        : "Exact";
  return example.topPredictions
    .slice(0, 3)
    .map((item) => `${item.display} ${formatPercent(item.probability)}`)
    .join(" / ")
    .concat(` · ${broadText}`);
}

function renderExamples(payload) {
  failureTitle.textContent = failedOnly ? "Failed artwork" : "Evaluation artwork";
  const currentPage = payload.total === 0 ? 0 : Math.floor(payload.offset / payload.limit) + 1;
  const totalPages = Math.ceil(payload.total / payload.limit);
  pageStatus.textContent =
    payload.total === 0
      ? "0"
      : `${formatNumber(payload.offset + 1)}-${formatNumber(
          Math.min(payload.offset + payload.examples.length, payload.total),
        )} of ${formatNumber(payload.total)}`;
  pageStatus.setAttribute("title", `Page ${currentPage} of ${totalPages}`);
  prevPage.disabled = payload.offset === 0;
  nextPage.disabled = payload.offset + payload.limit >= payload.total;

  if (!payload.examples.length) {
    failureGrid.innerHTML = `<p class="empty-state">No examples available for this model yet.</p>`;
    return;
  }

  failureGrid.innerHTML = payload.examples
    .map(
      (example) => `
        <article class="failure-card">
          <img src="${example.imageUrl}" alt="">
          <div>
            <span class="failure-labels">
              <strong>${example.predictedDisplay}</strong>
              <span>actual ${example.actualDisplay}</span>
            </span>
            <p>${predictionText(example)}</p>
          </div>
        </article>
      `,
    )
    .join("");
}

async function loadExamples() {
  const params = new URLSearchParams({
    failedOnly: String(failedOnly),
    limit: String(pageSize),
    offset: String(exampleOffset),
  });
  const response = await fetch(`/api/models/${selectedModelId}/examples?${params}`);
  if (!response.ok) throw new Error(`Example request failed: ${response.status}`);
  renderExamples(await response.json());
}

async function selectModel(modelId) {
  selectedModelId = modelId;
  exampleOffset = 0;
  renderModelCards();
  const model = models.find((item) => item.id === selectedModelId);
  if (!model) return;
  renderMetrics(model);
  await loadExamples();
}

async function loadModels() {
  const response = await fetch("/api/models");
  if (!response.ok) throw new Error(`Model request failed: ${response.status}`);
  const payload = await response.json();
  models = payload.models || [];
  selectedModelId = models[0]?.id ?? null;
  renderModelCards();
  if (selectedModelId) await selectModel(selectedModelId);
}

showAllToggle.addEventListener("click", async () => {
  failedOnly = !failedOnly;
  exampleOffset = 0;
  showAllToggle.textContent = failedOnly ? "Failures" : "All";
  showAllToggle.setAttribute("aria-pressed", String(!failedOnly));
  await loadExamples();
});

prevPage.addEventListener("click", async () => {
  exampleOffset = Math.max(0, exampleOffset - pageSize);
  await loadExamples();
});

nextPage.addEventListener("click", async () => {
  exampleOffset += pageSize;
  await loadExamples();
});

if (themeToggle) {
  themeToggle.addEventListener("click", () => {
    const currentTheme = document.documentElement.dataset.theme || "light";
    applyTheme(currentTheme === "dark" ? "light" : "dark");
  });
}

loadModels().catch((error) => {
  selectedModelName.textContent = "Admin unavailable";
  failureGrid.innerHTML = `<p class="empty-state">${error.message}</p>`;
});
