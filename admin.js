const modelGrid = document.querySelector("#modelGrid");
const selectedModelName = document.querySelector("#selectedModelName");
const metricList = document.querySelector("#metricList");
const failureGrid = document.querySelector("#failureGrid");
const failureTitle = document.querySelector("#failureTitle");
const failureCount = document.querySelector("#failureCount");
const showAllToggle = document.querySelector("#showAllToggle");

let models = [];
let selectedModelId = null;
let failedOnly = true;

function formatPercent(value) {
  if (value == null) return "--";
  return `${Math.round(value * 1000) / 10}%`;
}

function formatNumber(value) {
  if (value == null) return "--";
  return new Intl.NumberFormat().format(value);
}

function modelStatus(model) {
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
            Top-3 ${formatPercent(metrics.accuracy_top_3)} · Test ${formatNumber(metrics.test_size)}
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
    ["Artifact", model.artifactAvailable ? "Available" : "Missing"],
    ["Top-1 Accuracy", formatPercent(metrics.accuracy_top_1)],
    ["Top-3 Accuracy", formatPercent(metrics.accuracy_top_3)],
    ["Train Size", formatNumber(metrics.train_size)],
    ["Test Size", formatNumber(metrics.test_size)],
    ["Failures", `${formatNumber(model.failureCount)} (${formatPercent(model.failureRate)})`],
  ];

  if (metrics.classifier) rows.push(["Classifier", metrics.classifier]);
  if (metrics.clip_model) rows.push(["Embedding", metrics.clip_model]);
  if (metrics.image_size) rows.push(["Image Size", `${metrics.image_size}px`]);
  if (metrics.epochs) rows.push(["Epochs", metrics.epochs]);
  if (metrics.batch_size) rows.push(["Batch Size", metrics.batch_size]);
  if (metrics.learning_rate) rows.push(["Learning Rate", Number(metrics.learning_rate).toExponential(2)]);
  if (metrics.weight_decay) rows.push(["Weight Decay", Number(metrics.weight_decay).toExponential(2)]);
  if (metrics.dropout) rows.push(["Dropout", Math.round(metrics.dropout * 100) / 100]);
  if (tuning.trials) rows.push(["Tuning Trials", tuning.trials]);
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
  return example.topPredictions
    .slice(0, 3)
    .map((item) => `${item.display} ${formatPercent(item.probability)}`)
    .join(" / ");
}

function renderExamples(payload) {
  failureTitle.textContent = failedOnly ? "Failed artwork" : "Evaluation artwork";
  failureCount.textContent = formatNumber(payload.total);

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
  const response = await fetch(`/api/models/${selectedModelId}/examples?failedOnly=${failedOnly}&limit=120`);
  if (!response.ok) throw new Error(`Example request failed: ${response.status}`);
  renderExamples(await response.json());
}

async function selectModel(modelId) {
  selectedModelId = modelId;
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
  showAllToggle.textContent = failedOnly ? "Failures" : "All";
  showAllToggle.setAttribute("aria-pressed", String(!failedOnly));
  await loadExamples();
});

loadModels().catch((error) => {
  selectedModelName.textContent = "Admin unavailable";
  failureGrid.innerHTML = `<p class="empty-state">${error.message}</p>`;
});
