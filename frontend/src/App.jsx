import { useEffect, useMemo, useState } from "react";

const samples = [
  ["electronic", "Electronic", "data/album_covers_20_genres/images/electronic/electronic-00410.jpg"],
  ["metal", "Metal", "data/album_covers_20_genres/images/heavymetal/heavymetal-00366.jpg"],
  ["jazz", "Jazz", "data/album_covers_20_genres/images/jazz/jazz-00316.jpg"],
  ["pop", "Pop", "data/album_covers_20_genres/images/pop/pop-00591.jpg"],
].map(([key, label, imagePath]) => ({ key, label, imagePath }));

const floatingCovers = [
  ["Electronic", "cover-electronic"],
  ["Heavy Metal", "cover-metal"],
  ["Jazz", "cover-jazz"],
  ["Pop", "cover-pop"],
];

const pageSize = 18;

function formatPercent(value) {
  if (value == null) return "--";
  return `${Math.round(value * 100)}%`;
}

function formatPercentDetailed(value) {
  if (value == null) return "--";
  return `${Math.round(value * 1000) / 10}%`;
}

function formatNumber(value) {
  if (value == null) return "--";
  return new Intl.NumberFormat().format(value);
}

function artworkUrl(imagePath) {
  return `/api/artwork/${imagePath}`;
}

function modelStatus(model) {
  if (model.serving) return "Serving";
  if (!model.artifactAvailable) return "Missing artifact";
  if (!model.examplesAvailable) return "No evaluation examples";
  return "Ready";
}

function predictionEvidence(predictions, modelInfo) {
  if (!predictions.length) return ["Model rationale will appear after a cover is analyzed."];

  const top = predictions[0];
  const topAlternatives = predictions
    .slice(0, 3)
    .map((item) => `${item.genre} ${formatPercent(item.probability)}`)
    .join(", ");
  const evidence = [
    `The CLIP embedding model ranks this closest to ${top.genre}; its broad family is ${top.broadGenreDisplay}.`,
    `Top alternatives: ${topAlternatives}. Similar held-out covers below provide a quick plausibility check.`,
  ];

  if (modelInfo.accuracyTop1 != null && modelInfo.accuracyBroadTop1 != null) {
    evidence.push(
      `Held-out accuracy: ${formatPercent(modelInfo.accuracyTop1)} exact top-1, ${formatPercent(
        modelInfo.accuracyBroadTop1,
      )} broad-family top-1.`,
    );
  }

  return evidence;
}

function useTheme() {
  const [theme, setTheme] = useState(() => localStorage.getItem("coversense-theme") || "light");

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("coversense-theme", theme);
  }, [theme]);

  return [theme, () => setTheme((currentTheme) => (currentTheme === "dark" ? "light" : "dark"))];
}

function TopBar({ statusText, theme, onToggleTheme }) {
  return (
    <header className="topbar">
      <a className="brand-mark" href="/" aria-label="CoverSense home">
        <span className="brand-glyph" aria-hidden="true" />
        CoverSense
      </a>
      <nav className="top-nav" aria-label="Primary navigation">
        <a href="/#predictor">Predict</a>
        <details className="menu-dropdown">
          <summary>Menu</summary>
          <div className="menu-list">
            <a href="/">Inference app</a>
            <a href="/admin">Model dashboard</a>
            <a href="/admin">Evaluation reports</a>
            <a href="https://github.com/okihayashi/coversense">GitHub repo</a>
          </div>
        </details>
      </nav>
      <div className="topbar-actions">
        <div className="model-pill" aria-label="Current model type">
          <span className="status-dot" />
          {statusText}
        </div>
        <button className="theme-toggle" type="button" onClick={onToggleTheme} aria-label="Toggle dark and light mode">
          {theme === "dark" ? "Light" : "Dark"}
        </button>
        <a className="admin-link" href="/admin">Dashboard</a>
      </div>
    </header>
  );
}

function InferencePage({ statusText, setStatusText }) {
  const [backendReady, setBackendReady] = useState(false);
  const [previewUrl, setPreviewUrl] = useState("");
  const [selectedName, setSelectedName] = useState("");
  const [predictions, setPredictions] = useState([]);
  const [similarExamples, setSimilarExamples] = useState([]);
  const [modelInfo, setModelInfo] = useState({});
  const [isPredicting, setIsPredicting] = useState(false);
  const [error, setError] = useState("");

  const topPrediction = predictions[0];
  const evidence = useMemo(() => predictionEvidence(predictions, modelInfo), [predictions, modelInfo]);

  useEffect(() => {
    let active = true;
    fetch("/api/health")
      .then((response) => {
        if (!response.ok) throw new Error(`Health check failed: ${response.status}`);
        return response.json();
      })
      .then((payload) => {
        if (!active) return;
        setBackendReady(Boolean(payload.modelAvailable));
        setStatusText(payload.modelAvailable ? "Trained model ready" : "Model unavailable");
      })
      .catch(() => {
        if (!active) return;
        setBackendReady(false);
        setStatusText("Model unavailable");
      });
    return () => {
      active = false;
    };
  }, [setStatusText]);

  async function predictBlob(blob, name, localUrl) {
    if (!backendReady) {
      setError("The trained backend model is not available.");
      return;
    }

    setPreviewUrl(localUrl);
    setSelectedName(name);
    setPredictions([]);
    setSimilarExamples([]);
    setError("");
    setIsPredicting(true);
    setStatusText("Analyzing cover");

    const formData = new FormData();
    formData.append("file", blob, name || "cover.jpg");

    try {
      const response = await fetch("/api/predict", { method: "POST", body: formData });
      if (!response.ok) throw new Error(`Prediction failed: ${response.status}`);
      const payload = await response.json();
      setPredictions(payload.predictions || []);
      setSimilarExamples((payload.similarExamples || []).slice(0, 3));
      setModelInfo({
        model: payload.model,
        accuracyTop1: payload.accuracyTop1,
        accuracyTop3: payload.accuracyTop3,
        accuracyBroadTop1: payload.accuracyBroadTop1,
        hierarchicalScore: payload.hierarchicalScore,
      });
      setStatusText("Trained CLIP model");
    } catch (predictionError) {
      setError(predictionError.message);
      setStatusText("Prediction failed");
    } finally {
      setIsPredicting(false);
    }
  }

  function handleUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    predictBlob(file, file.name, URL.createObjectURL(file));
  }

  async function handleSample(sample) {
    setError("");
    setIsPredicting(true);
    try {
      const response = await fetch(artworkUrl(sample.imagePath));
      if (!response.ok) throw new Error(`Could not load sample: ${response.status}`);
      const blob = await response.blob();
      await predictBlob(blob, `${sample.key}.jpg`, artworkUrl(sample.imagePath));
    } catch (sampleError) {
      setError(sampleError.message);
      setStatusText("Sample failed");
      setIsPredicting(false);
    }
  }

  return (
    <>
      <section className="hero-section" aria-labelledby="heroTitle">
        <div className="hero-copy">
          <p className="eyebrow">Album intelligence</p>
          <h1 id="heroTitle">Genre inference from cover art.</h1>
          <p className="hero-subtitle">
            Upload artwork, inspect the prediction, and compare it against held-out covers from the same predicted category.
          </p>
          <div className="hero-actions">
            <a className="hero-primary" href="#predictor">Try a cover</a>
            <a className="hero-secondary" href="/admin">View model reports</a>
          </div>
        </div>

        <div className="hero-gallery" aria-hidden="true">
          <div className="hero-grid" />
          {floatingCovers.map(([label, className]) => (
            <article className={`floating-cover ${className}`} key={label}>
              <span>{label}</span>
            </article>
          ))}
          <div className="floating-insight">
            <span>Evidence view</span>
            <strong>Prediction + similar covers</strong>
            <small>Broad-family metrics stay visible for near misses.</small>
          </div>
        </div>
      </section>

      <div id="predictor" className="main-grid">
        <section className="input-panel" aria-label="Album cover input">
          <label className="drop-zone">
            <input type="file" accept="image/*" onChange={handleUpload} />
            <span className="drop-icon" aria-hidden="true">+</span>
            <span className="drop-title">Drop album cover</span>
            <span className="drop-meta">PNG, JPG, WebP</span>
          </label>

          {previewUrl && (
            <div className="preview-wrap">
              <img className="cover-preview" src={previewUrl} alt={selectedName || "Selected album cover"} />
            </div>
          )}

          <div className="sample-row" aria-label="Sample covers">
            {samples.map((sample) => (
              <button
                className="sample-card"
                type="button"
                key={sample.key}
                onClick={() => handleSample(sample)}
                disabled={isPredicting}
              >
                <span className={`sample-swatch swatch-${sample.key}`} />
                {sample.label}
              </button>
            ))}
          </div>
        </section>

        <section className="results-panel" aria-label="Prediction results">
          <div className="prediction-card">
            <p className="section-label">Prediction</p>
            <div className="primary-prediction">
              <div>
                <h2>{topPrediction ? topPrediction.genre : "Waiting for cover"}</h2>
                <p className="broad-genre">
                  {topPrediction ? `Broad genre: ${topPrediction.broadGenreDisplay}` : "Broad genre --"}
                </p>
              </div>
              <p className="confidence">
                {isPredicting ? "Analyzing" : topPrediction ? `${formatPercent(topPrediction.probability)} confidence` : "Upload or pick a sample"}
              </p>
            </div>

            {predictions.length > 0 && (
              <div className="probability-list" aria-live="polite">
                {predictions.slice(0, 4).map((item) => (
                  <div className="probability-row" key={item.label}>
                    <span className="probability-label">
                      <strong>{item.genre}</strong>
                      <small>{item.broadGenreDisplay}</small>
                    </span>
                    <div className="bar-track" aria-hidden="true">
                      <div className="bar-fill" style={{ "--score": formatPercent(item.probability) }} />
                    </div>
                    <strong>{formatPercent(item.probability)}</strong>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="similar-panel">
            <div className="section-heading-row">
              <p className="section-label">Evidence</p>
              <span>{topPrediction ? `${topPrediction.genre} / ${topPrediction.broadGenreDisplay}` : "Pick a cover to compare"}</span>
            </div>
            <ul className="evidence-list">
              {evidence.map((item) => <li key={item}>{item}</li>)}
            </ul>
            <div className="similar-grid" aria-live="polite">
              {similarExamples.length > 0 ? (
                similarExamples.map((example) => (
                  <article className="similar-card" key={example.imagePath}>
                    <img src={example.imageUrl} alt="" />
                    <div>
                      <strong>{example.actualDisplay}</strong>
                      <span>{example.predictedDisplay} match</span>
                    </div>
                  </article>
                ))
              ) : (
                <p className="empty-state">Held-out examples from the predicted category will appear here.</p>
              )}
            </div>
            {error && <p className="error-state">{error}</p>}
          </div>
        </section>
      </div>
    </>
  );
}

function AdminPage({ setStatusText }) {
  const [models, setModels] = useState([]);
  const [selectedModelId, setSelectedModelId] = useState(null);
  const [failedOnly, setFailedOnly] = useState(true);
  const [exampleOffset, setExampleOffset] = useState(0);
  const [examplesPayload, setExamplesPayload] = useState({ examples: [], total: 0, offset: 0, limit: pageSize });
  const [error, setError] = useState("");

  const selectedModel = models.find((model) => model.id === selectedModelId);

  useEffect(() => {
    setStatusText("Evaluation reports");
    fetch("/api/models")
      .then((response) => {
        if (!response.ok) throw new Error(`Model request failed: ${response.status}`);
        return response.json();
      })
      .then((payload) => {
        const modelList = payload.models || [];
        setModels(modelList);
        setSelectedModelId(modelList[0]?.id ?? null);
      })
      .catch((requestError) => setError(requestError.message));
  }, [setStatusText]);

  useEffect(() => {
    if (!selectedModelId) return;
    const params = new URLSearchParams({
      failedOnly: String(failedOnly),
      limit: String(pageSize),
      offset: String(exampleOffset),
    });
    fetch(`/api/models/${selectedModelId}/examples?${params}`)
      .then((response) => {
        if (!response.ok) throw new Error(`Example request failed: ${response.status}`);
        return response.json();
      })
      .then(setExamplesPayload)
      .catch((requestError) => setError(requestError.message));
  }, [selectedModelId, failedOnly, exampleOffset]);

  function selectModel(modelId) {
    setSelectedModelId(modelId);
    setExampleOffset(0);
  }

  function metricRows(model) {
    if (!model) return [];
    const metrics = model.metrics || {};
    const tuning = metrics.tuning || {};
    return [
      ["Family", model.family],
      ["Status", modelStatus(model)],
      ["Artifact", model.artifactAvailable ? "Available" : "Missing"],
      ["Top-1 Accuracy", formatPercentDetailed(metrics.accuracy_top_1)],
      ["Top-3 Accuracy", formatPercentDetailed(metrics.accuracy_top_3)],
      ["Broad Top-1", formatPercentDetailed(metrics.accuracy_broad_top_1)],
      ["Broad Top-3", formatPercentDetailed(metrics.accuracy_broad_top_3)],
      ["Near Misses", formatPercentDetailed(metrics.near_miss_rate)],
      ["Far Misses", formatPercentDetailed(metrics.far_miss_rate)],
      ["Hierarchical Score", formatPercentDetailed(metrics.hierarchical_score)],
      ["Train Size", formatNumber(metrics.train_size)],
      ["Test Size", formatNumber(metrics.test_size)],
      ["Failures", `${formatNumber(model.failureCount)} (${formatPercentDetailed(model.failureRate)})`],
      metrics.classifier && ["Classifier", metrics.classifier],
      metrics.clip_model && ["Embedding", metrics.clip_model],
      metrics.image_size && ["Image Size", `${metrics.image_size}px`],
      metrics.epochs && ["Epochs", metrics.epochs],
      metrics.batch_size && ["Batch Size", metrics.batch_size],
      metrics.learning_rate && ["Learning Rate", Number(metrics.learning_rate).toExponential(2)],
      metrics.weight_decay && ["Weight Decay", Number(metrics.weight_decay).toExponential(2)],
      metrics.dropout != null && ["Dropout", Math.round(metrics.dropout * 100) / 100],
      metrics.sibling_smoothing != null && ["Sibling Smoothing", Math.round(metrics.sibling_smoothing * 100) / 100],
      tuning.trials && ["Tuning Trials", tuning.trials],
      tuning.best_settings && ["Best CNN Width", tuning.best_settings.base_channels],
    ].filter(Boolean);
  }

  function predictionText(example) {
    const broadText =
      example.errorType === "near_miss"
        ? `Near miss: ${example.actualBroad}`
        : example.errorType === "far_miss"
          ? `${example.actualBroad} -> ${example.predictedBroad}`
          : "Exact";
    return `${(example.topPredictions || [])
      .slice(0, 3)
      .map((item) => `${item.display} ${formatPercent(item.probability)}`)
      .join(" / ")} · ${broadText}`;
  }

  const currentEnd = Math.min(examplesPayload.offset + examplesPayload.examples.length, examplesPayload.total);

  return (
    <>
      <section className="admin-heading">
        <p className="eyebrow">CoverSense Admin</p>
        <h1>Model observability</h1>
      </section>

      <section className="admin-model-grid" aria-label="Model list">
        {models.map((model) => {
          const metrics = model.metrics || {};
          return (
            <button
              className={`model-card${model.id === selectedModelId ? " is-selected" : ""}`}
              type="button"
              key={model.id}
              onClick={() => selectModel(model.id)}
            >
              <span className="model-card-topline">
                <span>{model.name}</span>
                <span>{modelStatus(model)}</span>
              </span>
              <strong>{formatPercentDetailed(metrics.accuracy_top_1)}</strong>
              <span className="model-card-meta">
                Broad {formatPercentDetailed(metrics.accuracy_broad_top_1)} · Top-3 {formatPercentDetailed(metrics.accuracy_top_3)}
              </span>
            </button>
          );
        })}
      </section>

      <section className="admin-detail-grid">
        <div className="admin-panel">
          <div className="admin-panel-header">
            <div>
              <p className="section-label">Selected Model</p>
              <h2>{selectedModel?.name || "Loading"}</h2>
            </div>
            <button
              className="toggle-button"
              type="button"
              aria-pressed={!failedOnly}
              onClick={() => {
                setFailedOnly((current) => !current);
                setExampleOffset(0);
              }}
            >
              {failedOnly ? "Failures" : "All"}
            </button>
          </div>
          <dl className="metric-list">
            {metricRows(selectedModel).map(([label, value]) => (
              <div key={label}>
                <dt>{label}</dt>
                <dd>{value ?? "--"}</dd>
              </div>
            ))}
          </dl>
        </div>

        <div className="admin-panel failure-panel">
          <div className="admin-panel-header">
            <div>
              <p className="section-label">Error Analysis</p>
              <h2>{failedOnly ? "Failed artwork" : "Evaluation artwork"}</h2>
            </div>
            <div className="pagination-actions">
              <button
                className="icon-page-button"
                type="button"
                aria-label="Previous page"
                disabled={exampleOffset === 0}
                onClick={() => setExampleOffset((offset) => Math.max(0, offset - pageSize))}
              >
                &lt;
              </button>
              <span className="count-pill">
                {examplesPayload.total === 0
                  ? "0"
                  : `${formatNumber(examplesPayload.offset + 1)}-${formatNumber(currentEnd)} of ${formatNumber(examplesPayload.total)}`}
              </span>
              <button
                className="icon-page-button"
                type="button"
                aria-label="Next page"
                disabled={exampleOffset + pageSize >= examplesPayload.total}
                onClick={() => setExampleOffset((offset) => offset + pageSize)}
              >
                &gt;
              </button>
            </div>
          </div>
          <div className="failure-grid" aria-live="polite">
            {examplesPayload.examples.length > 0 ? (
              examplesPayload.examples.map((example) => (
                <article className="failure-card" key={example.imagePath}>
                  <img src={example.imageUrl} alt="" />
                  <div>
                    <span className="failure-labels">
                      <strong>{example.predictedDisplay}</strong>
                      <span>actual {example.actualDisplay}</span>
                    </span>
                    <p>{predictionText(example)}</p>
                  </div>
                </article>
              ))
            ) : (
              <p className="empty-state">{error || "No examples available for this model yet."}</p>
            )}
          </div>
        </div>
      </section>
    </>
  );
}

export default function App() {
  const [theme, toggleTheme] = useTheme();
  const [statusText, setStatusText] = useState("Checking model");
  const isAdmin = window.location.pathname.startsWith("/admin");

  return (
    <main className="app-shell">
      <section className="workspace" aria-label={isAdmin ? "CoverSense model observability" : "Album cover genre predictor"}>
        <TopBar statusText={statusText} theme={theme} onToggleTheme={toggleTheme} />
        {isAdmin ? <AdminPage setStatusText={setStatusText} /> : <InferencePage statusText={statusText} setStatusText={setStatusText} />}
      </section>
    </main>
  );
}
