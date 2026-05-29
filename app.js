const genres = [
  "Pop",
  "Hip-Hop/Rap",
  "Rock",
  "Metal",
  "Electronic",
  "Jazz",
  "Classical",
  "R&B/Soul",
  "Country/Folk",
  "Indie/Alternative",
];

const fileInput = document.querySelector("#fileInput");
const dropZone = document.querySelector("#dropZone");
const canvas = document.querySelector("#previewCanvas");
const ctx = canvas.getContext("2d", { willReadFrequently: true });

const primaryGenre = document.querySelector("#primaryGenre");
const confidence = document.querySelector("#confidence");
const probabilityList = document.querySelector("#probabilityList");
const evidenceList = document.querySelector("#evidenceList");

const metricBrightness = document.querySelector("#metricBrightness");
const metricSaturation = document.querySelector("#metricSaturation");
const metricContrast = document.querySelector("#metricContrast");
const metricEdges = document.querySelector("#metricEdges");
const modelPill = document.querySelector(".model-pill");

function clamp(value, min = 0, max = 1) {
  return Math.max(min, Math.min(max, value));
}

function rgbToHsl(r, g, b) {
  r /= 255;
  g /= 255;
  b /= 255;

  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  let h = 0;
  let s = 0;
  const l = (max + min) / 2;

  if (max !== min) {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    switch (max) {
      case r:
        h = (g - b) / d + (g < b ? 6 : 0);
        break;
      case g:
        h = (b - r) / d + 2;
        break;
      default:
        h = (r - g) / d + 4;
        break;
    }
    h /= 6;
  }

  return { h: h * 360, s, l };
}

function luminance(r, g, b) {
  return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
}

function resetCanvas() {
  const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
  gradient.addColorStop(0, "#e9dcc7");
  gradient.addColorStop(0.5, "#d3e6df");
  gradient.addColorStop(1, "#f1b3a7");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.fillStyle = "rgba(21, 21, 20, 0.74)";
  ctx.fillRect(88, 440, 544, 96);
  ctx.fillStyle = "#fffdf8";
  ctx.font = "700 48px Inter, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Add cover art", canvas.width / 2, 501);
}

function drawImageToCanvas(img) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const sourceSize = Math.min(img.naturalWidth, img.naturalHeight);
  const sx = (img.naturalWidth - sourceSize) / 2;
  const sy = (img.naturalHeight - sourceSize) / 2;
  ctx.drawImage(img, sx, sy, sourceSize, sourceSize, 0, 0, canvas.width, canvas.height);
}

function analyzeCanvas() {
  const sampleSize = 96;
  const analysisCanvas = document.createElement("canvas");
  const analysisCtx = analysisCanvas.getContext("2d", { willReadFrequently: true });
  analysisCanvas.width = sampleSize;
  analysisCanvas.height = sampleSize;
  analysisCtx.drawImage(canvas, 0, 0, sampleSize, sampleSize);

  const { data } = analysisCtx.getImageData(0, 0, sampleSize, sampleSize);
  const lumas = [];
  const hueBins = new Array(12).fill(0);
  let totalR = 0;
  let totalG = 0;
  let totalB = 0;
  let totalSat = 0;
  let totalWarmth = 0;
  let darkPixels = 0;
  let neonPixels = 0;
  let redPixels = 0;
  let bluePixels = 0;
  let goldPixels = 0;
  let monochromePixels = 0;

  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const lum = luminance(r, g, b);
    const hsl = rgbToHsl(r, g, b);

    totalR += r;
    totalG += g;
    totalB += b;
    totalSat += hsl.s;
    totalWarmth += (r + 0.4 * g - b) / 255;
    lumas.push(lum);

    hueBins[Math.min(11, Math.floor(hsl.h / 30))] += hsl.s;
    if (lum < 0.26) darkPixels += 1;
    if (hsl.s > 0.62 && lum > 0.48) neonPixels += 1;
    if ((hsl.h < 24 || hsl.h > 338) && hsl.s > 0.35) redPixels += 1;
    if (hsl.h > 178 && hsl.h < 255 && hsl.s > 0.28) bluePixels += 1;
    if (hsl.h > 34 && hsl.h < 58 && hsl.s > 0.25 && lum > 0.32) goldPixels += 1;
    if (hsl.s < 0.16) monochromePixels += 1;
  }

  const pixels = lumas.length;
  const brightness = lumas.reduce((sum, value) => sum + value, 0) / pixels;
  const saturation = totalSat / pixels;
  const warmth = totalWarmth / pixels;
  const avgR = totalR / pixels;
  const avgG = totalG / pixels;
  const avgB = totalB / pixels;
  const contrast = Math.sqrt(
    lumas.reduce((sum, value) => sum + (value - brightness) ** 2, 0) / pixels,
  );

  let edgeTotal = 0;
  let edgeCount = 0;
  for (let y = 1; y < sampleSize - 1; y += 1) {
    for (let x = 1; x < sampleSize - 1; x += 1) {
      const center = lumas[y * sampleSize + x];
      const right = lumas[y * sampleSize + x + 1];
      const down = lumas[(y + 1) * sampleSize + x];
      edgeTotal += Math.abs(center - right) + Math.abs(center - down);
      edgeCount += 2;
    }
  }

  const edgeDensity = clamp((edgeTotal / edgeCount) * 4.4);
  const colorDiversity =
    hueBins.filter((value) => value > Math.max(6, pixels * 0.015)).length / hueBins.length;

  return {
    avgR,
    avgG,
    avgB,
    brightness,
    saturation,
    warmth,
    contrast: clamp(contrast * 2.5),
    edgeDensity,
    colorDiversity,
    darkRatio: darkPixels / pixels,
    neonRatio: neonPixels / pixels,
    redRatio: redPixels / pixels,
    blueRatio: bluePixels / pixels,
    goldRatio: goldPixels / pixels,
    monochromeRatio: monochromePixels / pixels,
  };
}

function scoreGenres(features) {
  const f = features;
  const scores = {
    Pop:
      0.7 +
      1.9 * f.saturation +
      1.15 * f.brightness +
      0.75 * f.neonRatio +
      0.35 * f.colorDiversity -
      0.52 * f.blueRatio -
      0.95 * f.darkRatio,
    "Hip-Hop/Rap":
      0.82 +
      0.92 * f.contrast +
      0.72 * f.darkRatio +
      0.48 * f.goldRatio +
      0.25 * f.redRatio +
      0.24 * f.edgeDensity -
      0.18 * f.brightness,
    Rock:
      0.78 +
      1.15 * f.contrast +
      0.72 * f.edgeDensity +
      0.38 * f.redRatio +
      0.25 * f.warmth +
      0.22 * f.darkRatio,
    Metal:
      0.45 +
      1.65 * f.darkRatio +
      1.28 * f.contrast +
      0.82 * f.edgeDensity +
      0.52 * f.redRatio -
      0.62 * f.brightness,
    Electronic:
      0.66 +
      1.7 * f.neonRatio +
      1.75 * f.blueRatio +
      0.82 * f.saturation +
      0.62 * f.colorDiversity +
      0.48 * f.contrast,
    Jazz:
      0.76 +
      1.18 * f.monochromeRatio +
      1.55 * f.goldRatio +
      0.62 * (1 - f.saturation) +
      0.48 * f.contrast +
      0.58 * (1 - f.edgeDensity) +
      0.22 * f.brightness -
      0.24 * f.neonRatio,
    Classical:
      0.52 +
      1.34 * (1 - f.saturation) +
      0.82 * f.brightness +
      0.38 * f.goldRatio -
      0.7 * f.edgeDensity -
      0.22 * f.neonRatio,
    "R&B/Soul":
      0.62 +
      0.82 * f.warmth +
      0.7 * f.darkRatio +
      0.58 * f.goldRatio +
      0.34 * f.saturation,
    "Country/Folk":
      0.58 +
      1.05 * f.warmth +
      0.72 * (1 - f.saturation) +
      0.4 * f.goldRatio +
      0.28 * f.brightness -
      0.3 * f.neonRatio,
    "Indie/Alternative":
      0.72 +
      0.72 * f.colorDiversity +
      0.54 * f.edgeDensity +
      0.42 * (1 - Math.abs(f.brightness - 0.5)) +
      0.32 * (1 - f.saturation),
  };

  const minScore = Math.min(...Object.values(scores));
  const shifted = Object.fromEntries(
    Object.entries(scores).map(([genre, score]) => [genre, Math.max(0.05, score - minScore + 0.18)]),
  );
  const total = Object.values(shifted).reduce((sum, value) => sum + value, 0);

  return Object.entries(shifted)
    .map(([genre, score]) => ({ genre, probability: score / total }))
    .sort((a, b) => b.probability - a.probability);
}

function featureEvidence(features, results, modelInfo = {}) {
  const f = features;
  const evidence = [];
  const top = results[0].genre;

  if (modelInfo.source === "trained") {
    const top3 = results
      .slice(0, 3)
      .map((item) => `${item.genre} ${formatPercent(item.probability)}`)
      .join(", ");
    evidence.push(`Trained CLIP classifier top matches: ${top3}.`);
    if (modelInfo.accuracyTop1 != null && modelInfo.accuracyTop3 != null) {
      evidence.push(
        `Full-dataset validation: ${formatPercent(modelInfo.accuracyTop1)} top-1, ${formatPercent(
          modelInfo.accuracyTop3,
        )} top-3.`,
      );
    }
    evidence.push("Visual metrics remain shown for inspection, but the prediction comes from the trained backend.");
    return evidence;
  }

  if (f.darkRatio > 0.38) evidence.push("Large dark areas suggest heavier or moodier genres.");
  if (f.neonRatio > 0.16) evidence.push("Bright saturated colors point toward pop or electronic packaging.");
  if (f.edgeDensity > 0.32) evidence.push("Dense edges imply busy typography, collage, texture, or aggressive artwork.");
  if (f.monochromeRatio > 0.48) evidence.push("Muted or monochrome color can match jazz, classical, indie, or archival covers.");
  if (f.goldRatio > 0.1) evidence.push("Warm gold and sepia tones often pull the score toward jazz, soul, folk, or rap.");
  if (f.blueRatio > 0.18 && f.saturation > 0.34) evidence.push("Cool saturated color is a strong electronic and alternative signal.");
  if (f.contrast > 0.42) evidence.push("High contrast gives extra weight to rock, metal, hip-hop, and jazz.");

  evidence.push(`Top result: ${top}, based on the combined visual signature.`);
  return evidence.slice(0, 5);
}

function formatPercent(value) {
  return `${Math.round(value * 100)}%`;
}

function renderModelPill(text) {
  modelPill.innerHTML = `<span class="status-dot"></span>${text}`;
}

function renderResults(features, results, modelInfo = {}) {
  const top = results[0];
  primaryGenre.textContent = top.genre;
  confidence.textContent = `${formatPercent(top.probability)} confidence`;
  renderModelPill(modelInfo.source === "trained" ? "Trained CLIP model" : "Visual heuristic model");

  probabilityList.innerHTML = results
    .slice(0, 6)
    .map(
      (item) => `
        <div class="probability-row">
          <span>${item.genre}</span>
          <div class="bar-track" aria-hidden="true">
            <div class="bar-fill" style="--score: ${formatPercent(item.probability)}"></div>
          </div>
          <strong>${formatPercent(item.probability)}</strong>
        </div>
      `,
    )
    .join("");

  metricBrightness.textContent = formatPercent(features.brightness);
  metricSaturation.textContent = formatPercent(features.saturation);
  metricContrast.textContent = formatPercent(features.contrast);
  metricEdges.textContent = formatPercent(features.edgeDensity);

  evidenceList.innerHTML = featureEvidence(features, results, modelInfo)
    .map((item) => `<li>${item}</li>`)
    .join("");
}

function canvasToBlob() {
  return new Promise((resolve) => {
    canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.92);
  });
}

async function predictWithBackend() {
  const blob = await canvasToBlob();
  if (!blob) throw new Error("Could not prepare image for backend prediction.");

  const formData = new FormData();
  formData.append("file", blob, "cover.jpg");

  const response = await fetch("/api/predict", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Backend prediction failed: ${response.status}`);
  }

  const payload = await response.json();
  return {
    results: payload.predictions,
    modelInfo: {
      source: "trained",
      accuracyTop1: payload.accuracyTop1,
      accuracyTop3: payload.accuracyTop3,
    },
  };
}

async function runPrediction() {
  const features = analyzeCanvas();
  const results = scoreGenres(features);
  renderResults(features, results, { source: "heuristic" });

  try {
    const backendPrediction = await predictWithBackend();
    renderResults(features, backendPrediction.results, backendPrediction.modelInfo);
  } catch (error) {
    renderModelPill("Visual heuristic model");
  }
}

function loadFile(file) {
  if (!file || !file.type.startsWith("image/")) return;

  const reader = new FileReader();
  reader.onload = () => {
    const img = new Image();
    img.onload = () => {
      drawImageToCanvas(img);
      runPrediction();
    };
    img.src = reader.result;
  };
  reader.readAsDataURL(file);
}

function drawSampleCover(type) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (type === "electronic") {
    const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
    gradient.addColorStop(0, "#06152c");
    gradient.addColorStop(0.45, "#00d5cf");
    gradient.addColorStop(1, "#fa4faf");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.78)";
    ctx.lineWidth = 8;
    for (let i = 0; i < 11; i += 1) {
      ctx.beginPath();
      ctx.moveTo(80 + i * 58, 94);
      ctx.lineTo(620 - i * 26, 628);
      ctx.stroke();
    }
  }

  if (type === "metal") {
    ctx.fillStyle = "#050505";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const gradient = ctx.createRadialGradient(360, 360, 40, 360, 360, 470);
    gradient.addColorStop(0, "#8e211e");
    gradient.addColorStop(1, "#050505");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "#d8d1c5";
    ctx.lineWidth = 7;
    for (let i = 0; i < 19; i += 1) {
      ctx.beginPath();
      ctx.moveTo(80 + i * 31, 160 + (i % 3) * 28);
      ctx.lineTo(164 + i * 24, 588 - (i % 4) * 42);
      ctx.stroke();
    }
  }

  if (type === "jazz") {
    ctx.fillStyle = "#eee2c6";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#172031";
    ctx.fillRect(0, 432, canvas.width, 168);
    ctx.fillStyle = "#c8a33f";
    ctx.beginPath();
    ctx.arc(258, 320, 136, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#172031";
    ctx.fillRect(400, 120, 48, 428);
    ctx.fillRect(470, 182, 42, 366);
  }

  if (type === "pop") {
    const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
    gradient.addColorStop(0, "#ff5d8f");
    gradient.addColorStop(0.52, "#f9de55");
    gradient.addColorStop(1, "#58bfd3");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "rgba(255, 255, 255, 0.66)";
    for (let i = 0; i < 12; i += 1) {
      ctx.beginPath();
      ctx.arc(110 + (i % 4) * 165, 120 + Math.floor(i / 4) * 165, 52, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.fillStyle = "#171514";
    ctx.fillRect(128, 286, 464, 112);
  }

  ctx.fillStyle = "rgba(255, 253, 248, 0.86)";
  ctx.font = "800 54px Inter, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(type.toUpperCase(), canvas.width / 2, 666);
  runPrediction();
}

fileInput.addEventListener("change", (event) => {
  loadFile(event.target.files[0]);
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("is-dragging");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("is-dragging");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("is-dragging");
  loadFile(event.dataTransfer.files[0]);
});

document.querySelectorAll("[data-sample]").forEach((button) => {
  button.addEventListener("click", () => drawSampleCover(button.dataset.sample));
});

resetCanvas();
