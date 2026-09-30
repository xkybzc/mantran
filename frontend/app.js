"use strict";

// Where the API lives. When the page is served by the backend itself this is
// the same origin; opened any other way it falls back to the local server.
const DEFAULT_API = "http://127.0.0.1:8000";
const MAX_FILE_BYTES = 20 * 1024 * 1024;
const ACCEPTED_TYPES = ["image/png", "image/jpeg", "image/webp", "image/bmp"];

const el = {
  status: document.getElementById("api-status"),
  statusText: document.querySelector("#api-status .status-text"),
  error: document.getElementById("error"),
  errorText: document.getElementById("error-text"),
  errorClose: document.getElementById("error-close"),
  dropzone: document.getElementById("dropzone"),
  dropzoneEmpty: document.getElementById("dropzone-empty"),
  fileInput: document.getElementById("file-input"),
  fileMeta: document.getElementById("file-meta"),
  originalImage: document.getElementById("original-image"),
  sourceLang: document.getElementById("source-lang"),
  targetLang: document.getElementById("target-lang"),
  clearBtn: document.getElementById("clear-btn"),
  translateBtn: document.getElementById("translate-btn"),
  resultMeta: document.getElementById("result-meta"),
  resultEmpty: document.getElementById("result-empty"),
  loading: document.getElementById("loading"),
  elapsed: document.getElementById("elapsed"),
  resultImage: document.getElementById("result-image"),
  downloadBtn: document.getElementById("download-btn"),
};

const state = {
  apiBase: location.protocol.startsWith("http") ? "" : DEFAULT_API,
  file: null,
  previewUrl: null,
  busy: false,
  timer: null,
};

// ---- Server status --------------------------------------------------------

function setStatus(kind, text) {
  el.status.dataset.state = kind;
  el.statusText.textContent = text;
}

async function checkServer() {
  const bases = [...new Set([state.apiBase, DEFAULT_API])];
  for (const base of bases) {
    try {
      const response = await fetch(`${base}/health`, { cache: "no-store" });
      if (response.ok) {
        state.apiBase = base;
        setStatus("online", "Server online");
        return true;
      }
    } catch {
      // Try the next candidate.
    }
  }
  setStatus("offline", "Server offline");
  return false;
}

// ---- Errors ---------------------------------------------------------------

function showError(message) {
  el.errorText.textContent = message;
  el.error.hidden = false;
}

function hideError() {
  el.error.hidden = true;
}

// ---- Choosing a file ------------------------------------------------------

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function setFile(file) {
  if (!file) return;
  if (!ACCEPTED_TYPES.includes(file.type)) {
    showError("That file is not a supported image. Use PNG, JPG, WEBP or BMP.");
    return;
  }
  if (file.size > MAX_FILE_BYTES) {
    showError(`That image is ${formatBytes(file.size)}. The limit is 20 MB.`);
    return;
  }

  hideError();
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.file = file;
  state.previewUrl = URL.createObjectURL(file);

  el.originalImage.src = state.previewUrl;
  el.originalImage.hidden = false;
  el.dropzoneEmpty.hidden = true;
  el.dropzone.classList.add("has-image");
  el.fileMeta.textContent = `${file.name || "Pasted image"} · ${formatBytes(file.size)}`;
  el.originalImage.onload = () => {
    const { naturalWidth: w, naturalHeight: h } = el.originalImage;
    el.fileMeta.textContent = `${file.name || "Pasted image"} · ${w} × ${h} · ${formatBytes(file.size)}`;
  };

  resetResult();
  updateButtons();
}

function clearFile() {
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.file = null;
  state.previewUrl = null;
  el.fileInput.value = "";
  el.originalImage.removeAttribute("src");
  el.originalImage.hidden = true;
  el.dropzoneEmpty.hidden = false;
  el.dropzone.classList.remove("has-image");
  el.fileMeta.textContent = "";
  hideError();
  resetResult();
  updateButtons();
}

function openFilePicker() {
  if (!state.busy) el.fileInput.click();
}

// ---- Result panel ---------------------------------------------------------

function resetResult() {
  el.resultImage.removeAttribute("src");
  el.resultImage.hidden = true;
  el.resultEmpty.hidden = false;
  el.loading.hidden = true;
  el.resultMeta.textContent = "";
  el.downloadBtn.removeAttribute("href");
  el.downloadBtn.setAttribute("aria-disabled", "true");
}

function setBusy(busy) {
  state.busy = busy;
  el.loading.hidden = !busy;
  if (busy) {
    el.resultEmpty.hidden = true;
    el.resultImage.hidden = true;
    const started = performance.now();
    el.elapsed.textContent = "0 s";
    state.timer = setInterval(() => {
      el.elapsed.textContent = `${Math.round((performance.now() - started) / 1000)} s`;
    }, 500);
  } else {
    clearInterval(state.timer);
  }
  el.translateBtn.textContent = busy ? "Translating" : "Translate";
  updateButtons();
}

function updateButtons() {
  el.translateBtn.disabled = !state.file || state.busy;
  el.clearBtn.disabled = !state.file || state.busy;
  el.targetLang.disabled = state.busy;
}

function outputName(file) {
  const base = (file.name || "page").replace(/\.[^.]+$/, "");
  return `${base}_${el.targetLang.value}.png`;
}

function showResult(data, seconds, file) {
  const url = `data:${data.mime_type || "image/png"};base64,${data.output_image}`;
  el.resultImage.src = url;
  el.resultImage.hidden = false;
  el.resultEmpty.hidden = true;

  el.downloadBtn.href = url;
  el.downloadBtn.download = outputName(file);
  el.downloadBtn.setAttribute("aria-disabled", "false");

  const regions = data.regions || [];
  const translated = regions.filter((region) => region.translation).length;
  const noun = regions.length === 1 ? "text region" : "text regions";
  el.resultMeta.textContent = `${translated} of ${regions.length} ${noun} translated in ${seconds.toFixed(1)} s`;
}

// ---- Translating ----------------------------------------------------------

async function readError(response) {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
  } catch {
    // Fall through to the status text.
  }
  return `${response.status} ${response.statusText}`.trim();
}

async function translate() {
  if (!state.file || state.busy) return;
  hideError();

  if (el.status.dataset.state !== "online" && !(await checkServer())) {
    showError("Cannot reach the server. Start it with: uvicorn app.main:app (run from the backend folder).");
    return;
  }

  const file = state.file;
  const form = new FormData();
  form.append("file", file, file.name || "page.png");
  form.append("source_lang", el.sourceLang.value);
  form.append("target_lang", el.targetLang.value);

  setBusy(true);
  const started = performance.now();
  try {
    const response = await fetch(`${state.apiBase}/api/translate`, { method: "POST", body: form });
    if (!response.ok) throw new Error(await readError(response));
    const data = await response.json();
    if (state.file !== file) return; // The user picked another image meanwhile.
    showResult(data, (performance.now() - started) / 1000, file);
  } catch (error) {
    if (state.file !== file) return;
    resetResult();
    const offline = error instanceof TypeError;
    if (offline) setStatus("offline", "Server offline");
    showError(offline ? "Lost connection to the server while translating." : `Translation failed: ${error.message}`);
  } finally {
    setBusy(false);
  }
}

// ---- Wiring ---------------------------------------------------------------

el.dropzone.addEventListener("click", openFilePicker);
el.dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    openFilePicker();
  }
});
el.fileInput.addEventListener("change", () => setFile(el.fileInput.files[0]));

["dragenter", "dragover"].forEach((type) =>
  el.dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    if (!state.busy) el.dropzone.classList.add("is-dragging");
  })
);
["dragleave", "drop"].forEach((type) =>
  el.dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    el.dropzone.classList.remove("is-dragging");
  })
);
el.dropzone.addEventListener("drop", (event) => {
  if (!state.busy) setFile(event.dataTransfer.files[0]);
});
// Dropping outside the drop zone should not navigate away to the image.
window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("drop", (event) => event.preventDefault());

document.addEventListener("paste", (event) => {
  if (state.busy) return;
  const item = [...event.clipboardData.items].find((entry) => entry.type.startsWith("image/"));
  if (item) setFile(item.getAsFile());
});

el.translateBtn.addEventListener("click", translate);
el.clearBtn.addEventListener("click", clearFile);
el.errorClose.addEventListener("click", hideError);

checkServer();
