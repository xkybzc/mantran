"use strict";

// Where the API lives. When the page is served by the backend itself this is
// the same origin; opened any other way it falls back to the local server.
const DEFAULT_API = "http://127.0.0.1:8000";
const MAX_FILE_BYTES = 20 * 1024 * 1024;
const ACCEPTED_TYPES = ["image/png", "image/jpeg", "image/webp", "image/bmp"];
const POLL_INTERVAL_MS = 1000;
const JOB_STORAGE_KEY = "mantran.chapterJob";
// Which chapter language to preselect when a title has several.
const SOURCE_PREFERENCE = ["en", "es-la", "es", "id", "pt-br", "pt", "fr", "it", "de", "ru", "tr", "pl", "ja", "ko", "zh", "zh-hk", "vi"];

const $ = (id) => document.getElementById(id);

const el = {
  status: $("api-status"),
  statusText: document.querySelector("#api-status .status-text"),
  error: $("error"),
  errorText: $("error-text"),
  errorClose: $("error-close"),
  tabs: [$("tab-page"), $("tab-chapter")],
  // Single page
  dropzone: $("dropzone"),
  dropzoneEmpty: $("dropzone-empty"),
  fileInput: $("file-input"),
  fileMeta: $("file-meta"),
  originalImage: $("original-image"),
  sourceLang: $("source-lang"),
  targetLang: $("target-lang"),
  clearBtn: $("clear-btn"),
  translateBtn: $("translate-btn"),
  resultMeta: $("result-meta"),
  resultEmpty: $("result-empty"),
  loading: $("loading"),
  elapsed: $("elapsed"),
  resultImage: $("result-image"),
  downloadBtn: $("download-btn"),
  // MangaDex chapter
  linkForm: $("link-form"),
  mangaLink: $("manga-link"),
  loadBtn: $("load-btn"),
  mangaCard: $("manga-card"),
  mangaTitle: $("manga-title"),
  mangaSub: $("manga-sub"),
  pickerRow: $("picker-row"),
  chapterSource: $("chapter-source"),
  chapterTarget: $("chapter-target"),
  chapterFilter: $("chapter-filter"),
  chapterList: $("chapter-list"),
  chapterEmpty: $("chapter-empty"),
  chapterCount: $("chapter-count"),
  selectionNote: $("selection-note"),
  chapterTranslateBtn: $("chapter-translate-btn"),
  jobMeta: $("job-meta"),
  jobProgress: $("job-progress"),
  jobProgressBar: $("job-progress-bar"),
  reader: $("reader"),
  readerEmpty: $("reader-empty"),
  readerPages: $("reader-pages"),
  jobStatus: $("job-status"),
  jobCancelBtn: $("job-cancel-btn"),
  zipBtn: $("zip-btn"),
};

const state = {
  apiBase: location.protocol.startsWith("http") ? "" : DEFAULT_API,
  // Single page
  file: null,
  previewUrl: null,
  busy: false,
  timer: null,
  // MangaDex chapter
  manga: null,
  chapters: [],
  chaptersCache: new Map(),
  selectedChapter: null,
  job: null,
  pollTimer: null,
};

// ---- Helpers --------------------------------------------------------------

function api(path) {
  return `${state.apiBase}${path}`;
}

async function readError(response) {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
  } catch {
    // Fall through to the status text.
  }
  return `${response.status} ${response.statusText}`.trim();
}

async function getJson(path, options) {
  const response = await fetch(api(path), options);
  if (!response.ok) {
    const error = new Error(await readError(response));
    error.status = response.status;
    throw error;
  }
  return response.json();
}

function storageGet(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function storageSet(key, value) {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch {
    // Storage may be unavailable (private mode); nothing else depends on it.
  }
}

function formatDuration(seconds) {
  seconds = Math.round(seconds);
  if (seconds < 60) return `${seconds} s`;
  return `${Math.floor(seconds / 60)} min ${seconds % 60} s`;
}

// Disable the target language that equals the source (e.g. English to English).
function syncTargetOptions(sourceSelect, targetSelect) {
  const source = sourceSelect.value;
  for (const option of targetSelect.options) option.disabled = option.value === source;
  if (targetSelect.selectedOptions[0]?.disabled) {
    const first = [...targetSelect.options].find((option) => !option.disabled);
    if (first) targetSelect.value = first.value;
  }
}

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

async function ensureServer() {
  if (el.status.dataset.state === "online" || (await checkServer())) return true;
  showError("Cannot reach the server. Start it with: uvicorn app.main:app (run from the backend folder).");
  return false;
}

function reportFailure(error, action) {
  if (error instanceof TypeError) {
    setStatus("offline", "Server offline");
    showError(`Lost connection to the server while ${action}.`);
  } else {
    showError(error.message);
  }
}

async function loadLanguages() {
  try {
    const { sources } = await getJson("/api/languages");
    const current = el.sourceLang.value;
    el.sourceLang.replaceChildren(
      ...sources.map(({ code, name }) => new Option(name, code, false, code === current))
    );
  } catch {
    // Keep the built-in Japanese option.
  }
  syncTargetOptions(el.sourceLang, el.targetLang);
}

// ---- Errors ---------------------------------------------------------------

function showError(message) {
  el.errorText.textContent = message;
  el.error.hidden = false;
}

function hideError() {
  el.error.hidden = true;
}

// ---- Tabs -----------------------------------------------------------------

function showView(name, { focus = false } = {}) {
  for (const tab of el.tabs) {
    const selected = tab.id === `tab-${name}`;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
    $(tab.getAttribute("aria-controls")).hidden = !selected;
    if (selected && focus) tab.focus();
  }
  hideError();
  if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
}

function viewFromHash() {
  return location.hash === "#chapter" ? "chapter" : "page";
}

// ==== Single page ==========================================================

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
  el.sourceLang.disabled = state.busy;
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

async function translatePage() {
  if (!state.file || state.busy) return;
  hideError();
  if (!(await ensureServer())) return;

  const file = state.file;
  const form = new FormData();
  form.append("file", file, file.name || "page.png");
  form.append("source_lang", el.sourceLang.value);
  form.append("target_lang", el.targetLang.value);

  setBusy(true);
  const started = performance.now();
  try {
    const data = await getJson("/api/translate", { method: "POST", body: form });
    if (state.file !== file) return; // The user picked another image meanwhile.
    showResult(data, (performance.now() - started) / 1000, file);
  } catch (error) {
    if (state.file !== file) return;
    resetResult();
    if (error instanceof TypeError) reportFailure(error, "translating");
    else showError(`Translation failed: ${error.message}`);
  } finally {
    setBusy(false);
  }
}

// ==== MangaDex chapter =====================================================

function setChapterListMessage(message) {
  el.chapterEmpty.textContent = message;
  el.chapterList.replaceChildren(el.chapterEmpty);
}

function updateChapterButton() {
  const running = state.job && ["queued", "running"].includes(state.job.status);
  el.chapterTranslateBtn.disabled = !state.selectedChapter || !el.chapterSource.value || running;
  if (!state.selectedChapter) {
    el.selectionNote.textContent = state.chapters.length ? "Pick a chapter from the list." : "";
  } else {
    const chapter = state.selectedChapter;
    el.selectionNote.textContent = `${chapter.label} · ${chapter.pages} pages`;
  }
}

async function loadLink(event) {
  event.preventDefault();
  hideError();
  const link = el.mangaLink.value.trim();
  if (!link || !(await ensureServer())) return;

  el.loadBtn.disabled = true;
  el.loadBtn.textContent = "Loading";
  try {
    const { manga, chapter } = await getJson(`/api/mangadex/lookup?url=${encodeURIComponent(link)}`);
    state.manga = manga;
    state.chaptersCache.clear();
    showManga(manga);

    const supported = manga.languages.filter((lang) => lang.supported).map((lang) => lang.code);
    const preferred = chapter && supported.includes(chapter.language)
      ? chapter.language
      : SOURCE_PREFERENCE.find((code) => supported.includes(code) && code !== el.chapterTarget.value)
        || SOURCE_PREFERENCE.find((code) => supported.includes(code));
    if (!preferred) {
      el.chapterSource.value = "";
      state.chapters = [];
      state.selectedChapter = null;
      setChapterListMessage("None of this title's languages can be read yet.");
      updateChapterButton();
      return;
    }
    el.chapterSource.value = preferred;
    syncTargetOptions(el.chapterSource, el.chapterTarget);
    await loadChapters(chapter ? chapter.id : null);
  } catch (error) {
    reportFailure(error, "loading the title");
  } finally {
    el.loadBtn.disabled = false;
    el.loadBtn.textContent = "Load";
  }
}

function showManga(manga) {
  el.mangaCard.hidden = false;
  el.pickerRow.hidden = false;
  el.mangaTitle.textContent = manga.title;
  const original = manga.original_language ? `Original: ${languageLabel(manga.original_language)}` : "";
  const count = `${manga.languages.length} ${manga.languages.length === 1 ? "language" : "languages"} on MangaDex`;
  el.mangaSub.textContent = [original, count].filter(Boolean).join(" · ");

  const languages = [...manga.languages].sort(
    (a, b) => Number(b.supported) - Number(a.supported) || a.name.localeCompare(b.name)
  );
  el.chapterSource.replaceChildren(
    ...languages.map((lang) => {
      const option = new Option(lang.supported ? lang.name : `${lang.name} (not supported)`, lang.code);
      option.disabled = !lang.supported;
      return option;
    })
  );
}

function languageLabel(code) {
  const known = state.manga?.languages.find((lang) => lang.code === code);
  if (known) return known.name.replace(/ \(.*\)$/, "");
  const option = [...el.sourceLang.options].find((opt) => opt.value === code);
  return option ? option.textContent : code;
}

async function loadChapters(selectId = null) {
  const lang = el.chapterSource.value;
  if (!state.manga || !lang) return;
  state.selectedChapter = null;
  el.chapterFilter.value = "";
  updateChapterButton();

  let chapters = state.chaptersCache.get(lang);
  if (!chapters) {
    setChapterListMessage("Loading chapters...");
    el.chapterCount.textContent = "";
    try {
      ({ chapters } = await getJson(`/api/mangadex/manga/${state.manga.id}/chapters?lang=${encodeURIComponent(lang)}`));
    } catch (error) {
      setChapterListMessage("Could not load chapters.");
      reportFailure(error, "loading chapters");
      return;
    }
    state.chaptersCache.set(lang, chapters);
  }
  if (el.chapterSource.value !== lang) return; // The language changed meanwhile.

  state.chapters = chapters;
  el.chapterCount.textContent = `${chapters.length} ${chapters.length === 1 ? "chapter" : "chapters"}`;
  renderChapterList();
  if (selectId) selectChapter(selectId, { scroll: true });
  updateChapterButton();
}

function renderChapterList() {
  if (!state.chapters.length) {
    setChapterListMessage("No readable chapters in this language.");
    return;
  }
  const rows = state.chapters.map((chapter) => {
    const row = document.createElement("label");
    row.className = "chapter-row";
    row.dataset.id = chapter.id;
    row.dataset.search = `${chapter.chapter || ""} ${chapter.title} ${chapter.volume || ""}`.toLowerCase();

    const input = document.createElement("input");
    input.type = "radio";
    input.name = "chapter";
    input.value = chapter.id;
    input.checked = state.selectedChapter?.id === chapter.id;

    const number = document.createElement("span");
    number.className = "chapter-number";
    number.textContent = chapter.chapter ? `Ch. ${chapter.chapter}` : "Oneshot";

    const title = document.createElement("span");
    title.className = "chapter-title";
    title.textContent = chapter.title || (chapter.volume ? `Volume ${chapter.volume}` : "");

    const extra = document.createElement("span");
    extra.className = "chapter-extra";
    extra.textContent = [chapter.group, `${chapter.pages} p`].filter(Boolean).join(" · ");

    row.append(input, number, title, extra);
    return row;
  });
  el.chapterList.replaceChildren(...rows);
}

function filterChapters() {
  const query = el.chapterFilter.value.trim().toLowerCase();
  let visible = 0;
  for (const row of el.chapterList.querySelectorAll(".chapter-row")) {
    const match = !query || row.dataset.search.includes(query);
    row.hidden = !match;
    visible += match;
  }
  const empty = el.chapterList.querySelector(".list-empty");
  if (!visible && state.chapters.length && !empty) {
    const note = document.createElement("p");
    note.className = "list-empty";
    note.textContent = "No chapter matches.";
    el.chapterList.append(note);
  } else if (visible && empty) {
    empty.remove();
  }
}

function selectChapter(id, { scroll = false } = {}) {
  state.selectedChapter = state.chapters.find((chapter) => chapter.id === id) || null;
  const input = el.chapterList.querySelector(`input[value="${CSS.escape(id)}"]`);
  if (input) {
    input.checked = true;
    if (scroll) input.closest(".chapter-row").scrollIntoView({ block: "center" });
  }
  updateChapterButton();
}

// ---- Chapter jobs ---------------------------------------------------------

async function startChapterJob() {
  if (!state.selectedChapter) return;
  hideError();
  if (!(await ensureServer())) return;
  el.chapterTranslateBtn.disabled = true;
  try {
    const job = await getJson("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chapter_id: state.selectedChapter.id,
        source_lang: el.chapterSource.value,
        target_lang: el.chapterTarget.value,
      }),
    });
    followJob(job);
  } catch (error) {
    reportFailure(error, "starting the translation");
    updateChapterButton();
  }
}

function followJob(job) {
  clearTimeout(state.pollTimer);
  if (!state.job || state.job.id !== job.id) resetReader();
  state.job = job;
  storageSet(JOB_STORAGE_KEY, job.id);
  renderJob(job);
  if (["queued", "running"].includes(job.status)) state.pollTimer = setTimeout(pollJob, POLL_INTERVAL_MS);
}

async function pollJob() {
  if (!state.job) return;
  try {
    followJob(await getJson(`/api/jobs/${state.job.id}`));
  } catch (error) {
    if (error.status === 404) {
      storageSet(JOB_STORAGE_KEY, null);
      state.job = null;
      resetReader();
      showError("The server no longer knows this chapter job (it probably restarted). Start it again; finished pages are reused.");
      updateChapterButton();
      return;
    }
    // Keep trying: the server may be busy or briefly unreachable.
    if (error instanceof TypeError) setStatus("offline", "Server offline");
    state.pollTimer = setTimeout(pollJob, POLL_INTERVAL_MS * 3);
  }
}

async function cancelJob() {
  if (!state.job) return;
  try {
    followJob(await getJson(`/api/jobs/${state.job.id}`, { method: "DELETE" }));
  } catch (error) {
    reportFailure(error, "cancelling");
  }
}

function resetReader() {
  el.readerPages.replaceChildren();
  el.readerEmpty.hidden = false;
  el.jobProgress.hidden = true;
  el.jobProgressBar.style.width = "0";
  el.jobMeta.textContent = "";
  el.jobStatus.textContent = "";
  el.jobCancelBtn.hidden = true;
  el.zipBtn.removeAttribute("href");
  el.zipBtn.setAttribute("aria-disabled", "true");
}

function ensurePageSlots(job) {
  const pages = el.readerPages;
  for (let index = pages.children.length; index < job.total; index += 1) {
    const figure = document.createElement("figure");
    figure.className = "reader-page";
    figure.dataset.index = String(index);
    const placeholder = document.createElement("div");
    placeholder.className = "page-placeholder";
    placeholder.textContent = `Page ${index + 1}`;
    figure.append(placeholder);
    pages.append(figure);
  }
}

function renderJob(job) {
  const chapter = job.chapter || {};
  el.jobMeta.textContent = [chapter.manga_title, chapter.label].filter(Boolean).join(" · ");
  el.readerEmpty.hidden = job.total > 0;
  ensurePageSlots(job);

  for (const index of job.completed) {
    const figure = el.readerPages.children[index];
    if (!figure || figure.querySelector("img")) continue;
    const image = document.createElement("img");
    image.loading = "lazy";
    image.alt = `Page ${index + 1}`;
    image.src = api(`/api/jobs/${job.id}/pages/${index}`);
    figure.replaceChildren(image);
  }
  for (const index of job.failed_pages) {
    const placeholder = el.readerPages.children[index]?.querySelector(".page-placeholder");
    if (placeholder) {
      placeholder.classList.add("is-failed");
      placeholder.textContent = `Page ${index + 1} could not be translated`;
    }
  }

  const finished = job.completed.length + job.failed_pages.length;
  const active = ["queued", "running"].includes(job.status);
  el.jobProgress.hidden = !job.total || job.status === "done";
  el.jobProgressBar.style.width = job.total ? `${(finished / job.total) * 100}%` : "0";
  el.jobCancelBtn.hidden = !active;

  const notes = [];
  if (job.status === "queued") notes.push("Waiting for the previous chapter to finish");
  else if (job.status === "running") {
    notes.push(job.total ? `Translating page ${Math.min(finished + 1, job.total)} of ${job.total}` : "Fetching the chapter");
    notes.push(formatDuration(job.elapsed));
  } else if (job.status === "done") notes.push(`Done: ${job.completed.length} pages in ${formatDuration(job.elapsed)}`);
  else if (job.status === "cancelled") notes.push("Cancelled");
  else if (job.status === "error") notes.push("Failed");
  if (job.failed_pages.length) notes.push(`${job.failed_pages.length} failed`);
  if (job.untranslated) notes.push(`${job.untranslated} bubbles left untranslated`);
  el.jobStatus.textContent = notes.join(" · ");
  if (job.status === "error" && job.error) showError(`Chapter translation failed: ${job.error}`);

  const ready = job.status === "done" && job.completed.length > 0;
  if (ready) el.zipBtn.href = api(`/api/jobs/${job.id}/download`);
  else el.zipBtn.removeAttribute("href");
  el.zipBtn.setAttribute("aria-disabled", String(!ready));
  updateChapterButton();
}

async function resumeSavedJob() {
  const jobId = storageGet(JOB_STORAGE_KEY);
  if (!jobId) return;
  try {
    followJob(await getJson(`/api/jobs/${jobId}`));
  } catch {
    storageSet(JOB_STORAGE_KEY, null);
  }
}

// ---- Wiring ---------------------------------------------------------------

for (const tab of el.tabs) {
  tab.addEventListener("click", () => showView(tab.id.replace("tab-", "")));
  tab.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
    const next = el.tabs[(el.tabs.indexOf(tab) + 1) % el.tabs.length];
    showView(next.id.replace("tab-", ""), { focus: true });
  });
}
window.addEventListener("hashchange", () => showView(viewFromHash()));

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
  if (state.busy || $("view-page").hidden) return;
  const item = [...event.clipboardData.items].find((entry) => entry.type.startsWith("image/"));
  if (item) setFile(item.getAsFile());
});

el.sourceLang.addEventListener("change", () => syncTargetOptions(el.sourceLang, el.targetLang));
el.translateBtn.addEventListener("click", translatePage);
el.clearBtn.addEventListener("click", clearFile);
el.errorClose.addEventListener("click", hideError);

el.linkForm.addEventListener("submit", loadLink);
el.chapterSource.addEventListener("change", () => {
  syncTargetOptions(el.chapterSource, el.chapterTarget);
  loadChapters();
});
el.chapterTarget.addEventListener("change", updateChapterButton);
el.chapterFilter.addEventListener("input", filterChapters);
el.chapterList.addEventListener("change", (event) => {
  if (event.target.name === "chapter") selectChapter(event.target.value);
});
el.chapterTranslateBtn.addEventListener("click", startChapterJob);
el.jobCancelBtn.addEventListener("click", cancelJob);

showView(viewFromHash());
checkServer().then((online) => {
  if (!online) return;
  loadLanguages();
  resumeSavedJob();
});
