"use strict";

// Where the API lives. When the page is served by the backend itself this is
// the same origin; opened any other way it falls back to the local server.
const DEFAULT_API = "http://127.0.0.1:8000";
const MAX_FILE_BYTES = 20 * 1024 * 1024;
const MAX_PAGES = 500;
const ACCEPTED_TYPES = ["image/png", "image/jpeg", "image/webp", "image/bmp"];
const IMAGE_NAME = /\.(png|jpe?g|webp|bmp)$/i;
const POLL_INTERVAL_MS = 1000;
// Which chapter language to preselect when a title has several.
const SOURCE_PREFERENCE = ["en", "es-la", "es", "id", "pt-br", "pt", "fr", "it", "de", "ru", "tr", "pl", "ja", "ko", "zh", "zh-hk", "vi"];
const REMOVE_ICON = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 2l8 8M10 2l-8 8" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>';

const $ = (id) => document.getElementById(id);
const DOT = " · ";

const el = {
  status: $("api-status"),
  statusText: document.querySelector("#api-status .status-text"),
  error: $("error"),
  errorText: $("error-text"),
  errorClose: $("error-close"),
  tabs: [$("tab-page"), $("tab-chapter")],
  // Upload
  dropzone: $("dropzone"),
  dropzoneEmpty: $("dropzone-empty"),
  thumbGrid: $("thumb-grid"),
  fileInput: $("file-input"),
  folderInput: $("folder-input"),
  chooseFilesBtn: $("choose-files-btn"),
  chooseFolderBtn: $("choose-folder-btn"),
  uploadMeta: $("upload-meta"),
  sourceLang: $("source-lang"),
  targetLang: $("target-lang"),
  clearBtn: $("clear-btn"),
  translateBtn: $("translate-btn"),
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
};

const state = {
  apiBase: location.protocol.startsWith("http") ? "" : DEFAULT_API,
  // Upload
  items: [], // { file, path, url }
  uploading: false,
  // MangaDex chapter
  manga: null,
  chapters: [],
  chaptersCache: new Map(),
  selectedChapter: null,
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

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDuration(seconds) {
  seconds = Math.round(seconds);
  if (seconds < 60) return `${seconds} s`;
  return `${Math.floor(seconds / 60)} min ${seconds % 60} s`;
}

function plural(count, word) {
  return `${count} ${count === 1 ? word : `${word}s`}`;
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

// ---- Server status and errors ---------------------------------------------

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

function showError(message) {
  el.errorText.textContent = message;
  el.error.hidden = false;
}

function hideError() {
  el.error.hidden = true;
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

// ==== Job view: progress, reader and download, shared by both tabs ==========

function createJobView(prefix, { storageKey, onChange }) {
  const ui = {
    meta: $(`${prefix}-job-meta`),
    progress: $(`${prefix}-job-progress`),
    progressBar: $(`${prefix}-job-progress-bar`),
    readerEmpty: $(`${prefix}-reader-empty`),
    readerPages: $(`${prefix}-reader-pages`),
    status: $(`${prefix}-job-status`),
    cancelBtn: $(`${prefix}-job-cancel-btn`),
    downloadBtn: $(`${prefix}-download-btn`),
  };
  const view = { job: null, pollTimer: null };

  view.isActive = () => Boolean(view.job && ["queued", "running"].includes(view.job.status));

  view.reset = () => {
    ui.readerPages.replaceChildren();
    ui.readerEmpty.hidden = false;
    ui.progress.hidden = true;
    ui.progressBar.style.width = "0";
    ui.meta.textContent = "";
    ui.status.textContent = "";
    ui.cancelBtn.hidden = true;
    ui.downloadBtn.removeAttribute("href");
    ui.downloadBtn.removeAttribute("download");
    ui.downloadBtn.setAttribute("aria-disabled", "true");
  };

  view.follow = (job) => {
    clearTimeout(view.pollTimer);
    if (!view.job || view.job.id !== job.id) view.reset();
    view.job = job;
    storageSet(storageKey, job.id);
    render(job);
    if (view.isActive()) view.pollTimer = setTimeout(poll, POLL_INTERVAL_MS);
    onChange();
  };

  async function poll() {
    if (!view.job) return;
    try {
      view.follow(await getJson(`/api/jobs/${view.job.id}`));
    } catch (error) {
      if (error.status === 404) {
        forget();
        showError("The server no longer knows this job (it probably restarted). Start it again.");
        return;
      }
      // Keep trying: the server may be busy or briefly unreachable.
      if (error instanceof TypeError) setStatus("offline", "Server offline");
      view.pollTimer = setTimeout(poll, POLL_INTERVAL_MS * 3);
    }
  }

  function forget() {
    storageSet(storageKey, null);
    view.job = null;
    view.reset();
    onChange();
  }

  view.cancel = async () => {
    if (!view.job) return;
    try {
      view.follow(await getJson(`/api/jobs/${view.job.id}`, { method: "DELETE" }));
    } catch (error) {
      reportFailure(error, "cancelling");
    }
  };

  view.resume = async () => {
    const jobId = storageGet(storageKey);
    if (!jobId) return;
    try {
      view.follow(await getJson(`/api/jobs/${jobId}`));
    } catch {
      storageSet(storageKey, null);
    }
  };

  function ensurePageSlots(job) {
    for (let index = ui.readerPages.children.length; index < job.total; index += 1) {
      const figure = document.createElement("figure");
      figure.className = "reader-page";
      const placeholder = document.createElement("div");
      placeholder.className = "page-placeholder";
      placeholder.textContent = `Page ${index + 1}`;
      figure.append(placeholder);
      ui.readerPages.append(figure);
    }
  }

  function render(job) {
    const info = job.chapter || {};
    ui.meta.textContent = [info.manga_title, info.label].filter(Boolean).join(DOT);
    ui.readerEmpty.hidden = job.total > 0;
    ensurePageSlots(job);

    for (const index of job.completed) {
      const figure = ui.readerPages.children[index];
      if (!figure || figure.querySelector("img")) continue;
      const image = document.createElement("img");
      image.loading = "lazy";
      image.alt = `Page ${index + 1}`;
      image.src = api(`/api/jobs/${job.id}/pages/${index}`);
      figure.replaceChildren(image);
    }
    for (const index of job.failed_pages) {
      const placeholder = ui.readerPages.children[index]?.querySelector(".page-placeholder");
      if (placeholder) {
        placeholder.classList.add("is-failed");
        placeholder.textContent = `Page ${index + 1} could not be translated`;
      }
    }

    const finished = job.completed.length + job.failed_pages.length;
    ui.progress.hidden = !job.total || job.status === "done";
    ui.progressBar.style.width = job.total ? `${(finished / job.total) * 100}%` : "0";
    ui.cancelBtn.hidden = !view.isActive();

    const notes = [];
    if (job.status === "queued") notes.push("Waiting for the previous job to finish");
    else if (job.status === "running") {
      notes.push(job.total ? `Translating page ${Math.min(finished + 1, job.total)} of ${job.total}` : "Getting the pages");
      notes.push(formatDuration(job.elapsed));
    } else if (job.status === "done") notes.push(`Done: ${plural(job.completed.length, "page")} in ${formatDuration(job.elapsed)}`);
    else if (job.status === "cancelled") notes.push("Cancelled");
    else if (job.status === "error") notes.push("Failed");
    if (job.failed_pages.length) notes.push(`${job.failed_pages.length} failed`);
    if (job.untranslated) notes.push(`${plural(job.untranslated, "bubble")} left untranslated`);
    ui.status.textContent = notes.join(DOT);
    if (job.status === "error" && job.error) showError(`Translation failed: ${job.error}`);

    const ready = job.status === "done" && job.completed.length > 0;
    const single = job.kind === "upload" && job.total === 1 && ready;
    if (!ready) {
      ui.downloadBtn.removeAttribute("href");
    } else if (single) {
      // One page: download the image itself rather than a ZIP.
      ui.downloadBtn.href = api(`/api/jobs/${job.id}/pages/${job.completed[0]}`);
      ui.downloadBtn.download = job.page_names?.[String(job.completed[0])] || `page_${job.target_lang}.png`;
    } else {
      ui.downloadBtn.href = api(`/api/jobs/${job.id}/download`);
      ui.downloadBtn.removeAttribute("download");
    }
    if (prefix === "up") ui.downloadBtn.textContent = single ? "Download page" : "Download ZIP";
    ui.downloadBtn.setAttribute("aria-disabled", String(!ready));
  }

  ui.cancelBtn.addEventListener("click", view.cancel);
  return view;
}

// ==== Upload pages ==========================================================

function isImage(file, path) {
  return ACCEPTED_TYPES.includes(file.type) || IMAGE_NAME.test(path);
}

function comparePaths(a, b) {
  return a.path.localeCompare(b.path, undefined, { numeric: true, sensitivity: "base" });
}

// Walks dropped folders. Entries must be taken from the event synchronously,
// before anything is awaited, or the browser discards them.
async function collectDropped(dataTransfer) {
  const entries = [...dataTransfer.items]
    .filter((item) => item.kind === "file")
    .map((item) => item.webkitGetAsEntry?.())
    .filter(Boolean);
  if (!entries.length) return [...dataTransfer.files].map((file) => ({ file, path: file.name }));

  const found = [];
  const readFile = (entry) => new Promise((resolve, reject) => entry.file(resolve, reject));
  const readBatch = (reader) => new Promise((resolve, reject) => reader.readEntries(resolve, reject));
  async function walk(entry, prefix) {
    if (entry.isFile) {
      found.push({ file: await readFile(entry), path: `${prefix}${entry.name}` });
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      for (let batch = await readBatch(reader); batch.length; batch = await readBatch(reader)) {
        for (const child of batch) await walk(child, `${prefix}${entry.name}/`);
      }
    }
  }
  for (const entry of entries) await walk(entry, "");
  return found;
}

function addItems(candidates) {
  if (state.uploading) return;
  hideError();
  const known = new Set(state.items.map((item) => `${item.path}:${item.file.size}`));
  let skipped = 0;
  let tooBig = 0;
  let duplicate = 0;
  for (const { file, path } of candidates) {
    if (!isImage(file, path)) {
      skipped += 1;
    } else if (file.size > MAX_FILE_BYTES) {
      tooBig += 1;
    } else if (known.has(`${path}:${file.size}`)) {
      duplicate += 1;
    } else if (state.items.length < MAX_PAGES) {
      state.items.push({ file, path, url: URL.createObjectURL(file) });
      known.add(`${path}:${file.size}`);
    }
  }
  state.items.sort(comparePaths);

  const notes = [];
  if (skipped) notes.push(`skipped ${plural(skipped, "file")} that ${skipped === 1 ? "is" : "are"} not an image`);
  if (tooBig) notes.push(`skipped ${plural(tooBig, "image")} over 20 MB`);
  if (candidates.length - skipped - tooBig - duplicate > 0 && state.items.length >= MAX_PAGES) {
    notes.push(`only the first ${MAX_PAGES} pages were kept`);
  }
  if (notes.length) showError(`Added the pages, but ${notes.join(" and ")}.`);
  renderItems();
}

function removeItem(index) {
  const [item] = state.items.splice(index, 1);
  if (item) URL.revokeObjectURL(item.url);
  renderItems();
}

function clearItems() {
  for (const item of state.items) URL.revokeObjectURL(item.url);
  state.items = [];
  el.fileInput.value = "";
  el.folderInput.value = "";
  hideError();
  renderItems();
}

function renderItems() {
  const count = state.items.length;
  el.dropzoneEmpty.hidden = count > 0;
  el.thumbGrid.hidden = count === 0;
  el.dropzone.classList.toggle("has-items", count > 0);
  const bytes = state.items.reduce((sum, item) => sum + item.file.size, 0);
  el.uploadMeta.textContent = count ? `${plural(count, "page")}${DOT}${formatBytes(bytes)}` : "";

  const tiles = state.items.map((item, index) => {
    const tile = document.createElement("li");
    tile.className = "thumb";
    const image = document.createElement("img");
    image.src = item.url;
    image.alt = "";
    image.loading = "lazy";
    const label = document.createElement("span");
    label.className = "thumb-label";
    label.title = item.path;
    const number = document.createElement("span");
    number.className = "thumb-number";
    number.textContent = String(index + 1);
    label.append(number, item.path.split("/").pop());
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "thumb-remove";
    remove.dataset.index = String(index);
    remove.setAttribute("aria-label", `Remove ${item.path}`);
    remove.innerHTML = REMOVE_ICON;
    remove.disabled = state.uploading;
    tile.append(image, label, remove);
    return tile;
  });
  if (count) {
    const add = document.createElement("li");
    add.className = "thumb-add";
    for (const [text, input] of [["Add files", el.fileInput], ["Add folder", el.folderInput]]) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "btn btn-ghost";
      button.textContent = text;
      button.disabled = state.uploading;
      button.addEventListener("click", () => input.click());
      add.append(button);
    }
    tiles.push(add);
  }
  el.thumbGrid.replaceChildren(...tiles);
  updateUploadButtons();
}

function updateUploadButtons() {
  const busy = state.uploading || uploadJob.isActive();
  const count = state.items.length;
  el.translateBtn.disabled = !count || busy;
  el.translateBtn.textContent = state.uploading
    ? "Uploading"
    : uploadJob.isActive() ? "Translating" : count > 1 ? `Translate ${count} pages` : "Translate";
  el.clearBtn.disabled = !count || state.uploading;
  el.sourceLang.disabled = state.uploading;
  el.targetLang.disabled = state.uploading;
}

function batchTitle() {
  const folders = new Set(state.items.map((item) => (item.path.includes("/") ? item.path.split("/")[0] : "")));
  if (folders.size === 1 && !folders.has("")) return [...folders][0];
  if (state.items.length === 1) return state.items[0].path.replace(/\.[^.]+$/, "");
  return "Translated pages";
}

async function translateItems() {
  if (!state.items.length || state.uploading || uploadJob.isActive()) return;
  hideError();
  if (!(await ensureServer())) return;

  const form = new FormData();
  for (const item of state.items) form.append("files", item.file, item.path);
  form.append("source_lang", el.sourceLang.value);
  form.append("target_lang", el.targetLang.value);
  form.append("title", batchTitle());

  state.uploading = true;
  renderItems();
  try {
    uploadJob.follow(await getJson("/api/uploads", { method: "POST", body: form }));
  } catch (error) {
    reportFailure(error, "uploading the pages");
  } finally {
    state.uploading = false;
    renderItems();
  }
}

// ==== MangaDex chapter ======================================================

function setChapterListMessage(message) {
  el.chapterEmpty.textContent = message;
  el.chapterList.replaceChildren(el.chapterEmpty);
}

function updateChapterButton() {
  el.chapterTranslateBtn.disabled = !state.selectedChapter || !el.chapterSource.value || chapterJob.isActive();
  if (!state.selectedChapter) {
    el.selectionNote.textContent = state.chapters.length ? "Pick a chapter from the list." : "";
  } else {
    el.selectionNote.textContent = `${state.selectedChapter.label}${DOT}${state.selectedChapter.pages} pages`;
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
  const count = `${plural(manga.languages.length, "language")} on MangaDex`;
  el.mangaSub.textContent = [original, count].filter(Boolean).join(DOT);

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
  el.chapterCount.textContent = plural(chapters.length, "chapter");
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
    extra.textContent = [chapter.group, `${chapter.pages} p`].filter(Boolean).join(DOT);

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
    if (scroll) {
      // Scroll only the list; scrollIntoView would also move the whole page.
      const row = input.closest(".chapter-row").getBoundingClientRect();
      const list = el.chapterList.getBoundingClientRect();
      el.chapterList.scrollTop += row.top - list.top - (list.height - row.height) / 2;
    }
  }
  updateChapterButton();
}

async function startChapterJob() {
  if (!state.selectedChapter) return;
  hideError();
  if (!(await ensureServer())) return;
  el.chapterTranslateBtn.disabled = true;
  try {
    chapterJob.follow(await getJson("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chapter_id: state.selectedChapter.id,
        source_lang: el.chapterSource.value,
        target_lang: el.chapterTarget.value,
      }),
    }));
  } catch (error) {
    reportFailure(error, "starting the translation");
    updateChapterButton();
  }
}

// ---- Wiring ---------------------------------------------------------------

const uploadJob = createJobView("up", { storageKey: "mantran.uploadJob", onChange: () => updateUploadButtons() });
const chapterJob = createJobView("ch", { storageKey: "mantran.chapterJob", onChange: () => updateChapterButton() });

for (const tab of el.tabs) {
  tab.addEventListener("click", () => showView(tab.id.replace("tab-", "")));
  tab.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
    const next = el.tabs[(el.tabs.indexOf(tab) + 1) % el.tabs.length];
    showView(next.id.replace("tab-", ""), { focus: true });
  });
}
window.addEventListener("hashchange", () => showView(viewFromHash()));

el.chooseFilesBtn.addEventListener("click", () => el.fileInput.click());
el.chooseFolderBtn.addEventListener("click", () => el.folderInput.click());
el.fileInput.addEventListener("change", () => {
  addItems([...el.fileInput.files].map((file) => ({ file, path: file.name })));
  el.fileInput.value = "";
});
el.folderInput.addEventListener("change", () => {
  addItems([...el.folderInput.files].map((file) => ({ file, path: file.webkitRelativePath || file.name })));
  el.folderInput.value = "";
});
el.thumbGrid.addEventListener("click", (event) => {
  const remove = event.target.closest(".thumb-remove");
  if (remove && !state.uploading) removeItem(Number(remove.dataset.index));
});

["dragenter", "dragover"].forEach((type) =>
  el.dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    if (!state.uploading) el.dropzone.classList.add("is-dragging");
  })
);
el.dropzone.addEventListener("dragleave", (event) => {
  if (!el.dropzone.contains(event.relatedTarget)) el.dropzone.classList.remove("is-dragging");
});
el.dropzone.addEventListener("drop", async (event) => {
  event.preventDefault();
  el.dropzone.classList.remove("is-dragging");
  if (state.uploading) return;
  try {
    addItems(await collectDropped(event.dataTransfer));
  } catch {
    showError("Could not read the dropped files. Try Choose files or Choose folder instead.");
  }
});
// Dropping outside the drop zone should not navigate away to the image.
window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("drop", (event) => event.preventDefault());

document.addEventListener("paste", (event) => {
  if (state.uploading || $("view-page").hidden) return;
  const files = [...event.clipboardData.items]
    .filter((entry) => entry.type.startsWith("image/"))
    .map((entry) => entry.getAsFile())
    .filter(Boolean);
  const stamp = new Date().toISOString().replace(/\D/g, "").slice(0, 14);
  if (files.length) {
    addItems(files.map((file, i) => ({ file, path: `pasted_${stamp}_${i + 1}.${file.type.split("/")[1] || "png"}` })));
  }
});

el.sourceLang.addEventListener("change", () => syncTargetOptions(el.sourceLang, el.targetLang));
el.translateBtn.addEventListener("click", translateItems);
el.clearBtn.addEventListener("click", clearItems);
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

showView(viewFromHash());
renderItems();
checkServer().then((online) => {
  if (!online) return;
  loadLanguages();
  uploadJob.resume();
  chapterJob.resume();
});
