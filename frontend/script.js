"use strict";
const $ = (id) => document.getElementById(id);
const state = {
  user: null,
  session: null,
  documents: [],
  selected: new Set(),
  busy: false,
  controller: null,
  replaceId: null,
  poll: null,
};

async function api(path, options = {}) {
  const headers = { "X-RAG-Request": "1", ...(options.headers || {}) };
  if (options.body && !(options.body instanceof FormData))
    headers["Content-Type"] = "application/json";
  const response = await fetch(`/api/v1${path}`, {
    credentials: "same-origin",
    ...options,
    headers,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (response.status === 401 && !path.includes("/auth/login")) showLogin();
    const error = new Error(
      data.error?.message || `Request failed (${response.status}).`,
    );
    error.code = data.error?.code;
    error.requestId = data.error?.request_id;
    throw error;
  }
  return data;
}
function notify(message, error = false) {
  $("notice").textContent = message;
  $("notice").hidden = !message;
  $("notice").classList.toggle("error", error);
}
function showLogin() {
  state.user = null;
  clearTimeout(state.poll);
  $("workspace").hidden = true;
  $("login-screen").hidden = false;
  $("messages")
    .querySelectorAll(".message")
    .forEach((node) => node.remove());
  state.documents = [];
  state.selected.clear();
  state.session = null;
}
async function enterWorkspace() {
  const me = await api("/auth/me");
  state.user = me.username;
  $("user-label").textContent = me.username;
  $("file-hint").textContent =
    `PDF, TXT, MD, DOCX · up to ${me.max_upload_mb} MB`;
  $("question").maxLength = me.max_query_chars;
  $("workspace").hidden = false;
  $("login-screen").hidden = true;
  $("access-key").value = "";
  await Promise.all([refreshDocuments(), refreshSessions()]);
  if (!state.session) await newConversation();
  await loadConversation();
  checkHealth();
  schedulePoll();
}
$("login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("login-button").disabled = true;
  $("login-error").textContent = "";
  try {
    await api("/auth/login", {
      method: "POST",
      body: JSON.stringify({
        username: $("username").value.trim(),
        access_key: $("access-key").value,
      }),
    });
    await enterWorkspace();
  } catch (error) {
    $("login-error").textContent = error.message;
  } finally {
    $("login-button").disabled = false;
  }
});
$("logout-button").addEventListener("click", async () => {
  try {
    await api("/auth/logout", { method: "POST" });
    showLogin();
  } catch (error) {
    notify(error.message, true);
  }
});
async function refreshSessions() {
  const sessions = await api("/sessions");
  const select = $("conversation-select");
  select.replaceChildren();
  for (const session of sessions) {
    const option = document.createElement("option");
    option.value = session.id;
    option.textContent = session.title;
    select.appendChild(option);
  }
  if (!sessions.some((s) => s.id === state.session))
    state.session = sessions[0]?.id || null;
  select.value = state.session || "";
}
async function newConversation() {
  if (state.busy) return;
  const title = `Conversation · ${new Date().toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}`;
  const session = await api("/sessions", {
    method: "POST",
    body: JSON.stringify({ title }),
  });
  state.session = session.id;
  await refreshSessions();
  await loadConversation();
  $("question").focus();
}
$("new-chat").addEventListener("click", () =>
  newConversation().catch((e) => notify(e.message, true)),
);
$("conversation-select").addEventListener("change", async (event) => {
  state.session = event.target.value;
  try {
    await loadConversation();
  } catch (error) {
    notify(error.message, true);
  }
});
async function loadConversation() {
  if (!state.session) return;
  const messages = await api(`/sessions/${state.session}/messages`);
  $("messages")
    .querySelectorAll(".message")
    .forEach((node) => node.remove());
  $("empty-state").hidden = messages.length > 0;
  for (const message of messages)
    addMessage(message.role, message.content, message.sources, message.status);
}
function confirmAction(title, description) {
  return new Promise((resolve) => {
    $("confirm-title").textContent = title;
    $("confirm-description").textContent = description;
    const dialog = $("confirm-dialog");
    dialog.returnValue = "cancel";
    dialog.addEventListener(
      "close",
      () => resolve(dialog.returnValue === "confirm"),
      { once: true },
    );
    dialog.showModal();
  });
}
$("delete-chat").addEventListener("click", async () => {
  if (!state.session || state.busy) return;
  if (
    !(await confirmAction(
      "Delete this conversation?",
      "Its stored messages will be removed.",
    ))
  )
    return;
  try {
    await api(`/sessions/${state.session}`, { method: "DELETE" });
    state.session = null;
    await refreshSessions();
    if (!state.session) await newConversation();
    else await loadConversation();
  } catch (error) {
    notify(error.message, true);
  }
});
async function refreshDocuments() {
  const documents = await api("/documents");
  const oldVersions = new Map(
    state.documents.map((doc) => [doc.id, doc.active_version]),
  );
  const replaced = documents.some(
    (doc) =>
      oldVersions.get(doc.id) && oldVersions.get(doc.id) !== doc.active_version,
  );
  const changed = JSON.stringify(documents) !== JSON.stringify(state.documents);
  state.documents = documents;
  for (const id of state.selected)
    if (!documents.some((d) => d.id === id && d.active_version))
      state.selected.delete(id);
  if (changed) renderDocuments();
  if (replaced && !state.busy) {
    await loadConversation();
    notify(
      "Document replacement is ready. Previous conversation history was cleared.",
    );
  }
}
function renderDocuments() {
  const list = $("document-list");
  list.replaceChildren();
  $("document-count").textContent = state.documents.length;
  if (!state.documents.length) {
    const empty = document.createElement("p");
    empty.className = "library-empty";
    empty.textContent =
      "Your library is a blank page. Add your first source to get started.";
    list.appendChild(empty);
  }
  for (const doc of state.documents) {
    const item = document.createElement("article");
    item.className = "document-item";
    const row = document.createElement("label");
    row.className = "document-main";
    const check = document.createElement("input");
    check.type = "checkbox";
    check.checked = $("all-sources").checked
      ? Boolean(doc.active_version)
      : state.selected.has(doc.id);
    check.disabled = !doc.active_version || state.busy;
    check.setAttribute("aria-label", `Use ${doc.filename}`);
    check.addEventListener("change", () => {
      if ($("all-sources").checked) {
        state.selected = new Set(
          state.documents.filter((d) => d.active_version).map((d) => d.id),
        );
        $("all-sources").checked = false;
      }
      if (check.checked) state.selected.add(doc.id);
      else state.selected.delete(doc.id);
      updateScope();
    });
    const text = document.createElement("span");
    const name = document.createElement("strong");
    name.textContent = doc.filename;
    name.title = doc.filename;
    const status = document.createElement("small");
    const job = doc.job;
    status.textContent =
      job?.status === "ready"
        ? `${job.chunks} passages · Ready`
        : `${job?.status || "Pending"}${doc.active_version ? " · Previous version available" : ""}`;
    status.className = `doc-status ${job?.status || ""}`;
    text.append(name, status);
    row.append(check, text);
    item.appendChild(row);
    if (job?.error) {
      const error = document.createElement("p");
      error.className = "document-error";
      error.textContent = job.error;
      item.appendChild(error);
    }
    const actions = document.createElement("div");
    actions.className = "document-actions";
    if (doc.active_version) {
      const source = document.createElement("a");
      source.href = `/api/v1/documents/${doc.id}/source`;
      source.textContent = "Source";
      actions.appendChild(source);
      const overview = document.createElement("button");
      overview.className = "text-button";
      overview.textContent = "Overview";
      overview.disabled = state.busy;
      overview.addEventListener("click", () => showOverview(doc));
      actions.appendChild(overview);
    }
    const replace = document.createElement("button");
    replace.className = "text-button";
    replace.textContent = job?.status === "failed" ? "Retry" : "Replace";
    replace.disabled =
      state.busy || ["queued", "processing"].includes(job?.status);
    replace.addEventListener("click", () => {
      state.replaceId = doc.id;
      $("replace-input").click();
    });
    const remove = document.createElement("button");
    remove.className = "text-button";
    remove.textContent = "Delete";
    remove.disabled = state.busy;
    remove.addEventListener("click", async () => {
      if (
        !(await confirmAction(
          `Delete ${doc.filename}?`,
          "The document and its passages will be removed. Conversation history in your workspace is also cleared to prevent reuse of deleted content.",
        ))
      )
        return;
      try {
        await api(`/documents/${doc.id}`, { method: "DELETE" });
        await refreshDocuments();
        await loadConversation();
        notify("Document removed and conversation history cleared.");
      } catch (error) {
        notify(error.message, true);
      }
    });
    actions.append(replace, remove);
    item.appendChild(actions);
    list.appendChild(item);
  }
  updateScope();
}
function updateScope() {
  const all = $("all-sources").checked;
  $("scope-label").textContent = all
    ? "All ready documents"
    : `${state.selected.size} selected source${state.selected.size === 1 ? "" : "s"}`;
}
$("all-sources").addEventListener("change", () => {
  state.selected.clear();
  renderDocuments();
});
async function uploadFile(file, replaceId = null) {
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  $("upload-status").textContent = `Uploading ${file.name}…`;
  $("file-input").disabled = true;
  try {
    const result = await api(
      replaceId ? `/documents/${replaceId}` : "/documents",
      { method: replaceId ? "PUT" : "POST", body: form },
    );
    $("upload-status").textContent = result.duplicate
      ? "This document is already in your library."
      : "Upload received. Indexing will continue in the background.";
    await refreshDocuments();
    if (replaceId)
      notify(
        "The previous version stays available until the replacement is ready. Existing conversation history will then be cleared.",
      );
  } catch (error) {
    $("upload-status").textContent = error.message;
  } finally {
    $("file-input").disabled = false;
    $("file-input").value = "";
    $("replace-input").value = "";
  }
}
$("file-input").addEventListener("change", (event) =>
  uploadFile(event.target.files[0]),
);
$("replace-input").addEventListener("change", (event) =>
  uploadFile(event.target.files[0], state.replaceId),
);
for (const type of ["dragenter", "dragover"])
  $("upload-zone").addEventListener(type, (event) => {
    event.preventDefault();
    $("upload-zone").classList.add("dragging");
  });
for (const type of ["dragleave", "drop"])
  $("upload-zone").addEventListener(type, (event) => {
    event.preventDefault();
    $("upload-zone").classList.remove("dragging");
  });
$("upload-zone").addEventListener("drop", (event) => {
  if (!state.busy) uploadFile(event.dataTransfer.files[0]);
});
function addMessage(role, text, sources = [], status = null) {
  $("empty-state").hidden = true;
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = role === "user" ? "YOU" : "FOLIO";
  const content = document.createElement("div");
  content.className = "message-content";
  content.textContent = text;
  article.append(label, content);
  if (sources?.length) {
    const details = document.createElement("details");
    details.className = "source-details";
    const summary = document.createElement("summary");
    summary.textContent = `${sources.length} supporting passage${sources.length === 1 ? "" : "s"}`;
    details.appendChild(summary);
    for (const source of sources) {
      const card = document.createElement("div");
      card.className = "source-card";
      const link = document.createElement("a");
      // Build links from validated identifiers, never from model-generated URLs.
      if (
        /^[a-f0-9]{32}$/.test(source.document_id) &&
        /^[a-f0-9]{32}$/.test(source.version_id)
      )
        link.href = `/api/v1/documents/${source.document_id}/source?version_id=${source.version_id}`;
      link.textContent = `[${source.citation_id}] ${source.filename}${source.page ? ` · page ${source.page}` : ""} ↗`;
      const excerpt = document.createElement("p");
      excerpt.textContent = source.excerpt;
      card.append(link, excerpt);
      details.appendChild(card);
    }
    article.appendChild(details);
  }
  if (status && status !== "answered") {
    const badge = document.createElement("span");
    badge.className = "answer-status";
    badge.textContent = status.replaceAll("_", " ");
    article.appendChild(badge);
  }
  $("messages").appendChild(article);
  article.scrollIntoView({ block: "end", behavior: "smooth" });
  return article;
}
function setBusy(busy) {
  state.busy = busy;
  for (const id of [
    "send-button",
    "new-chat",
    "delete-chat",
    "conversation-select",
    "all-sources",
    "file-input",
    "logout-button",
  ])
    $(id).disabled = busy;
  $("cancel-chat").hidden = !busy;
  $("question").disabled = busy;
  renderDocuments();
}
$("chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = $("question").value.trim();
  if (!query || state.busy || !state.session) return;
  if (!$("all-sources").checked && !state.selected.size)
    return notify(
      "Select at least one source, or choose all ready documents.",
      true,
    );
  notify("");
  setBusy(true);
  addMessage("user", query);
  $("question").value = "";
  const pending = addMessage(
    "assistant",
    "Finding relevant passages and checking the evidence…",
  );
  pending.classList.add("pending");
  state.controller = new AbortController();
  const timeout = setTimeout(() => state.controller?.abort(), 240000);
  try {
    const response = await api("/chat", {
      method: "POST",
      signal: state.controller.signal,
      body: JSON.stringify({
        query,
        session_id: state.session,
        document_ids: $("all-sources").checked ? [] : [...state.selected],
      }),
    });
    pending.remove();
    addMessage("assistant", response.answer, response.sources, response.status);
  } catch (error) {
    pending.classList.remove("pending");
    pending.querySelector(".message-content").textContent =
      error.name === "AbortError"
        ? "Stopped waiting. The server may still finish and save this answer; reopen the conversation to check before retrying."
        : error.message;
    $("question").value = query;
    if (error.requestId) notify(`Request reference: ${error.requestId}`, true);
  } finally {
    clearTimeout(timeout);
    state.controller = null;
    setBusy(false);
    $("question").focus();
  }
});
$("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("chat-form").requestSubmit();
  }
});
$("cancel-chat").addEventListener("click", () => state.controller?.abort());
for (const button of document.querySelectorAll("[data-question]"))
  button.addEventListener("click", () => {
    $("question").value = button.dataset.question;
    $("question").focus();
  });
async function checkHealth() {
  try {
    const response = await fetch("/health/ready");
    $("health").textContent = response.ok ? "Service ready" : "Model not ready";
    $("health").classList.toggle("ready", response.ok);
  } catch {
    $("health").textContent = "Service unavailable";
    $("health").classList.remove("ready");
  }
}
let polls = 0;
function schedulePoll() {
  clearTimeout(state.poll);
  state.poll = setTimeout(
    async () => {
      if (!state.user) return;
      try {
        await refreshDocuments();
        if (++polls % 5 === 0) checkHealth();
      } catch (error) {
        if (state.user) notify(error.message, true);
      }
      if (state.user) schedulePoll();
    },
    state.documents.some((d) =>
      ["queued", "processing"].includes(d.job?.status),
    )
      ? 4000
      : 12000,
  );
}
enterWorkspace().catch(() => showLogin());

async function showOverview(doc) {
  $("overview-title").textContent = doc.filename;
  $("overview-content").textContent = "Preparing an evidence-based overview…";
  $("overview-dialog").showModal();
  try {
    const result = await api(`/documents/${doc.id}/overview`, {
      method: "POST",
    });
    $("overview-content").replaceChildren();
    const coverage = document.createElement("p");
    coverage.className = "helper";
    coverage.textContent = `${result.passages_used} of ${result.total_passages} indexed passages used${result.cached ? " · Saved overview" : ""}`;
    const answer = document.createElement("p");
    answer.className = "message-content";
    answer.textContent = result.answer;
    $("overview-content").append(coverage, answer);
    for (const source of result.sources) {
      const passage = document.createElement("p");
      passage.className = "source-card";
      passage.textContent = `[${source.citation_id}] ${source.filename}${source.page ? ` · page ${source.page}` : ""}\n${source.excerpt}`;
      $("overview-content").appendChild(passage);
    }
  } catch (error) {
    $("overview-content").textContent = error.message;
  }
}
$("close-overview").addEventListener("click", () =>
  $("overview-dialog").close(),
);
