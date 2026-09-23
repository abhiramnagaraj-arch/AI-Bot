"use strict";
const $ = (id) => document.getElementById(id);
const state = { user: null, session: "", selected: new Set(), busy: false };

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}

function signedOut() {
  state.user = null;
  state.session = "";
  state.selected.clear();
  $("workspace").hidden = $("logout").hidden = true;
  $("login-form").hidden = false;
  $("messages").replaceChildren();
  $("documents").replaceChildren();
  $("sessions").replaceChildren();
  $("question").value = "";
}

async function api(path, options = {}) {
  const headers = { "X-RAG-Request": "1" };
  if (options.body && !(options.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(options.body);
  }
  const response = await fetch(`/api/v1${path}`, { ...options, headers, credentials: "same-origin" });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (response.status === 401) signedOut();
    throw new Error(data.error?.message || `Request failed (${response.status}). Please retry.`);
  }
  return data;
}

async function run(task, activity = "Working…") {
  if (state.busy) return;
  state.busy = true;
  $("notice").textContent = "";
  $("activity").textContent = activity;
  document.querySelectorAll("button,input,textarea,select").forEach((el) => { el.disabled = true; });
  try { await task(); }
  catch (error) { $("notice").textContent = error.message || "Connection failed. Please retry."; }
  finally {
    state.busy = false;
    document.querySelectorAll("button,input,textarea,select").forEach((el) => { el.disabled = false; });
    $("activity").textContent = "Ready";
  }
}

function chooseSession(id) {
  state.session = id;
  localStorage.setItem(`folio-session:${state.user}`, id);
  $("sessions").value = id;
}

function renderMessage(role, content, sources = []) {
  const message = node("article", undefined, `message ${role}`);
  message.append(node("strong", role === "user" ? "You" : "Folio"), node("p", content, "answer"));
  if (sources.length) {
    const details = node("details", undefined, "source-details");
    details.append(node("summary", `References (${sources.length})`));
    for (const source of sources) {
      const card = node("div", undefined, "source-card");
      const link = node("a", `[${source.citation_id}] ${source.filename}${source.page ? ` · page ${source.page}` : ""}`);
      const url = new URL(source.url, location.origin);
      if (url.origin === location.origin && url.pathname.startsWith("/api/v1/documents/")) link.href = url.href;
      card.append(link, node("p", source.excerpt));
      details.append(card);
    }
    message.append(details);
  }
  $("messages").append(message);
  $("messages").scrollTop = $("messages").scrollHeight;
}

async function loadMessages() {
  const messages = state.session ? await api(`/sessions/${state.session}/messages`) : [];
  $("messages").replaceChildren();
  for (const message of messages) renderMessage(message.role, message.content, message.sources);
}

async function loadSessions() {
  const sessions = await api("/sessions");
  const empty = node("option", "New conversation");
  empty.value = "";
  $("sessions").replaceChildren(empty);
  for (const session of sessions) {
    const option = node("option", session.title);
    option.value = session.id;
    $("sessions").append(option);
  }
  if (!sessions.some((session) => session.id === state.session)) chooseSession("");
  $("sessions").value = state.session;
}

function updateScope() {
  $("scope").textContent = state.selected.size ? `Using ${state.selected.size} selected document(s).` : "Using all documents.";
}

async function loadDocuments() {
  const documents = await api("/documents");
  const ids = new Set(documents.map((doc) => doc.id));
  state.selected = new Set([...state.selected].filter((id) => ids.has(id)));
  $("documents").replaceChildren();
  if (!documents.length) $("documents").append(node("p", "No documents yet. Add your first source.", "muted"));
  for (const doc of documents) {
    const row = node("div", undefined, "document");
    const label = node("label");
    const checkbox = node("input");
    checkbox.type = "checkbox";
    checkbox.checked = state.selected.has(doc.id);
    checkbox.disabled = state.busy;
    checkbox.addEventListener("change", () => {
      checkbox.checked ? state.selected.add(doc.id) : state.selected.delete(doc.id);
      updateScope();
    });
    label.append(checkbox, node("span", doc.filename));
    const remove = node("button", "Delete");
    remove.disabled = state.busy;
    remove.addEventListener("click", () => {
      if (!confirm(`Delete ${doc.filename}? Saved answers will also be cleared.`)) return;
      run(async () => {
        await api(`/documents/${doc.id}`, { method: "DELETE" });
        await loadDocuments();
        await loadMessages();
      }, "Deleting document…");
    });
    row.append(label, node("small", `${doc.chunks} passages · Ready`), remove);
    $("documents").append(row);
  }
  updateScope();
}

async function openWorkspace() {
  const user = await api("/auth/me");
  state.user = user.username;
  state.session = localStorage.getItem(`folio-session:${state.user}`) || "";
  $("question").maxLength = user.max_query_chars;
  $("upload-help").textContent = `PDF, TXT, Markdown or DOCX · up to ${user.max_upload_mb} MB each. Scanned PDFs need OCR first.`;
  $("workspace").hidden = $("logout").hidden = false;
  $("login-form").hidden = true;
  await loadDocuments();
  await loadSessions();
  await loadMessages();
}

$("login-form").addEventListener("submit", (event) => {
  event.preventDefault();
  run(async () => {
    await api("/auth/login", { method: "POST", body: { username: $("username").value.trim(), access_key: $("access-key").value } });
    $("access-key").value = "";
    await openWorkspace();
  }, "Signing in…");
});
$("logout").addEventListener("click", () => run(async () => {
  await api("/auth/logout", { method: "POST" });
  signedOut();
}));
$("file-input").addEventListener("change", () => {
  const files = [...$("file-input").files];
  run(async () => {
    try {
      for (const file of files) {
        $("activity").textContent = `Reading and indexing ${file.name}…`;
        const form = new FormData();
        form.append("file", file);
        await api("/documents", { method: "POST", body: form });
        await loadDocuments();
      }
    } finally { $("file-input").value = ""; }
  }, "Uploading…");
});
$("sessions").addEventListener("change", () => run(async () => {
  chooseSession($("sessions").value);
  await loadMessages();
}));
$("new-chat").addEventListener("click", () => run(async () => {
  chooseSession("");
  await loadMessages();
}));
$("delete-chat").addEventListener("click", () => {
  if (!state.session || !confirm("Delete this conversation?")) return;
  run(async () => {
    await api(`/sessions/${state.session}`, { method: "DELETE" });
    chooseSession("");
    await loadSessions();
    await loadMessages();
  });
});
$("chat-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const query = $("question").value.trim();
  if (!query) return;
  run(async () => {
    if (!state.session) {
      const session = await api("/sessions", { method: "POST", body: { title: query.slice(0, 80) } });
      chooseSession(session.id);
      await loadSessions();
    }
    const answer = await api("/chat", { method: "POST", body: { session_id: state.session, query, document_ids: [...state.selected] } });
    renderMessage("user", query);
    renderMessage("assistant", answer.answer, answer.sources);
    $("question").value = "";
  }, "Finding evidence and answering…");
});
run(async () => {
  try { await openWorkspace(); }
  catch (error) { if (state.user) throw error; signedOut(); }
});
