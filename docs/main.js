// UI ↔ worker glue. Keeps the worker lifecycle simple (one global) and pipes
// streaming chunks into the output pane.

const $ = (id) => document.getElementById(id);
const els = {
  status: $("status"),
  output: $("output"),
  prompt: $("prompt"),
  send: $("send"),
  stop: $("stop"),
  clear: $("clear"),
  maxTokens: $("max-tokens"),
  temp: $("temp"),
  form: $("form"),
};

const worker = new Worker(new URL("./worker.js", import.meta.url), {
  type: "module",
});

let generating = false;
let promptEcho = "";

function setStatus(text, kind = "") {
  els.status.textContent = text;
  els.status.className = kind;
}

function appendOutput(text, klass = "gen") {
  const span = document.createElement("span");
  span.className = klass;
  span.textContent = text;
  els.output.appendChild(span);
  els.output.scrollTop = els.output.scrollHeight;
}

function setBusy(busy) {
  generating = busy;
  els.send.disabled = busy;
  els.stop.disabled = !busy;
  els.prompt.disabled = busy;
}

worker.addEventListener("message", (e) => {
  const m = e.data;
  switch (m.status) {
    case "loading":
      setStatus(m.data);
      break;
    case "progress": {
      const pct = Math.round((m.progress ?? 0));
      const mb = m.total ? `${(m.loaded / 1e6).toFixed(1)}/${(m.total / 1e6).toFixed(1)} MB` : "";
      setStatus(`Loading ${m.file ?? ""} ${pct}% ${mb}`);
      break;
    }
    case "ready":
      setStatus("Ready. WebGPU initialized — type a prompt below.");
      els.send.disabled = false;
      break;
    case "start":
      els.output.textContent = "";
      appendOutput(promptEcho, "prompt");
      break;
    case "update":
      appendOutput(m.chunk, "gen");
      setStatus(`Generating · ${m.numTokens} tok · ${m.tps?.toFixed(1) ?? "—"} tok/s`);
      break;
    case "complete":
      setStatus(`Done · ${m.numTokens} tokens · ${m.tps?.toFixed(1) ?? "—"} tok/s`);
      setBusy(false);
      break;
    case "error":
      setStatus(`Error: ${m.data}`, "err");
      setBusy(false);
      break;
  }
});

els.form.addEventListener("submit", (e) => {
  e.preventDefault();
  if (generating) return;
  const prompt = els.prompt.value.trim();
  if (!prompt) return;
  promptEcho = prompt;
  setBusy(true);
  worker.postMessage({
    type: "generate",
    data: {
      prompt,
      max_new_tokens: Math.max(1, Math.min(2048, parseInt(els.maxTokens.value, 10) || 200)),
      temperature: Math.max(0, Math.min(2, parseFloat(els.temp.value) || 0)),
    },
  });
});

// Submit on Enter, newline on Shift+Enter — standard chat-textarea behavior.
els.prompt.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    els.form.requestSubmit();
  }
});

els.stop.addEventListener("click", () => {
  worker.postMessage({ type: "interrupt" });
});

els.clear.addEventListener("click", () => {
  els.output.textContent = "";
});

// Kick off the load immediately — the user-visible status reflects progress.
worker.postMessage({ type: "load" });
