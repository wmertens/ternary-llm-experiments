// Web worker that owns the transformers.js pipeline so the UI thread stays
// responsive during model load and generation.

import {
  pipeline,
  TextStreamer,
  InterruptableStoppingCriteria,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.1.0";

// Self-host the model alongside this worker (./model/...). Disable the
// HF Hub fallback so a typo in the local layout fails loudly instead of
// silently going over the network.
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = new URL("./", self.location.href).href;

const MODEL_ID = "model";
const stopper = new InterruptableStoppingCriteria();
let pipe = null;

async function load() {
  if (pipe) {
    self.postMessage({ status: "ready" });
    return;
  }
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error("WebGPU not available — try a recent Chrome / Edge.");
  } catch (e) {
    self.postMessage({ status: "error", data: String(e) });
    return;
  }

  self.postMessage({ status: "loading", data: "Fetching model files…" });
  try {
    pipe = await pipeline("text-generation", MODEL_ID, {
      device: "webgpu",
      dtype: "q2",
      progress_callback: (info) => {
        if (info.status === "progress" || info.status === "progress_total") {
          self.postMessage({
            status: "progress",
            file: info.file,
            loaded: Number(info.loaded ?? 0),
            total: Number(info.total ?? 0),
            progress: Number(info.progress ?? 0),
          });
        }
      },
    });

    self.postMessage({ status: "loading", data: "Warming up WebGPU kernels…" });
    const ids = pipe.tokenizer("a", { add_special_tokens: false });
    await pipe.model.generate({ ...ids, max_new_tokens: 1 });

    self.postMessage({ status: "ready" });
  } catch (e) {
    self.postMessage({ status: "error", data: String(e) });
  }
}

async function generate({ prompt, max_new_tokens, temperature }) {
  if (!pipe) {
    self.postMessage({ status: "error", data: "Model not loaded" });
    return;
  }
  // Match chat.py: cached training sequences begin with BOS, so the student
  // sees an out-of-distribution prefix without it and degenerates into loops.
  const bos = pipe.tokenizer.bos_token ?? "";
  const text = prompt.startsWith(bos) ? prompt : bos + prompt;

  let started = 0;
  let nTok = 0;
  let tps = 0;

  const streamer = new TextStreamer(pipe.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (chunk) => {
      self.postMessage({ status: "update", chunk, tps, numTokens: nTok });
    },
    token_callback_function: () => {
      if (!started) started = performance.now();
      nTok++;
      const dt = performance.now() - started;
      if (dt > 0) tps = (nTok / dt) * 1000;
    },
  });

  self.postMessage({ status: "start" });
  stopper.reset();

  try {
    await pipe(text, {
      max_new_tokens,
      do_sample: temperature > 0,
      temperature,
      top_p: 0.9,
      streamer,
      stopping_criteria: stopper,
      return_full_text: false,
    });
    self.postMessage({ status: "complete", tps, numTokens: nTok });
  } catch (e) {
    self.postMessage({ status: "error", data: String(e) });
  }
}

self.addEventListener("message", (e) => {
  const { type, data } = e.data;
  switch (type) {
    case "load": return load();
    case "generate": return generate(data);
    case "interrupt": return stopper.interrupt();
  }
});
