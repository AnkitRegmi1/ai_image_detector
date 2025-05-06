(async () => {
  const folder = new URL('./', document.currentScript.src).href;
  const ortURL = folder + 'ort.min.js';
  const wasmBase = folder;
  const modelURL = folder + 'onnx_model.onnx';

  function load(src) {
    return new Promise((res, rej) => {
      const s = document.createElement('script');
      s.src = src; s.onload = res;
      s.onerror = () => rej(new Error('❌ load ' + src));
      document.head.appendChild(s);
    });
  }

  try {
    console.log('[AI‑Detector page] loading ORT …');
    await load(ortURL);
    if (!window.ort) throw new Error('window.ort missing');
    ort.env.wasm.wasmPaths = wasmBase;
    ort.env.wasm.numThreads = 1;

    console.log('[AI‑Detector page] loading model …');
    window.ortSession = await ort.InferenceSession.create(modelURL, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all'
    });

    window.postMessage({ type: 'PAGE_CTX_READY' }, '*');
  } catch (e) {
    console.error('[AI‑Detector page]', e);
    window.postMessage({ type: 'PAGE_CTX_FAILED', error: e.message }, '*');
  }
})();

window.addEventListener('message', async ev => {
  if (ev.data?.type !== 'AI_DETECTOR_ANALYZE') return;
  const { id, imageData } = ev.data;
  try {
    const t = new ort.Tensor('float32', new Float32Array(imageData.data), [1, 3, imageData.width, imageData.width]);
    const r = await window.ortSession.run({ input: t });
    window.postMessage({ type: 'AI_DETECTOR_RESULT', id, logits: Array.from(r.logits.data) }, '*');
  } catch (e) {
    window.postMessage({ type: 'AI_DETECTOR_ERROR', id, error: e.message }, '*');
  }
});
