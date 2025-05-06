console.log('[AI‑Detector] content script loading…');
(() => {
  const s = document.createElement('script');
  s.src = chrome.runtime.getURL('page_context.js');
  (document.head || document.documentElement).appendChild(s);
  s.remove();
})();

window.addEventListener('message', ev => {
  if (ev.data?.type === 'PAGE_CTX_READY') {
    console.log('[AI‑Detector] model ready – hover any image');
    initPipeline();
  }
});

function initPipeline() {
  const THRESHOLD = 0.8;
  const SIZE = 224;
  const MEAN = [.485, .456, .406], STD = [.229, .224, .225];
  const pending = new Map();

  const softmax = a => {
    const m = Math.max(...a), ex = a.map(v => Math.exp(v - m));
    const s = ex.reduce((p, c) => p + c, 0); return ex.map(v => v / s);
  };

  const toPayload = (img) => {
    const cv = document.createElement('canvas'); cv.width = cv.height = SIZE;
    const ctx = cv.getContext('2d'); ctx.drawImage(img, 0, 0, SIZE, SIZE);
    const d = ctx.getImageData(0, 0, SIZE, SIZE).data;
    const out = new Float32Array(3 * SIZE * SIZE);
    for (let i = 0, p = 0; i < d.length; i += 4, p++) {
      const r = d[i]/255, g = d[i+1]/255, b = d[i+2]/255;
      out[p] = (r - MEAN[0]) / STD[0];
      out[p + SIZE*SIZE] = (g - MEAN[1]) / STD[1];
      out[p + 2*SIZE*SIZE] = (b - MEAN[2]) / STD[2];
    }
    return { data: Array.from(out), width: SIZE };
  };

  const addBadge = (img, text, color) => {
    img.parentElement?.querySelectorAll('.ai-detector-badge').forEach(b => b.remove());
    const div = document.createElement('div');
    div.className = 'ai-detector-badge';
    div.textContent = text;
    Object.assign(div.style, {
      position: 'absolute', top: '6px', right: '6px',
      background: color, color: '#fff', padding: '2px 6px',
      fontSize: '11px', fontWeight: 600, borderRadius: '4px',
      pointerEvents: 'none', zIndex: 9999
    });
    const p = img.parentElement;
    if (p && getComputedStyle(p).position === 'static') p.style.position = 'relative';
    (p || document.body).appendChild(div);
    return div;
  };

  const fetchAsImage = async (url) => {
    const blob = await (await fetch(url, { mode: 'cors' })).blob();
    const obj = URL.createObjectURL(blob);
    const tmp = new Image(); tmp.crossOrigin = 'anonymous';
    await new Promise((ok, err) => { tmp.onload = ok; tmp.onerror = err; tmp.src = obj; });
    URL.revokeObjectURL(obj);
    return { img: tmp, blob };
  };

  document.addEventListener('mouseover', async ev => {
    const img = ev.target.closest('img');
    if (!img || img.dataset.aiChecked) return;

    img.dataset.aiChecked = '1';
    const spin = addBadge(img, '…', '#444');

    try {
      const { img: decoded, blob } = await fetchAsImage(img.currentSrc || img.src);
      const payload = toPayload(decoded);
      const id = crypto.randomUUID();
      pending.set(id, { img, spin, blob });
      window.postMessage({ type: 'AI_DETECTOR_ANALYZE', id, imageData: payload }, '*');
    } catch (e) {
      console.warn('[AI‑Detector] decode error:', e);
      spin.remove();
    }
  }, true);

  window.addEventListener('message', async ev => {
    if (ev.data?.type !== 'AI_DETECTOR_RESULT') return;
    const entry = pending.get(ev.data.id); pending.delete(ev.data.id);
    if (!entry) return;

    const { img, spin, blob } = entry; spin.remove();
    const probAI = softmax(ev.data.logits)[1];
    if (probAI >= THRESHOLD) {
      addBadge(img, `⚠️ AI ${(probAI*100).toFixed(1)}%`, 'red');
      explainWithGemini(img, blob);
    } else {
      addBadge(img, `✔️ Human ${(100 - probAI*100).toFixed(1)}%`, 'green');
    }
  });

  console.log('[AI‑Detector] ✅ hover-to-analyze pipeline ready');
}


