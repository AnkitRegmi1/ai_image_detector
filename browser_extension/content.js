// ----------------------------------------------------------------------------
// AI Image Detector (hover → ONNX + OpenAI truth-table)
// ----------------------------------------------------------------------------
console.log('[AI-Detector] content script loading…');

// Inject ONNX loader into page world
(function(){
  const s = document.createElement('script');
  s.src = chrome.runtime.getURL('page_context.js');
  (document.head||document.documentElement).appendChild(s);
  s.remove();
})();

// Wait for ONNX model ready
window.addEventListener('message', ev => {
  if (ev.data?.type === 'PAGE_CTX_READY') {
    console.log('[AI-Detector] ONNX model ready – hover images');
    initPipeline();
  }
});

function initPipeline() {
  const THRESH = 0.8;
  const SIZE   = 224;
  const MEAN   = [0.485,0.456,0.406];
  const STD    = [0.229,0.224,0.225];
  const pending = new Map();

  const softmax = arr => {
    const m = Math.max(...arr),
          ex = arr.map(v=>Math.exp(v-m)),
          s  = ex.reduce((a,b)=>a+b,0);
    return ex.map(v=>v/s);
  };

  function toPayload(img) {
    const cv = document.createElement('canvas');
    cv.width = cv.height = SIZE;
    const ctx = cv.getContext('2d');
    ctx.drawImage(img,0,0,SIZE,SIZE);
    const D = ctx.getImageData(0,0,SIZE,SIZE).data;
    const out = new Float32Array(3*SIZE*SIZE);
    for (let i=0,p=0; i<D.length; i+=4,p++){
      const r=D[i]/255, g=D[i+1]/255, b=D[i+2]/255;
      out[p]               = (r - MEAN[0])/STD[0];
      out[p+   SIZE*SIZE]  = (g - MEAN[1])/STD[1];
      out[p+2*SIZE*SIZE]   = (b - MEAN[2])/STD[2];
    }
    return { data: Array.from(out), width: SIZE };
  }

  async function fetchAsImage(url) {
    const res = await new Promise((resolve, reject) => {
      chrome.runtime.sendMessage({ type:'FETCH_IMAGE', url }, response => {
        if (chrome.runtime.lastError) {
          return reject(new Error(chrome.runtime.lastError.message));
        }
        resolve(response);
      });
    });

    if (!res || !res.ok) {
      throw new Error(res?.error || 'No response from background');
    }

    const img = new Image();
    img.crossOrigin = 'anonymous'; // Needed for canvas access
    await new Promise((Y,N) => {
      img.onload  = Y;
      img.onerror = () => N(new Error('Image loading/decoding failed'));
      img.src     = res.dataURL; // Use the dataURL from the background script
    });

    // We still create a Blob to easily get the clean MIME type and pass it
    const blob = (() => {
      const [hdr, b64] = res.dataURL.split(',');
      // Corrected regex to capture only the MIME type before any parameters
      const match = hdr.match(/data:([^;]+);base64/);
      if (!match) {
         console.warn('[AI-Detector] Could not extract simple MIME type from data URL header:', hdr);
         // Fallback: Try to extract anything between data: and ;base64
         const fallbackMatch = hdr.match(/data:(.*);base64/);
          if (fallbackMatch && fallbackMatch[1]) {
              const mime = fallbackMatch[1];
              console.log(`[AI-Detector] Extracted fallback MIME: ${mime}`);
               // Convert base64 to ArrayBuffer for the Blob creation
              try {
                  const bin = atob(b64);
                  const arr = new Uint8Array(bin.length);
                  for (let i=0; i<bin.length; i++) arr[i] = bin.charCodeAt(i);
                   return new Blob([arr], { type: mime });
               } catch(e) {
                   console.error('[AI-Detector] Failed to decode base64 for fallback blob:', e);
                   throw new Error('Failed to decode base64 for fallback blob.');
               }
          } else {
              throw new Error('Could not extract MIME type from data URL header');
          }
      }
      const mime = match[1]; // Capture group 1: the part before the first ;
      console.log(`[AI-Detector] Extracted clean MIME: ${mime}`);

      // Convert base64 to ArrayBuffer for the Blob
      try {
          const bin = atob(b64);
          const arr = new Uint8Array(bin.length);
          for (let i=0; i<bin.length; i++) arr[i] = bin.charCodeAt(i);
          return new Blob([arr], { type: mime });
      } catch(e) {
           console.error('[AI-Detector] Failed to decode base64 for blob:', e);
           throw new Error('Failed to decode base64 for blob.');
      }
    })();

    // We return the original dataURL as well, which includes the base64 data
    return { img, blob, dataURL: res.dataURL };
  }

  function addBadge(el, txt, bg) {
    // Remove existing badges on this element's parent
    el.parentElement?.querySelectorAll('.ai-detector-badge').forEach(b=>b.remove());
    const d = document.createElement('div');
    d.className   = 'ai-detector-badge';
    d.textContent = txt;
    Object.assign(d.style, {
      position:'absolute', top:'6px', right:'6px',
      background:bg, color:'#fff',
      padding:'2px 6px', fontSize:'11px',
      fontWeight:600, borderRadius:'4px',
      pointerEvents:'none', zIndex:9999
    });
    const p = el.parentElement;
    // Ensure parent has position: relative for absolute positioning of the badge
    if (p && getComputedStyle(p).position === 'static') {
        p.style.position = 'relative';
    }
     // Append badge to the parent or body if parent is not suitable
    (p || document.body).appendChild(d);
    return d;
  }

  async function classifyWithOpenAI(dataURL, mimeType) {
    // The dataURL is already in the format "data:mime/type;base64,..."
    // OpenAI API needs this format directly in the image_url field

    console.log('[AI-Detector content] Sending dataURL to background (truncated):', dataURL.substring(0, 100) + '...');
    console.log('[AI-Detector content] Sending mime type to background:', mimeType);


    const res = await new Promise(r =>
      chrome.runtime.sendMessage({
        type: 'OPENAI_CLASSIFY',
        dataUrl: dataURL, // Send the full data URL string
        mimeType: mimeType // Send the clean MIME type (redundant but good practice)
      }, r)
    );
    if (!res || !res.ok) {
      throw new Error(res?.error || 'OpenAI did not respond');
    }
    return res; // { label, text }
  }

  document.addEventListener('mouseover', async ev => {
    const imgEl = ev.target.closest('img');
    // Use a unique ID or a flag to prevent processing the same image multiple times quickly
    if (!imgEl || imgEl.dataset.aiChecked) return;
     // Mark the element immediately to prevent re-triggering while processing
    imgEl.dataset.aiChecked = 'pending'; // Use 'pending' state

    const spin = addBadge(imgEl,'…','#444');
    try {
      // Fetch image and get dataURL, decoded image, and blob
      const { img: decoded, blob, dataURL } = await fetchAsImage(imgEl.currentSrc||imgEl.src);

      // --- ONNX Analysis ---
      const payload = toPayload(decoded);
      const onnxId = crypto.randomUUID(); // Separate ID for ONNX pending request
      pending.set(onnxId, { imgEl, spin, blob, dataURL }); // Store dataURL for OpenAI later
      // Post message to page context for ONNX analysis
      window.postMessage({ type:'AI_DETECTOR_ANALYZE', id: onnxId, imageData: payload }, '*');

      

    } catch(e) {
      console.warn('[AI-Detector] fetch/decode error:', e);
       // Update badge on error
      spin.textContent = 'N/A';
      spin.style.background = 'gray';
      // Mark as checked even on error, to prevent infinite retries on bad images
      imgEl.dataset.aiChecked = 'error'; // Use 'error' state

      // Clean up any pending entry related to this image if it was added before the error
       for (const [id, entry] of pending.entries()) {
           if (entry.imgEl === imgEl) {
               pending.delete(id);
           }
       }
    }
  }, true);

  window.addEventListener('message', async ev => {
    // Handle messages from the page context (e.g., ONNX results)
    if (ev.data?.type!=='AI_DETECTOR_RESULT') return;

    const onnxId = ev.data.id;
    const ent = pending.get(onnxId);
    // Don't delete pending yet, might need dataURL for OpenAI
    if (!ent) return;

    const { imgEl, spin, blob, dataURL } = ent; // Retrieve stored dataURL

    // ONNX results
    // Ensure data exists before processing
    if (!ev.data.logits) {
         console.error('[AI-Detector] ONNX result missing logits', ev.data);
         // Handle missing ONNX result: remove spinner, show error state or N/A badge
         spin.remove();
         addBadge(imgEl, 'ONNX Err', 'gray');
         imgEl.dataset.aiChecked = 'onnx-error';
         pending.delete(onnxId); // Clean up
         return;
    }

    spin.remove(); // Remove spinner after receiving ONNX result (before OpenAI call)

    // --- ONNX Calculation (keep this if you still want to use ONNX data later) ---
    const probAI = softmax(ev.data.logits)[1];
    let modelLabel, modelConf;
    if (probAI >= THRESH) modelLabel = 'ai';
    else if ((1 - probAI) >= THRESH) modelLabel = 'human';
    else modelLabel = 'uncertain';

    // I did this to check onnx result before using the API resul
    // --- Display ONNX result initially ---
    // COMMENTED OUT to only show the OpenAI result
    /*
    if (modelLabel === 'ai') addBadge(imgEl, `AI ${(modelConf * 100).toFixed(1)}%`, 'red');
    else if (modelLabel === 'human') addBadge(imgEl, `Human ${((modelConf) * 100).toFixed(1)}%`, 'green');
    else addBadge(imgEl, `${(modelConf * 100).toFixed(1)}%`, 'gold');
    */
    // -------------------------------------


    // --- OpenAI Classification ---
    // Now attempt OpenAI classification using the stored dataURL
    try {
      // Use the dataURL and mimeType obtained from fetchAsImage
      // The dataURL is already base64 encoded in the format OpenAI expects
      const api = await classifyWithOpenAI(dataURL, blob.type); // Use dataURL and blob.type
      console.log('[AI-Detector][OpenAI] raw response:', api);
      console.log('[AI-Detector][OpenAI] label:', api.label);

      // Update badge with OpenAI result (Keep this section)
      const final = api.label;
      const color = final === 'ai' ? 'red' : 'green';
      addBadge(imgEl, final === 'ai' ? 'AI' : 'Human', color); // This adds/updates the badge

      imgEl.dataset.aiChecked = final; // Mark as checked with the final label

    } catch(e) {
      console.error('[AI-Detector][OpenAI] error:', e);
      // If OpenAI fails, add an error badge
       addBadge(imgEl, 'AI Err', 'gray'); // Add an error badge if OpenAI fails
       imgEl.dataset.aiChecked = 'openai-error'; // Mark as checked with OpenAI error state

    } finally {
        // Clean up the pending entry after OpenAI classification attempt (success or failure)
        pending.delete(onnxId);
         // Ensure element is marked as checked if not already by success/error
        if (!imgEl.dataset.aiChecked || imgEl.dataset.aiChecked === 'pending') {
             imgEl.dataset.aiChecked = 'done'; // Generic done state if no specific result
        }
    }
  });

  console.log('[AI-Detector]  pipeline ready');
}