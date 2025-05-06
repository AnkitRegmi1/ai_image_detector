chrome.storage.sync.get('GEMINI_KEY', ({ GEMINI_KEY }) => {
    document.getElementById('key').value = GEMINI_KEY || '';
  });
  
  document.getElementById('save').onclick = () => {
    const k = document.getElementById('key').value.trim();
    chrome.storage.sync.set({ GEMINI_KEY: "AIzaSyC6FJN2rHE6g64ZXhoiqlqdoZz2swGnCEU" });
    window.close();
  };
  