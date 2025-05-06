// background.js

// WARNING: Hardcoding API keys is a security risk if the extension is distributed.
// For personal testing only. Replace "YOUR_OPENAI_API_KEY_HERE" with your actual key.
const HARDCODED_OPENAI_KEY = '';

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    // Handle messages from content scripts
    if (msg.type === 'FETCH_IMAGE') {
        console.log(`[AI-Detector Background] Received FETCH_IMAGE for URL: ${msg.url}`);
        // Fetch the image using the background script's privileges
        fetch(msg.url, { mode: 'cors' }) // Use cors mode
            .then(resp => {
                if (!resp.ok) {
                    // Try to get more details from the response if possible, then throw
                    const errorDetail = `HTTP error ${resp.status} (${resp.statusText}) while fetching ${msg.url}`;
                    console.error('[AI-Detector Background] FETCH_IMAGE HTTP error:', errorDetail);
                    // Attempt to read response body for more details if available and not too large
                    return resp.text().then(text => {
                        const fullError = `${errorDetail}. Body: ${text.substring(0, 500)}...`; // Limit body output
                        throw new Error(fullError);
                    }).catch(() => {
                         // If reading body fails, just throw the original error
                         throw new Error(errorDetail);
                    });
                }
                // Get the response as a Blob
                return resp.blob();
            })
            // Convert Blob to Data URL in the background script
            .then(blob => new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onloadend = () => {
                    // FileReader result is the data URL (e.g., data:image/png;base64,...)
                    console.log('[AI-Detector Background] FETCH_IMAGE success for:', msg.url, 'MIME:', blob.type, 'Size:', blob.size);
                    resolve({ ok: true, dataURL: reader.result });
                };
                reader.onerror = () => {
                    console.error('[AI-Detector Background] FETCH_IMAGE FileReader error for:', msg.url, reader.error);
                    reject(new Error('FileReader failed for ' + msg.url));
                };
                 // Read the blob as a data URL
                reader.readAsDataURL(blob);
            }))
            .then(successResponse => {
                // Send the resulting data URL back to the content script
                sendResponse(successResponse);
            })
            .catch(err => {
                // Catch any errors in the fetch or file reading process
                console.error('[AI-Detector Background] Final catch in FETCH_IMAGE:', err.message);
                sendResponse({ ok: false, error: err.message });
            });
        // Return true to indicate that sendResponse will be called asynchronously
        return true;
    }

    // Handle the message containing base64 data for OpenAI classification
    if (msg.type === 'OPENAI_CLASSIFY') {
        console.log('[AI-Detector Background] Received OPENAI_CLASSIFY message.');
        // The content script now sends the full data URL string directly
        const dataUrl = msg.dataUrl;
        const mimeType = msg.mimeType; // Still good to have the mime type separately

        console.log('[AI-Detector Background] Received dataUrl length:', dataUrl ? dataUrl.length : 'undefined');
        console.log('[AI-Detector Background] Received mimeType:', mimeType);
        console.log('[AI-Detector Background] Received dataUrl (truncated):', dataUrl ? dataUrl.substring(0, 100) + '...' : 'undefined');


        // Check if the API key is set
        if (!HARDCODED_OPENAI_KEY || HARDCODED_OPENAI_KEY === "YOUR_OPENAI_API_KEY_HERE") {
            const errMsg = 'OpenAI API Key not set or is a placeholder in background.js.';
            console.error(`[AI-Detector Background] ${errMsg}`);
            sendResponse({ ok: false, error: errMsg });
            return false; // Not async if key is missing
        }

        // Validate that dataUrl and mimeType are present
        if (!dataUrl || !mimeType) {
             const errMsg = 'Missing dataUrl or mime type in OPENAI_CLASSIFY message.';
             console.error('[AI-Detector Background] ' + errMsg, msg);
             sendResponse({ ok: false, error: errMsg });
             return false; // Not async if data is missing
        }

        // The dataUrl is already in the format "data:mime/type;base64,..." which OpenAI expects
        const imageUrl = dataUrl;

        // Log the data URL being sent to OpenAI (truncated)
        console.log('[AI-Detector Background] Data URL constructed for OpenAI (truncated):', imageUrl.substring(0, 100) + '...');

        // Make the fetch request to the OpenAI API
        fetch('https://api.openai.com/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${HARDCODED_OPENAI_KEY}`,
                'Content-Type':  'application/json'
            },
            // Include the image_url in the messages payload
            body: JSON.stringify({
                model:       'gpt-4o', // Use a vision-capable model
                temperature: 0,       // Keep temperature low for deterministic classification
                max_tokens:  10,      // Limit response length as we expect "AI" or "Human"
                messages: [
                    {
                        role: 'system',
                        content: 'You are an AI forensic assistant. Given an image, respond with exactly one word: "AI" or "Human". Do not include any other text, explanation, or punctuation.'
                    },
                    {
                        role: 'user',
                        content: [
                            { type: "text", text: "Please classify this image." },
                            { type: "image_url", "image_url": { "url": imageUrl } } // Pass the data URL here
                        ]
                    }
                ]
            })
        })
        .then(response => {
            // Check for HTTP errors
            if (!response.ok) {
                // Attempt to parse error body for more details
                return response.json().then(errBody => {
                    const errMsg = `OpenAI API Error: ${response.status} ${errBody.error?.message || JSON.stringify(errBody)}`;
                    console.error('[AI-Detector Background] OpenAI API HTTP Error:', errMsg, 'Full body:', errBody);
                    throw new Error(errMsg);
                }).catch(jsonParseError => {
                    // Handle cases where the error body cannot be parsed
                    const errMsg = `OpenAI API Error: ${response.status} ${response.statusText} (Error body parsing failed: ${jsonParseError.message})`;
                    console.error('[AI-Detector Background] OpenAI API HTTP Error (could not parse error body):', errMsg);
                    throw new Error(errMsg);
                });
            }
            // Parse the successful JSON response
            return response.json();
        })
        .then(json => {
            console.log('[AI-Detector Background] OpenAI API raw response:', json);
            // Extract and process the response text
            const rawText = json.choices?.[0]?.message?.content || '';
            const Txt = rawText.trim().replace(/[.,!"'`]/g, '').toLowerCase(); // Clean and lowercase
            let label = Txt === 'ai' ? 'ai' : 'human'; // Determine label based on cleaned text
            console.log(`[AI-Detector Background] OpenAI API processed text: "${Txt}", Determined label: "${label}"`);
            // Send the result back to the content script
            sendResponse({ ok: true, label: label, text: rawText.trim() }); // Send back original text too
        })
        .catch(err => {
            // Catch any errors during the fetch or processing
            console.error('[AI-Detector Background] Final catch in OPENAI_CLASSIFY:', err.message, err);
            sendResponse({ ok: false, error: err.message });
        });
         // Return true to indicate that sendResponse will be called asynchronously
        return true;
    }

    // Optional: Handle unknown message types if necessary
    // console.log('[AI-Detector Background] Received unknown message type:', msg.type);
    // sendResponse({ ok: false, error: 'Unknown message type received by background script.' });
    // return false; // if sendResponse is called synchronously for unknown types
});

// Log to confirm the script itself loaded without immediate syntax errors
console.log('[AI-Detector Background] Service worker script loaded and listener attached.');
