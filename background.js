chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'detect') {
      // Call backend API to process the text
      fetch('http://localhost:5000/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: request.text })
      })
      .then(response => response.json())
      .then(data => {
        sendResponse({ label: data.label, probability: data.probability });
      })
      .catch(error => {
        console.error('Error:', error);
        sendResponse({ error: 'Detection failed' });
      });
      return true; // Keep the message channel open
    }
  });