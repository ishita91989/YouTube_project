document.addEventListener('DOMContentLoaded', function() {
    chrome.runtime.sendMessage({ command: "checkYouTube" }, function(details) {
        if (chrome.runtime.lastError) {
            console.error(chrome.runtime.lastError);
            return;
        }
        const status = document.getElementById('status');
        if (details && details.open) {
            const classification = details.classification;
            status.textContent = `Classification: ${classification}`;
            status.style.backgroundColor = classification === 'productive' ? 'green' : 'red';
            status.style.color = 'white';
        } else {
            status.textContent = "No YouTube video detected.";
            status.style.backgroundColor = 'transparent';
            status.style.color = 'black';
        }
    });
});
