let youtubeDetails = { open: false, classification: "" };

// Function to send data to the API
function sendScreenshotToAPI(image) {
    fetch('http://127.0.0.1:5000/classify', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: image })
    })
    .then(response => response.json())
    .then(data => {
        youtubeDetails.classification = data.classification ? 'productive' : 'unproductive';
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}

// Function to update YouTube details
function updateYouTubeDetails(tabId, changeInfo, tab) {
    if (tab.url && tab.url.includes("youtube.com/watch")) {
        youtubeDetails.open = true;
        takeScreenshot();
    } else {
        youtubeDetails.open = false;
        youtubeDetails.classification = "";
    }
}

// Function to take a screenshot and send it to the API
function takeScreenshot() {
    chrome.tabs.captureVisibleTab(null, {}, (image) => {
        sendScreenshotToAPI(image);
    });
}

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (changeInfo.status === 'complete') {
        updateYouTubeDetails(tabId, changeInfo, tab);
    }
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.command === "checkYouTube") {
        sendResponse(youtubeDetails);
    }
});


chrome.tabs.onActivated.addListener(activeInfo => {
    chrome.tabs.get(activeInfo.tabId, tab => {
        updateYouTubeDetails(activeInfo.tabId, {}, tab);
    });
});

// Set an interval to update the screenshot every minute
setInterval(() => {
    if (youtubeDetails.open) {
        takeScreenshot();
    }
}, 60000);
