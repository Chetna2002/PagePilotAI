
console.log("PagePilot AI service worker started!");

chrome.runtime.onInstalled.addListener(() => {
    console.log("PagePilot AI installed!");

    chrome.sidePanel.setPanelBehavior({
        openPanelOnActionClick: true
    });
});

// Find the active webpage from the Chrome browser window
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {

    if (message.type === "GET_ACTIVE_TAB") {

        chrome.windows.getLastFocused(
            { windowTypes: ["normal"], populate: true },
            (window) => {

                if (chrome.runtime.lastError) {
                    sendResponse({
                        error: chrome.runtime.lastError.message
                    });
                    return;
                }

                const activeTab = window?.tabs?.find(tab => tab.active);

                if (!activeTab) {
                    sendResponse({
                        error: "No active Chrome tab found."
                    });
                    return;
                }

                console.log("Active tab detected:", activeTab.url);

                sendResponse({
                    tabId: activeTab.id,
                    url: activeTab.url,
                    title: activeTab.title
                });
            }
        );

        return true;
    }
});
