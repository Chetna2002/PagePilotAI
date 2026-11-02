
console.log("PagePilot AI loaded!");

const readPageButton = document.getElementById("readPage");
const sendQuestionButton = document.getElementById("sendQuestion");

const pageTitle = document.getElementById("pageTitle");
const pageStatus = document.getElementById("pageStatus");
const pagePreview = document.getElementById("pagePreview");

const questionInput = document.getElementById("questionInput");
const chatMessages = document.getElementById("chatMessages");

let currentPageText = "";
let currentPageUrl = "";


function addMessage(message, sender) {
    const messageElement = document.createElement("div");

    messageElement.className = `message ${sender}`;
    messageElement.textContent = message;

    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}


// READ WEBPAGE
readPageButton.addEventListener("click", async () => {

    try {
        readPageButton.disabled = true;
        readPageButton.textContent = "Reading...";
        pageStatus.textContent = "Finding your webpage...";

        // Find the Chrome window containing the side panel
        const currentWindow = await chrome.windows.getCurrent();

        // Find the active tab in that Chrome window
        const tabs = await chrome.tabs.query({
            active: true,
            windowId: currentWindow.id
        });

        console.log("Detected tabs:", tabs);

        if (!tabs || tabs.length === 0 || !tabs[0].id) {
            throw new Error("No active browser tab found.");
        }

        const tab = tabs[0];

        console.log("Selected tab URL:", tab.url);

        if (!tab.url ||
            (!tab.url.startsWith("http://") &&
             !tab.url.startsWith("https://"))) {

            throw new Error(
                "Detected tab is not a normal website. Open a website like https://www.wikipedia.org and try again."
            );
        }

        pageStatus.textContent = "Reading webpage...";

        // Extract webpage content
        const results = await chrome.scripting.executeScript({
            target: { tabId: tab.id },
            func: () => ({
                title: document.title,
                url: location.href,
                text: document.body?.innerText || ""
            })
        });

        const pageData = results[0].result;

        currentPageText = pageData.text.slice(0, 12000);
        currentPageUrl = pageData.url;

        if (!currentPageText.trim()) {
            throw new Error("No readable text found on this webpage.");
        }

        pageTitle.textContent = pageData.title;

        pageStatus.textContent =
            `Successfully read ${currentPageText.length} characters.`;

        pagePreview.textContent =
            currentPageText.slice(0, 1500);

        addMessage(
            "I've read your webpage! Ask me anything.",
            "assistant"
        );

    } catch (error) {

        console.error("Page reading error:", error);

        pageStatus.textContent = error.message;

        addMessage(
            "Unable to read webpage: " + error.message,
            "assistant"
        );

    } finally {

        readPageButton.disabled = false;
        readPageButton.textContent = "Read this page";
    }
});


// ASK AI
async function askPagePilot() {

    const question = questionInput.value.trim();

    if (!question) {
        alert("Please enter a question.");
        return;
    }

    if (!currentPageText) {
        alert("Please read a webpage first.");
        return;
    }

    addMessage(question, "user");
    questionInput.value = "";

    sendQuestionButton.disabled = true;
    sendQuestionButton.textContent = "Thinking...";

    try {

        const response = await fetch(
            "http://127.0.0.1:8000/chat",
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    question: question,
                    page_text: currentPageText
                })
            }
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Backend request failed.");
        }

        addMessage(data.answer, "assistant");

    } catch (error) {

        console.error("AI request error:", error);

        addMessage(
            "AI error: " + error.message,
            "assistant"
        );

    } finally {

        sendQuestionButton.disabled = false;
        sendQuestionButton.textContent = "Ask AI";
    }
}


sendQuestionButton.addEventListener("click", askPagePilot);


questionInput.addEventListener("keydown", (event) => {

    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        askPagePilot();
    }

});
