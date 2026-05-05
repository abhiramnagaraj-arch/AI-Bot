const textarea = document.getElementById("user-input");

// Auto-resize logic for textarea
textarea.addEventListener("input", function() {
  this.style.height = "auto"; // Reset height to recalculate
  this.style.height = (this.scrollHeight) + "px";
});

// Event listener for Enter to send, Shift+Enter for newline
textarea.addEventListener("keydown", function(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault(); // Prevent newline
    sendMessage();
  }
});

async function sendMessage() {
  const input = document.getElementById("user-input");
  const chatBox = document.getElementById("chat-box");
  const sendBtn = document.getElementById("send-btn");

  const userText = input.value.trim();
  if (!userText) {
    return;
  }

  // Disable UI
  input.disabled = true;
  sendBtn.disabled = true;

  // Add user message
  addMessage(userText, 'user');
  
  // Clear input and reset height
  input.value = "";
  input.style.height = "auto";

  // Loading indicator
  const loading = document.createElement("div");
  loading.className = "message bot";
  loading.innerHTML = `
    <div class="typing-dots">
      <span></span><span></span><span></span>
    </div>
  `;
  chatBox.appendChild(loading);
  scrollToBottom();

  try {
    const startTime = Date.now();
    const response = await fetch("http://localhost:8000/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        query: userText,
        session_id: "user123"
      })
    });

    const data = await response.json();
    const timeMs = Date.now() - startTime;
    const timeSec = (timeMs / 1000).toFixed(1);

    loading.innerText = `${data.answer}\n\n⏱️ Response time: ${timeSec}s`;
    console.log(`Response time: ${timeMs}ms`);

  } catch (error) {
    loading.innerText = "Error connecting to server. Please ensure the backend is running on port 8000.";
  } finally {
    // Re-enable UI
    input.disabled = false;
    sendBtn.disabled = false;
    input.focus();
  }

  scrollToBottom();
}

function addMessage(text, side) {
  const chatBox = document.getElementById("chat-box");
  const msgDiv = document.createElement("div");
  msgDiv.className = `message ${side}`;
  msgDiv.innerText = text; // white-space: pre-wrap handles newlines
  chatBox.appendChild(msgDiv);
  scrollToBottom();
}

function scrollToBottom() {
  const chatBox = document.getElementById("chat-box");
  chatBox.scrollTop = chatBox.scrollHeight;
}
