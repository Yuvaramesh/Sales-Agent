// Global state
let userEmail = "";
let sessionId = null;

// DOM Elements
const welcomeModal = document.getElementById("welcomeModal");
const welcomeForm = document.getElementById("welcomeForm");
const chatContainer = document.getElementById("chatContainer");
const messagesContainer = document.getElementById("messagesContainer");
const messageForm = document.getElementById("messageForm");
const messageInput = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const typingIndicator = document.getElementById("typingIndicator");
const userEmailDisplay = document.getElementById("userEmailDisplay");
const endSessionBtn = document.getElementById("endSessionBtn");
const loadingOverlay = document.getElementById("loadingOverlay");
const todayDate = document.getElementById("todayDate");

// Initialize
document.addEventListener("DOMContentLoaded", () => {
  setTodayDate();
  setupEventListeners();
  adjustTextareaHeight();
});

// Set today's date
function setTodayDate() {
  const today = new Date();
  const options = {
    weekday: "long",
    year: "numeric",
    month: "long",
    day: "numeric",
  };
  todayDate.textContent = today.toLocaleDateString("en-US", options);
}

// Setup event listeners
function setupEventListeners() {
  // Welcome form submission
  welcomeForm.addEventListener("submit", handleWelcomeSubmit);

  // Message form submission
  messageForm.addEventListener("submit", handleMessageSubmit);

  // Message input changes
  messageInput.addEventListener("input", () => {
    adjustTextareaHeight();
    toggleSendButton();
  });

  // Handle Enter key (Shift+Enter for new line)
  messageInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (messageInput.value.trim()) {
        messageForm.dispatchEvent(new Event("submit"));
      }
    }
  });

  // End session button
  endSessionBtn.addEventListener("click", handleEndSession);
}

// Handle welcome form submission
async function handleWelcomeSubmit(e) {
  e.preventDefault();
  const emailInput = document.getElementById("userEmail");
  const email = emailInput.value.trim();

  if (!email || !isValidEmail(email)) {
    showError("Please enter a valid email address");
    return;
  }

  userEmail = email;
  userEmailDisplay.textContent = email;

  // Animate transition
  welcomeModal.classList.remove("active");
  setTimeout(() => {
    chatContainer.classList.add("active");
    messageInput.focus();
  }, 300);
}

// Handle message submission
async function handleMessageSubmit(e) {
  e.preventDefault();

  const message = messageInput.value.trim();
  if (!message) return;

  // Disable input while sending
  setInputState(false);

  // Add user message to chat
  addMessage(message, "user");

  // Clear input
  messageInput.value = "";
  adjustTextareaHeight();

  // Show typing indicator
  typingIndicator.classList.add("active");
  scrollToBottom();

  try {
    // Send message to backend
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        user_email: userEmail,
        message: message,
        session_id: sessionId,
      }),
    });

    if (!response.ok) {
      throw new Error("Failed to send message");
    }

    const data = await response.json();

    // Store session ID
    if (data.session_id) {
      sessionId = data.session_id;
    }

    // Hide typing indicator
    typingIndicator.classList.remove("active");

    // Add bot response
    if (data.response) {
      addMessage(data.response, "bot");
    }
  } catch (error) {
    console.error("Error:", error);
    typingIndicator.classList.remove("active");
    addMessage("Sorry, I encountered an error. Please try again.", "bot");
  } finally {
    setInputState(true);
    messageInput.focus();
  }
}

// Add message to chat
function addMessage(text, type) {
  const messageDiv = document.createElement("div");
  messageDiv.className = `message ${type}-message`;

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";

  if (type === "bot") {
    avatar.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
                <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
            </svg>
        `;
  } else {
    avatar.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
                <circle cx="12" cy="7" r="4"/>
            </svg>
        `;
  }

  const content = document.createElement("div");
  content.className = "message-content";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.textContent = text;

  const time = document.createElement("div");
  time.className = "message-time";
  time.textContent = getCurrentTime();

  content.appendChild(bubble);
  content.appendChild(time);

  messageDiv.appendChild(avatar);
  messageDiv.appendChild(content);

  messagesContainer.appendChild(messageDiv);
  scrollToBottom();
}

// Handle end session
async function handleEndSession() {
  if (!confirm("Are you sure you want to end this session?")) {
    return;
  }

  loadingOverlay.classList.add("active");

  try {
    const response = await fetch("/api/end-session", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        user_email: userEmail,
        session_id: sessionId,
      }),
    });

    if (response.ok) {
      // Reset everything
      sessionId = null;
      userEmail = "";
      messagesContainer.innerHTML = `
                <div class="date-divider">
                    <span id="todayDate"></span>
                </div>
                <div class="message bot-message animate-in">
                    <div class="message-avatar">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
                            <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
                        </svg>
                    </div>
                    <div class="message-content">
                        <div class="message-bubble">
                            Hello! 👋 I'm AutoBot, your personal car buying assistant. How can I help you find your dream car today?
                        </div>
                        <div class="message-time"></div>
                    </div>
                </div>
            `;
      setTodayDate();

      // Show welcome modal again
      chatContainer.classList.remove("active");
      setTimeout(() => {
        welcomeModal.classList.add("active");
      }, 300);
    }
  } catch (error) {
    console.error("Error ending session:", error);
    alert("Failed to end session. Please try again.");
  } finally {
    loadingOverlay.classList.remove("active");
  }
}

// Utility functions
function adjustTextareaHeight() {
  messageInput.style.height = "auto";
  messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + "px";
}

function toggleSendButton() {
  sendBtn.disabled = !messageInput.value.trim();
}

function setInputState(enabled) {
  messageInput.disabled = !enabled;
  sendBtn.disabled = !enabled || !messageInput.value.trim();
}

function scrollToBottom() {
  setTimeout(() => {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }, 100);
}

function getCurrentTime() {
  const now = new Date();
  return now.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function isValidEmail(email) {
  const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return re.test(email);
}

function showError(message) {
  alert(message);
}

// Auto-scroll on new messages
const observer = new MutationObserver(scrollToBottom);
observer.observe(messagesContainer, { childList: true });
