function sendMessage() {
    const input = document.getElementById("user-input");
    const chatBox = document.getElementById("chat-box");
    const message = input.value.trim();

    if (!message) return;

    // User message
    chatBox.innerHTML += `
        <div class="message user">${message}</div>
    `;
    input.value = "";
    chatBox.scrollTop = chatBox.scrollHeight;

    // Typing indicator
    const typingDiv = document.createElement("div");
    typingDiv.className = "message bot typing";
    typingDiv.innerText = "🤖 Thinking...";
    chatBox.appendChild(typingDiv);
    chatBox.scrollTop = chatBox.scrollHeight;

    // Send to backend
    fetch("/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ message: message })
    })
    .then(res => res.json())
    .then(data => {
        typingDiv.remove();
        chatBox.innerHTML += `
            <div class="message bot">${data.reply}</div>
        `;
        chatBox.scrollTop = chatBox.scrollHeight;
    })
    .catch(() => {
        typingDiv.innerText = "⚠️ Error getting response";
    });
}
