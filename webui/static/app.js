const API_BASE = window.location.origin;
const API_DIRECT = `${window.location.protocol}//${window.location.hostname}:8000`;
const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const tempSlider = document.getElementById('temperature');
const tempValue = document.getElementById('temp-value');
const tokensSlider = document.getElementById('max-tokens');
const tokensValue = document.getElementById('tokens-value');
const historyEl = document.getElementById('chat-history');
const newChatBtn = document.getElementById('new-chat');
const clearChatBtn = document.getElementById('clear-chat');

let messages = [];
let chatHistory = [];
let currentChatId = Date.now().toString();
let isGenerating = false;
let sessionId = crypto.randomUUID();
let selectedFile = null;

const uploadBtn = document.getElementById('upload-btn');
const fileInput = document.getElementById('file-input');
const filePreview = document.getElementById('file-preview');
const previewName = document.getElementById('preview-name');
const removeFileBtn = document.getElementById('remove-file');

uploadBtn.onclick = () => fileInput.click();
fileInput.onchange = (e) => handleFileSelect(e.target.files[0]);
removeFileBtn.onclick = clearSelectedFile;

function handleFileSelect(file) {
    if (!file) return;
    selectedFile = file;
    previewName.textContent = file.name;
    filePreview.style.display = 'flex';
}

function clearSelectedFile() {
    selectedFile = null;
    fileInput.value = '';
    filePreview.style.cssText = 'display: none !important';
    previewName.textContent = '';
}

tempSlider.oninput = () => tempValue.textContent = tempSlider.value;
tokensSlider.oninput = () => tokensValue.textContent = tokensSlider.value;

inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

sendBtn.onclick = sendMessage;
newChatBtn.onclick = newChat;
clearChatBtn.onclick = clearChat;

function addMessage(role, content, file = null) {
    const welcome = messagesEl.querySelector('.welcome');
    if (welcome) welcome.remove();

    let fileHtml = '';
    if (file) {
        if (file.type && file.type.startsWith('image/')) {
            const url = URL.createObjectURL(file);
            fileHtml = `<div class="msg-file"><img src="${url}" style="max-width:300px;max-height:200px;border-radius:8px;margin-top:8px"></div>`;
        } else {
            fileHtml = `<div class="msg-file"><span class="file-icon">📄</span> ${file.name}</div>`;
        }
    }

    const div = document.createElement('div');
    div.className = `message ${role}`;
    div.innerHTML = `
        <div class="avatar">${role === 'user' ? 'U' : 'AI'}</div>
        <div class="content">${fileHtml}${escapeHtml(content).replace(/\n/g, '<br>')}</div>
    `;
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return div;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatResponse(text) {
    let html = text;
    html = html.replace(/&/g, '&amp;');
    html = html.replace(/</g, '&lt;');
    html = html.replace(/>/g, '&gt;');
    html = html.replace(/&lt;think&gt;([\s\S]*?)&lt;\/think&gt;/g, '<div class="thinking-block"><strong>Thinking:</strong>$1</div>');
    html = html.replace(/\n\n/g, '</p><p>');
    html = html.replace(/\n/g, '<br>');
    html = '<p>' + html + '</p>';
    return html;
}

async function sendMessage() {
    const text = inputEl.value.trim();
    if (!text && !selectedFile) return;
    if (isGenerating) return;

    const hasFile = selectedFile !== null;
    const isImage = hasFile && selectedFile.type.startsWith('image/');
    const isDoc = hasFile && !isImage;

    isGenerating = true;
    sendBtn.disabled = true;
    inputEl.value = '';
    inputEl.style.height = 'auto';

    let userMsg = text || (isImage ? 'Describe this image' : `Process ${selectedFile.name}`);
    const fileToSend = selectedFile;
    messages.push({ role: 'user', content: userMsg });
    addMessage('user', userMsg, hasFile ? fileToSend : null);
    clearSelectedFile();

    const assistantDiv = document.createElement('div');
    assistantDiv.className = 'message assistant';
    assistantDiv.innerHTML = `
        <div class="avatar">AI</div>
        <div class="content"><div class="typing-indicator"><span></span><span></span><span></span></div></div>
    `;
    messagesEl.appendChild(assistantDiv);
    messagesEl.scrollTop = messagesEl.scrollHeight;

    const contentEl = assistantDiv.querySelector('.content');

    try {
        if (isImage) {
            await uploadImage(fileToSend, userMsg, contentEl);
        } else if (isDoc) {
            await uploadDocument(fileToSend, contentEl);
        } else {
            await streamResponse(text, contentEl);
        }
    } catch (err) {
        contentEl.textContent = 'Error: ' + err.message;
    } finally {
        isGenerating = false;
        sendBtn.disabled = false;
        messagesEl.scrollTop = messagesEl.scrollHeight;
        saveToHistory(userMsg);
    }
}

async function uploadImage(file, question, contentEl) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('message', question || 'Describe this image');
    formData.append('max_length', tokensSlider.value);
    formData.append('temperature', tempSlider.value);

    contentEl.innerHTML = 'Analyzing image...';

    const res = await fetch(`${API_DIRECT}/chat/image/upload?session_id=${sessionId}&_t=${Date.now()}`, {
        method: 'POST',
        body: formData
    });

    const data = await res.json();
    const reply = String(data.text || (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)) || 'No response');
    contentEl.innerHTML = formatResponse(reply);
    messages.push({ role: 'assistant', content: reply });
}

async function uploadDocument(file, contentEl) {
    const formData = new FormData();
    formData.append('file', file);

    contentEl.textContent = 'Uploading document...';

    const res = await fetch(`${API_DIRECT}/knowledge/upload`, {
        method: 'POST',
        body: formData
    });

    const data = await res.json();
    if (data.detail) {
        contentEl.textContent = 'Error: ' + data.detail;
    } else {
        contentEl.innerHTML = `<strong>Document uploaded:</strong> ${data.filename}<br>Chunks: ${data.chunk_count}`;
        messages.push({ role: 'assistant', content: `Document "${data.filename}" uploaded with ${data.chunk_count} chunks.` });
    }
}

async function loadModels() {
    try {
        const res = await fetch(`${API_DIRECT}/v1/models`);
        const data = await res.json();
        const select = document.getElementById('model-select');
        select.innerHTML = '';
        if (data.models && data.models.length > 0) {
            data.models.forEach(model => {
                const opt = document.createElement('option');
                opt.value = model.name;
                opt.textContent = model.name + (model.loaded ? '' : ' (not loaded)');
                select.appendChild(opt);
            });
        } else {
            const opt = document.createElement('option');
            opt.value = 'chat';
            opt.textContent = 'chat';
            select.appendChild(opt);
        }
    } catch (e) {
        console.error('Failed to load models:', e);
    }
}

document.getElementById('model-select').onchange = (e) => {
    console.log('Model selected:', e.target.value);
};

loadModels();

async function streamResponse(text, contentEl) {
    const res = await fetch(`${API_DIRECT}/v1/chat/completions?session_id=${sessionId}&_t=${Date.now()}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            model: 'chat',
            messages: messages,
            temperature: parseFloat(tempSlider.value),
            max_tokens: parseInt(tokensSlider.value),
            stream: true
        })
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let fullText = '';
    let buffer = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed.startsWith('data: ')) {
                const data = trimmed.slice(6);
                if (data === '[DONE]') continue;
                    try {
                        const parsed = JSON.parse(data);
                        const delta = parsed.choices[0].delta;
                        if (delta && delta.content) {
                            fullText += delta.content;
                            contentEl.innerHTML = formatResponse(fullText);
                            messagesEl.scrollTop = messagesEl.scrollHeight;
                        }
                    } catch (e) {}
            }
        }
    }

    messages.push({ role: 'assistant', content: fullText });
}

function newChat() {
    if (messages.length > 0) {
        saveChatToHistory();
    }
    messages = [];
    currentChatId = Date.now().toString();
    sessionId = crypto.randomUUID();
    messagesEl.innerHTML = '<div class="welcome"><h1>OvService</h1><p>Powered by OpenVINO GenAI</p><p>Start chatting below</p></div>';
}

function clearChat() {
    messages = [];
    currentChatId = Date.now().toString();
    sessionId = crypto.randomUUID();
    messagesEl.innerHTML = '<div class="welcome"><h1>OvService</h1><p>Powered by OpenVINO GenAI</p><p>Start chatting below</p></div>';
}

function saveToHistory(lastUserMsg) {
    const existing = chatHistory.find(c => c.id === currentChatId);
    if (existing) {
        existing.messages = [...messages];
    } else {
        chatHistory.unshift({
            id: currentChatId,
            title: lastUserMsg.substring(0, 30) + (lastUserMsg.length > 30 ? '...' : ''),
            messages: [...messages]
        });
    }
    renderHistory();
}

function saveChatToHistory() {
    const existing = chatHistory.find(c => c.id === currentChatId);
    if (!existing && messages.length > 0) {
        chatHistory.unshift({
            id: currentChatId,
            title: messages[0].content.substring(0, 30) + '...',
            messages: [...messages]
        });
        renderHistory();
    }
}

function renderHistory() {
    historyEl.innerHTML = chatHistory.map(chat => `
        <div class="history-item" onclick="loadChat('${chat.id}')">${chat.title}</div>
    `).join('');
}

window.loadChat = function(chatId) {
    const chat = chatHistory.find(c => c.id === chatId);
    if (!chat) return;
    currentChatId = chatId;
    messages = [...chat.messages];
    messagesEl.innerHTML = '';
    messages.forEach(msg => addMessage(msg.role, msg.content));
};

async function checkHealth() {
    try {
        const res = await fetch(`${API_DIRECT}/health`);
        const data = await res.json();
        const btn = document.getElementById('model-status');
        if (data.model_loaded) {
            btn.textContent = 'Connected';
            btn.style.background = '#27ae60';
        } else {
            btn.textContent = 'No Model';
            btn.style.background = '#e74c3c';
        }
    } catch (e) {
        const btn = document.getElementById('model-status');
        btn.textContent = 'Offline';
        btn.style.background = '#e74c3c';
    }
}

checkHealth();
setInterval(checkHealth, 30000);
