/**
 * EAUT Admission Chatbot - Frontend Client Logic
 * Modern vanilla JS with streaming, markdown formatting, TTS, STT, and responsive handbook
 */

// State Management
const state = {
  isStreaming: false,
  soundEnabled: true,
  theme: localStorage.getItem('eaut_theme') || 'dark',
  systemReady: false,
  sampleCategories: [],
  activeCategory: 'all',
  handbookData: null,
  speechSynth: window.speechSynthesis || null,
  currentUtterance: null,
  isListening: false,
  recognition: null
};

// DOM Selectors
const DOM = {
  messagesFeed: document.getElementById('messagesFeed'),
  welcomeHero: document.getElementById('welcomeHero'),
  chatThread: document.getElementById('chatThread'),
  typingIndicator: document.getElementById('typingIndicator'),
  messageInput: document.getElementById('messageInput'),
  btnSendMessage: document.getElementById('btnSendMessage'),
  btnVoiceInput: document.getElementById('btnVoiceInput'),
  btnClearChat: document.getElementById('btnClearChat'),
  btnToggleTheme: document.getElementById('btnToggleTheme'),
  btnToggleSound: document.getElementById('btnToggleSound'),
  iconSun: document.getElementById('iconSun'),
  iconMoon: document.getElementById('iconMoon'),
  iconSoundOn: document.getElementById('iconSoundOn'),
  iconSoundOff: document.getElementById('iconSoundOff'),
  systemStatusBadge: document.getElementById('systemStatusBadge'),
  statusText: document.getElementById('statusText'),
  devicePill: document.getElementById('devicePill'),
  kbDocCount: document.getElementById('kbDocCount'),
  sidebar: document.getElementById('sidebar'),
  btnToggleSidebar: document.getElementById('btnToggleSidebar'),
  btnCloseSidebar: document.getElementById('btnCloseSidebar'),
  sidebarPromptList: document.getElementById('sidebarPromptList'),
  categoryChips: document.getElementById('categoryChips'),
  suggestionChips: document.getElementById('suggestionChips'),
  quickCardsGrid: document.getElementById('quickCardsGrid'),
  handbookModal: document.getElementById('handbookModal'),
  btnOpenHandbook: document.getElementById('btnOpenHandbook'),
  btnCloseHandbook: document.getElementById('btnCloseHandbook'),
  btnCloseHandbookBtn: document.getElementById('btnCloseHandbookBtn'),
  modalTabContent: document.getElementById('modalTabContent'),
  toastContainer: document.getElementById('toastContainer')
};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initEventListeners();
  initSpeechRecognition();
  checkSystemStatus();
  fetchSampleQuestions();
  fetchHandbookData();
});

// ==========================================================
// Theme Management
// ==========================================================
function initTheme() {
  document.body.setAttribute('data-theme', state.theme);
  updateThemeIcons();
}

function toggleTheme() {
  state.theme = state.theme === 'dark' ? 'light' : 'dark';
  localStorage.setItem('eaut_theme', state.theme);
  document.body.setAttribute('data-theme', state.theme);
  updateThemeIcons();
  showToast(`Đã chuyển sang giao diện ${state.theme === 'dark' ? 'Tối' : 'Sáng'}`);
}

function updateThemeIcons() {
  if (state.theme === 'dark') {
    DOM.iconSun.classList.add('hidden');
    DOM.iconMoon.classList.remove('hidden');
  } else {
    DOM.iconSun.classList.remove('hidden');
    DOM.iconMoon.classList.add('hidden');
  }
}

// ==========================================================
// Sound Management
// ==========================================================
function toggleSound() {
  state.soundEnabled = !state.soundEnabled;
  if (state.soundEnabled) {
    DOM.iconSoundOn.classList.remove('hidden');
    DOM.iconSoundOff.classList.add('hidden');
    playTone(600, 0.08);
    showToast('Đã bật âm thanh');
  } else {
    DOM.iconSoundOn.classList.add('hidden');
    DOM.iconSoundOff.classList.remove('hidden');
    if (state.speechSynth) state.speechSynth.cancel();
    showToast('Đã tắt âm thanh');
  }
}

function playTone(freq = 440, duration = 0.08) {
  if (!state.soundEnabled) return;
  try {
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
    gain.gain.setValueAtTime(0.08, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + duration);
  } catch (e) {
    // AudioContext blocked or unsupported
  }
}

// ==========================================================
// Event Listeners
// ==========================================================
function initEventListeners() {
  // Theme and Sound toggles
  DOM.btnToggleTheme.addEventListener('click', toggleTheme);
  DOM.btnToggleSound.addEventListener('click', toggleSound);

  // Clear chat
  DOM.btnClearChat.addEventListener('click', handleClearChat);

  // Sidebar toggle on mobile
  if (DOM.btnToggleSidebar) {
    DOM.btnToggleSidebar.addEventListener('click', () => DOM.sidebar.classList.add('open'));
  }
  if (DOM.btnCloseSidebar) {
    DOM.btnCloseSidebar.addEventListener('click', () => DOM.sidebar.classList.remove('open'));
  }

  // Textarea input auto-grow and keypress handling
  DOM.messageInput.addEventListener('input', () => {
    DOM.messageInput.style.height = 'auto';
    DOM.messageInput.style.height = Math.min(DOM.messageInput.scrollHeight, 140) + 'px';
    const hasValue = DOM.messageInput.value.trim().length > 0;
    DOM.btnSendMessage.disabled = !hasValue || state.isStreaming;
  });

  DOM.messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!DOM.btnSendMessage.disabled) {
        submitUserMessage();
      }
    }
  });

  // Send button
  DOM.btnSendMessage.addEventListener('click', submitUserMessage);

  // Category chips
  DOM.categoryChips.addEventListener('click', (e) => {
    if (e.target.classList.contains('cat-chip')) {
      document.querySelectorAll('.cat-chip').forEach(c => c.classList.remove('active'));
      e.target.classList.add('active');
      state.activeCategory = e.target.getAttribute('data-category');
      renderPrompts();
    }
  });

  // Quick suggestion chips
  DOM.suggestionChips.addEventListener('click', (e) => {
    const chip = e.target.closest('.sug-chip');
    if (chip) {
      const prompt = chip.getAttribute('data-prompt');
      sendQuickPrompt(prompt);
    }
  });

  // Quick cards in welcome screen
  DOM.quickCardsGrid.addEventListener('click', (e) => {
    const card = e.target.closest('.quick-card');
    if (card) {
      const prompt = card.getAttribute('data-prompt');
      sendQuickPrompt(prompt);
    }
  });

  // Handbook modal controls
  DOM.btnOpenHandbook.addEventListener('click', () => openHandbook());
  DOM.btnCloseHandbook.addEventListener('click', () => closeHandbook());
  DOM.btnCloseHandbookBtn.addEventListener('click', () => closeHandbook());
  DOM.handbookModal.addEventListener('click', (e) => {
    if (e.target === DOM.handbookModal) closeHandbook();
  });

  // Modal tab switching
  document.querySelectorAll('.m-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.m-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      renderHandbookTab(tab.getAttribute('data-tab'));
    } );
  });
}

// ==========================================================
// Speech Recognition (Voice Input)
// ==========================================================
function initSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    DOM.btnVoiceInput.style.display = 'none';
    return;
  }

  state.recognition = new SpeechRecognition();
  state.recognition.lang = 'vi-VN';
  state.recognition.continuous = false;
  state.recognition.interimResults = false;

  state.recognition.onstart = () => {
    state.isListening = true;
    DOM.btnVoiceInput.classList.add('listening');
    showToast('Đang lắng nghe câu hỏi của bạn...');
  };

  state.recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    DOM.messageInput.value = transcript;
    DOM.messageInput.dispatchEvent(new Event('input'));
    DOM.messageInput.focus();
    playTone(520, 0.1);
  };

  state.recognition.onerror = () => {
    state.isListening = false;
    DOM.btnVoiceInput.classList.remove('listening');
    showToast('Không nhận diện được giọng nói. Vui lòng thử lại!');
  };

  state.recognition.onend = () => {
    state.isListening = false;
    DOM.btnVoiceInput.classList.remove('listening');
  };

  DOM.btnVoiceInput.addEventListener('click', () => {
    if (state.isListening) {
      state.recognition.stop();
    } else {
      try {
        state.recognition.start();
      } catch (err) {
        state.recognition.stop();
      }
    }
  });
}

// ==========================================================
// System Status Polling
// ==========================================================
async function checkSystemStatus() {
  try {
    const res = await fetch('/api/status');
    if (!res.ok) throw new Error('Status fetch failed');
    const data = await res.json();

    if (data.is_loaded) {
      state.systemReady = true;
      DOM.statusText.textContent = 'Hệ thống sẵn sàng';
      DOM.devicePill.textContent = data.device.toUpperCase();
      if (DOM.kbDocCount) DOM.kbDocCount.textContent = `${data.doc_count || 184} văn bản`;
      DOM.systemStatusBadge.classList.add('ready');
    } else if (data.is_loading) {
      DOM.statusText.textContent = data.progress || 'Đang nạp mô hình Qwen2.5...';
      setTimeout(checkSystemStatus, 3000);
    } else {
      DOM.statusText.textContent = 'Khởi động AI...';
      setTimeout(checkSystemStatus, 3000);
    }
  } catch (err) {
    DOM.statusText.textContent = 'Sẵn sàng';
    setTimeout(checkSystemStatus, 5000);
  }
}

// ==========================================================
// Sample Questions & Sidebar
// ==========================================================
async function fetchSampleQuestions() {
  try {
    const res = await fetch('/api/sample-questions');
    if (res.ok) {
      state.sampleCategories = await res.json();
      renderPrompts();
    }
  } catch (err) {
    console.error('Error fetching questions:', err);
  }
}

function renderPrompts() {
  DOM.sidebarPromptList.innerHTML = '';
  let allQuestions = [];

  state.sampleCategories.forEach(cat => {
    let match = false;
    if (state.activeCategory === 'all') match = true;
    else if (state.activeCategory === 'nganh' && cat.category.includes('Ngành')) match = true;
    else if (state.activeCategory === 'hocphi' && cat.category.includes('Học phí')) match = true;
    else if (state.activeCategory === 'hoso' && cat.category.includes('Phương thức')) match = true;
    else if (state.activeCategory === 'thoigian' && cat.category.includes('Thời gian')) match = true;

    if (match) {
      cat.questions.forEach(q => allQuestions.push(q));
    }
  });

  allQuestions.forEach(q => {
    const btn = document.createElement('button');
    btn.className = 'prompt-btn';
    btn.innerHTML = `
      <span class="prompt-btn-icon">💡</span>
      <span>${escapeHtml(q)}</span>
    `;
    btn.addEventListener('click', () => {
      sendQuickPrompt(q);
      if (window.innerWidth <= 900) {
        DOM.sidebar.classList.remove('open');
      }
    });
    DOM.sidebarPromptList.appendChild(btn);
  });
}

function sendQuickPrompt(promptText) {
  if (state.isStreaming) return;
  DOM.messageInput.value = promptText;
  DOM.messageInput.dispatchEvent(new Event('input'));
  submitUserMessage();
}

// ==========================================================
// Chat Interaction & Messaging
// ==========================================================
async function submitUserMessage() {
  const message = DOM.messageInput.value.trim();
  if (!message || state.isStreaming) return;

  // Clear input
  DOM.messageInput.value = '';
  DOM.messageInput.style.height = 'auto';
  DOM.btnSendMessage.disabled = true;

  // Hide Welcome Hero if first message
  if (DOM.welcomeHero) {
    DOM.welcomeHero.style.display = 'none';
  }

  // Play send audio feedback
  playTone(480, 0.06);

  // Append user message
  appendUserMessage(message);

  // Show typing indicator
  showTypingIndicator();

  // Scroll to bottom
  scrollToBottom();

  state.isStreaming = true;

  try {
    await streamChatResponse(message);
  } catch (error) {
    console.error('Chat error:', error);
    hideTypingIndicator();
    appendBotMessage(
      'Xin lỗi bạn, đã xảy ra lỗi kết nối với máy chủ AI. Vui lòng kiểm tra lại kết nối mạng hoặc thử lại sau!',
      []
    );
  } finally {
    state.isStreaming = false;
    DOM.btnSendMessage.disabled = DOM.messageInput.value.trim().length === 0;
    hideTypingIndicator();
  }
}

function appendUserMessage(text) {
  const row = document.createElement('div');
  row.className = 'message-row user-row';

  const timeStr = getCurrentTimeString();

  row.innerHTML = `
    <div class="message-content-wrapper">
      <div class="message-meta">
        <span class="message-time">${timeStr}</span>
        <span class="message-sender">Bạn</span>
      </div>
      <div class="message-bubble">${escapeHtml(text)}</div>
    </div>
    <div class="msg-avatar user-avatar">BẠN</div>
  `;

  DOM.chatThread.appendChild(row);
}

async function streamChatResponse(userQuery) {
  let botRow = null;
  let bubbleElement = null;
  let accumulatedAnswer = '';
  let sources = [];

  try {
    const response = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: userQuery, top_k: 3 })
    });

    if (!response.ok || !response.body) {
      // Fallback to regular POST /api/chat
      return await fallbackRegularChat(userQuery);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const event = JSON.parse(line.substring(6));

            if (event.type === 'meta') {
              sources = event.sources || [];
            } else if (event.type === 'token') {
              if (!botRow) {
                hideTypingIndicator();
                botRow = createBotMessageElement();
                bubbleElement = botRow.querySelector('.message-bubble');
                DOM.chatThread.appendChild(botRow);
                playTone(680, 0.05);
              }
              accumulatedAnswer += event.content;
              bubbleElement.innerHTML = renderMarkdown(accumulatedAnswer);
              scrollToBottom();
            } else if (event.type === 'error') {
              accumulatedAnswer = event.content || 'Đã xảy ra lỗi.';
              if (bubbleElement) bubbleElement.innerHTML = renderMarkdown(accumulatedAnswer);
            }
          } catch (e) {
            console.error('Parse SSE line error:', e);
          }
        }
      }
    }

    // Finalize bot message with sources & actions
    if (botRow) {
      finalizeBotMessage(botRow, accumulatedAnswer, sources);
    } else {
      hideTypingIndicator();
      appendBotMessage(accumulatedAnswer || 'Không nhận được câu trả lời từ hệ thống.', sources);
    }

  } catch (err) {
    console.warn('Stream failed, falling back to synchronous chat:', err);
    await fallbackRegularChat(userQuery);
  }
}

async function fallbackRegularChat(userQuery) {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: userQuery, top_k: 3 })
  });

  const data = await response.json();
  hideTypingIndicator();
  appendBotMessage(data.answer, data.sources || []);
}

function createBotMessageElement() {
  const row = document.createElement('div');
  row.className = 'message-row bot-row';

  const timeStr = getCurrentTimeString();

  row.innerHTML = `
    <div class="msg-avatar">
      <img src="/static/images/bot_avatar.jpg" alt="EAUT Bot">
    </div>
    <div class="message-content-wrapper">
      <div class="message-meta">
        <span class="message-sender">EAUT AI Assistant</span>
        <span class="message-time">${timeStr}</span>
      </div>
      <div class="message-bubble"></div>
      <div class="sources-placeholder"></div>
      <div class="actions-placeholder"></div>
    </div>
  `;
  return row;
}

function finalizeBotMessage(row, fullAnswer, sources) {
  const wrapper = row.querySelector('.message-content-wrapper');

  // Render sources accordion if available
  if (sources && sources.length > 0) {
    const sourcesContainer = document.createElement('div');
    sourcesContainer.className = 'sources-container';
    sourcesContainer.innerHTML = `
      <button class="sources-toggle" type="button">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
        <span>Căn cứ tài liệu tuyển sinh (${sources.length} trích đoạn)</span>
        <svg class="sources-chevron" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg>
      </button>
      <div class="sources-list hidden">
        ${sources.map(s => `
          <div class="source-item">
            <span class="source-file-badge">📄 ${escapeHtml(s.source)}</span>
            <div class="source-snippet">${escapeHtml(s.preview)}</div>
          </div>
        `).join('')}
      </div>
    `;

    const toggleBtn = sourcesContainer.querySelector('.sources-toggle');
    const sourcesList = sourcesContainer.querySelector('.sources-list');

    toggleBtn.addEventListener('click', () => {
      toggleBtn.classList.toggle('open');
      sourcesList.classList.toggle('hidden');
      scrollToBottom();
    });

    const sourcesPlaceholder = wrapper.querySelector('.sources-placeholder');
    if (sourcesPlaceholder) {
      sourcesPlaceholder.replaceWith(sourcesContainer);
    } else {
      wrapper.appendChild(sourcesContainer);
    }
  }

  // Render Action buttons (Copy, TTS, Feedback)
  const actionsBar = document.createElement('div');
  actionsBar.className = 'message-actions';
  actionsBar.innerHTML = `
    <button class="bubble-btn btn-copy" title="Sao chép nội dung">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
      <span>Sao chép</span>
    </button>
    <button class="bubble-btn btn-tts" title="Đọc to câu trả lời">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg>
      <span>Nghe đọc</span>
    </button>
    <button class="bubble-btn btn-feedback-like" title="Câu trả lời hữu ích">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path></svg>
    </button>
    <button class="bubble-btn btn-feedback-dislike" title="Chưa hài lòng">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3"></path></svg>
    </button>
  `;

  // Attach button events
  const btnCopy = actionsBar.querySelector('.btn-copy');
  btnCopy.addEventListener('click', () => {
    navigator.clipboard.writeText(fullAnswer);
    showToast('Đã sao chép câu trả lời vào clipboard!');
  });

  const btnTTS = actionsBar.querySelector('.btn-tts');
  btnTTS.addEventListener('click', () => {
    speakText(fullAnswer, btnTTS);
  });

  const btnLike = actionsBar.querySelector('.btn-feedback-like');
  const btnDislike = actionsBar.querySelector('.btn-feedback-dislike');

  btnLike.addEventListener('click', () => {
    btnLike.classList.toggle('active');
    btnDislike.classList.remove('active');
    showToast('Cảm ơn bạn đã phản hồi!');
  });

  btnDislike.addEventListener('click', () => {
    btnDislike.classList.toggle('active');
    btnLike.classList.remove('active');
    showToast('Cảm ơn đóng góp của bạn để cải thiện AI!');
  });

  const actionsPlaceholder = wrapper.querySelector('.actions-placeholder');
  if (actionsPlaceholder) {
    actionsPlaceholder.replaceWith(actionsBar);
  } else {
    wrapper.appendChild(actionsBar);
  }
}

function appendBotMessage(answer, sources) {
  const botRow = createBotMessageElement();
  const bubble = botRow.querySelector('.message-bubble');
  bubble.innerHTML = renderMarkdown(answer);
  DOM.chatThread.appendChild(botRow);
  finalizeBotMessage(botRow, answer, sources);
  scrollToBottom();
  playTone(640, 0.08);
}

function handleClearChat() {
  if (DOM.chatThread.children.length === 0) return;
  if (confirm('Bạn có chắc muốn làm mới cuộc trò chuyện này không?')) {
    DOM.chatThread.innerHTML = '';
    if (DOM.welcomeHero) DOM.welcomeHero.style.display = 'flex';
    if (state.speechSynth) state.speechSynth.cancel();
    showToast('Đã tạo cuộc trò chuyện mới');
  }
}

function showTypingIndicator() {
  DOM.typingIndicator.classList.remove('hidden');
}

function hideTypingIndicator() {
  DOM.typingIndicator.classList.add('hidden');
}

function scrollToBottom() {
  DOM.messagesFeed.scrollTop = DOM.messagesFeed.scrollHeight;
}

// ==========================================================
// Text-to-Speech (Vietnamese Voice Synthesis)
// ==========================================================
function speakText(text, btnElement) {
  if (!state.speechSynth) {
    showToast('Trình duyệt của bạn chưa hỗ trợ đọc âm thanh');
    return;
  }

  if (state.speechSynth.speaking) {
    state.speechSynth.cancel();
    btnElement.classList.remove('active');
    return;
  }

  // Clean Markdown characters before speaking
  const cleanText = text
    .replace(/[#*`_~\[\]\(\)>|]/g, ' ')
    .replace(/--+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

  state.currentUtterance = new SpeechSynthesisUtterance(cleanText);
  state.currentUtterance.lang = 'vi-VN';
  state.currentUtterance.rate = 1.0;

  // Find Vietnamese voice if installed
  const voices = state.speechSynth.getVoices();
  const viVoice = voices.find(v => v.lang.includes('vi') || v.lang.includes('VN'));
  if (viVoice) state.currentUtterance.voice = viVoice;

  state.currentUtterance.onstart = () => {
    btnElement.classList.add('active');
  };

  state.currentUtterance.onend = () => {
    btnElement.classList.remove('active');
  };

  state.currentUtterance.onerror = () => {
    btnElement.classList.remove('active');
  };

  state.speechSynth.speak(state.currentUtterance);
}

// ==========================================================
// Admission Handbook Modal Logic
// ==========================================================
async function fetchHandbookData() {
  try {
    const res = await fetch('/api/handbook');
    if (res.ok) {
      state.handbookData = await res.json();
    }
  } catch (e) {
    console.error('Handbook fetch error:', e);
  }
}

function openHandbook() {
  DOM.handbookModal.classList.remove('hidden');
  renderHandbookTab('tab-majors');
}

function closeHandbook() {
  DOM.handbookModal.classList.add('hidden');
}

function renderHandbookTab(tabName) {
  if (!state.handbookData) return;
  const content = DOM.modalTabContent;
  content.innerHTML = '';

  if (tabName === 'tab-majors') {
    content.innerHTML = `
      <div style="margin-bottom: 16px;">
        <h4 style="color: var(--brand-gold-light); margin-bottom: 6px;">Danh mục 34 Ngành & Chuyên ngành đào tạo tiêu biểu EAUT</h4>
        <p style="font-size: 0.84rem; color: var(--text-secondary);">Chương trình đào tạo theo mô hình thực hành công nghệ kết hợp với các tập đoàn lớn.</p>
      </div>
      <div class="hb-grid">
        <div class="hb-item-card">
          <h5>Công nghệ thông tin</h5>
          <p>Mã ngành: <strong>7480201</strong><br>Chuyên ngành: Trí tuệ nhân tạo (AI), Thiết kế đồ họa số, Kỹ thuật phần mềm.</p>
        </div>
        <div class="hb-item-card">
          <h5>Công nghệ kỹ thuật Ô tô</h5>
          <p>Mã ngành: <strong>7510205</strong><br>Thực hành chuyên sâu trên xưởng ô tô điện & động cơ hiện đại.</p>
        </div>
        <div class="hb-item-card">
          <h5>Kỹ thuật Bán dẫn & Vi mạch</h5>
          <p>Mã ngành: <strong>7510301</strong><br>Ngành mũi nhọn chiến lược, đón đầu làn sóng công nghệ bán dẫn quốc tế.</p>
        </div>
        <div class="hb-item-card">
          <h5>Công nghệ Điều khiển & Tự động hóa</h5>
          <p>Mã ngành: <strong>7510303</strong><br>Ứng dụng Robotics, cánh tay robot công nghiệp và IoT nhà máy thông minh.</p>
        </div>
        <div class="hb-item-card">
          <h5>Quản trị kinh doanh & Marketing</h5>
          <p>Mã ngành: <strong>7340101</strong><br>Đào tạo kinh doanh số, truyền thông đa phương tiện và logistics.</p>
        </div>
        <div class="hb-item-card">
          <h5>Ngôn ngữ Anh & Ngôn ngữ Trung</h5>
          <p>Định hướng biên phiên dịch thương mại, công nghệ và du lịch quốc tế.</p>
        </div>
      </div>
    `;
  } else if (tabName === 'tab-methods') {
    content.innerHTML = `
      <div style="margin-bottom: 16px;">
        <h4 style="color: var(--brand-gold-light); margin-bottom: 6px;">4 Phương thức tuyển sinh chính quy năm 2026</h4>
        <p style="font-size: 0.84rem; color: var(--text-secondary);">Thí sinh có thể đồng thời sử dụng nhiều phương thức để tối đa cơ hội trúng tuyển.</p>
      </div>
      <div style="display: flex; flex-direction: column; gap: 12px;">
        ${state.handbookData.methods.map(m => `
          <div class="hb-item-card">
            <h5 style="color: var(--brand-gold);">${escapeHtml(m.title)}</h5>
            <p>${escapeHtml(m.desc)}</p>
          </div>
        `).join('')}
      </div>
    `;
  } else if (tabName === 'tab-tuition') {
    content.innerHTML = `
      <div style="margin-bottom: 16px;">
        <h4 style="color: var(--brand-gold-light); margin-bottom: 6px;">Chính sách học phí minh bạch & Ổn định toàn khóa</h4>
        <p style="font-size: 0.84rem; color: var(--text-secondary);">Mức học phí thu theo số lượng tín chỉ thực tế sinh viên đăng ký học.</p>
      </div>
      <table style="width: 100%; border-collapse: collapse; margin-top: 8px;">
        <thead>
          <tr style="background: rgba(255,255,255,0.06);">
            <th style="padding: 10px; border: 1px solid var(--border-subtle); text-align: left;">Khối ngành đào tạo</th>
            <th style="padding: 10px; border: 1px solid var(--border-subtle); text-align: left;">Mức học phí tham khảo</th>
            <th style="padding: 10px; border: 1px solid var(--border-subtle); text-align: left;">Ghi chú chính sách</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style="padding: 10px; border: 1px solid var(--border-subtle);">Khối Công nghệ & Kỹ thuật</td>
            <td style="padding: 10px; border: 1px solid var(--border-subtle); color: var(--brand-gold);">Từ 380.000đ - 450.000đ / tín chỉ</td>
            <td style="padding: 10px; border: 1px solid var(--border-subtle);">Bao gồm thực hành phòng Lab</td>
          </tr>
          <tr>
            <td style="padding: 10px; border: 1px solid var(--border-subtle);">Khối Kinh tế & Quản trị</td>
            <td style="padding: 10px; border: 1px solid var(--border-subtle); color: var(--brand-gold);">Từ 360.000đ - 420.000đ / tín chỉ</td>
            <td style="padding: 10px; border: 1px solid var(--border-subtle);">Cam kết hỗ trợ thực tập doanh nghiệp</td>
          </tr>
          <tr>
            <td style="padding: 10px; border: 1px solid var(--border-subtle);">Khối Ngôn ngữ & Du lịch</td>
            <td style="padding: 10px; border: 1px solid var(--border-subtle); color: var(--brand-gold);">Từ 370.000đ - 430.000đ / tín chỉ</td>
            <td style="padding: 10px; border: 1px solid var(--border-subtle);">Trải nghiệm văn hóa thực địa</td>
          </tr>
        </tbody>
      </table>
    `;
  } else if (tabName === 'tab-scholarship') {
    content.innerHTML = `
      <div style="margin-bottom: 16px;">
        <h4 style="color: var(--brand-gold-light); margin-bottom: 6px;">Quỹ Học Bổng Tài Năng EAUT 2026</h4>
        <p style="font-size: 0.84rem; color: var(--text-secondary);">Hàng trăm suất học bổng toàn phần và bán phần chắp cánh ước mơ sinh viên.</p>
      </div>
      <div class="hb-grid">
        ${state.handbookData.scholarships.map(s => `
          <div class="hb-item-card">
            <h5 style="color: var(--brand-gold);">${escapeHtml(s.tier)}</h5>
            <p>${escapeHtml(s.benefit)}</p>
          </div>
        `).join('')}
      </div>
    `;
  } else if (tabName === 'tab-timeline') {
    content.innerHTML = `
      <div style="margin-bottom: 16px;">
        <h4 style="color: var(--brand-gold-light); margin-bottom: 6px;">Kế hoạch & Mốc thời gian tuyển sinh chuẩn năm 2026</h4>
        <p style="font-size: 0.84rem; color: var(--text-secondary);">Lịch trình chung theo chỉ đạo của Bộ GD&ĐT và Hội đồng tuyển sinh EAUT.</p>
      </div>
      <div class="timeline-track">
        ${state.handbookData.timeline_2026.map(t => `
          <div class="tl-node">
            <div class="tl-date">${escapeHtml(t.milestone)}</div>
            <div class="tl-desc">${escapeHtml(t.event)}</div>
          </div>
        `).join('')}
      </div>
    `;
  }
}

// ==========================================================
// Helpers: Markdown Parser, Toasts, Formatting
// ==========================================================
function renderMarkdown(md) {
  if (!md) return '';

  let html = escapeHtml(md);

  // Bold **text**
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

  // Italic *text*
  html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');

  // Inline code `text`
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

  // Unordered list items: lines starting with "- " or "* "
  html = html.replace(/(?:^|\n)[-*]\s+(.+)/g, '\n<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');

  // Numbered list items
  html = html.replace(/(?:^|\n)\d+\.\s+(.+)/g, '\n<li>$1</li>');

  // Paragraph breaks
  html = html.replace(/\n\n+/g, '</p><p>');
  html = html.replace(/\n/g, '<br>');

  return `<p>${html}</p>`;
}

function escapeHtml(text) {
  if (!text) return '';
  const map = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;'
  };
  return text.toString().replace(/[&<>"']/g, m => map[m]);
}

function getCurrentTimeString() {
  const now = new Date();
  const h = String(now.getHours()).padStart(2, '0');
  const m = String(now.getMinutes()).padStart(2, '0');
  return `${h}:${m}`;
}

function showToast(message, type = 'success') {
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `
    <span>${escapeHtml(message)}</span>
  `;
  DOM.toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(20px)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}
