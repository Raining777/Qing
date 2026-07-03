/* chat.js — 聊天消息、SSE 流式（delta）、统一路由到 /api/chat */

function addMessage(role, text, sources, tokenUsage) {
  const container = document.getElementById('chat-messages');
  const welcome = container.querySelector('.welcome-message');
  if (welcome) welcome.remove();

  const div = document.createElement('div');
  div.className = `message ${role}`;
  const content = document.createElement('div');
  content.className = 'message-content';
  content.innerHTML = role === 'assistant' ? renderMarkdown(text) : escapeHtml(text);
  div.appendChild(content);

  if (sources && sources.length) {
    const srcDiv = document.createElement('details');
    srcDiv.className = 'message-sources';
    srcDiv.innerHTML = '<summary>📎 来源</summary>';
    for (const s of sources) {
      srcDiv.innerHTML += `<div class="source-excerpt"><strong>${escapeHtml(s.file)}</strong>${s.page ? ' (p.' + s.page + ')' : ''}: ${escapeHtml(s.excerpt?.substring(0, 200) || '')}</div>`;
    }
    div.appendChild(srcDiv);
  }

  if (tokenUsage && tokenUsage.total) {
    addTokenCount(tokenUsage.input, tokenUsage.output);
  }

  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return content;
}

function addUserMessage(text) { addMessage('user', text); }

function addAssistantStream() {
  const container = document.getElementById('chat-messages');
  const welcome = container.querySelector('.welcome-message');
  if (welcome) welcome.remove();
  const div = document.createElement('div');
  div.className = 'message assistant';
  const content = document.createElement('div');
  content.className = 'message-content';
  content.id = 'streaming-content';
  div.appendChild(content);
  const srcDiv = document.createElement('details');
  srcDiv.className = 'message-sources hidden';
  srcDiv.id = 'streaming-sources';
  div.appendChild(srcDiv);
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return content;
}

function finishStream(contentEl, sources, tokenUsage) {
  contentEl.innerHTML = renderMarkdown(contentEl.textContent || '');
  const modelTag = document.createElement('div');
  modelTag.className = 'model-tag';
  const providerName = STATE.providers.find(p => p.id === STATE.activeProvider)?.name || STATE.activeProvider;
  modelTag.textContent = `${providerName} · ${STATE.activeModel}`;
  contentEl.parentElement.appendChild(modelTag);

  if (sources && sources.length) {
    const srcDiv = document.getElementById('streaming-sources');
    if (srcDiv) {
      srcDiv.classList.remove('hidden');
      srcDiv.innerHTML = '<summary>📎 来源</summary>';
      for (const s of sources) {
        srcDiv.innerHTML += `<div class="source-excerpt"><strong>${escapeHtml(s.file)}</strong>${s.page ? ' (p.' + s.page + ')' : ''}: ${escapeHtml(s.excerpt?.substring(0, 200) || '')}</div>`;
      }
    }
  }
  if (tokenUsage && tokenUsage.total) addTokenCount(tokenUsage.input, tokenUsage.output);
  document.getElementById('chat-messages').scrollTop = document.getElementById('chat-messages').scrollHeight;
}

async function sendMessage() {
  const input = document.getElementById('chat-input');
  const question = input.value.trim();
  if (!question) return;
  input.value = ''; input.style.height = 'auto';

  // Always send to unified /api/chat — server handles intent detection
  const body = {
    question,
    course: STATE.activeCourse || undefined,
    provider: STATE.activeProvider,
    model: STATE.activeModel,
    session_id: STATE.sessionId,
  };

  addUserMessage(question);
  await streamAction('/api/chat', body);
}

async function streamAction(endpoint, body) {
  const btn = document.getElementById('btn-send');
  btn.disabled = true;
  const contentEl = addAssistantStream();

  try {
    const resp = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const err = await resp.json();
      contentEl.textContent = `出错: ${err.detail || '请稍后重试'}`;
      btn.disabled = false;
      return;
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '', fullText = '', finalSources = null, finalTokens = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = JSON.parse(line.slice(6));
        if (data.session_id) {
          STATE.sessionId = data.session_id;
          loadSessions();
        }
        if (data.done) break;
        if (data.error) { contentEl.textContent = '错误: ' + data.error; break; }
        if (data.status) {
          contentEl.textContent = data.status;
          continue;
        }
        // Delta streaming (v2.0): accumulate incremental tokens
        if (data.delta) {
          fullText += data.delta;
          contentEl.textContent = fullText;
          document.getElementById('chat-messages').scrollTop = document.getElementById('chat-messages').scrollHeight;
        }
        // Final full response with sources (backward compat)
        if (data.response) {
          fullText = data.response;
          contentEl.textContent = fullText;
        }
        if (data.sources) finalSources = data.sources;
        if (data.token_usage) finalTokens = data.token_usage;
      }
    }
    finishStream(contentEl, finalSources, finalTokens);
  } catch (e) {
    contentEl.textContent = '连接错误: ' + e.message;
  }
  contentEl.id = '';
  btn.disabled = false;
}

async function loadSessions() {
  try {
    const resp = await fetch('/api/sessions');
    const sessions = await resp.json();
    const container = document.getElementById('sessions-container');
    if (!sessions.length) { container.innerHTML = ''; return; }
    container.innerHTML = sessions.map(s => `
      <div class="session-item ${s.id === STATE.sessionId ? 'active' : ''}" onclick="switchSession('${s.id}')">
        <span>💬</span><span>${escapeHtml(s.preview?.substring(0, 20) || '对话')}</span>
        <button class="session-delete-btn" title="删除对话" onclick="event.stopPropagation();deleteSession('${s.id}')">✕</button>
      </div>
    `).join('');
  } catch (e) {}
}

async function switchSession(sid) {
  STATE.sessionId = sid;
  document.getElementById('chat-messages').innerHTML = `
    <div class="welcome-message">
      <div class="welcome-logo">清</div>
      <p>已切换到对话 ${sid}</p>
      <div class="quick-actions" id="quick-actions">
        <div class="quick-action" onclick="quickAction('summarize')"><span class="qa-icon">📝</span><span class="qa-text"><strong>总结知识点</strong><small>提取核心概念</small></span></div>
        <div class="quick-action" onclick="quickAction('mindmap')"><span class="qa-icon">🧠</span><span class="qa-text"><strong>生成思维导图</strong><small>可视化知识结构</small></span></div>
        <div class="quick-action" onclick="quickAction('practice')"><span class="qa-icon">✍️</span><span class="qa-text"><strong>出题练习</strong><small>多种题型</small></span></div>
        <div class="quick-action" onclick="quickAction('sprint')"><span class="qa-icon">🎯</span><span class="qa-text"><strong>考前冲刺</strong><small>薄弱诊断</small></span></div>
      </div>
    </div>`;
  loadSessions();
}

async function deleteSession(sid) {
  if (!confirm('确定删除这个对话？')) return;
  try {
    await fetch(`/api/sessions/${sid}`, { method: 'DELETE' });
    if (STATE.sessionId === sid) { STATE.sessionId = ''; showWelcome(); }
    loadSessions();
  } catch (e) {}
}

function showWelcome() {
  document.getElementById('chat-messages').innerHTML = `
    <div class="welcome-message">
      <div class="welcome-logo">清</div>
      <p>上传你的学习资料，我来帮你总结知识点、出题练习、制定复习计划</p>
      <div class="quick-actions" id="quick-actions">
        <div class="quick-action" onclick="quickAction('summarize')"><span class="qa-icon">📝</span><span class="qa-text"><strong>总结知识点</strong><small>提取核心概念</small></span></div>
        <div class="quick-action" onclick="quickAction('mindmap')"><span class="qa-icon">🧠</span><span class="qa-text"><strong>生成思维导图</strong><small>可视化知识结构</small></span></div>
        <div class="quick-action" onclick="quickAction('practice')"><span class="qa-icon">✍️</span><span class="qa-text"><strong>出题练习</strong><small>多种题型</small></span></div>
        <div class="quick-action" onclick="quickAction('sprint')"><span class="qa-icon">🎯</span><span class="qa-text"><strong>考前冲刺</strong><small>薄弱诊断</small></span></div>
      </div>
    </div>`;
}

async function newSession() {
  try {
    const resp = await fetch('/api/sessions/new', { method: 'POST' });
    const data = await resp.json();
    STATE.sessionId = data.session_id;
    showWelcome();
    loadSessions();
  } catch (e) {}
}

document.getElementById('btn-new-chat').addEventListener('click', newSession);
