/* chat.js — 聊天消息、SSE 流式、意图识别 */

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
  // Add model badge
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

  const lower = question.toLowerCase();
  let endpoint = '/api/chat';
  let body = {
    question,
    course: STATE.activeCourse || undefined,
    provider: STATE.activeProvider,
    model: STATE.activeModel,
    session_id: STATE.sessionId,
  };

  const wantsSummary = /^(帮我|请|给我)?(总结|梳理|概括)/.test(question) || lower.includes('summarize') || lower.includes('summary');
  const wantsMindmap = lower.includes('思维导图') || lower.includes('mindmap') || lower.includes('mind map');
  const wantsPlan = lower.includes('复习计划') || lower.includes('study plan') || /制定.*计划/.test(question);
  const wantsPractice = lower.includes('出题') || lower.includes('练习题') || /生成.*题/.test(question) || lower.includes('practice questions') || lower.includes('quiz');
  const wantsFlashcards = lower.includes('闪卡') || lower.includes('flashcard');
  const wantsFormulas = lower.includes('公式表') || lower.includes('整理公式') || lower.includes('提取公式') || lower.includes('formula sheet');
  const wantsCompare = lower.includes('对比') || lower.includes('compare') || lower.includes(' vs ');
  const wantsMnemonic = lower.includes('记忆口诀') || lower.includes('记忆方法') || lower.includes('mnemonic') || lower.includes('帮我记');
  const wantsSprint = lower.includes('冲刺') || lower.includes('sprint') || lower.includes('考前');

  if (wantsSummary) {
    endpoint = '/api/summary';
    body = { course: STATE.activeCourse, provider: STATE.activeProvider, model: STATE.activeModel };
  } else if (wantsMindmap) {
    endpoint = '/api/mindmap';
    body = { course: STATE.activeCourse, provider: STATE.activeProvider, model: STATE.activeModel };
  } else if (wantsPlan) {
    endpoint = '/api/plan';
    body = { course: STATE.activeCourse, provider: STATE.activeProvider, model: STATE.activeModel, exam_date: extractDate(question) };
  } else if (wantsPractice) {
    endpoint = '/api/practice';
    const count = (question.match(/(\d+)\s*(道|题|questions)/) || [])[1] || '5';
    body = { course: STATE.activeCourse, provider: STATE.activeProvider, model: STATE.activeModel, practice_count: parseInt(count), practice_topic: question };
  } else if (wantsFlashcards) {
    endpoint = '/api/flashcards';
    body = { course: STATE.activeCourse, provider: STATE.activeProvider, model: STATE.activeModel };
  } else if (wantsFormulas) {
    endpoint = '/api/formulas';
    body = { course: STATE.activeCourse, provider: STATE.activeProvider, model: STATE.activeModel };
  } else if (wantsCompare) {
    endpoint = '/api/compare';
    const compareText = question.replace(/^(帮我|请|给我)?(对比|比较|compare)\s*/i, '').trim();
    const parts = compareText.split(/\s+vs\.?\s+|和|与|以及|、/).map(s => s.trim()).filter(Boolean);
    body = { course: STATE.activeCourse, provider: STATE.activeProvider, model: STATE.activeModel, concept_a: parts[0]?.trim() || '', concept_b: parts[1]?.trim() || '' };
  } else if (wantsMnemonic) {
    endpoint = '/api/mnemonic';
    body = { query: question, provider: STATE.activeProvider, model: STATE.activeModel };
  } else if (wantsSprint) {
    endpoint = '/api/sprint';
    body = { course: STATE.activeCourse, sprint_course: STATE.activeCourse, provider: STATE.activeProvider, model: STATE.activeModel };
  }

  addUserMessage(question);
  await streamAction(endpoint, body);
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
          // Show progress status briefly
          contentEl.textContent = data.status;
          continue;
        }
        if (data.response) {
          fullText = data.response;
          contentEl.textContent = fullText;
          document.getElementById('chat-messages').scrollTop = document.getElementById('chat-messages').scrollHeight;
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

function extractDate(text) {
  const match = text.match(/(\d{1,2})\/(\d{1,2})(?:\/(\d{2,4}))?/);
  return match ? match[0] : '';
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
  // Clear chat messages and show welcome
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
    if (STATE.sessionId === sid) {
      STATE.sessionId = '';
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
    loadSessions();
  } catch (e) { console.error('Failed to delete session:', e); }
}

function clearCurrentChat() {
  if (!confirm('确定清空当前聊天记录？')) return;
  newSession();
}

async function newSession() {
  try {
    const resp = await fetch('/api/sessions/new', { method: 'POST' });
    const data = await resp.json();
    STATE.sessionId = data.session_id;
    document.getElementById('chat-messages').innerHTML = `
      <div class="welcome-message">
        <div class="welcome-logo">清</div>
        <p>新对话已创建。上传资料或直接提问吧。</p>
        <div class="quick-actions" id="quick-actions">
          <div class="quick-action" onclick="quickAction('summarize')"><span class="qa-icon">📝</span><span class="qa-text"><strong>总结知识点</strong><small>提取核心概念</small></span></div>
          <div class="quick-action" onclick="quickAction('mindmap')"><span class="qa-icon">🧠</span><span class="qa-text"><strong>生成思维导图</strong><small>可视化知识结构</small></span></div>
          <div class="quick-action" onclick="quickAction('practice')"><span class="qa-icon">✍️</span><span class="qa-text"><strong>出题练习</strong><small>多种题型</small></span></div>
          <div class="quick-action" onclick="quickAction('sprint')"><span class="qa-icon">🎯</span><span class="qa-text"><strong>考前冲刺</strong><small>薄弱诊断</small></span></div>
        </div>
      </div>`;
    loadSessions();
  } catch (e) {}
}

document.getElementById('btn-new-chat').addEventListener('click', newSession);
