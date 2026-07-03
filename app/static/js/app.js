/* app.js — 清 主状态管理 */

const STATE = {
  configured: false,
  defaultProvider: 'deepseek',
  providers: [],
  activeProvider: 'deepseek',
  activeModel: '',
  activeCourse: '',
  sessionId: '',
  tokenTotal: 0,
};

async function init() {
  try {
    const resp = await fetch('/api/status');
    const data = await resp.json();
    STATE.configured = data.configured;
    STATE.defaultProvider = data.default_provider;
    STATE.providers = data.providers;

    // Use default provider from backend
    STATE.activeProvider = data.default_provider || 'deepseek';

    // Find default model for active provider
    for (const p of data.providers) {
      if (p.id === STATE.activeProvider && p.models && p.models.length) {
        STATE.activeModel = p.models[0];
        break;
      }
    }
    if (!STATE.activeModel) STATE.activeModel = 'deepseek-v4-pro';

    // Build model selector
    buildModelSelect();

    if (!STATE.configured) {
      document.getElementById('setup-overlay').classList.remove('hidden');
    } else {
      document.getElementById('setup-overlay').classList.add('hidden');
      document.getElementById('app').classList.remove('hidden');
      await loadCourses();
      await loadSessions();
      initUpload();
    }
  } catch (e) {
    document.getElementById('setup-overlay').classList.remove('hidden');
  }
}

function buildModelSelect() {
  const providerSelect = document.getElementById('provider-select');
  const modelInput = document.getElementById('model-input');
  if (!providerSelect) return;

  // Populate provider dropdown
  providerSelect.innerHTML = STATE.providers.map(p =>
    `<option value="${p.id}" ${p.id === STATE.activeProvider ? 'selected' : ''}>${p.name}${p.vision ? ' 👁' : ''}${p.local ? ' 🖥' : ''}</option>`
  ).join('');

  // Set model input
  if (modelInput) {
    modelInput.value = STATE.activeModel || '';
    modelInput.placeholder = STATE.activeModel || '模型名称';
  }
}

function onProviderChange() {
  const select = document.getElementById('provider-select');
  STATE.activeProvider = select.value;
  // Auto-select first model for this provider
  for (const p of STATE.providers) {
    if (p.id === STATE.activeProvider && p.models && p.models.length) {
      STATE.activeModel = p.models[0];
      const modelInput = document.getElementById('model-input');
      if (modelInput) {
        modelInput.value = STATE.activeModel;
        modelInput.placeholder = STATE.activeModel;
      }
      break;
    }
  }
}

function onModelChange() {
  const modelInput = document.getElementById('model-input');
  const val = modelInput.value.trim();
  if (val) STATE.activeModel = val;
}

function handleKeyDown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
  if (e.key === 'k' && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    newSession();
  }
}

function addTokenCount(input, output) {
  STATE.tokenTotal += (input || 0) + (output || 0);
  const el = document.getElementById('token-display');
  if (el && STATE.tokenTotal > 0) {
    const k = Math.round(STATE.tokenTotal / 1000);
    el.textContent = `📊 约 ${k}K token`;
  }
}

function quickAction(action) {
  const course = STATE.activeCourse;
  if (!course) {
    addMessage('assistant', '请先在左侧选择一个课程，或上传学习资料。');
    return;
  }
  const map = {
    'summarize': '帮我总结这门课的所有核心知识点，标注重要程度',
    'mindmap': '帮我生成这门课的思维导图',
    'practice': '根据这门课的内容出 5 道练习题',
    'sprint': '帮我做考前冲刺复习',
  };
  const text = map[action] || action;
  document.getElementById('chat-input').value = text;
  sendMessage();
}

document.addEventListener('DOMContentLoaded', init);
