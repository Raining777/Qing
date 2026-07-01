/* setup.js — API Key 配置（首次 + 随时可改） */

let _setupFirstTime = true;

async function openSettings() {
  _setupFirstTime = false;
  // Show current status for each provider
  try {
    const resp = await fetch('/api/status');
    const data = await resp.json();
    for (const p of data.providers) {
      if (p.id === 'ollama') continue;
      const statusEl = document.getElementById('status-' + p.id);
      if (statusEl) {
        statusEl.textContent = '✓ 已配置';
        statusEl.className = 'status ok';
      }
    }
  } catch (e) {}

  document.getElementById('setup-overlay').classList.remove('hidden');
  document.getElementById('setup-save').textContent = '保存并重新加载';
}

function closeSettings() {
  document.getElementById('setup-overlay').classList.add('hidden');
  if (_setupFirstTime) {
    document.getElementById('app').classList.remove('hidden');
    buildModelSelect();
    loadCourses();
    initUpload();
  }
}

async function saveSetup() {
  const providers = [
    { id: 'anthropic', el: 'setup-anthropic-key', status: 'status-anthropic' },
    { id: 'deepseek', el: 'setup-deepseek-key', status: 'status-deepseek' },
    { id: 'openai', el: 'setup-openai-key', status: 'status-openai' },
  ];

  let anySaved = false;
  const btn = document.getElementById('setup-save');
  btn.textContent = '正在校验...';
  btn.disabled = true;

  for (const p of providers) {
    const key = document.getElementById(p.el).value.trim();
    if (!key) continue;

    const statusEl = document.getElementById(p.status);
    statusEl.textContent = '校验中...';
    statusEl.className = 'status';

    try {
      const resp = await fetch('/api/setup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider: p.id, api_key: key }),
      });
      const data = await resp.json();

      if (resp.ok) {
        statusEl.textContent = '✓ ' + (data.message || '已保存');
        statusEl.className = 'status ok';
        anySaved = true;
      } else {
        statusEl.textContent = '✗ ' + (data.detail || '保存失败');
        statusEl.className = 'status error';
      }
    } catch (e) {
      statusEl.textContent = '✗ 网络错误，请稍后重试';
      statusEl.className = 'status error';
    }
  }

  btn.textContent = '保存并重新加载';
  btn.disabled = false;

  // 只要有一个保存成功就刷新，让新 Key 生效
  if (anySaved) {
    setTimeout(() => location.reload(), 1500);
  }
}

function skipSetup() {
  if (_setupFirstTime) {
    document.getElementById('setup-overlay').classList.add('hidden');
    document.getElementById('app').classList.remove('hidden');
    buildModelSelect();
    loadCourses();
    initUpload();
  } else {
    closeSettings();
  }
}
