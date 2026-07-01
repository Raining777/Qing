/* upload.js — 文件拖拽上传 + 实时进度 */

let progressSource = null;

function initUpload() {
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');

  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    handleFiles(e.dataTransfer.files);
  });
  fileInput.addEventListener('change', () => handleFiles(fileInput.files));

  document.addEventListener('paste', (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (const item of items) {
      if (item.type.startsWith('image/')) {
        handleFiles([item.getAsFile()]);
        break;
      }
    }
  });
}

async function handleFiles(fileList) {
  const files = Array.from(fileList).filter(f => f.size > 0);
  if (!files.length) return;

  showProgress(0, files.length, '上传中...', 'uploading');

  const form = new FormData();
  for (const f of files) form.append('files', f);

  try {
    const resp = await fetch('/api/upload', { method: 'POST', body: form });
    if (!resp.ok) throw new Error('上传失败');
    listenProgress();
  } catch (e) {
    document.getElementById('progress-text').textContent = '上传失败: ' + e.message;
  }
}

function listenProgress() {
  if (progressSource) progressSource.close();
  const es = new EventSource('/api/upload/progress');
  progressSource = es;

  es.onmessage = (e) => {
    const p = JSON.parse(e.data);
    showProgress(p.done, p.total, p.current, p.step);
    if (p.phase === 'complete') {
      es.close(); progressSource = null;
      setTimeout(() => {
        document.getElementById('upload-progress').classList.add('hidden');
        loadCourses();
      }, 1000);
    }
  };
  es.onerror = () => { es.close(); progressSource = null; loadCourses(); };
}

function showProgress(done, total, current, step) {
  const el = document.getElementById('upload-progress');
  el.classList.remove('hidden');
  const pct = total > 0 ? Math.round((done / total) * 100) : 0;
  document.getElementById('progress-fill').style.width = pct + '%';

  const stepLabels = {
    'uploading': '上传中',
    'starting': '启动中',
    'parsing': '解析文件',
    'ocr_scanning': 'OCR 识别',
    'chunking': '文本分块',
    'embedding': '向量化',
    'storing': '存入知识库',
    'complete': '✓ 完成',
  };
  const label = stepLabels[step] || step;
  document.getElementById('progress-text').textContent =
    current ? `${done}/${total} ${label}: ${current}` : `${done}/${total} ${label}`;
}
