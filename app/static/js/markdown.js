/* markdown.js — Markdown rendering with KaTeX + code highlighting
   Mermaid is lazy-loaded on first use to save ~1MB on initial page load. */

let _mermaidReady = null;  // Promise that resolves when Mermaid is loaded

function _ensureMermaid() {
  if (_mermaidReady) return _mermaidReady;
  _mermaidReady = new Promise((resolve) => {
    if (window.mermaid) {
      mermaid.initialize({ startOnLoad: false, theme: 'default', securityLevel: 'loose' });
      resolve();
      return;
    }
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/mermaid@11.6/dist/mermaid.min.js';
    script.onload = () => {
      mermaid.initialize({ startOnLoad: false, theme: 'default', securityLevel: 'loose' });
      resolve();
    };
    script.onerror = () => resolve();  // Don't block if CDN fails
    document.head.appendChild(script);
  });
  return _mermaidReady;
}

marked.setOptions({ breaks: true, gfm: true });

function renderMarkdown(text) {
  if (!text) return '';

  // Protect Mermaid blocks from marked processing
  const mermaidBlocks = [];
  let processed = text.replace(/```mermaid\n([\s\S]*?)```/g, (match, code) => {
    const idx = mermaidBlocks.length;
    mermaidBlocks.push(code);
    return `%%MERMAID_${idx}%%`;
  });

  let html = marked.parse(processed);

  // Restore Mermaid blocks
  html = html.replace(/<p>%%MERMAID_(\d+)%%<\/p>/g, (_, idx) => {
    const code = mermaidBlocks[parseInt(idx)];
    const id = 'mermaid-' + Math.random().toString(36).slice(2, 8);
    return `<div class="mindmap-container"><div class="mermaid" id="${id}">${code}</div><div class="mindmap-actions"><button onclick="exportMindmapPNG('${id}')">📥 Export PNG</button></div></div>`;
  });

  // Add copy buttons to code blocks
  html = html.replace(/<pre><code/g, '<pre><button class="copy-btn" onclick="copyCode(this)">Copy</button><code');

  // Render KaTeX
  setTimeout(() => {
    try {
      renderMathInElement(document.getElementById('chat-messages'), {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '$', right: '$', display: false },
          { left: '\\[', right: '\\]', display: true },
          { left: '\\(', right: '\\)', display: false },
        ],
        throwOnError: false,
      });
    } catch (e) {}
  }, 50);

  // Highlight code
  setTimeout(() => {
    document.querySelectorAll('pre code').forEach(block => {
      try { hljs.highlightElement(block); } catch (e) {}
    });
  }, 100);

  // Render Mermaid (lazy-loaded)
  if (mermaidBlocks.length > 0) {
    setTimeout(async () => {
      await _ensureMermaid();
      const mermaidEls = document.querySelectorAll('.mermaid');
      for (const el of mermaidEls) {
        if (el.getAttribute('data-processed')) continue;
        el.setAttribute('data-processed', 'true');
        try {
          const { svg } = await mermaid.render(el.id + '-svg', el.textContent);
          el.innerHTML = svg;
        } catch (e) {
          el.innerHTML = '<p style="color:var(--danger)">Mermaid render error</p><pre>' + el.textContent + '</pre>';
        }
      }
    }, 200);
  }

  return html;
}

function copyCode(btn) {
  const code = btn.parentElement.querySelector('code');
  if (!code) return;
  navigator.clipboard.writeText(code.textContent).then(() => {
    btn.textContent = 'Copied!';
    setTimeout(() => btn.textContent = 'Copy', 2000);
  });
}
