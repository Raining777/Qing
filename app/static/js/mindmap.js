/* mindmap.js — Mermaid mindmap rendering and export */

async function renderMindmap(containerId, mermaidCode) {
  const container = document.getElementById(containerId);
  if (!container) return;

  container.innerHTML = '';
  try {
    const { svg } = await mermaid.render(containerId + '-svg', mermaidCode);
    container.innerHTML = svg;
  } catch (e) {
    container.innerHTML = `<p style="color:var(--danger)">Render error: ${e.message}</p><pre>${escapeHtml(mermaidCode)}</pre>`;
  }
}

async function exportMindmapPNG(containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;

  try {
    // Use html2canvas if available, or download SVG directly
    const svgEl = container.querySelector('svg');
    if (svgEl) {
      // Download SVG
      const svgData = new XMLSerializer().serializeToString(svgEl);
      const svgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
      const url = URL.createObjectURL(svgBlob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'mindmap.svg';
      a.click();
      URL.revokeObjectURL(url);
    }
  } catch (e) {
    console.error('Export error:', e);
    alert('Export failed. Try right-clicking the mindmap and saving.');
  }
}
