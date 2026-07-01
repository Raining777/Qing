/* courses.js — 课程侧栏 */

async function loadCourses() {
  try {
    const resp = await fetch('/api/courses');
    const courses = await resp.json();
    renderCourses(courses);
  } catch (e) {}
}

function renderCourses(courses) {
  const container = document.getElementById('courses-container');
  if (!courses || !courses.length) {
    container.innerHTML = '<div class="course-item" style="color:var(--text-hint)">还没有课程<br>上传文件自动分类</div>';
    return;
  }
  container.innerHTML = courses.map(c => `
    <div class="course-item ${c.name === STATE.activeCourse ? 'active' : ''}" onclick="selectCourse('${escapeHtml(c.name)}')">
      <span class="course-dot ready"></span>
      <span>${escapeHtml(c.name)}</span>
      <span style="font-size:11px;color:var(--text-hint)">${c.file_count}文件 ${c.chunk_count}条</span>
      <span class="course-actions">
        <button class="course-btn" onclick="event.stopPropagation();quickMindmap('${escapeHtml(c.name)}')" title="思维导图">🧠</button>
        <button class="course-btn" onclick="event.stopPropagation();quickSummary('${escapeHtml(c.name)}')" title="知识点总结">📝</button>
        <button class="course-btn" onclick="event.stopPropagation();deleteCourse('${escapeHtml(c.name)}')" title="删除">🗑</button>
      </span>
    </div>
  `).join('');
}

function selectCourse(name) {
  STATE.activeCourse = name;
  loadCourses();
  addMessage('assistant', `已切换到 **${escapeHtml(name)}**，可以开始提问了。`);
}

async function quickMindmap(course) {
  STATE.activeCourse = course; loadCourses();
  addUserMessage(`🧠 生成思维导图: ${course}`);
  await streamAction('/api/mindmap', { course, provider: STATE.activeProvider, model: STATE.activeModel });
}

async function quickSummary(course) {
  STATE.activeCourse = course; loadCourses();
  addUserMessage(`📝 总结知识点: ${course}`);
  await streamAction('/api/summary', { course, provider: STATE.activeProvider, model: STATE.activeModel });
}

async function deleteCourse(name) {
  if (!confirm(`确定要删除「${name}」及其全部数据吗？`)) return;
  await fetch(`/api/courses/${encodeURIComponent(name)}`, { method: 'DELETE' });
  if (STATE.activeCourse === name) STATE.activeCourse = '';
  loadCourses();
}

function escapeHtml(s) {
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}
