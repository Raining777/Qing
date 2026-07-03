"""Generation handlers: summary, mindmap, plan, practice, flashcard, formula,
compare, mnemonic, sprint. All use external LLM APIs — no local models."""
import logging
from typing import AsyncIterator

from app.services.vectordb import get_course_chunks_text, get_course_chunks_with_meta, hybrid_search
from app.services.embedder import get_embedder
from app.services.llm import create_llm
from app.history import get_sessions

logger = logging.getLogger(__name__)

SYSTEM = (
    "你是清，一个智能学习助手，擅长总结资料、设计复习路径、生成练习题并帮助学生查漏补缺。"
    "请默认使用中文回复，使用 Markdown 格式；公式用 LaTeX，代码使用代码块。"
    "输出要适合复习：重点明确、结构清楚、步骤可执行。"
)


# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════

async def generate_summary(
    course: str, provider_id: str = "", model_name: str = "", session_id: str = "",
) -> AsyncIterator[dict]:
    """Layered course summary: chapters → per-chapter → aggregate."""
    chunks = get_course_chunks_with_meta(course)
    if not chunks:
        yield {"response": f"还没有找到「{course}」的资料索引，请先上传文件。"}
        return

    llm = create_llm(provider_id, model_name)
    full_text = "\n\n".join(c["text"] for c in chunks)

    # Short text: direct summary
    if len(full_text) < 15000:
        yield {"response": "# 正在生成复习总结...\n\n"}
        result = ""
        async for token in llm.chat_stream(SYSTEM, [{"role": "user", "content": SUMMARY_PROMPT.format(
            course=course, content=full_text[:15000]
        )}]):
            result += token
            yield {"delta": token}
        yield {"response": result}

        # Cache summaries
        if session_id:
            get_sessions().set_summaries(session_id, {course: {"all": result}})
        return

    # Large course: detect chapters
    yield {"response": "## 正在分析资料结构...\n\n"}
    chapters = _detect_chapters(full_text)
    yield {"response": f"## 找到 {len(chapters)} 个部分，正在逐段总结...\n\n"}

    chapter_summaries = {}
    for i, (title, text) in enumerate(chapters):
        yield {"response": f"**正在总结：{title}**（{i+1}/{len(chapters)}）...\n\n"}
        summary = await llm.chat(SYSTEM, [{"role": "user", "content": CHAPTER_PROMPT.format(
            title=title, content=text[:4000]
        )}])
        chapter_summaries[title] = summary

    # Aggregate
    yield {"response": "\n\n## 📋 完整复习总结\n\n"}
    agg_prompt = AGGREGATE_PROMPT.format(course=course, summaries=_format_summaries(chapter_summaries))
    result = ""
    async for token in llm.chat_stream(SYSTEM, [{"role": "user", "content": agg_prompt}]):
        result += token
        yield {"delta": token}
    yield {"response": result}

    if session_id:
        get_sessions().set_summaries(session_id, {course: chapter_summaries})


# ═══════════════════════════════════════════════════════════════════════
# Mindmap
# ═══════════════════════════════════════════════════════════════════════

async def generate_mindmap(
    course: str, provider_id: str = "", model_name: str = "", session_id: str = "",
) -> AsyncIterator[dict]:
    summaries = get_sessions().get_summaries(session_id).get(course, {}) if session_id else {}
    llm = create_llm(provider_id, model_name)

    if summaries:
        text = "\n\n".join(f"## {t}\n{s}" for t, s in summaries.items())[:5000]
    else:
        text = get_course_chunks_text(course)[:8000]

    prompt = (
        f"请根据下面内容，为「{course}」生成一张 Mermaid 思维导图。\n"
        f"要求使用 mindmap 语法，层级 2-3 层，节点标签要短，突出复习主线和概念关系。\n\n{text}\n\n"
        "只输出 Mermaid 代码块（```mermaid ... ```），不要添加额外解释。"
    )

    yield {"response": "正在生成思维导图...\n\n"}
    result = ""
    async for token in llm.chat_stream(SYSTEM, [{"role": "user", "content": prompt}]):
        result += token
        yield {"delta": token}
    yield {"response": result}


# ═══════════════════════════════════════════════════════════════════════
# Study Plan
# ═══════════════════════════════════════════════════════════════════════

async def generate_plan(
    course: str, exam_date: str = "", provider_id: str = "", model_name: str = "", session_id: str = "",
) -> AsyncIterator[dict]:
    summaries = get_sessions().get_summaries(session_id).get(course, {}) if session_id else {}
    topics_text = "\n".join(f"- {t}" for t in summaries.keys()) if summaries else "Topics from uploaded materials"
    llm = create_llm(provider_id, model_name)

    prompt = (
        f"请为课程「{course}」制定一份中文复习计划。\n"
        f"考试日期：{exam_date or '未指定，请按未来 4 周规划'}\n"
        f"需要覆盖的主题：\n{topics_text}\n\n"
        "请按以下结构输出：\n"
        "## 📅 复习计划\n"
        "### 阶段一：建立框架（前 40% 时间）\n"
        "- 每天列出具体主题、学习动作和预计用时\n"
        "### 阶段二：深入理解（中间 30% 时间）\n"
        "- 聚焦难点、概念联系、典型题型\n"
        "### 阶段三：练习与回顾（最后 30% 时间）\n"
        "- 安排刷题、错题复盘、模拟检查\n\n"
        "### 🔄 间隔复习安排\n"
        "### ✅ 每日执行表\n"
        "每一天都要写清：主题、动作（阅读/总结/练题/回顾）、预计用时、完成标准。"
    )

    yield {"response": "正在生成复习计划...\n\n"}
    result = ""
    async for token in llm.chat_stream(SYSTEM, [{"role": "user", "content": prompt}]):
        result += token
        yield {"delta": token}
    yield {"response": result}


# ═══════════════════════════════════════════════════════════════════════
# Practice Questions
# ═══════════════════════════════════════════════════════════════════════

async def generate_practice(
    course: str, topic: str = "", count: int = 5, qtype: str = "mixed",
    provider_id: str = "", model_name: str = "",
) -> AsyncIterator[dict]:
    embedder = get_embedder()
    query = topic or f"key concepts in {course}"
    q_emb = await embedder.encode_query(query)
    docs = hybrid_search(course, q_emb, query, top_k=5)
    context = "\n\n".join(d["content"][:500] for d in docs) if docs else get_course_chunks_text(course)[:3000]
    llm = create_llm(provider_id, model_name)

    prompt = (
        f"请为课程「{course}」生成 {count} 道中文练习题" + (f"，主题是「{topic}」" if topic else "") + "。\n\n"
        f"参考资料：\n{context[:4000]}\n\n"
        f"题型要求：{qtype if qtype != 'mixed' else '混合题型，包括选择题、简答题，若适合则加入代码题'}。\n\n"
        "每道题请包含：题目、答案（详细解析）、难度（简单/中等/困难）、考察点、易错提醒。\n"
        "如果课程内容适合代码题，至少加入 1 道代码或算法题。\n"
        "请使用 ## 第 N 题 作为标题。"
    )

    yield {"response": "正在生成练习题...\n\n"}
    result = ""
    async for token in llm.chat_stream(SYSTEM, [{"role": "user", "content": prompt}]):
        result += token
        yield {"delta": token}
    yield {"response": result}


# ═══════════════════════════════════════════════════════════════════════
# Flashcards
# ═══════════════════════════════════════════════════════════════════════

async def generate_flashcards(
    course: str, provider_id: str = "", model_name: str = "", session_id: str = "",
) -> AsyncIterator[dict]:
    summaries = get_sessions().get_summaries(session_id).get(course, {}) if session_id else {}
    text = "\n\n".join(f"## {t}\n{s}" for t, s in summaries.items())[:5000] if summaries else get_course_chunks_text(course)[:5000]
    llm = create_llm(provider_id, model_name)

    prompt = (
        f"请根据下面内容，为「{course}」创建 15-20 张中文复习闪卡。\n\n{text}\n\n"
        "请返回 CSV，列名固定为：front,back\n"
        "front = 问题 / 概念 / 术语；back = 简洁答案 / 定义 / 公式 / 关键解释。\n"
        "包含逗号或换行的字段请用双引号包裹。\n"
        "只输出合法 CSV：\n"
    )

    yield {"response": "正在生成复习闪卡...\n\n"}
    result = ""
    async for token in llm.chat_stream(SYSTEM, [{"role": "user", "content": prompt}]):
        result += token
        yield {"delta": token}
    yield {"response": result}


# ═══════════════════════════════════════════════════════════════════════
# Formula Sheet
# ═══════════════════════════════════════════════════════════════════════

async def generate_formulas(
    course: str, provider_id: str = "", model_name: str = "", session_id: str = "",
) -> AsyncIterator[dict]:
    summaries = get_sessions().get_summaries(session_id).get(course, {}) if session_id else {}
    text = "\n\n".join(f"## {t}\n{s}" for t, s in summaries.items())[:5000] if summaries else get_course_chunks_text(course)[:5000]
    llm = create_llm(provider_id, model_name)

    prompt = (
        f"请从「{course}」的内容中提取所有公式、定理、关键等式和重要符号定义。\n"
        f"请按主题分组，公式使用 LaTeX（行内 $...$，块级 $$...$$）。\n"
        f"每个条目包含：名称、公式、符号含义、什么时候使用、常见误用或注意点。\n\n"
        f"资料：\n{text}"
    )

    yield {"response": "正在整理公式表...\n\n"}
    result = ""
    async for token in llm.chat_stream(SYSTEM, [{"role": "user", "content": prompt}]):
        result += token
        yield {"delta": token}
    yield {"response": result}


# ═══════════════════════════════════════════════════════════════════════
# Concept Comparison
# ═══════════════════════════════════════════════════════════════════════

async def generate_compare(
    course: str, concept_a: str, concept_b: str,
    provider_id: str = "", model_name: str = "",
) -> AsyncIterator[dict]:
    embedder = get_embedder()
    q_emb = await embedder.encode_query(f"{concept_a} vs {concept_b}")
    docs = hybrid_search(course, q_emb, f"{concept_a} {concept_b}", top_k=4)
    context = "\n\n".join(d["content"][:400] for d in docs) if docs else get_course_chunks_text(course)[:2000]
    llm = create_llm(provider_id, model_name)

    prompt = (
        f"请结合「{course}」课程语境，对比「{concept_a}」和「{concept_b}」。\n\n"
        f"参考资料：\n{context[:3000]}\n\n"
        "请输出：\n## 对比表\n使用 Markdown 表格，从 6-10 个维度比较。\n"
        "## 怎么区分\n## 什么时候用哪个\n## 易错点\n"
        "如果适用，请加入简短代码示例。"
    )

    yield {"response": "正在生成概念对比...\n\n"}
    result = ""
    async for token in llm.chat_stream(SYSTEM, [{"role": "user", "content": prompt}]):
        result += token
        yield {"delta": token}
    yield {"response": result}


# ═══════════════════════════════════════════════════════════════════════
# Mnemonic
# ═══════════════════════════════════════════════════════════════════════

async def generate_mnemonic(
    query: str, provider_id: str = "", model_name: str = "",
) -> AsyncIterator[dict]:
    concept = query.replace("give me a mnemonic for", "").replace("mnemonic for", "") \
        .replace("给我一个", "").replace("帮我记住", "").replace("记忆口诀", "").replace("记忆方法", "").strip()
    llm = create_llm(provider_id, model_name)

    prompt = (
        f"请为这个知识点设计中文记忆方法：{concept}\n"
        "请包含：口诀/缩写/谐音/画面化联想、如何使用、为什么有效、一个快速自测问题。\n"
        "要求好记、准确，不要为了押韵牺牲知识正确性。"
    )

    yield {"response": "正在生成记忆方法...\n\n"}
    result = ""
    async for token in llm.chat_stream(SYSTEM, [{"role": "user", "content": prompt}]):
        result += token
        yield {"delta": token}
    yield {"response": result}


# ═══════════════════════════════════════════════════════════════════════
# Exam Sprint
# ═══════════════════════════════════════════════════════════════════════

async def generate_sprint(
    course: str, provider_id: str = "", model_name: str = "", session_id: str = "",
) -> AsyncIterator[dict]:
    summaries = get_sessions().get_summaries(session_id).get(course, {}) if session_id else {}
    text = "\n\n".join(f"## {t}\n{s}" for t, s in summaries.items())[:6000] if summaries else get_course_chunks_text(course)[:6000]
    llm = create_llm(provider_id, model_name)

    prompt = (
        f"请为「{course}」生成一份中文考前冲刺材料。资料如下：\n{text}\n\n"
        "请输出：\n## 🎯 快速回顾\n每个主题用 1-2 句话说明必须掌握什么。\n"
        "## ⚠️ Top 10 考试陷阱与误区\n"
        "## 📝 5 道必会题\n"
        "## 🔥 临考速记清单\n"
        "要求极其精炼，适合最后一轮复习。"
    )

    yield {"response": "正在准备考前冲刺材料...\n\n"}
    result = ""
    async for token in llm.chat_stream(SYSTEM, [{"role": "user", "content": prompt}]):
        result += token
        yield {"delta": token}
    yield {"response": result}


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _detect_chapters(text: str) -> list[tuple[str, str]]:
    """Detect chapter/section boundaries using headers."""
    import re
    header_re = re.compile(
        r'(?m)^(?:#{1,3}\s+|Chapter\s+\d+|CHAPTER\s+\d+|Section\s+\d+|SECTION\s+\d+)\s*(.+)',
    )
    matches = list(header_re.finditer(text))

    if matches and len(matches) >= 2:
        chapters = []
        for i, m in enumerate(matches):
            title = m.group(1).strip() or m.group(0).strip()
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            if len(content) > 100:
                chapters.append((title, content[:5000]))
        if chapters:
            return chapters[:15]

    # Fallback: split by triple newlines
    sections = text.split("\n\n\n")
    result = []
    for i, section in enumerate(sections[:15]):
        lines = section.strip().split("\n")
        title = lines[0][:80] if lines else f"第 {i+1} 部分"
        content = section.strip()
        if len(content) > 50:
            result.append((title, content[:5000]))
    return result


def _format_summaries(summaries: dict) -> str:
    return "\n\n".join(f"## {title}\n{text}" for title, text in summaries.items())


# ── Prompt templates ──

SUMMARY_PROMPT = """请根据下面的资料，为课程「{course}」生成一份完整的中文复习总结。

要求：
- 优先依据资料本身，不要凭空扩写资料里没有的事实。
- 语言要适合学生复习，重点清楚，解释准确。

请按以下结构输出：
## 📋 知识地图
按主题列出：核心概念、前置知识、⭐ 重要程度（1-5）、相关知识点。

## 🔍 核心概念详解
- 定义、定理、公式（行内 $...$，块级 $$...$$）
- 有代码或算法时给出简短示例
- 容易混淆的点和正确理解

## 🧠 复习抓手
## ⚠️ 易错点与考试陷阱
## 📝 可能考法
## ✅ 最后复习清单

资料：
{content}"""

CHAPTER_PROMPT = """请用中文简洁总结这一节「{title}」：
- 3-5 个关键点，并给出必要定义
- 重要公式、定理或代码逻辑（公式用 LaTeX）
- 容易问成什么题
- 常见误区
- 如果适用，补充一个实际应用场景

资料：
{content}"""

AGGREGATE_PROMPT = """请根据以下章节总结，为「{course}」整理一份最终中文复习指南：

{summaries}

请按以下结构输出：
## 📋 完整复习总结
## 🔍 最重要概念排行
## 🔗 知识联系图（文字版）
## ⚠️ Top 10 易错点
## 📝 练习题
## ✅ 考前检查清单
"""


# ── Intent → handler mapping ──

INTENT_HANDLERS = {
    "summary":   generate_summary,
    "mindmap":   generate_mindmap,
    "plan":      generate_plan,
    "practice":  generate_practice,
    "flashcard": generate_flashcards,
    "formula":   generate_formulas,
    "compare":   generate_compare,
    "mnemonic":  generate_mnemonic,
    "sprint":    generate_sprint,
}
