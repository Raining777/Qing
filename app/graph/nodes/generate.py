"""Generation nodes: mindmap, plan, practice, flashcard, formula, compare, mnemonic, sprint.

All use cached chapter summaries when available to minimize token usage.
"""
import logging
from typing import AsyncIterator

from app.graph.state import QingState
from app.services.vectordb import get_course_chunks_text, get_course_chunks_with_meta
from app.services.llm import create_llm, get_task_llm

logger = logging.getLogger(__name__)

SYSTEM = (
    "你是清，一个智能学习助手，擅长总结资料、设计复习路径、生成练习题并帮助学生查漏补缺。"
    "请默认使用中文回复，使用 Markdown 格式；公式用 LaTeX，代码使用代码块。"
    "输出要适合复习：重点明确、结构清楚、步骤可执行。"
)


# ── Mindmap ──

async def generate_mindmap(state: QingState) -> AsyncIterator[dict]:
    """Generate Mermaid mindmap from chapter summaries or full text."""
    course = state.get("target_course", "")
    summaries = state.get("chapter_summaries", {}).get(course, {})

    llm = create_llm(state.get("provider_id"), state.get("model_name"))

    if summaries:
        summary_text = "\n\n".join(f"## {t}\n{s}" for t, s in summaries.items())
        prompt = (
            f"请根据这些章节总结，为「{course}」生成一张 Mermaid 思维导图。\n"
            f"要求使用 mindmap 语法，层级 2-3 层，节点标签要短，突出复习主线和概念关系。\n\n{summary_text[:5000]}\n\n"
            "只输出 Mermaid 代码块（```mermaid ... ```），不要添加额外解释。"
        )
    else:
        text = get_course_chunks_text(course)[:8000]
        prompt = (
            f"请根据下面资料，为「{course}」生成一张 Mermaid 思维导图。\n"
            f"要求使用 mindmap 语法，层级 2-3 层，节点标签要短，突出复习主线和概念关系。\n\n{text}\n\n"
            "只输出 Mermaid 代码块（```mermaid ... ```），不要添加额外解释。"
        )

    yield {"response": "正在生成思维导图...\n\n"}

    result = ""
    async for token in llm.chat_stream(SYSTEM, [{"role": "user", "content": prompt}]):
        result += token
        yield {"response": result}

    yield {"response": result}


# ── Study Plan ──

async def generate_plan(state: QingState) -> AsyncIterator[dict]:
    """Generate spaced-repetition study plan with SM-2 reminders."""
    course = state.get("target_course", "")
    exam_date = state.get("exam_date", "")
    summaries = state.get("chapter_summaries", {}).get(course, {})

    llm = create_llm(state.get("provider_id"), state.get("model_name"))

    if summaries:
        topics_text = "\n".join(f"- {t}" for t in summaries.keys())
    else:
        topics_text = "Topics from uploaded materials"

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
        "- 第一次学习后：1 天后复习\n"
        "- 第二次复习后：3 天后复习\n"
        "- 第三次复习后：7 天后复习\n"
        "- 第四次复习后：14 天后复习\n\n"
        "### ✅ 每日执行表\n"
        "每一天都要写清：主题、动作（阅读/总结/练题/回顾）、预计用时、完成标准。"
    )

    yield {"response": "正在生成复习计划...\n\n"}

    result = ""
    async for token in llm.chat_stream(SYSTEM, [{"role": "user", "content": prompt}]):
        result += token
        yield {"response": result}

    yield {"response": result}


# ── Practice Questions ──

async def generate_practice(state: QingState) -> AsyncIterator[dict]:
    """Generate practice questions from retrieval, not full text."""
    course = state.get("target_course", "")
    topic = state.get("practice_topic", "")
    count = state.get("practice_count", 5)
    qtype = state.get("practice_type", "mixed")

    # Use retrieval to find relevant content (token-efficient)
    from app.services.embedder import get_embedder
    from app.services.vectordb import hybrid_search

    query = topic or f"key concepts in {course}"
    embedder = get_embedder()
    q_emb = embedder.encode_query(query)
    docs = hybrid_search(course, q_emb, query, top_k=5)

    context = "\n\n".join(d["content"][:500] for d in docs) if docs else get_course_chunks_text(course)[:3000]

    llm = create_llm(state.get("provider_id"), state.get("model_name"))

    prompt = (
        f"请为课程「{course}」生成 {count} 道中文练习题" + (f"，主题是「{topic}」" if topic else "") + "。\n\n"
        f"参考资料：\n{context[:4000]}\n\n"
        f"题型要求：{qtype if qtype != 'mixed' else '混合题型，包括选择题、简答题，若适合则加入代码题'}。\n\n"
        "每道题请包含：\n"
        "- 题目：表述清楚，贴近资料\n"
        "- 答案：给出详细解析\n"
        "- 难度：简单 / 中等 / 困难\n"
        "- 考察点：说明对应知识点\n"
        "- 易错提醒：指出常见错误思路\n"
        "如果课程内容适合代码题，至少加入 1 道代码或算法题。\n"
        "请使用 ## 第 N 题 作为标题。"
    )

    yield {"response": "正在生成练习题...\n\n"}

    result = ""
    async for token in llm.chat_stream(SYSTEM, [{"role": "user", "content": prompt}]):
        result += token
        yield {"response": result}

    yield {"response": result}


# ── Flashcards ──

async def generate_flashcards(state: QingState) -> AsyncIterator[dict]:
    """Generate flashcards from chapter summaries."""
    course = state.get("target_course", "")
    summaries = state.get("chapter_summaries", {}).get(course, {})

    if summaries:
        text = "\n\n".join(f"## {t}\n{s}" for t, s in summaries.items())[:5000]
    else:
        text = get_course_chunks_text(course)[:5000]

    llm = create_llm(state.get("provider_id"), state.get("model_name"))

    prompt = (
        f"请根据下面内容，为「{course}」创建 15-20 张中文复习闪卡。\n\n{text}\n\n"
        "请返回 CSV，列名固定为：front,back\n"
        "front = 问题 / 概念 / 术语；back = 简洁答案 / 定义 / 公式 / 关键解释。\n"
        "要覆盖核心概念、易混点和关键公式；有代码知识时可以包含短代码片段。\n"
        "包含逗号或换行的字段请用双引号包裹。\n"
        "只输出合法 CSV，不要输出 Markdown 标题或代码块：\n"
    )

    yield {"response": "正在生成复习闪卡...\n\n"}

    result = ""
    async for token in llm.chat_stream(SYSTEM, [{"role": "user", "content": prompt}]):
        result += token
        yield {"response": result}

    yield {"response": result}


# ── Formula Sheet ──

async def generate_formulas(state: QingState) -> AsyncIterator[dict]:
    """Extract all formulas from chapter summaries."""
    course = state.get("target_course", "")
    summaries = state.get("chapter_summaries", {}).get(course, {})

    if summaries:
        text = "\n\n".join(f"## {t}\n{s}" for t, s in summaries.items())[:5000]
    else:
        text = get_course_chunks_text(course)[:5000]

    llm = create_llm(state.get("provider_id"), state.get("model_name"))

    prompt = (
        f"请从「{course}」的内容中提取所有公式、定理、关键等式和重要符号定义。\n"
        f"请按主题分组，公式使用 LaTeX（行内公式用 $...$，块级公式用 $$...$$）。\n"
        f"每个条目包含：\n"
        f"- 名称或描述\n"
        f"- 公式本身\n"
        f"- 符号含义\n"
        f"- 什么时候使用\n"
        f"- 常见误用或注意点\n\n"
        f"资料：\n{text}"
    )

    yield {"response": "正在整理公式表...\n\n"}

    result = ""
    async for token in llm.chat_stream(SYSTEM, [{"role": "user", "content": prompt}]):
        result += token
        yield {"response": result}

    yield {"response": result}


# ── Concept Comparison ──

async def generate_compare(state: QingState) -> AsyncIterator[dict]:
    """Generate comparison table for two concepts."""
    course = state.get("target_course", "")
    concept_a = state.get("concept_a", "")
    concept_b = state.get("concept_b", "")

    # Search for relevant context
    from app.services.embedder import get_embedder
    from app.services.vectordb import hybrid_search
    embedder = get_embedder()
    q_emb = embedder.encode_query(f"{concept_a} vs {concept_b}")
    docs = hybrid_search(course, q_emb, f"{concept_a} {concept_b}", top_k=4)

    context = "\n\n".join(d["content"][:400] for d in docs) if docs else get_course_chunks_text(course)[:2000]

    llm = create_llm(state.get("provider_id"), state.get("model_name"))

    prompt = (
        f"请结合「{course}」课程语境，对比「{concept_a}」和「{concept_b}」。\n\n"
        f"参考资料：\n{context[:3000]}\n\n"
        "请输出：\n"
        "## 对比表\n"
        "使用 Markdown 表格，从 6-10 个维度比较。\n"
        "## 怎么区分\n"
        "给出学生最容易理解的判断方法。\n"
        "## 什么时候用哪个\n"
        "说明适用场景、取舍和限制。\n"
        "## 易错点\n"
        "列出常见混淆和纠正方式。\n"
        "如果适用，请加入简短代码示例。"
    )

    yield {"response": "正在生成概念对比...\n\n"}

    result = ""
    async for token in llm.chat_stream(SYSTEM, [{"role": "user", "content": prompt}]):
        result += token
        yield {"response": result}

    yield {"response": result}


# ── Mnemonic ──

async def generate_mnemonic(state: QingState) -> AsyncIterator[dict]:
    """Generate memory aids for concepts."""
    query = state.get("query", "")
    # Extract concept from query: "give me a mnemonic for OSI 7 layers" -> "OSI 7 layers"
    concept = (
        query.replace("give me a mnemonic for", "")
        .replace("mnemonic for", "")
        .replace("给我一个", "")
        .replace("帮我记住", "")
        .replace("记忆口诀", "")
        .replace("记忆方法", "")
        .strip()
    )

    llm = create_llm(state.get("provider_id"), state.get("model_name"))

    prompt = (
        f"请为这个知识点设计中文记忆方法：{concept}\n"
        "请包含：\n"
        "1. 口诀、缩写、谐音或画面化联想\n"
        "2. 如何使用这个记忆法\n"
        "3. 为什么它有效\n"
        "4. 一个快速自测问题\n"
        "要求好记、准确，不要为了押韵牺牲知识正确性。"
    )

    yield {"response": "正在生成记忆方法...\n\n"}

    result = ""
    async for token in llm.chat_stream(SYSTEM, [{"role": "user", "content": prompt}]):
        result += token
        yield {"response": result}

    yield {"response": result}


# ── Exam Sprint ──

async def generate_sprint(state: QingState) -> AsyncIterator[dict]:
    """Pre-exam sprint: weak-point diagnosis + quick review + trap list."""
    course = state.get("sprint_course", "") or state.get("target_course", "")
    summaries = state.get("chapter_summaries", {}).get(course, {})

    if summaries:
        text = "\n\n".join(f"## {t}\n{s}" for t, s in summaries.items())[:6000]
    else:
        text = get_course_chunks_text(course)[:6000]

    llm = create_llm(state.get("provider_id"), state.get("model_name"))

    prompt = (
        f"请为「{course}」生成一份中文考前冲刺材料。资料如下：\n{text}\n\n"
        "请输出：\n"
        "## 🎯 快速回顾\n"
        "每个主题用 1-2 句话说明必须掌握什么。\n"
        "## ⚠️ Top 10 考试陷阱与误区\n"
        "写出错误想法、正确理解和提醒。\n"
        "## 📝 5 道必会题\n"
        "给出简短答案或解题方向。\n"
        "## 🔥 临考速记清单\n"
        "整理关键公式、定义、步骤和判断方法。\n"
        "要求极其精炼，适合最后一轮复习。"
    )

    yield {"response": "正在准备考前冲刺材料...\n\n"}

    result = ""
    async for token in llm.chat_stream(SYSTEM, [{"role": "user", "content": prompt}]):
        result += token
        yield {"response": result}

    yield {"response": result}
