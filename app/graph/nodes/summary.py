"""Layered knowledge summary: chapter-by-chapter, then aggregate."""
import logging
from typing import AsyncIterator

from app.graph.state import QingState
from app.services.vectordb import get_course_chunks_with_meta
from app.services.llm import create_llm, get_task_llm

logger = logging.getLogger(__name__)


async def generate_summary(state: QingState) -> AsyncIterator[dict]:
    """Generate layered summary for a course.

    Strategy: Extract chapter structure, summarize each chapter, then aggregate.
    Chapter summaries are cached in state for reuse by mindmap/flashcard/formula.
    """
    course = state.get("target_course", "")
    provider_id = state.get("provider_id")
    model_name = state.get("model_name")

    if not course:
        yield {"response": "还没有指定要总结的课程。"}
        return

    chunks = get_course_chunks_with_meta(course)
    if not chunks:
        yield {"response": f"还没有找到「{course}」的资料索引，请先上传文件。"}
        return

    llm = create_llm(provider_id, model_name)
    full_text = "\n\n".join(c["text"] for c in chunks)

    # If text is short enough, summarize directly
    if len(full_text) < 15000:
        yield {"response": "# 正在生成复习总结...\n\n"}
        prompt = SUMMARY_DIRECT_PROMPT.format(course=course, content=full_text[:15000])
        result = ""
        async for token in llm.chat_stream(SYSTEM, [{"role": "user", "content": prompt}]):
            result += token
            yield {"response": result}
        yield {"response": result, "chapter_summaries": {course: {"all": result}}}
        return

    # Large course: detect chapters, summarize each, then aggregate
    yield {"response": "## 正在分析资料结构...\n\n"}
    chapters = await _detect_chapters(full_text, llm)

    yield {"response": f"## 找到 {len(chapters)} 个部分，正在逐段总结...\n\n"}

    chapter_summaries = {}
    for i, (title, text) in enumerate(chapters):
        yield {"response": f"**正在总结：{title}**（{i+1}/{len(chapters)}）...\n\n"}
        cheap_llm = get_task_llm("summary", provider_id, model_name) if i > 2 else llm  # Use cheaper for later chapters
        summary = await cheap_llm.chat(SYSTEM, [{"role": "user", "content": CHAPTER_PROMPT.format(title=title, content=text[:4000])}])
        chapter_summaries[title] = summary

    # Aggregate
    yield {"response": "\n\n## 📋 完整复习总结\n\n"}
    agg_prompt = AGGREGATE_PROMPT.format(course=course, summaries=_format_summaries(chapter_summaries))
    result = ""
    async for token in llm.chat_stream(SYSTEM, [{"role": "user", "content": agg_prompt}]):
        result += token
        yield {"response": result}

    yield {"response": result, "chapter_summaries": {course: chapter_summaries}}


async def _detect_chapters(text: str, llm) -> list[tuple[str, str]]:
    """Detect chapter/section boundaries using headers or Claude."""
    # Try header-based detection first (0 API cost)
    import re
    headers = re.findall(r'(?m)^(?:#{1,3}|Chapter|CHAPTER|Section|SECTION)\s*(.+)', text[:5000])
    if headers:
        chapters = []
        for h in headers:
            start = text.find(h)
            if start >= 0:
                next_start = min(
                    (s for x in headers if (s := text.find(x, start + len(h))) > 0),
                    default=len(text)
                )
                chapters.append((h.strip(), text[start:next_start]))
        if chapters:
            return chapters[:15]

    # Fallback: split by double newline groups and use first line as title
    sections = text.split("\n\n\n")
    result = []
    for i, section in enumerate(sections[:15]):
        lines = section.strip().split("\n")
        title = lines[0][:80] if lines else f"第 {i+1} 部分"
        result.append((title, section[:5000]))
    return result


def _format_summaries(summaries: dict) -> str:
    return "\n\n".join(f"## {title}\n{text}" for title, text in summaries.items())


SYSTEM = (
    "你是清，一个智能学习助手，擅长把课程资料整理成可复习、可记忆、可练习的学习材料。"
    "请默认使用中文，使用 Markdown 排版，公式用 LaTeX。"
    "总结时要抓住考试重点、概念联系、易错点和复习路径。"
)

SUMMARY_DIRECT_PROMPT = """请根据下面的资料，为课程「{course}」生成一份完整的中文复习总结。

要求：
- 优先依据资料本身，不要凭空扩写资料里没有的事实。
- 如果资料明显缺失，请指出缺失处，并给出复习时需要补查的方向。
- 语言要适合学生复习，重点清楚，解释准确。

请按以下结构输出：

## 📋 知识地图
按主题列出：核心概念、前置知识、⭐ 重要程度（1-5）、相关知识点。

## 🔍 核心概念详解
- 定义、定理、公式（行内公式用 $...$，块级公式用 $$...$$）
- 有代码或算法时给出简短示例
- 容易混淆的点和正确理解

## 🧠 复习抓手
把本课程最值得记住的内容整理成清单，说明为什么重要。

## ⚠️ 易错点与考试陷阱
列出常见误区、陷阱题思路和纠正方法。

## 📝 可能考法
给出 5 道不同类型的潜在考题，并标注考察点。

## ✅ 最后复习清单
用勾选清单列出考前需要确认自己掌握的内容。

资料：
{content}"""

CHAPTER_PROMPT = """请用中文简洁总结这一节「{title}」，输出适合复习的内容：
- 3-5 个关键点，并给出必要定义
- 重要公式、定理或代码逻辑（公式用 LaTeX）
- 这一节最容易问成什么题
- 常见误区或容易漏掉的细节
- 如果适用，补充一个实际应用场景

资料：
{content}"""

AGGREGATE_PROMPT = """请根据以下章节总结，为「{course}」整理一份最终中文复习指南：

{summaries}

请按以下结构输出：
## 📋 完整复习总结
## 🔍 最重要概念排行
按重要性排序，并说明每个概念为什么重要。
## 🔗 知识联系图（文字版）
说明章节之间、概念之间的依赖关系。
## ⚠️ Top 10 易错点
每条包含：错误理解、正确理解、复习提醒。
## 📝 练习题
给出覆盖核心知识的练习题，并附简短答案或解题方向。
## ✅ 考前检查清单
用清单形式列出最后一轮复习必须确认的内容。
"""
