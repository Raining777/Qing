"""Auto-classify files into courses using a single Claude API call."""
import json
import logging
from pathlib import Path

from app.services.parser import parse_first_pages

logger = logging.getLogger(__name__)


async def classify_files(file_paths: list[str], llm) -> dict[str, str]:
    """Classify multiple files into courses with ONE API call.

    Returns: {filename: course_name}
    """
    if not file_paths:
        return {}

    # Collect first pages + filenames
    file_info = []
    for fp in file_paths:
        name = Path(fp).name
        preview = parse_first_pages(fp)
        file_info.append({"name": name, "preview": preview[:2000]})

    # Build prompt
    lines = []
    for fi in file_info:
        lines.append(f"FILE: {fi['name']}\nCONTENT PREVIEW:\n{fi['preview']}\n---")

    prompt = (
        "你正在帮助智能学习助手「清」整理用户上传的学习资料。"
        "请阅读每个文件名和内容预览，把文件归类到最合适的课程或学习主题。"
        "课程名要稳定、简洁、适合在侧边栏展示；如果资料里已有明确课程名，优先使用原课程名。"
        "类似资料要归到同一个课程名下。"
        "如果无法判断，请使用「其他」。\n\n"
        + "\n".join(lines)
        + "\n\n只返回合法 JSON，不要解释：[{\"file\": \"filename\", \"course\": \"课程名\"}, ...]"
    )

    try:
        response = await llm.chat(
            system="你是课程资料分类器。只输出合法 JSON，不要输出 Markdown 或解释。",
            messages=[{"role": "user", "content": prompt}],
        )
        # Parse JSON from response
        text = response.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.split("```")[0]
        text = text.strip()

        classifications = json.loads(text)
        return {item["file"]: item["course"] for item in classifications}
    except Exception as e:
        logger.error(f"Classification error: {e}")
        # Fallback: all files to "Other"
        return {Path(fp).name: "其他" for fp in file_paths}
