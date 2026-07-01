"""RAG answer generation with streaming support and full conversation context."""
import logging
from typing import AsyncIterator

from langchain_core.messages import HumanMessage, AIMessage

from app.graph.state import QingState
from app.services.llm import create_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "你是清，一个智能学习助手，擅长把资料整理成清晰的总结、复习线索和可执行的学习建议。\n"
    "你的目标是帮助用户真正学会：先讲清核心结论，再补齐概念、步骤、例子、易错点和复习建议。\n\n"
    "回答规则：\n"
    "1. 默认必须用中文回答；除非用户明确要求其他语言。\n"
    "2. 优先依据用户上传的参考资料作答，并在能判断来源时标注文件名、页码或片段来源。\n"
    "3. 如果资料里没有足够依据，要明确说明「资料中没有找到直接依据」，再给出通用解释或推测。\n"
    "4. 输出要适合复习：结构清晰、重点突出，必要时给出公式、代码、例题、记忆方法或检查清单。\n"
    "5. 公式使用 LaTeX，代码使用 Markdown 代码块。"
)

# Additional context note when continuing a conversation
CONTINUITY_NOTE = (
    "\n\n## 对话上下文\n"
    "用户正在进行多轮对话。请结合前面的对话内容理解当前问题，"
    "在回答时保持连贯性，可以用「你刚才问的...」「前面提到...」来呼应之前的讨论。"
)


def _extract_history_for_llm(messages, max_turns: int = 10) -> list[dict]:
    """Extract conversation history as dict messages for the LLM.

    Handles both LangChain message objects and plain dicts.
    Returns up to max_turns * 2 messages (user + assistant pairs).
    """
    max_msgs = max_turns * 2
    history = []
    for m in messages[-max_msgs:]:
        if isinstance(m, dict):
            role = m.get("role", "user")
            content = m.get("content", "")
            if role != "system" and content:
                history.append({"role": role, "content": content})
        elif hasattr(m, "content") and m.content:
            role = "assistant" if getattr(m, "type", "") == "ai" else "user"
            history.append({"role": role, "content": m.content})
    return history


async def answer_stream(state: QingState) -> AsyncIterator[dict]:
    """Generate RAG answer with streaming, using full conversation context.

    Key improvements over the original:
    - Uses up to 10 turns of conversation history (was 3)
    - Injects continuity note when there's existing conversation
    - Preserves RAG context (retrieved_docs, compressed_context) from pipeline
    """
    query = state.get("query", "")
    ctx = state.get("compressed_context", "")
    docs = state.get("reranked_docs", [])
    messages = state.get("messages", [])

    # Build system prompt with RAG context
    if ctx:
        system = f"{SYSTEM_PROMPT}\n\n## 参考资料\n{ctx}"
    else:
        system = SYSTEM_PROMPT

    # Add continuity note if there's an ongoing conversation
    history = _extract_history_for_llm(messages, max_turns=10)
    if len(history) > 2:  # More than just the current exchange
        system += CONTINUITY_NOTE

    # Remove the current query from history if it's the last user message
    # (it was added to messages before the graph ran; we pass it separately)
    if history and history[-1]["role"] == "user":
        history.pop()

    # Append the current query
    history.append({"role": "user", "content": query})

    # Generate
    llm = create_llm(state.get("provider_id"), state.get("model_name"))

    response_text = ""
    try:
        async for token in llm.chat_stream(system, history):
            response_text += token
            yield {"response": response_text}
    except Exception as e:
        logger.error(f"Answer generation error: {e}")
        response_text = f"生成回答时出错：{e}"
        yield {"response": response_text}

    # Build sources
    sources = []
    for d in docs:
        meta = d.get("meta", {})
        sources.append({
            "file": meta.get("file", "unknown"),
            "page": meta.get("page", ""),
            "excerpt": d.get("content", "")[:300],
        })

    # Estimate token usage
    try:
        input_tokens = llm.count_tokens(system + "".join(m["content"] for m in history))
        output_tokens = len(response_text) // 4
    except Exception:
        input_tokens = output_tokens = 0

    yield {
        "response": response_text,
        "sources": sources,
        "token_usage": {"input": input_tokens, "output": output_tokens, "total": input_tokens + output_tokens},
    }
