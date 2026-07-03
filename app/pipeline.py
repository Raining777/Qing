"""RAG pipeline — simple async pipeline replacing the LangGraph workflow.

Flow: expand_query → detect_course → retrieve → (rerank if BM25-only) → answer_stream

When embedding API is configured: hybrid semantic + BM25 search.
When no embedding key: pure BM25 + LLM reranking (no extra API needed).
"""
import json
import logging
from typing import AsyncIterator

from app.history import get_sessions
from app.intents import expand_query as do_expand, is_follow_up
from app.services.embedder import get_embedder
from app.services.vectordb import hybrid_search, search_keyword, find_course_for_query, list_courses
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

CONTINUITY_NOTE = (
    "\n\n## 对话上下文\n"
    "用户正在进行多轮对话。请结合前面的对话内容理解当前问题，"
    "在回答时保持连贯性，可以用「你刚才问的...」「前面提到...」来呼应之前的讨论。"
)


async def run_rag_pipeline(
    query: str,
    course: str = "",
    session_id: str = "",
    provider_id: str = "",
    model_name: str = "",
) -> AsyncIterator[dict]:
    """Run the RAG pipeline and stream answer chunks."""
    store = get_sessions()

    # ── Step 1: Query expansion ──
    expanded_query = query
    if session_id:
        context = store.get_recent_context(session_id)
        if context and is_follow_up(query):
            expanded_query = do_expand(query, context)

    # ── Step 2: Auto-detect course ──
    target_course = course or store.get_course(session_id)
    available = list_courses()
    if not target_course and query and available:
        embedder = get_embedder()
        if embedder.has_semantic:
            q_emb = await embedder.encode_query(query)
            target_course = find_course_for_query(q_emb)
        else:
            # BM25-based course detection: check which course has most keyword hits
            target_course = _bm25_detect_course(query, available)
        target_course = target_course or (available[0]["name"] if available else "")
        if target_course and session_id:
            store.set_course(session_id, target_course)

    # ── Step 3: Retrieval ──
    docs = []
    embedder = get_embedder()

    if target_course and available:
        if embedder.has_semantic:
            # Semantic + BM25 hybrid search
            q_emb = await embedder.encode_query(expanded_query)
            docs = hybrid_search(target_course, q_emb, expanded_query, top_k=5)
        else:
            # BM25-only → fetch top-10 then LLM rerank to top-5
            bm25_docs = search_keyword(target_course, expanded_query, top_k=10)
            if bm25_docs and len(bm25_docs) > 5:
                docs = await _llm_rerank(query, bm25_docs, provider_id, model_name)
            else:
                docs = bm25_docs

    # Build context
    if docs:
        ctx = "\n\n---\n\n".join(
            f"来源：{d.get('meta', {}).get('file', 'unknown')} 页码：{d.get('meta', {}).get('page', '')}\n{d.get('content') or d.get('text', '')}"
            for d in docs
        )
        system = f"{SYSTEM_PROMPT}\n\n## 参考资料\n{ctx}"
    else:
        system = SYSTEM_PROMPT

    # ── Step 4: Conversation history ──
    history = store.get_messages(session_id) if session_id else []
    if len(history) > 2:
        system += CONTINUITY_NOTE
    if history and history[-1]["role"] == "user":
        history.pop()
    history.append({"role": "user", "content": query})

    # ── Step 5: Stream answer ──
    llm = create_llm(provider_id, model_name)
    response_text = ""

    try:
        async for token in llm.chat_stream(system, history):
            response_text += token
            yield {"delta": token}
    except Exception as e:
        logger.error(f"Answer generation error: {e}")
        response_text = f"生成回答时出错：{e}"
        yield {"delta": response_text}

    # ── Step 6: Save ──
    if session_id:
        store.add_message(session_id, "user", query)
        store.add_message(session_id, "assistant", response_text)

    # ── Step 7: Sources ──
    sources = []
    for d in docs:
        meta = d.get("meta", {}) or {}
        sources.append({
            "file": meta.get("file", "unknown"),
            "page": meta.get("page", ""),
            "excerpt": (d.get("content") or d.get("text", ""))[:300],
        })

    input_tokens = llm.count_tokens(system + "".join(m["content"] for m in history))
    output_tokens = len(response_text) // 4

    yield {
        "response": response_text,
        "sources": sources,
        "token_usage": {"input": input_tokens, "output": output_tokens, "total": input_tokens + output_tokens},
    }


# ── BM25-only helpers ──

def _bm25_detect_course(query: str, courses: list[dict]) -> str:
    """Detect course by BM25 keyword match (no embedding needed)."""
    best_course, best_hits = None, -1
    for c in courses:
        results = search_keyword(c["name"], query, top_k=3)
        hits = sum(r.get("score", 0) for r in results)
        if hits > best_hits:
            best_hits = hits
            best_course = c["name"]
    return best_course


async def _llm_rerank(
    query: str, docs: list[dict], provider_id: str, model_name: str,
) -> list[dict]:
    """Use LLM to select the 5 most relevant passages from BM25 results."""
    if not docs:
        return docs

    llm = create_llm(provider_id, model_name)

    passages = ""
    for i, d in enumerate(docs):
        src = d.get("meta", {}).get("file", "unknown")
        text = d.get("content") or d.get("text", "")
        passages += f"[{i}] {src}: {text[:300]}...\n\n"

    prompt = (
        f"用户问题：{query}\n\n"
        f"下面有 {len(docs)} 段资料，请选出与问题最相关的 5 段（不够 5 段就全选）。"
        f"只返回索引编号的 JSON 数组，如 [0, 3, 7, 2, 5]。\n\n{passages}"
    )

    try:
        resp = await llm.chat(
            system="你负责给学习资料排序。只输出合法 JSON 数组，不要解释。",
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.strip()
        if "```" in text:
            text = text.split("```")[1].strip("json").strip("```").strip()
        indices = json.loads(text)
        reranked = [docs[i] for i in indices if i < len(docs)][:5]
        return reranked or docs[:5]
    except Exception as e:
        logger.warning(f"LLM rerank failed, using BM25 top-5: {e}")
        return docs[:5]
