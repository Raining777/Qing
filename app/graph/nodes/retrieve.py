"""Hybrid retrieval + context compression for RAG.

Uses expanded_query (conversation-aware) for retrieval when available,
and original query for relevance assessment.
"""
import logging
import json

from app.graph.state import QingState
from app.services.vectordb import hybrid_search
from app.services.embedder import get_embedder
from app.services.llm import get_task_llm, create_llm

logger = logging.getLogger(__name__)


async def do_retrieve(state: QingState) -> dict:
    """Hybrid semantic + keyword search.

    Uses expanded_query (with conversation context) when available,
    which helps match follow-up questions to relevant documents.
    """
    query = state.get("expanded_query", "") or state.get("query", "")
    course = state.get("target_course", "")

    if not query or not course:
        return {"retrieved_docs": []}

    embedder = get_embedder()
    q_emb = embedder.encode_query(query)
    docs = hybrid_search(course, q_emb, query, top_k=8)

    return {"retrieved_docs": docs}


async def do_rerank(state: QingState) -> dict:
    """Rerank retrieved docs to top-3 using LLM relevance scoring.

    Uses the original user query (not expanded) for relevance ranking,
    since we want to match the user's actual intent, not the noisy context.
    """
    docs = state.get("retrieved_docs", [])
    query = state.get("query", "")  # Original query for relevance
    expanded_query = state.get("expanded_query", "")

    if not docs or len(docs) <= 3:
        return {"reranked_docs": docs}

    # For reranking, use the original query + expanded context for better relevance
    rank_query = query
    if expanded_query and len(query) < 30:
        # For very short queries, include some context in the ranking
        rank_query = expanded_query

    try:
        llm = get_task_llm("rerank")
        passages_text = ""
        for i, doc in enumerate(docs):
            meta = doc.get("meta", {})
            source = meta.get("file", "unknown")
            page = meta.get("page", "")
            passages_text += f"[{i}] 来源：{source} 页码：{page}\n{doc['content'][:500]}...\n\n"

        prompt = (
            f"用户问题：{rank_query}\n\n"
            f"请按与问题的相关性对下面资料片段排序，只返回最相关的 3 个片段索引，"
            f"格式必须是 JSON 数组，例如 [2, 0, 5]。\n\n{passages_text}"
        )
        response = await llm.chat(
            system="你负责给学习资料检索结果排序。只输出合法 JSON 数组，不要解释。",
            messages=[{"role": "user", "content": prompt}],
        )

        # Parse indices
        text = response.strip()
        if "```" in text:
            text = text.split("```")[1].strip("json").strip("```").strip()
        indices = json.loads(text)
        reranked = [docs[i] for i in indices if i < len(docs)][:3]
        return {"reranked_docs": reranked}
    except Exception as e:
        logger.warning(f"Rerank failed, using top-3: {e}")
        return {"reranked_docs": docs[:3]}


async def do_compress(state: QingState) -> dict:
    """Compress retrieved context to only key information relevant to query.

    Uses the original query for what's relevant; the expanded query would
    introduce noise in the compression prompt.
    """
    docs = state.get("reranked_docs", [])
    query = state.get("query", "")

    if not docs:
        return {"compressed_context": ""}

    # If total content is already small, skip compression
    total_len = sum(len(d["content"]) for d in docs)
    if total_len < 1500:
        ctx = "\n\n---\n\n".join(
            f"来源：{d['meta'].get('file', 'unknown')} 页码：{d['meta'].get('page', '')}\n{d['content']}"
            for d in docs
        )
        return {"compressed_context": ctx}

    try:
        llm = get_task_llm("compress")
        combined = "\n\n---\n\n".join(
            f"[{i}] 来源：{d['meta'].get('file', 'unknown')} 页码：{d['meta'].get('page', '')}\n{d['content']}"
            for i, d in enumerate(docs)
        )
        prompt = (
            f"用户问题：{query}\n\n"
            f"请从下面资料片段中提取与回答问题直接相关的信息。"
            f"要求简洁但完整，保留关键事实、定义、公式、代码和来源标记；不要加入资料外的新内容。\n\n{combined}"
        )
        compressed = await llm.chat(
            system="你负责压缩学习资料上下文，只保留与问题相关的信息，并保留来源。",
            messages=[{"role": "user", "content": prompt}],
        )
        return {"compressed_context": compressed}
    except Exception as e:
        logger.warning(f"Compression failed: {e}")
        ctx = "\n\n---\n\n".join(
            f"来源：{d['meta'].get('file', 'unknown')} 页码：{d['meta'].get('page', '')}\n{d['content']}"
            for d in docs
        )
        return {"compressed_context": ctx}
