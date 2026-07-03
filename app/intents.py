"""Unified intent detection — server-side keyword matching.

All messages go through /api/chat. This module determines the intent
(summary, mindmap, practice, etc.) so the backend can dispatch accordingly.
"""

INTENT_PATTERNS: list[tuple[str, list[str]]] = [
    ("summary",    ["总结", "梳理", "概括", "summary", "summarize", "知识地图"]),
    ("mindmap",    ["思维导图", "mindmap", "mind map", "知识结构"]),
    ("plan",       ["复习计划", "study plan", "学习计划", "备考计划"]),
    ("practice",   ["出题", "练习", "quiz", "practice", "考题", "做题"]),
    ("flashcard",  ["闪卡", "flashcard", "卡片", "记忆卡"]),
    ("formula",    ["公式", "formula", "定理", "equation"]),
    ("compare",    ["对比", "比较", "compare", " vs ", "区别"]),
    ("mnemonic",   ["记忆口诀", "记忆方法", "mnemonic", "帮我记", "口诀"]),
    ("sprint",     ["冲刺", "sprint", "考前", "速成", "突击"]),
]

# Follow-up indicators for query expansion
FOLLOW_UP_MARKERS = [
    "那", "这", "它", "这个", "那个", "这些", "那些",
    "上面", "前面", "刚才", "刚刚", "之前", "上次",
    "怎么用", "怎么做", "为什么", "然后呢", "还有呢",
    "再", "还有", "继续", "接着", "详细", "具体",
    "举例", "比如", "例如", "能再", "多说",
    "what about", "how about", "and", "then", "also", "so",
    "explain more", "tell me more", "elaborate", "go on",
    "why", "how", "when", "where", "which",
]


def detect_intent(query: str) -> str:
    """Detect intent from user query. Returns intent key string."""
    ql = query.lower()
    for intent, keywords in INTENT_PATTERNS:
        for kw in keywords:
            if kw.lower() in ql:
                return intent
    return "question"


def is_follow_up(query: str) -> bool:
    """Check if query looks like a follow-up needing conversation context."""
    ql = query.lower().strip()
    if len(query) < 15:
        return True
    for marker in FOLLOW_UP_MARKERS:
        if marker in ql:
            return True
    return False


def expand_query(query: str, context: str) -> str:
    """Expand a follow-up query with conversation context for better retrieval."""
    if not is_follow_up(query):
        return query
    if not context:
        return query
    return f"对话上下文: {context}\n用户追问: {query}"


def extract_exam_date(query: str) -> str:
    """Extract exam date from query text."""
    import re
    m = re.search(r'(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?', query)
    return m.group(0) if m else ""


def extract_practice_count(query: str) -> int:
    """Extract desired number of practice questions from query."""
    import re
    m = re.search(r'(\d+)\s*(道|题|questions|problems)', query)
    return int(m.group(1)) if m else 5


def extract_concepts(query: str) -> tuple[str, str]:
    """Extract two concepts from a comparison query."""
    # Remove intent keywords
    for kw in ["对比", "比较", "compare", "帮我", "请", "给我"]:
        query = query.replace(kw, " ")
    # Split on common separators
    import re
    parts = re.split(r'\s+vs\.?\s+|\s+和\s+|\s+与\s+|\s+以及\s+|\s+、\s+', query)
    parts = [p.strip() for p in parts if p.strip()]
    return (parts[0] if len(parts) > 0 else "",
            parts[1] if len(parts) > 1 else "")
