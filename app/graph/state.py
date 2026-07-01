"""LangGraph state definition for Qing AI Study Assistant."""
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class QingState(TypedDict):
    # ── Messages ──
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # ── Session ──
    session_id: str
    intent: str  # question | summary | mindmap | plan | practice | flashcard | formula | compare | mnemonic | sprint

    # ── LLM Selection ──
    provider_id: str   # anthropic | deepseek | openai | ollama
    model_name: str

    # ── RAG Context ──
    query: str
    expanded_query: str           # query enriched with conversation context for retrieval
    target_course: str           # auto-detected or user-specified
    retrieved_docs: list[dict]   # [{content, meta, score}, ...]
    reranked_docs: list[dict]    # after reranking (top-3)
    compressed_context: str      # compressed version of retrieved text

    # ── Generation ──
    response: str                # final generated text / mermaid code / markdown
    sources: list[dict]          # [{file, page, excerpt}]
    token_usage: dict            # {input, output, total}

    # ── Course Catalog ──
    available_courses: list[dict]  # [{name, file_count, chunk_count}]

    # ── Practice / Review ──
    practice_type: str           # mcq | short_answer | coding
    practice_count: int
    practice_topic: str
    generated_questions: list[dict]
    user_answers: list[dict]

    # ── SM-2 Spaced Repetition ──
    review_chapter: str
    review_rating: int           # 0-3 (0=forgot, 3=easy)

    # ── Compare ──
    concept_a: str
    concept_b: str

    # ── Sprint ──
    exam_date: str
    sprint_course: str

    # ── Chapter Summaries Cache ──
    chapter_summaries: dict      # {course: {chapter: summary_text}}
