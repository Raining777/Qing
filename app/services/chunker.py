"""Code-aware text chunking for RAG."""
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

# Separators ordered by priority — code-aware
SEPARATORS = [
    "\n\n## ",     # Markdown h2
    "\n\n### ",    # Markdown h3
    "\n\n#### ",   # Markdown h4
    "\n\n",        # paragraph break
    "\nclass ",    # class definition
    "\ndef ",      # function definition
    "\nfunc ",     # Go function
    "\npublic ",   # Java/C# method
    "\nprivate ",  # Java/C# method
    "\n",          # line break
    ". ",          # sentence
    ", ",          # clause
    " ",           # word
    "",            # character
]

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=SEPARATORS,
    length_function=len,
)


def chunk_text(text: str, metadata: dict = None) -> list[dict]:
    """Split text into chunks with metadata.

    Returns: [{"text": "...", "meta": {...}}, ...]
    """
    if not text or not text.strip():
        return []

    base_meta = metadata or {}
    chunks = _splitter.split_text(text)

    results = []
    for i, chunk_text_content in enumerate(chunks):
        meta = {**base_meta, "chunk_index": i, "chunk_count": len(chunks)}
        results.append({"text": chunk_text_content, "meta": meta})

    return results


def chunk_document(parsed_result: dict) -> list[dict]:
    """Chunk a parsed document result from parser.parse_file().

    parsed_result: {"ok": True, "text": "...", "type": "...", "file": "..."}
    Returns: list of {"text": "...", "meta": {...}}
    """
    base_meta = {
        "file": parsed_result.get("file", ""),
        "type": parsed_result.get("type", "text"),
        "is_scanned": parsed_result.get("is_scanned", False),
    }
    return chunk_text(parsed_result.get("text", ""), base_meta)
