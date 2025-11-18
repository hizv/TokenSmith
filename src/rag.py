"""Self-RAG helpers for TokenSmith."""

from typing import Dict, List
from collections import defaultdict
from src.retriever import Retriever
from src.generator import answer


_MIN_PRIMARY_SCORE = 0.3  # avoid feeding obviously off-topic evidence
_MIN_SECONDARY_SCORE = 0.15


def _normalize_scores(scores: Dict[int, float]) -> Dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    low, high = min(values), max(values)
    if high <= low:
        return {idx: 0.0 for idx in scores}
    return {idx: (val - low) / (high - low) for idx, val in scores.items()}


def _get_chunks_for_span(
    span_text: str,
    chunks: List[str],
    retrievers: List[Retriever],
    pool_size: int,
    top_k: int = 3,
    question: str | None = None,
):
    """Find supporting chunks by querying with the question + span and filter weak matches."""
    if not retrievers:
        return []

    query_text = span_text.strip()
    if question:
        query_text = f"{question.strip()} {query_text}".strip()

    combined = defaultdict(float)
    for r in retrievers:
        raw = r.get_scores(query_text, pool_size, chunks)
        if not raw:
            continue
        normalized = _normalize_scores(raw)
        for idx, score in normalized.items():
            combined[idx] += score

    if not combined:
        return []

    num_retrievers = max(len(retrievers), 1)
    averaged = {idx: score / num_retrievers for idx, score in combined.items()}
    ordered = sorted(averaged.items(), key=lambda item: item[1], reverse=True)

    strong_idxs = [idx for idx, score in ordered if score >= _MIN_PRIMARY_SCORE][:top_k]
    if not strong_idxs and ordered and ordered[0][1] >= _MIN_SECONDARY_SCORE:
        strong_idxs = [ordered[0][0]]

    if not strong_idxs:
        return []

    return [chunks[i] for i in strong_idxs]


def self_rag_correct_answer(question: str, original_answer: str, hallucinated_spans: List[dict], chunks: List[str], retrievers: List[Retriever], pool_size: int, model_path: str, max_tokens: int = 300) -> str:
    """Attempt to correct an answer using local retrieval for hallucinated spans.

    - For each hallucinated span, search for supporting chunks using the span text
    - If supporting chunks exist, prompt the local model to rewrite that span with evidence from those chunks
    - If none exist, annotate the span as unsupported
    - Returns a revised answer with either rewritten content or inline annotations
    """
    if not hallucinated_spans:
        return original_answer

    # We will progressively modify the answer using character offsets provided in spans.
    # Build a mutable copy of the answer as a list so replacements don't change offsets for remaining spans.
    corrected = original_answer
    # Process spans in reverse order by start index to preserve offsets
    spans_sorted = sorted(hallucinated_spans, key=lambda s: s.get("start", 0), reverse=True)
    for span in spans_sorted:
        start = span.get("start", None)
        end = span.get("end", None)
        text = span.get("text", None)
        if start is None or end is None or not text:
            # skip malformed spans
            continue

        # Find supporting chunks in the local index
        support_chunks = _get_chunks_for_span(text, chunks, retrievers, pool_size, question=question)
        if support_chunks:
            # Prompt the local model to rewrite this span with evidence.
            prompt_chunks = support_chunks[:3]
            rewrite_query = (
                f"Rewrite the following snippet from an assistant answer using only facts that are directly supported by the provided excerpts. "
                f"If evidence contradicts the snippet, prefer the evidence. If there is no evidence, annotate with '[UNSUPPORTED]'.\n\nSnippet: {text}\n\nExcerpts:\n" + "\n---\n".join(prompt_chunks)
            )
            # Use local LLM to rewrite (short responses)
            new_text = answer(rewrite_query, prompt_chunks, model_path, max_tokens=max_tokens)
            if not new_text or new_text.strip() == "":
                # If rewrite fails, annotate with unsupported
                new_text = f"[UNSUPPORTED: {text}]"
        else:
            # No supporting chunks found - annotate unsupported
            new_text = f"[UNSUPPORTED: {text}]"

        # Replace the substring in the answer. Use the start/end offsets directly.
        corrected = corrected[:start] + new_text + corrected[end:]
    return corrected
