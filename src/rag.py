"""
Corrective RAG and Self-RAG helpers for TokenSmith.

This file contains small, pure-Python helpers that operate on the
existing local retrievers and ranker objects in TokenSmith. They
are intentionally conservative: they avoid external APIs and focus
on re-query/re-rank loops and local generation to repair or mark
unsupported spans.

The two core functions are:
- grade_documents_local: compute combined score for chunks and filter them
- self_rag_correct_answer: for hallucinated spans, attempt to retrieve local
  support and rewrite/annotate the answer.

The implementation uses the existing retrievers and the local answer() wrapper
for generation.
"""

from typing import Dict, List, Tuple
from collections import defaultdict
import re

from src.retriever import Retriever
from src.ranking.ranker import EnsembleRanker
from src.generator import answer


_STOPWORDS = {
    "the", "is", "at", "which", "on", "for", "a", "an", "and", "or", "in",
    "to", "of", "by", "with", "that", "this", "it", "as", "are", "was", "what",
    "who", "when", "where", "how", "why"}


def _extract_keywords(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if t and t not in _STOPWORDS]


def _keyword_overlap_score(keywords: List[str], chunk_text: str) -> float:
    if not keywords:
        return 1.0
    content = chunk_text.lower()
    matches = sum(1 for word in keywords if word in content)
    return matches / max(1, len(keywords))


def _normalize(scores: Dict[int, float]) -> Dict[int, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def compute_combined_scores(raw_scores: Dict[str, Dict[int, float]], weights: Dict[str, float]) -> Dict[int, float]:
    """Compute a weighted linear fusion of normalized scores and return a mapping idx->score.
    This mirrors the 'linear' fusion performed by EnsembleRanker, but returns explicit scores.
    """
    combined = defaultdict(float)
    for name, scores in raw_scores.items():
        weight = weights.get(name, 0.0)
        if weight <= 0:
            continue
        norm = _normalize(scores)
        for idx, val in norm.items():
            combined[idx] += weight * val
    return dict(combined)


def grade_documents_local(
    candidate_indices: List[int],
    raw_scores: Dict[str, Dict[int, float]],
    ranker: EnsembleRanker,
    threshold: float = 0.2,
    question: str | None = None,
    chunks: List[str] | None = None,
) -> Tuple[List[int], Dict[int, float]]:
    """Given candidate chunk indices and raw retriever scores, grade relevance locally.

    Combines retriever-based similarity with a simple keyword-overlap heuristic so obviously
    unrelated chunks can be filtered out before generation.
    """
    if not candidate_indices:
        return [], {}

    combined = compute_combined_scores(raw_scores, ranker.weights)
    adjusted_scores = {}

    keywords = _extract_keywords(question or "") if question else []
    for idx in candidate_indices:
        base_score = combined.get(idx, 0.0)
        if keywords and chunks:
            overlap = _keyword_overlap_score(keywords, chunks[idx])
            adjusted_scores[idx] = base_score * overlap
        else:
            adjusted_scores[idx] = base_score

    filtered_idxs = [idx for idx in candidate_indices if adjusted_scores.get(idx, 0.0) >= threshold]
    return filtered_idxs, adjusted_scores


def _get_chunks_for_span(span_text: str, chunks: List[str], retrievers: List[Retriever], pool_size: int, top_k: int = 3):
    """Find supporting chunks for a span by running retrievers on the span text.
    Returns a list of supporting chunk texts (no duplicates) up to top_k.
    """
    raw_scores = {}
    for r in retrievers:
        raw_scores[r.name] = r.get_scores(span_text, pool_size, chunks)
    # Merge using a simple voting / linear fusion w/ equal weights
    merged = defaultdict(float)
    for scores in raw_scores.values():
        for k, v in scores.items():
            merged[k] += float(v)
    # pick top_k
    if not merged:
        return []
    ordered = sorted(merged.keys(), key=lambda i: merged[i], reverse=True)
    top_idxs = ordered[:top_k]
    return [chunks[i] for i in top_idxs]


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
        support_chunks = _get_chunks_for_span(text, chunks, retrievers, pool_size)
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
