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

from src.retriever import Retriever
from src.ranking.ranker import EnsembleRanker
from src.generator import answer


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


def grade_documents_local(question: str, chunks: List[str], retrievers: List[Retriever], ranker: EnsembleRanker, pool_size: int, threshold: float = 0.2) -> Tuple[List[str], Dict[int, float]]:
    """Grade/relevance-filter a set of candidate chunks using a local ensemble score.

    Returns the filtered list of chunk texts and the final combined scores for the
    evaluated pool.
    """
    # Gather retriever scores
    raw_scores = {}
    for r in retrievers:
        raw_scores[r.name] = r.get_scores(question, pool_size, chunks)

    combined = compute_combined_scores(raw_scores, ranker.weights)
    # Debug: show top combined scores
    if combined:
        ordered_combined = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)
        top_preview = ordered_combined[:min(5, len(ordered_combined))]
        print(f"[CRAG] top combined scores: {top_preview} (threshold={threshold})")

    # Filter by threshold - if document >= threshold it is considered relevant
    filtered_idxs = [idx for idx, s in combined.items() if s >= threshold]
    # If no docs passed threshold, fallback to selecting top K
    if not filtered_idxs:
        # Choose top-1 by combined score (safe fallback)
        ordered = sorted(combined.keys(), key=lambda i: combined[i], reverse=True)
        filtered_idxs = ordered[:1]

    # Sort the filtered indexes by combined score descending
    filtered_idxs_sorted = sorted(filtered_idxs, key=lambda i: combined.get(i, 0.0), reverse=True)
    filtered_chunks = [chunks[i] for i in filtered_idxs_sorted]
    print(f"[CRAG] filtered chunk count: {len(filtered_chunks)}")
    return filtered_chunks, combined


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
    # Debug
    print(f"[Self-RAG] For span: {span_text[:80]}... found support chunks indexes: {top_idxs}")
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
    print(f"[Self-RAG] Attempting to repair {len(spans_sorted)} hallucinated spans")
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
                print(f"[Self-RAG] rewrite failed, annotating span as unsupported: {text[:40]}...")
            else:
                print(f"[Self-RAG] rewrite success; replaced span: {text[:40]}... with new text of length {len(new_text)}")
        else:
            # No supporting chunks found - annotate unsupported
            new_text = f"[UNSUPPORTED: {text}]"
            print(f"[Self-RAG] No support found for span: {text[:40]}...; annotating as unsupported.")

        # Replace the substring in the answer. Use the start/end offsets directly.
        corrected = corrected[:start] + new_text + corrected[end:]
    return corrected
