import pytest
from src.rag import grade_documents_local
from src.ranking.ranker import EnsembleRanker
from src.config import QueryPlanConfig


class FakeRetriever:
    def __init__(self, name: str, scores: dict):
        self.name = name
        self._scores = scores

    def get_scores(self, query: str, pool_size: int, chunks: list):
        return self._scores


def test_grade_documents_local_default(config):
    # Initialize config and artifacts
    cfg = QueryPlanConfig.from_yaml("config/config.yaml")
    # Build fake chunks and fake scores so the test does not depend on local models
    chunks = ["the memory is external", "agents have working memory", "a different topic"]
    # Center top candidate is index 1
    fake_scores_faiss = {0: 0.2, 1: 0.9, 2: 0.05}
    fake_scores_bm25 = {0: 0.25, 1: 0.75, 2: 0.01}
    retrievers = [FakeRetriever("faiss", fake_scores_faiss), FakeRetriever("bm25", fake_scores_bm25)]
    ranker = EnsembleRanker(cfg.ensemble_method, cfg.ranker_weights, rrf_k=cfg.rrf_k)

    # Use a simple query and expect some chunks pass the default threshold of 0.05
    q = "What is agent memory?"
    filtered, scores = grade_documents_local(q, chunks, retrievers, ranker, cfg.pool_size, threshold=0.05)

    assert isinstance(filtered, list)
    assert isinstance(scores, dict)
    # With a low threshold we expect at least one candidate
    assert len(filtered) >= 1


def test_self_rag_annotations_when_no_support(config):
    from src.rag import self_rag_correct_answer
    from src.retriever import Retriever

    cfg = QueryPlanConfig.from_yaml("config/config.yaml")
    chunks = ["This chunk is unrelated.", "Another chunk about different topic."]
    retrievers = []  # No retrievers → no support found
    answer_text = "The answer is that agents have long term memory."
    spans = [{"start": 20, "end": 41, "text": "agents have long term memory"}]

    corrected = self_rag_correct_answer(
        question="What is agent memory?",
        original_answer=answer_text,
        hallucinated_spans=spans,
        chunks=chunks,
        retrievers=retrievers,
        pool_size=cfg.pool_size,
        model_path=str(cfg.model_path),
        max_tokens=64,
    )
    assert "[UNSUPPORTED" in corrected
