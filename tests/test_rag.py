from src.rag import grade_documents_local, self_rag_correct_answer
from src.ranking.ranker import EnsembleRanker


def test_grade_documents_local_filters_candidates():
    ranker = EnsembleRanker("linear", {"faiss": 0.6, "bm25": 0.4}, rrf_k=60)
    candidate_indices = [0, 1, 2]
    fake_scores_faiss = {0: 0.2, 1: 0.9, 2: 0.05}
    fake_scores_bm25 = {0: 0.25, 1: 0.75, 2: 0.01}
    raw_scores = {"faiss": fake_scores_faiss, "bm25": fake_scores_bm25}
    chunks = [
        "Transaction processing details",
        "Agents rely on working memory to store interim beliefs",
        "Random unrelated paragraph",
    ]
    question = "What is agent memory?"

    filtered, combined = grade_documents_local(
        candidate_indices=candidate_indices,
        raw_scores=raw_scores,
        ranker=ranker,
        threshold=0.5,
        question=question,
        chunks=chunks,
    )

    assert filtered == [1]
    assert combined[1] > combined[0]


def test_self_rag_annotations_when_no_support():
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
        pool_size=10,
        model_path="dummy.gguf",
        max_tokens=64,
    )
    assert "[UNSUPPORTED" in corrected
