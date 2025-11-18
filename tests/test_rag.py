from src.rag import self_rag_correct_answer


class _FakeRetriever:
    def __init__(self, scores):
        self.name = "fake"
        self._scores = scores
        self.last_query = None

    def get_scores(self, query, pool_size, chunks):
        self.last_query = query
        return self._scores


def test_placeholder_no_corrective_rag():
    # Since corrective RAG was removed, ensure that this repository still
    # contains the self-RAG helper and that the import works.
    assert callable(self_rag_correct_answer)


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


def test_self_rag_filters_irrelevant_support():
    chunks = [
        "Detailed ACID explanation.",
        "Tangential note about power outages and hardware.",
    ]
    retriever = _FakeRetriever({0: 0.0, 1: 0.0})  # Flat scores → should be filtered out
    question = "What are the ACID properties of transactions?"
    spans = [{"start": 0, "end": 15, "text": "ACID transactions"}]

    corrected = self_rag_correct_answer(
        question=question,
        original_answer="ACID stands for...",
        hallucinated_spans=spans,
        chunks=chunks,
        retrievers=[retriever],
        pool_size=10,
        model_path="dummy.gguf",
        max_tokens=64,
    )

    assert "[UNSUPPORTED" in corrected
    assert retriever.last_query is not None and "ACID properties" in retriever.last_query
