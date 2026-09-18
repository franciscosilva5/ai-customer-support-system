import faiss
from sentence_transformers import (
    CrossEncoder,
    SentenceTransformer,
)

from src.knowledge_base import load_knowledge_base


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class KnowledgeRetriever:
    def __init__(self):
        self.documents = load_knowledge_base()

        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

        self.reranker = CrossEncoder(
            RERANKER_MODEL_NAME
        )

        texts = [
            document["text"]
            for document in self.documents
        ]

        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        self.index = faiss.IndexFlatIP(
            embeddings.shape[1]
        )

        self.index.add(embeddings)

    def search(
        self,
        query,
        top_k=3,
        candidate_k=5,
    ):
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        candidate_k = min(
            candidate_k,
            len(self.documents),
        )

        scores, indices = self.index.search(
            query_embedding,
            candidate_k,
        )

        candidates = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):
            document = self.documents[index]

            candidates.append(
                {
                    "id": document["id"],
                    "title": document["title"],
                    "text": document["text"],
                    "source": document["source"],
                    "retrieval_score": float(score),
                }
            )

        pairs = [
            [query, candidate["text"]]
            for candidate in candidates
        ]

        rerank_scores = self.reranker.predict(
            pairs
        )

        for candidate, rerank_score in zip(
            candidates,
            rerank_scores,
        ):
            candidate["rerank_score"] = float(
                rerank_score
            )

        candidates.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return candidates[:top_k]
