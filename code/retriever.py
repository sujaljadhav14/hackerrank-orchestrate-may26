"""
retriever.py — Hybrid BM25 + semantic retrieval (Stage 3).

Strategy:
  - BM25 (rank_bm25) handles exact keyword/lexical matches (product names, error codes).
  - Sentence-Transformers (all-MiniLM-L6-v2) handles semantic / paraphrase matches.
  - Hybrid score = 0.5 * BM25_norm + 0.5 * cosine_sim.
  - If domain_filter is provided, scores for chunks from that domain are boosted by 1.2×.
  - Confidence = mean hybrid score of the top-k results.

No external API calls — runs fully local.
"""
from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import RETRIEVAL_TOP_K, RANDOM_SEED
from corpus import Chunk

# Fix numpy seed for reproducibility in model internals
np.random.seed(RANDOM_SEED)

# Model used for local semantic embeddings (downloads on first run, ~22 MB)
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


class HybridRetriever:
    """
    Builds BM25 + semantic indices over a list of Chunk objects.
    Call .retrieve() to get the top-k most relevant chunks for a query.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        """
        Args:
            chunks: All Chunk objects returned by corpus.load_corpus().
        """
        self.chunks = chunks
        self.texts = [c.text for c in chunks]

        # ── BM25 index ─────────────────────────────────────────────────────
        tokenized = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

        # ── Semantic embeddings ─────────────────────────────────────────────
        # Runs entirely offline after the model is downloaded once.
        self.encoder = SentenceTransformer(_EMBED_MODEL_NAME)
        self.embeddings = self.encoder.encode(
            self.texts,
            show_progress_bar=True,
            batch_size=64,
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        domain_filter: str | None = None,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> tuple[list[Chunk], float]:
        """
        Retrieve the most relevant chunks for a query using hybrid scoring.

        Args:
            query:         Free-text search query (typically subject + issue).
            domain_filter: If set, boost chunks from this domain by 1.2×.
            top_k:         Number of chunks to return.

        Returns:
            (chunks, confidence_score)
            - chunks: top_k Chunk objects sorted by descending score.
            - confidence_score: mean hybrid score of the returned chunks (0–1+).
              Values < RETRIEVAL_CONFIDENCE_THRESHOLD indicate insufficient context.
        """
        if not self.chunks:
            return [], 0.0

        # ── BM25 scores ─────────────────────────────────────────────────────
        bm25_raw = np.array(self.bm25.get_scores(query.lower().split()))
        bm25_min, bm25_max = bm25_raw.min(), bm25_raw.max()
        bm25_norm = (bm25_raw - bm25_min) / (bm25_max - bm25_min + 1e-9)

        # ── Semantic (cosine) scores ─────────────────────────────────────────
        q_emb = self.encoder.encode([query])
        sem_scores = cosine_similarity(q_emb, self.embeddings)[0]

        # ── Hybrid combination ───────────────────────────────────────────────
        hybrid = 0.5 * bm25_norm + 0.5 * sem_scores

        # ── Domain boost ─────────────────────────────────────────────────────
        if domain_filter:
            for i, chunk in enumerate(self.chunks):
                if chunk.domain == domain_filter:
                    hybrid[i] *= 1.2

        # ── Top-k selection ──────────────────────────────────────────────────
        actual_k = min(top_k, len(self.chunks))
        top_idx = np.argsort(hybrid)[::-1][:actual_k]
        results = [self.chunks[i] for i in top_idx]
        confidence = float(np.mean([hybrid[i] for i in top_idx]))

        return results, confidence
