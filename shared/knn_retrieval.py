"""
shared/knn_retrieval.py — kNN retrieval using bge-large embeddings from exp_0138.

Usage:
    retriever = KNNRetriever.from_exp0138(workspace_dir)
    neighbors = retriever.query(text, subtask="st3", k=3, exclude_ids={"some_id"})
    # Returns list of {"id", "subtask", "label", "text_preview", "similarity"} dicts
"""

import json
import numpy as np
from pathlib import Path


class KNNRetriever:
    def __init__(self, embeddings: np.ndarray, metadata: list[dict], encoder=None):
        """
        embeddings: (N, D) float32 array, L2-normalized
        metadata: list of N dicts with {id, subtask, label, text_preview}
        encoder: callable(texts) -> (M, D) ndarray, or None (must provide embed_fn at query time)
        """
        # Normalize for cosine similarity via dot product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self.embeddings = (embeddings / norms).astype(np.float32)
        self.metadata = metadata
        self.encoder = encoder

        # Build subtask index for filtered retrieval
        self._subtask_indices = {}
        for i, m in enumerate(metadata):
            st = m.get("subtask", "")
            self._subtask_indices.setdefault(st, []).append(i)

    @classmethod
    def from_exp0138(cls, workspace_dir, encoder=None):
        """Load embeddings and metadata from exp_0138 outputs."""
        base = Path(workspace_dir) / "experiments" / "exp_0138" / "outputs"
        embeddings = np.load(str(base / "embeddings.npy"))
        metadata = [json.loads(l) for l in open(base / "metadata.jsonl")]
        assert len(embeddings) == len(metadata), \
            f"Mismatch: {len(embeddings)} embeddings vs {len(metadata)} metadata"
        return cls(embeddings, metadata, encoder)

    def query_by_embedding(self, query_emb: np.ndarray, subtask: str = None,
                           k: int = 3, exclude_ids: set = None):
        """
        query_emb: (D,) float32 vector (will be normalized)
        Returns top-k neighbors as list of dicts with 'similarity' added.
        """
        q = query_emb.astype(np.float32).reshape(1, -1)
        q = q / max(np.linalg.norm(q), 1e-8)

        if subtask and subtask in self._subtask_indices:
            indices = np.array(self._subtask_indices[subtask])
            subset_emb = self.embeddings[indices]
        else:
            indices = np.arange(len(self.embeddings))
            subset_emb = self.embeddings

        # Cosine similarity via dot product (both normalized)
        sims = (subset_emb @ q.T).squeeze(1)
        top_k_local = np.argsort(-sims)

        results = []
        for local_idx in top_k_local:
            global_idx = int(indices[local_idx])
            meta = self.metadata[global_idx]
            if exclude_ids and meta["id"] in exclude_ids:
                continue
            results.append({
                **meta,
                "similarity": float(sims[local_idx]),
            })
            if len(results) >= k:
                break

        return results

    def query(self, text: str, subtask: str = None, k: int = 3,
              exclude_ids: set = None):
        """Encode text and retrieve k nearest neighbors."""
        if self.encoder is None:
            raise ValueError("No encoder set. Use query_by_embedding() or pass encoder to constructor.")
        emb = self.encoder([text])[0]
        return self.query_by_embedding(emb, subtask=subtask, k=k,
                                       exclude_ids=exclude_ids)


def load_bge_encoder(device="cuda"):
    """Load bge-large-en-v1.5 encoder for query-time embedding."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)

    def encode(texts):
        return model.encode(texts, normalize_embeddings=True,
                           show_progress_bar=False)
    return encode
