import os
import logging
import warnings

# Suppress HuggingFace / Transformers warnings and loading messages
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Programmatically disable logging messages from huggingface_hub and transformers
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Filter warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*unauthenticated.*")
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", message=".*position_ids.*")

import re
import numpy as np
import hashlib
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────────────────────────────────────
MODEL_LOADED = False
_model_instance = None

# In-memory embedding cache: md5(text) → np.ndarray (L2-normalized)
# Prevents re-encoding the same document on every scan call.
_embedding_cache: dict = {}

# Maximum characters to feed into BERT (≈ 512 tokens for typical English text).
# Sentence-transformers truncates internally, but pre-truncating avoids
# unnecessary tokenisation overhead on very long documents.
_BERT_MAX_CHARS = 4096


def _hash_text(text: str) -> str:
    """Deterministic MD5 key used to look up cached embeddings."""
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()


def _clean_for_bert(text: str) -> str:
    """Normalise whitespace and truncate before encoding."""
    text = re.sub(r"\s+", " ", text).strip()
    return text[:_BERT_MAX_CHARS]


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
DISABLE_BERT = (
    os.environ.get("DISABLE_BERT", "false").lower() == "true"
    or os.environ.get("LIGHTWEIGHT_MODE", "false").lower() == "true"
    or os.environ.get("RENDER") == "true"
)

if not DISABLE_BERT:
    try:
        from sentence_transformers import SentenceTransformer
        import torch  # noqa: F401 – imported to confirm torch availability

        def get_bert_model() -> Optional[SentenceTransformer]:
            global _model_instance, MODEL_LOADED
            if _model_instance is not None:
                return _model_instance
            try:
                # ~80 MB model; cached in ~/.cache/huggingface after first download.
                _model_instance = SentenceTransformer("all-MiniLM-L6-v2")
                MODEL_LOADED = True
                print("[BERT] Model loaded: all-MiniLM-L6-v2")
                return _model_instance
            except Exception as exc:
                print(f"[BERT] Failed to load model: {exc}. Falling back to semantic proxy.")
                MODEL_LOADED = False
                return None

    except ImportError:
        def get_bert_model():
            return None
else:
    def get_bert_model():
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Batch encoding with cache
# ─────────────────────────────────────────────────────────────────────────────
def _batch_encode(texts: List[str], model) -> np.ndarray:
    """
    Encode a list of texts using the BERT model with an in-memory cache.

    Only texts whose MD5 hash is *not* already in ``_embedding_cache`` are
    sent to the model; the rest are served from cache. This dramatically
    reduces redundant computation when the same documents appear across
    multiple scan calls (e.g. batch scans).

    Returns a 2-D numpy array of shape (len(texts), embedding_dim) where
    every row is a pre-L2-normalised embedding vector.
    """
    keys = [_hash_text(t) for t in texts]

    # Identify which texts still need encoding
    miss_indices = [i for i, k in enumerate(keys) if k not in _embedding_cache]

    if miss_indices:
        texts_to_encode = [texts[i] for i in miss_indices]
        new_embeddings = model.encode(
            texts_to_encode,
            batch_size=64,          # Encode up to 64 docs at a time (GPU/CPU batching)
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Pre-normalise so cosine sim = dot product
        )
        for i, emb in zip(miss_indices, new_embeddings):
            _embedding_cache[keys[i]] = emb

    return np.array([_embedding_cache[k] for k in keys])


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def calculate_bert_similarity_batch(target_text: str, peer_texts: List[str]) -> List[float]:
    """
    Compute BERT semantic similarity between *target_text* and every text in
    *peer_texts* using a **single batched model.encode() call**.

    Why this matters for speed
    --------------------------
    The previous approach called ``model.encode()`` for *each pair individually*
    (N model calls for N peers). This function encodes all peers at once (1 model
    call), then uses a vectorised dot-product to produce all similarity scores.
    For N=30 peers the speedup is roughly 15-25×.

    Caching
    -------
    Both the target and every peer embedding are stored in ``_embedding_cache``.
    Subsequent scans that reference the same documents skip re-encoding entirely.

    Returns
    -------
    List of float similarity scores in [0.0, 1.0], one per peer.
    """
    if not peer_texts:
        return []
    if not target_text.strip():
        return [0.0] * len(peer_texts)

    model = get_bert_model()

    if MODEL_LOADED and model is not None:
        try:
            # Prepare texts
            target_prep = _clean_for_bert(target_text)
            peers_prep = [_clean_for_bert(p) for p in peer_texts]

            # Encode target + all peers in one shot
            all_texts = [target_prep] + peers_prep
            all_embeddings = _batch_encode(all_texts, model)

            target_emb = all_embeddings[0]       # shape: (dim,)
            peer_embs = all_embeddings[1:]        # shape: (N, dim)

            # Cosine similarity via dot product (vectors are pre-normalised)
            similarities = np.dot(peer_embs, target_emb)  # shape: (N,)
            return [float(min(1.0, max(0.0, float(s)))) for s in similarities]

        except Exception as exc:
            print(f"[BERT] Batch encode error: {exc}. Falling back to proxy.")

    # Fallback: mathematical proxy (no model required)
    return [compute_fallback_semantic_similarity(target_text, p) for p in peer_texts]


def calculate_bert_similarity(doc1: str, doc2: str) -> float:
    """
    Compute BERT semantic similarity for a single document pair.
    Internally delegates to ``calculate_bert_similarity_batch`` so that the
    embedding cache is always used.
    """
    if not doc1.strip() or not doc2.strip():
        return 0.0
    results = calculate_bert_similarity_batch(doc1, [doc2])
    return results[0] if results else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Fallback semantic proxy (no BERT dependency)
# ─────────────────────────────────────────────────────────────────────────────
def compute_fallback_semantic_similarity(doc1: str, doc2: str) -> float:
    """
    A fast mathematical semantic proxy combining Jaccard overlap and soft word
    alignment. Used when ``sentence-transformers`` is unavailable or fails.
    """
    words1 = re.findall(r"\b\w+\b", doc1.lower())
    words2 = re.findall(r"\b\w+\b", doc2.lower())

    if not words1 or not words2:
        return 0.0

    _STOP = {
        "the", "is", "are", "and", "a", "an", "in", "on", "at", "to", "for",
        "of", "with", "this", "that", "it", "by", "as", "be", "has", "have",
        "from", "or", "but", "was", "were", "will", "would", "could", "should",
        "their", "they", "we", "our", "its", "not", "so", "if", "do", "does",
    }

    f1 = [w for w in words1 if w not in _STOP] or words1
    f2 = [w for w in words2 if w not in _STOP] or words2

    set1, set2 = set(f1), set(f2)
    jaccard = len(set1 & set2) / len(set1 | set2) if (set1 | set2) else 0.0

    # Soft alignment: count words in set1 that appear or nearly appear in set2
    matched = 0.0
    for w1 in set1:
        if w1 in set2:
            matched += 1.0
        else:
            for w2 in set2:
                if len(w1) > 4 and len(w2) > 4 and (
                    w1.startswith(w2[:4]) or w2.startswith(w1[:4])
                ):
                    matched += 0.8
                    break

    alignment = matched / max(len(set1), len(set2), 1)
    return min(1.0, max(0.0, 0.3 * jaccard + 0.7 * alignment))
