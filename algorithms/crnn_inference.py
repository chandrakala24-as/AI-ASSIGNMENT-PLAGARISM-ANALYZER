"""
CRNN Inference Module
=====================

Lightweight wrapper that loads the trained CRNN plagiarism model and exposes
simple similarity functions for use in the scan pipeline.

If the model file is not found (not yet trained), all functions gracefully
return 0.0 so the rest of the pipeline is unaffected.
"""

import os
import re
from typing import List, Optional

# ── Lazy imports — only load torch when actually needed ──────────────────────
_model = None
_tokenizer = None
_device = None
_load_attempted = False

# Paths (relative to this file's directory)
_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_DIR, "crnn_plagiarism.pt")
_VOCAB_PATH = os.path.join(_DIR, "crnn_vocab.json")

# Sequence length must match training
_MAX_SEQ_LEN = 512


def _load_model():
    """
    Load the trained CRNN model and tokenizer into memory.
    Called once on first inference request; cached thereafter.
    """
    global _model, _tokenizer, _device, _load_attempted
    _load_attempted = True

    if not os.path.exists(_MODEL_PATH) or not os.path.exists(_VOCAB_PATH):
        print(f"[CRNN] Model files not found at {_MODEL_PATH} — skipping CRNN inference.")
        return False

    try:
        import torch
        from algorithms.crnn_model import SiameseCRNN, SimpleTokenizer

        _device = torch.device("cpu")

        # Load tokenizer
        _tokenizer = SimpleTokenizer.load(_VOCAB_PATH)

        # Load model
        checkpoint = torch.load(_MODEL_PATH, map_location=_device, weights_only=False)
        vocab_size = checkpoint.get("vocab_size", 30_000)
        _model = SiameseCRNN(vocab_size=vocab_size)
        _model.load_state_dict(checkpoint["model_state_dict"])
        _model.to(_device)
        _model.eval()

        params = _model.count_parameters()
        print(f"[CRNN] Model loaded successfully ({params:,} parameters)")
        return True

    except Exception as exc:
        print(f"[CRNN] Failed to load model: {exc}")
        _model = None
        _tokenizer = None
        return False


def _ensure_loaded() -> bool:
    """Ensure model is loaded; return True if ready for inference."""
    if _model is not None:
        return True
    if _load_attempted:
        return False
    return _load_model()


def _clean_text(text: str) -> str:
    """Normalise whitespace."""
    return re.sub(r"\s+", " ", text).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def calculate_crnn_similarity(doc1: str, doc2: str) -> float:
    """
    Compute CRNN-based plagiarism similarity between two documents.

    Returns:
        float in [0.0, 1.0], or 0.0 if model is unavailable.
    """
    if not doc1.strip() or not doc2.strip():
        return 0.0

    if not _ensure_loaded():
        return 0.0

    try:
        import torch

        doc1_clean = _clean_text(doc1)
        doc2_clean = _clean_text(doc2)

        ids1 = _tokenizer.encode(doc1_clean, max_len=_MAX_SEQ_LEN)
        ids2 = _tokenizer.encode(doc2_clean, max_len=_MAX_SEQ_LEN)

        t1 = torch.tensor([ids1], dtype=torch.long, device=_device)
        t2 = torch.tensor([ids2], dtype=torch.long, device=_device)

        with torch.no_grad():
            score = _model(t1, t2).item()

        return float(min(1.0, max(0.0, score)))

    except Exception as exc:
        print(f"[CRNN] Inference error: {exc}")
        return 0.0


def calculate_crnn_similarity_batch(target_text: str, peer_texts: List[str]) -> List[float]:
    """
    Compute CRNN similarity between *target_text* and every text in *peer_texts*.

    Processes all pairs in a single batched forward pass for efficiency.

    Returns:
        List of float scores in [0.0, 1.0], one per peer.
    """
    if not peer_texts:
        return []
    if not target_text.strip():
        return [0.0] * len(peer_texts)

    if not _ensure_loaded():
        return [0.0] * len(peer_texts)

    try:
        import torch

        target_clean = _clean_text(target_text)
        peers_clean = [_clean_text(p) for p in peer_texts]

        # Encode all texts
        target_ids = _tokenizer.encode(target_clean, max_len=_MAX_SEQ_LEN)
        peer_ids = [_tokenizer.encode(p, max_len=_MAX_SEQ_LEN) for p in peers_clean]

        # Build batched tensors: repeat target for each peer
        batch_size = len(peer_texts)
        t1 = torch.tensor([target_ids] * batch_size, dtype=torch.long, device=_device)
        t2 = torch.tensor(peer_ids, dtype=torch.long, device=_device)

        with torch.no_grad():
            scores = _model(t1, t2).squeeze(-1)  # (batch,)

        return [float(min(1.0, max(0.0, s.item()))) for s in scores]

    except Exception as exc:
        print(f"[CRNN] Batch inference error: {exc}")
        return [0.0] * len(peer_texts)
