"""
CRNN (Convolutional Recurrent Neural Network) for Plagiarism Detection
======================================================================

Siamese architecture that encodes two text documents independently using
shared Conv1D + BiLSTM layers, then compares the encodings via a dense
comparison head to produce a similarity score in [0, 1].

Architecture
------------
  Shared Encoder (weight-tied):
    Embedding(vocab, 128) → Conv1D(128→64, k=3) → ReLU → MaxPool
                           → Conv1D(64→64, k=5)  → ReLU → MaxPool
                           → BiLSTM(64→128)       → GlobalAvgPool → 128-d

  Comparison Head:
    [enc1 ‖ enc2 ‖ |enc1−enc2| ‖ enc1⊙enc2]  (512-d)
    → Dense(512→128, ReLU, Drop=0.3)
    → Dense(128→64,  ReLU, Drop=0.2)
    → Dense(64→1,    Sigmoid)                  → similarity ∈ [0, 1]
"""

import re
import math
import json
import os
from collections import Counter
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────────────────────

class SimpleTokenizer:
    """
    Word-level tokenizer with a fixed vocabulary.

    Special tokens:
        0 = <PAD>
        1 = <UNK>
    """

    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self, vocab_size: int = 30_000):
        self.vocab_size = vocab_size
        self.word2idx: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word: Dict[int, str] = {0: "<PAD>", 1: "<UNK>"}
        self._fitted = False

    # ── Building the vocabulary ──────────────────────────────────────────

    def fit(self, texts: List[str]) -> "SimpleTokenizer":
        """Build vocabulary from a corpus of texts."""
        counter: Counter = Counter()
        for text in texts:
            tokens = self._tokenize(text)
            counter.update(tokens)

        # Keep the top (vocab_size - 2) most common words (reserve 0, 1)
        most_common = counter.most_common(self.vocab_size - 2)
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        self._fitted = True
        return self

    # ── Encoding ─────────────────────────────────────────────────────────

    def encode(self, text: str, max_len: int = 512) -> List[int]:
        """Convert text → list of integer token IDs, padded/truncated to max_len."""
        tokens = self._tokenize(text)
        ids = [self.word2idx.get(t, self.UNK_IDX) for t in tokens]
        # Truncate
        ids = ids[:max_len]
        # Pad
        ids += [self.PAD_IDX] * (max_len - len(ids))
        return ids

    def encode_batch(self, texts: List[str], max_len: int = 512) -> torch.Tensor:
        """Encode a batch of texts → (batch, max_len) LongTensor."""
        return torch.tensor([self.encode(t, max_len) for t in texts], dtype=torch.long)

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: str):
        """Save vocabulary to a JSON file."""
        data = {
            "vocab_size": self.vocab_size,
            "word2idx": self.word2idx,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "SimpleTokenizer":
        """Load vocabulary from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(vocab_size=data["vocab_size"])
        tok.word2idx = data["word2idx"]
        tok.idx2word = {int(v): k for k, v in data["word2idx"].items()}
        tok._fitted = True
        return tok

    # ── Internal ─────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Lowercase + split on word boundaries."""
        return re.findall(r"\b\w+\b", text.lower())


# ─────────────────────────────────────────────────────────────────────────────
# CRNN Encoder (shared between both branches of the Siamese network)
# ─────────────────────────────────────────────────────────────────────────────

class CRNNEncoder(nn.Module):
    """
    Conv1D → Conv1D → BiLSTM → GlobalAvgPool
    Produces a fixed-size 128-dimensional encoding of a variable-length text.
    """

    def __init__(self, vocab_size: int = 30_000, embed_dim: int = 128,
                 conv1_out: int = 64, conv2_out: int = 64,
                 lstm_hidden: int = 64, lstm_layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Conv block 1: captures 3-gram patterns
        self.conv1 = nn.Conv1d(embed_dim, conv1_out, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(conv1_out)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Conv block 2: captures 5-gram patterns
        self.conv2 = nn.Conv1d(conv1_out, conv2_out, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(conv2_out)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # BiLSTM for sequential context
        self.lstm = nn.LSTM(
            input_size=conv2_out,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # Output dimension: lstm_hidden * 2 (bidirectional) = 128
        self.output_dim = lstm_hidden * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) LongTensor of token IDs

        Returns:
            (batch, output_dim) FloatTensor — document encoding
        """
        # Embedding: (batch, seq_len) → (batch, seq_len, embed_dim)
        emb = self.embedding(x)
        emb = self.dropout(emb)

        # Conv expects (batch, channels, length) → transpose
        emb = emb.transpose(1, 2)  # (batch, embed_dim, seq_len)

        # Conv block 1
        c1 = self.pool1(F.relu(self.bn1(self.conv1(emb))))  # (batch, 64, seq_len//2)

        # Conv block 2
        c2 = self.pool2(F.relu(self.bn2(self.conv2(c1))))   # (batch, 64, seq_len//4)

        # Prepare for LSTM: (batch, length, features)
        c2 = c2.transpose(1, 2)  # (batch, seq_len//4, 64)

        # BiLSTM
        lstm_out, _ = self.lstm(c2)  # (batch, seq_len//4, lstm_hidden*2)

        # Global average pooling over the time dimension
        encoding = lstm_out.mean(dim=1)  # (batch, lstm_hidden*2=128)

        return encoding


# ─────────────────────────────────────────────────────────────────────────────
# Full Siamese CRNN Model
# ─────────────────────────────────────────────────────────────────────────────

class SiameseCRNN(nn.Module):
    """
    Siamese CRNN for document similarity.

    Takes two documents, encodes each with a *shared* CRNNEncoder, then
    computes similarity through a comparison head.
    """

    def __init__(self, vocab_size: int = 30_000, embed_dim: int = 128,
                 encoder_dropout: float = 0.1):
        super().__init__()

        self.encoder = CRNNEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            dropout=encoder_dropout,
        )

        enc_dim = self.encoder.output_dim  # 128

        # Comparison head input: [enc1, enc2, |enc1-enc2|, enc1*enc2] = 4 * 128 = 512
        self.classifier = nn.Sequential(
            nn.Linear(enc_dim * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: (batch, seq_len) — first document token IDs
            x2: (batch, seq_len) — second document token IDs

        Returns:
            (batch, 1) — similarity scores ∈ [0, 1]
        """
        enc1 = self.encoder(x1)  # (batch, 128)
        enc2 = self.encoder(x2)  # (batch, 128)

        # Multi-perspective comparison features
        diff = torch.abs(enc1 - enc2)
        prod = enc1 * enc2
        combined = torch.cat([enc1, enc2, diff, prod], dim=1)  # (batch, 512)

        return self.classifier(combined)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
