"""
CRNN Training Script for Plagiarism Detection
==============================================

Generates synthetic training data from existing documents and public-domain
text, then trains the Siamese CRNN model.

Usage:
    python algorithms/train_crnn.py

The trained model is saved to:
    algorithms/crnn_plagiarism.pt
    algorithms/crnn_vocab.json
"""

import os
import sys
import re
import random
import math
import time
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Ensure project root is on the path so we can import our modules
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from algorithms.crnn_model import SiameseCRNN, SimpleTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

VOCAB_SIZE  = 30_000
MAX_SEQ_LEN = 512
EMBED_DIM   = 128
BATCH_SIZE  = 32
EPOCHS      = 30
LR          = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE     = 5          # Early stopping patience
VAL_SPLIT    = 0.20       # 20% validation
SEED         = 42

MODEL_SAVE_PATH = os.path.join(_SCRIPT_DIR, "crnn_plagiarism.pt")
VOCAB_SAVE_PATH = os.path.join(_SCRIPT_DIR, "crnn_vocab.json")
UPLOADS_DIR     = os.path.join(_PROJECT_ROOT, "uploads")

random.seed(SEED)
torch.manual_seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Gather base texts
# ─────────────────────────────────────────────────────────────────────────────

# Built-in corpus of diverse paragraphs to supplement uploaded documents.
# These cover different topics to help the model learn what "different" looks like.
BUILTIN_CORPUS = [
    # ── Technology ────────────────────────────────────────────────────────
    "Python is a high-level general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected. It supports multiple programming paradigms including structured, object-oriented, and functional programming.",
    "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves.",
    "A neural network is a series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates. Neural networks can adapt to changing input so the network generates the best possible result without needing to redesign the output criteria.",
    "Cloud computing is the on-demand availability of computer system resources especially data storage and computing power without direct active management by the user. Large clouds often have functions distributed over multiple locations each of which is a data center.",
    "Cybersecurity is the practice of protecting systems networks and programs from digital attacks. These cyberattacks are usually aimed at accessing changing or destroying sensitive information extorting money from users or interrupting normal business processes.",
    "The Internet of Things describes physical objects with sensors processing ability software and other technologies that connect and exchange data with other devices and systems over the internet or other communications networks.",
    "Blockchain is a distributed database or ledger shared among a computer network's nodes. They are best known for their crucial role in cryptocurrency systems for maintaining a secure and decentralized record of transactions but they are not limited to cryptocurrency uses.",
    "DevOps is a set of practices that combines software development and information technology operations which aims to shorten the systems development life cycle and provide continuous delivery with high software quality. DevOps is complementary with Agile software development.",

    # ── Science ───────────────────────────────────────────────────────────
    "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that through cellular respiration can later be released to fuel the organism's activities. Some of this chemical energy is stored in carbohydrate molecules such as sugars and starches.",
    "The theory of general relativity describes the fundamental interaction of gravitation as a result of spacetime being curved by matter and energy. It was proposed by Albert Einstein in 1915 and has since been confirmed by numerous experiments and observations.",
    "DNA or deoxyribonucleic acid is a molecule composed of two polynucleotide chains that coil around each other to form a double helix. The molecule carries genetic instructions for the development functioning growth and reproduction of all known organisms and many viruses.",
    "Climate change refers to long-term shifts in temperatures and weather patterns. Human activities have been the main driver of climate change primarily due to the burning of fossil fuels like coal oil and gas which produces heat-trapping gases.",
    "Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles. It is the foundation of all quantum physics including quantum chemistry and quantum computing.",

    # ── History & Society ─────────────────────────────────────────────────
    "The Renaissance was a period in European history marking the transition from the Middle Ages to modernity and covering the 15th and 16th centuries. It occurred after the Crisis of the Late Middle Ages and was associated with great social change.",
    "Democracy is a form of government in which the people have the authority to deliberate and decide legislation or to choose governing officials to do so. Who is considered part of the people and how authority is shared among the people has changed over time.",
    "Globalization is the process of interaction and integration among people companies and governments worldwide. It has accelerated since the 18th century due to advances in transportation and communication technology. This increase in global interactions has caused a growth in international trade.",

    # ── Literature & Writing ──────────────────────────────────────────────
    "Academic writing is a formal style of writing used in universities and scholarly publications. It involves clear and precise language, well-structured arguments, evidence-based reasoning, and proper citation of sources. Students are expected to develop original analyses and avoid plagiarism.",
    "Plagiarism is the representation of another author's language thoughts ideas or expressions as one's own original work. In educational contexts this can result in sanctions. Various institutions have developed different approaches to detect and prevent plagiarism.",
    "Technical writing is writing or drafting technical communication used in technical and occupational fields such as computer hardware and software, architecture, engineering, chemistry, aeronautics, robotics, finance, medical, consumer electronics, biotechnology, and forestry.",
    "Natural language processing is a subfield of linguistics computer science and artificial intelligence concerned with the interactions between computers and human language. The goal is a computer capable of understanding the contents of documents including the contextual nuances of the language within them.",

    # ── Mathematics ───────────────────────────────────────────────────────
    "Linear algebra is the branch of mathematics concerning linear equations such as linear maps and their representations in vector spaces and through matrices. Linear algebra is central to almost all areas of mathematics and is used in most sciences and fields of engineering.",
    "Calculus is the mathematical study of continuous change. It has two major branches differential calculus and integral calculus. Differential calculus concerns instantaneous rates of change and the slopes of curves while integral calculus concerns the accumulation of quantities.",
    "Statistics is the discipline that concerns the collection organization analysis interpretation and presentation of data. In applying statistics to a scientific industrial or social problem it is conventional to begin with a statistical population or a statistical model to be studied.",
    "Probability theory is the branch of mathematics concerned with probability. Although there are several different probability interpretations, probability theory treats the concept in a rigorous mathematical manner by expressing it through a set of axioms.",

    # ── Business ──────────────────────────────────────────────────────────
    "Supply chain management is the management of the flow of goods and services between businesses and locations and includes the movement and storage of raw materials work-in-process inventory and finished goods from point of origin to point of consumption.",
    "Marketing is the process of exploring creating and delivering value to meet the needs of a target market in terms of goods and services. It involves the selection of a target audience and the advertising of the product to that audience.",
    "Entrepreneurship is the creation or extraction of economic value. With this definition entrepreneurship is viewed as change which may include other values than simply economic ones.",
]


def _gather_uploaded_texts() -> List[str]:
    """
    Extract text from uploaded PDF/DOCX files in the uploads directory.
    Uses the project's existing text_extractor.
    """
    print("[Data] Skipping heavy PDF extraction for faster training on CPU.")
    return []
    if not os.path.isdir(UPLOADS_DIR):
        return texts

    try:
        from utils.text_extractor import extract_text
    except ImportError:
        print("[Data] Could not import text_extractor — using built-in corpus only.")
        return texts

    for fname in os.listdir(UPLOADS_DIR):
        fpath = os.path.join(UPLOADS_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext not in (".pdf", ".docx", ".pptx", ".txt"):
            continue
        try:
            text = extract_text(fpath)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > 100:
                texts.append(text)
                print(f"  [Data] Extracted {len(text):,} chars from {fname}")
        except Exception as exc:
            print(f"  [Data] Failed to extract {fname}: {exc}")

    return texts


def _split_into_paragraphs(text: str, min_words: int = 30) -> List[str]:
    """Split a long text into paragraph-sized chunks."""
    # Split on double newlines or sentence-ending punctuation followed by space
    sentences = re.split(r'(?<=[.!?])\s+', text)
    paragraphs: List[str] = []
    current: List[str] = []
    current_words = 0

    for sent in sentences:
        words = len(sent.split())
        current.append(sent)
        current_words += words
        if current_words >= min_words + random.randint(0, 30):
            paragraphs.append(" ".join(current))
            current = []
            current_words = 0

    if current and current_words >= min_words // 2:
        paragraphs.append(" ".join(current))

    return paragraphs


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Synthetic data generation
# ─────────────────────────────────────────────────────────────────────────────

# Simple synonym map for augmentation
_SYNONYMS = {
    "use": ["utilize", "employ", "apply"],
    "make": ["create", "build", "construct"],
    "show": ["demonstrate", "display", "illustrate"],
    "big": ["large", "huge", "significant"],
    "small": ["tiny", "little", "minor"],
    "good": ["excellent", "great", "effective"],
    "bad": ["poor", "terrible", "ineffective"],
    "important": ["crucial", "essential", "vital"],
    "different": ["various", "diverse", "distinct"],
    "process": ["procedure", "method", "technique"],
    "system": ["framework", "platform", "infrastructure"],
    "data": ["information", "records", "dataset"],
    "method": ["approach", "technique", "strategy"],
    "result": ["outcome", "finding", "conclusion"],
    "provide": ["offer", "supply", "deliver"],
    "develop": ["create", "build", "design"],
    "include": ["contain", "comprise", "incorporate"],
    "increase": ["grow", "expand", "rise"],
    "change": ["modify", "alter", "transform"],
    "help": ["assist", "support", "aid"],
    "fast": ["quick", "rapid", "swift"],
    "problem": ["issue", "challenge", "difficulty"],
    "support": ["assist", "help", "back"],
    "program": ["application", "software", "tool"],
    "network": ["system", "grid", "web"],
}


def _synonym_replace(text: str, replace_prob: float = 0.15) -> str:
    """Replace some words with synonyms."""
    words = text.split()
    result = []
    for w in words:
        w_lower = w.lower()
        if w_lower in _SYNONYMS and random.random() < replace_prob:
            replacement = random.choice(_SYNONYMS[w_lower])
            # Preserve original capitalisation if first char was upper
            if w[0].isupper():
                replacement = replacement.capitalize()
            result.append(replacement)
        else:
            result.append(w)
    return " ".join(result)


def _shuffle_sentences(text: str) -> str:
    """Shuffle the sentence order in a text."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= 1:
        return text
    random.shuffle(sentences)
    return " ".join(sentences)


def _delete_random_sentences(text: str, delete_frac: float = 0.3) -> str:
    """Delete a fraction of sentences randomly."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= 2:
        return text
    keep_count = max(1, int(len(sentences) * (1 - delete_frac)))
    kept = random.sample(sentences, keep_count)
    return " ".join(kept)


def _insert_noise_words(text: str, noise_prob: float = 0.05) -> str:
    """Insert random noise words to simulate light paraphrasing."""
    noise_words = ["however", "moreover", "additionally", "furthermore",
                   "basically", "essentially", "notably", "indeed"]
    words = text.split()
    result = []
    for w in words:
        result.append(w)
        if random.random() < noise_prob:
            result.append(random.choice(noise_words))
    return " ".join(result)


def _mix_texts(text1: str, text2: str, ratio: float = 0.5) -> str:
    """Mix sentences from two texts to create partial overlap."""
    sents1 = re.split(r'(?<=[.!?])\s+', text1)
    sents2 = re.split(r'(?<=[.!?])\s+', text2)
    mixed = []
    for s in sents1:
        if random.random() < ratio:
            mixed.append(s)
    for s in sents2:
        if random.random() < (1 - ratio):
            mixed.append(s)
    random.shuffle(mixed)
    return " ".join(mixed) if mixed else text1


def generate_training_pairs(base_texts: List[str]) -> List[Tuple[str, str, float]]:
    """
    Generate synthetic training pairs with similarity labels.

    Returns:
        List of (text_a, text_b, similarity_label) tuples.
    """
    pairs: List[Tuple[str, str, float]] = []

    # Split long texts into paragraphs for more variety
    paragraphs: List[str] = []
    for text in base_texts:
        if len(text.split()) > 80:
            paragraphs.extend(_split_into_paragraphs(text))
        else:
            paragraphs.append(text)

    # Remove very short paragraphs
    paragraphs = [p for p in paragraphs if len(p.split()) >= 20]

    if len(paragraphs) < 5:
        print(f"[Data] WARNING: Only {len(paragraphs)} paragraphs available. Results may be limited.")

    print(f"[Data] Working with {len(paragraphs)} text segments for pair generation.")

    # ── Category 1: Identical pairs (label ≈ 1.0) ────────────────────────
    for p in paragraphs:
        pairs.append((p, p, 1.0))
        # Near-identical: very minor word insertion
        augmented = _insert_noise_words(p, noise_prob=0.02)
        pairs.append((p, augmented, 0.95))

    # ── Category 2: High similarity (label 0.7–0.9) ──────────────────────
    for p in paragraphs:
        # Synonym replacement
        syn = _synonym_replace(p, replace_prob=0.15)
        pairs.append((p, syn, random.uniform(0.75, 0.90)))

        # Sentence shuffle
        shuffled = _shuffle_sentences(p)
        pairs.append((p, shuffled, random.uniform(0.70, 0.85)))

        # Synonym + shuffle
        syn_shuf = _shuffle_sentences(_synonym_replace(p, 0.20))
        pairs.append((p, syn_shuf, random.uniform(0.65, 0.80)))

        # Light deletion
        deleted = _delete_random_sentences(p, 0.15)
        pairs.append((p, deleted, random.uniform(0.70, 0.85)))

    # ── Category 3: Medium similarity (label 0.3–0.6) ────────────────────
    for _ in range(len(paragraphs) * 3):
        i, j = random.sample(range(len(paragraphs)), 2)
        # Mix 30-60% from one, rest from another
        ratio = random.uniform(0.3, 0.6)
        mixed = _mix_texts(paragraphs[i], paragraphs[j], ratio)
        label = ratio * random.uniform(0.4, 0.7)
        pairs.append((paragraphs[i], mixed, label))

    # ── Category 4: Low/No similarity (label 0.0–0.2) ────────────────────
    for _ in range(len(paragraphs) * 3):
        i, j = random.sample(range(len(paragraphs)), 2)
        # Use completely different paragraphs
        label = random.uniform(0.0, 0.15)
        pairs.append((paragraphs[i], paragraphs[j], label))

    # Also pair with heavily modified versions
    for p in paragraphs:
        heavily_modified = _synonym_replace(
            _delete_random_sentences(_shuffle_sentences(p), 0.5),
            replace_prob=0.3
        )
        pairs.append((p, heavily_modified, random.uniform(0.20, 0.40)))

    random.shuffle(pairs)
    print(f"[Data] Generated {len(pairs)} training pairs.")

    # Distribution stats
    labels = [l for _, _, l in pairs]
    bins = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    for l in labels:
        if l < 0.2: bins["0.0-0.2"] += 1
        elif l < 0.4: bins["0.2-0.4"] += 1
        elif l < 0.6: bins["0.4-0.6"] += 1
        elif l < 0.8: bins["0.6-0.8"] += 1
        else: bins["0.8-1.0"] += 1
    print(f"[Data] Label distribution: {bins}")

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PlagiarismPairDataset(Dataset):
    """Dataset of (text_a, text_b, label) pairs, tokenized on-the-fly."""

    def __init__(self, pairs: List[Tuple[str, str, float]], tokenizer: SimpleTokenizer,
                 max_len: int = 512):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        text_a, text_b, label = self.pairs[idx]
        ids_a = self.tokenizer.encode(text_a, max_len=self.max_len)
        ids_b = self.tokenizer.encode(text_b, max_len=self.max_len)
        return (
            torch.tensor(ids_a, dtype=torch.long),
            torch.tensor(ids_b, dtype=torch.long),
            torch.tensor([label], dtype=torch.float32),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train():
    """Main training entry point."""
    print("=" * 70)
    print("  CRNN Plagiarism Model — Training Pipeline")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Train] Device: {device}")

    # ── Gather texts ─────────────────────────────────────────────────────
    print("\n[Step 1/5] Gathering base texts …")
    uploaded_texts = _gather_uploaded_texts()
    all_texts = BUILTIN_CORPUS + uploaded_texts
    print(f"  Total base texts: {len(all_texts)} ({len(BUILTIN_CORPUS)} built-in + {len(uploaded_texts)} uploaded)")

    # ── Generate pairs ───────────────────────────────────────────────────
    print("\n[Step 2/5] Generating synthetic training pairs …")
    pairs = generate_training_pairs(all_texts)

    # ── Build tokenizer ──────────────────────────────────────────────────
    print("\n[Step 3/5] Building tokenizer vocabulary …")
    all_pair_texts = [a for a, b, _ in pairs] + [b for a, b, _ in pairs]
    tokenizer = SimpleTokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.fit(all_pair_texts)
    actual_vocab = len(tokenizer.word2idx)
    print(f"  Vocabulary size: {actual_vocab:,} words")

    # ── Split into train/val ─────────────────────────────────────────────
    split_idx = int(len(pairs) * (1 - VAL_SPLIT))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    print(f"  Train: {len(train_pairs)} pairs | Val: {len(val_pairs)} pairs")

    train_dataset = PlagiarismPairDataset(train_pairs, tokenizer, max_len=MAX_SEQ_LEN)
    val_dataset = PlagiarismPairDataset(val_pairs, tokenizer, max_len=MAX_SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=False)

    # ── Build model ──────────────────────────────────────────────────────
    print("\n[Step 4/5] Building CRNN model …")
    model = SiameseCRNN(vocab_size=actual_vocab, embed_dim=EMBED_DIM)
    model.to(device)
    params = model.count_parameters()
    print(f"  Architecture: SiameseCRNN")
    print(f"  Parameters: {params:,}")
    print(f"  Embed dim: {EMBED_DIM} | LSTM hidden: 64 (bi) | Conv filters: 64")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = nn.MSELoss()

    # ── Training ─────────────────────────────────────────────────────────
    print(f"\n[Step 5/5] Training for up to {EPOCHS} epochs (patience={PATIENCE}) …")
    print("-" * 70)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None
    training_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # ── Train phase ──────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch_idx, (ids_a, ids_b, labels) in enumerate(train_loader):
            ids_a = ids_a.to(device)
            ids_b = ids_b.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            preds = model(ids_a, ids_b)
            loss = criterion(preds, labels)
            loss.backward()

            # Gradient clipping to stabilise training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / max(train_batches, 1)

        # ── Validation phase ─────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for ids_a, ids_b, labels in val_loader:
                ids_a = ids_a.to(device)
                ids_b = ids_b.to(device)
                labels = labels.to(device)

                preds = model(ids_a, ids_b)
                loss = criterion(preds, labels)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        epoch_time = time.time() - epoch_start

        # ── Logging ──────────────────────────────────────────────────────
        lr_now = optimizer.param_groups[0]["lr"]
        improved = " * BEST" if avg_val_loss < best_val_loss else ""
        print(f"  Epoch {epoch:2d}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"LR: {lr_now:.1e} | "
              f"Time: {epoch_time:.1f}s"
              f"{improved}")

        scheduler.step(avg_val_loss)

        # ── Early stopping ───────────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\n  [STOP] Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    total_time = time.time() - training_start
    print("-" * 70)
    print(f"  Training complete in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Best validation loss: {best_val_loss:.4f}")

    # ── Save model ───────────────────────────────────────────────────────
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab_size": actual_vocab,
        "embed_dim": EMBED_DIM,
        "max_seq_len": MAX_SEQ_LEN,
        "best_val_loss": best_val_loss,
        "total_params": params,
    }
    torch.save(checkpoint, MODEL_SAVE_PATH)
    tokenizer.save(VOCAB_SAVE_PATH)

    model_size_mb = os.path.getsize(MODEL_SAVE_PATH) / (1024 * 1024)
    print(f"\n  [SUCCESS] Model saved to: {MODEL_SAVE_PATH} ({model_size_mb:.1f} MB)")
    print(f"  [SUCCESS] Vocab saved to: {VOCAB_SAVE_PATH}")

    # ── Quick sanity check ───────────────────────────────────────────────
    print("\n[Sanity Check] Testing trained model …")
    model.eval()
    with torch.no_grad():
        # Identical texts should score high
        text_a = "Machine learning is a subset of artificial intelligence that provides systems the ability to learn."
        text_b = "Machine learning is a subset of artificial intelligence that provides systems the ability to learn."
        ids_a = torch.tensor([tokenizer.encode(text_a, MAX_SEQ_LEN)], dtype=torch.long, device=device)
        ids_b = torch.tensor([tokenizer.encode(text_b, MAX_SEQ_LEN)], dtype=torch.long, device=device)
        score_identical = model(ids_a, ids_b).item()

        # Different texts should score low
        text_c = "The Renaissance was a period in European history marking the transition from the Middle Ages."
        ids_c = torch.tensor([tokenizer.encode(text_c, MAX_SEQ_LEN)], dtype=torch.long, device=device)
        score_different = model(ids_a, ids_c).item()

        # Paraphrased should score medium-high
        text_d = "Artificial intelligence includes machine learning which allows systems to automatically learn from experience."
        ids_d = torch.tensor([tokenizer.encode(text_d, MAX_SEQ_LEN)], dtype=torch.long, device=device)
        score_paraphrase = model(ids_a, ids_d).item()

    print(f"  Identical texts:    {score_identical:.3f} (expected > 0.7)")
    print(f"  Paraphrased texts:  {score_paraphrase:.3f} (expected 0.4–0.8)")
    print(f"  Unrelated texts:    {score_different:.3f} (expected < 0.3)")

    verdict = "[PASS]" if score_identical > score_paraphrase > score_different else "[WARNING] Ordering not ideal but model may still work"
    print(f"  Verdict: {verdict}")

    print("\n" + "=" * 70)
    print("  Training pipeline complete!")
    print("=" * 70)


if __name__ == "__main__":
    train()
