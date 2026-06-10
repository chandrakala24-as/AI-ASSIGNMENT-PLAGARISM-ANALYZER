import re
from typing import Set, List, Tuple

def get_word_ngrams(text: str, n: int = 3) -> Set[Tuple[str, ...]]:
    """Tokenizes text and returns a set of word-level n-grams."""
    # Preprocess and split into words
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    if len(words) < n:
        return {tuple(words)}
    return {tuple(words[i:i+n]) for i in range(len(words) - n + 1)}

def get_char_ngrams(text: str, n: int = 5) -> Set[str]:
    """Cleans text of whitespace and returns a set of character-level n-grams."""
    # Remove all whitespace
    text = "".join(text.lower().split())
    if len(text) < n:
        return {text}
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def calculate_jaccard_similarity(set1: Set, set2: Set) -> float:
    """Computes the Jaccard similarity coefficient between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

def calculate_ngram_similarity(doc1: str, doc2: str) -> float:
    """
    Computes a combined word and character n-gram similarity score.
    Returns average Jaccard score (0.0 to 1.0).
    """
    if not doc1.strip() or not doc2.strip():
        return 0.0

    # Get word 3-grams
    word1 = get_word_ngrams(doc1, n=3)
    word2 = get_word_ngrams(doc2, n=3)
    word_sim = calculate_jaccard_similarity(word1, word2)

    # Get character 5-grams
    char1 = get_char_ngrams(doc1, n=5)
    char2 = get_char_ngrams(doc2, n=5)
    char_sim = calculate_jaccard_similarity(char1, char2)

    # Return weighted average: 60% word structure, 40% character spelling
    return (0.6 * word_sim) + (0.4 * char_sim)
