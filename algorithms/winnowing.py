import hashlib
import re
from typing import List, Tuple, Set, Dict

def sanitize_text(text: str) -> Tuple[str, List[int]]:
    """
    Cleans text by removing whitespace and punctuation, keeping only lowercase alphanumeric chars.
    Returns:
        A tuple of (sanitized_text, char_index_mapping)
        where char_index_mapping[i] yields the index in the original text 
        corresponding to character i in the sanitized text.
    """
    sanitized = []
    mapping = []
    
    for idx, char in enumerate(text):
        if char.isalnum():
            sanitized.append(char.lower())
            mapping.append(idx)
            
    return "".join(sanitized), mapping

def get_k_grams(text: str, k: int) -> List[str]:
    """Generates k-grams from text."""
    if len(text) < k:
        return [text]
    return [text[i:i+k] for i in range(len(text) - k + 1)]

def hash_string(s: str) -> int:
    """Generates a 32-bit integer hash from a string using MD5."""
    return int(hashlib.md5(s.encode('utf-8')).hexdigest()[:8], 16)

def winnow(hashes: List[int], w: int) -> Set[Tuple[int, int]]:
    """
    Implements the Winnowing algorithm.
    Given a list of k-gram hashes and a window size w,
    selects the minimum hash in each sliding window of size w.
    Returns a set of tuples: (hash_value, index_in_k_grams)
    """
    fingerprints = set()
    n = len(hashes)
    
    if n < w:
        # If hashes are fewer than window size, select the minimum of what we have
        if hashes:
            min_val = min(hashes)
            min_idx = hashes.index(min_val)
            fingerprints.add((min_val, min_idx))
        return fingerprints
        
    # Sliding window
    for i in range(n - w + 1):
        window = hashes[i : i + w]
        # Find minimum value in the window
        min_val = min(window)
        # Find the rightmost occurrence of the minimum value in the window (for robustness)
        min_idx_in_window = len(window) - 1 - window[::-1].index(min_val)
        global_idx = i + min_idx_in_window
        fingerprints.add((min_val, global_idx))
        
    return fingerprints

class WinnowingMatcher:
    def __init__(self, k: int = 12, w: int = 4):
        self.k = k
        self.w = w

    def get_fingerprints(self, text: str) -> Tuple[Set[Tuple[int, int]], str, List[int]]:
        """
        Processes text and extracts winnowed fingerprints.
        Returns (fingerprints, sanitized_text, index_mapping)
        """
        sanitized, mapping = sanitize_text(text)
        if len(sanitized) < self.k:
            # Handle short texts
            k_grams = [sanitized]
        else:
            k_grams = get_k_grams(sanitized, self.k)
            
        hashes = [hash_string(gram) for gram in k_grams]
        fingerprints = winnow(hashes, self.w)
        return fingerprints, sanitized, mapping

    def calculate_similarity(self, text1: str, text2: str) -> Tuple[float, List[Dict]]:
        """
        Calculates similarity using Winnowing fingerprint intersection.
        Returns:
            - similarity_ratio (float: 0.0 to 1.0)
            - list of matched spans with original document indices
        """
        if not text1.strip() or not text2.strip():
            return 0.0, []

        # Get fingerprints
        fg1, san1, map1 = self.get_fingerprints(text1)
        fg2, san2, map2 = self.get_fingerprints(text2)

        # Hash sets for quick lookup
        hashes1 = {h for h, idx in fg1}
        hashes2 = {h for h, idx in fg2}

        # Calculate intersection
        intersection = hashes1.intersection(hashes2)
        union = hashes1.union(hashes2)

        if not union:
            return 0.0, []

        similarity_ratio = len(intersection) / len(union)

        # Trace back matching positions to highlight text
        # We group adjacent matching hashes to form cohesive text spans
        matched_spans = []
        
        # Maps for quick index lookups
        fg1_dict = {h: idx for h, idx in fg1}
        fg2_dict = {h: idx for h, idx in fg2}

        # Find matching k-grams
        matching_kgram_indices = []
        for h in intersection:
            idx1 = fg1_dict[h]
            idx2 = fg2_dict[h]
            matching_kgram_indices.append((idx1, idx2))

        # Sort matching k-grams by text1 index to group them chronologically
        matching_kgram_indices.sort(key=lambda x: x[0])

        # Group contiguous matching k-grams
        current_span = None
        for idx1, idx2 in matching_kgram_indices:
            # Map back to original text character offsets
            start_orig1 = map1[idx1]
            end_orig1 = map1[min(idx1 + self.k, len(map1) - 1)] + 1

            start_orig2 = map2[idx2]
            end_orig2 = map2[min(idx2 + self.k, len(map2) - 1)] + 1

            text_content = text1[start_orig1:end_orig1]

            if current_span is None:
                current_span = {
                    "start1": start_orig1,
                    "end1": end_orig1,
                    "start2": start_orig2,
                    "end2": end_orig2,
                    "text": text_content
                }
            else:
                # If they are very close in the document, merge them
                if start_orig1 - current_span["end1"] <= 20: # margin of 20 characters
                    current_span["end1"] = end_orig1
                    current_span["end2"] = max(current_span["end2"], end_orig2)
                    current_span["text"] = text1[current_span["start1"]:current_span["end1"]]
                else:
                    matched_spans.append(current_span)
                    current_span = {
                        "start1": start_orig1,
                        "end1": end_orig1,
                        "start2": start_orig2,
                        "end2": end_orig2,
                        "text": text_content
                    }
                    
        if current_span:
            matched_spans.append(current_span)

        return similarity_ratio, matched_spans
