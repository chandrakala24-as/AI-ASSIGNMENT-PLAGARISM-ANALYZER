# Simple Verification Script to test backend algorithms and text parsing

print("Starting AI Plagiarism Analyser verification check...")

try:
    from algorithms.tfidf_cosine import calculate_tfidf_similarity
    print("[-] TF-IDF & Cosine Similarity module: Loaded successfully.")
except Exception as e:
    print(f"[!] TF-IDF & Cosine Similarity module load failed: {e}")

try:
    from algorithms.ngram_matching import calculate_ngram_similarity
    print("[-] N-gram Matching module: Loaded successfully.")
except Exception as e:
    print(f"[!] N-gram Matching module load failed: {e}")

try:
    from algorithms.winnowing import WinnowingMatcher
    print("[-] Winnowing Fingerprinting module: Loaded successfully.")
except Exception as e:
    print(f"[!] Winnowing Fingerprinting module load failed: {e}")

try:
    from algorithms.bert_semantic import calculate_bert_similarity
    print("[-] BERT Semantic Similarity module: Loaded successfully.")
except Exception as e:
    print(f"[!] BERT Semantic Similarity module load failed: {e}")

# Try standard test texts
doc_a = "Python is an outstanding high-level programming language that features great dynamic typing and clean readability."
doc_b = "Python is a fantastic high-level coding language emphasizing readability and dynamic typing principles."
doc_c = "FastAPI is a modern, high-performance python framework for creating RESTful web applications with speed."

print("\n--- Running Similarity Matrix Checks ---")
try:
    print(f"Comparing Doc A and Doc B (paraphrased):")
    
    tfidf_sim = calculate_tfidf_similarity(doc_a, doc_b)
    ngram_sim = calculate_ngram_similarity(doc_a, doc_b)
    
    w_matcher = WinnowingMatcher(k=8, w=3)
    winnow_sim, spans = w_matcher.calculate_similarity(doc_a, doc_b)
    
    bert_sim = calculate_bert_similarity(doc_a, doc_b)
    
    print(f"  * TF-IDF Cosine similarity: {tfidf_sim:.3f}")
    print(f"  * N-gram similarity score: {ngram_sim:.3f}")
    print(f"  * Winnowing fingerprint score: {winnow_sim:.3f} (Spans found: {len(spans)})")
    print(f"  * BERT Semantic similarity score: {bert_sim:.3f}")
    
    print("\nComparing Doc A and Doc C (different topic):")
    tfidf_diff = calculate_tfidf_similarity(doc_a, doc_c)
    bert_diff = calculate_bert_similarity(doc_a, doc_c)
    print(f"  * TF-IDF Cosine similarity: {tfidf_diff:.3f}")
    print(f"  * BERT Semantic similarity: {bert_diff:.3f}")
    
    print("\n[SUCCESS] All similarity checkers are compiling and executing correctly!")
    
except Exception as err:
    print(f"[FAIL] Algorithm calculations triggered runtime error: {err}")
    import traceback
    traceback.print_exc()

print("\nVerification process complete.")
