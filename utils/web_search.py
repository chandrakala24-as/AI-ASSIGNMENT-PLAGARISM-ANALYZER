from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from database import get_db
from algorithms.tfidf_cosine import calculate_tfidf_similarity_batch
from algorithms.ngram_matching import calculate_ngram_similarity
from algorithms.winnowing import WinnowingMatcher
from algorithms.bert_semantic import calculate_bert_similarity_batch

# Pre-filter threshold: run expensive BERT/Winnowing only when TF-IDF
# similarity already suggests potential overlap.
_PREFILTER_THRESHOLD = 0.03   # 3 %
_REPORT_THRESHOLD    = 5.0    # minimum combined % to include in report


def search_internet_similarity(student_text: str) -> Tuple[float, List[Dict]]:
    """
    Compare *student_text* against every record in the mock_internet_sources
    table and return a (highest_score_pct, matches) tuple.

    Performance improvements over the previous implementation
    ---------------------------------------------------------
    1. **Batch TF-IDF** – one vectoriser is fitted on all documents at once
       instead of rebuilding it for every source pair.
    2. **Pre-filtering** – only sources with TF-IDF ≥ 3 % are passed to the
       slower BERT and Winnowing algorithms.
    3. **Batch BERT** – all candidate sources are encoded in a single
       ``model.encode()`` call instead of one call per source.
    4. **Parallel Winnowing / N-gram** – structural checks run concurrently
       using a thread pool.
    """
    if not student_text.strip():
        return 0.0, []

    # ── Fetch all internet sources ──────────────────────────────────────────
    db = get_db()
    sources_cur = db.mock_internet_sources.find()
    sources = list(sources_cur)

    if not sources:
        return 0.0, []

    src_contents = [s["content"] for s in sources]

    # ── STEP 1: Batch TF-IDF (fast pre-pass over all sources) ──────────────
    tfidf_scores = calculate_tfidf_similarity_batch(student_text, src_contents)

    # ── STEP 2: Pre-filter ─────────────────────────────────────────────────
    candidate_indices = [
        i for i, score in enumerate(tfidf_scores)
        if score >= _PREFILTER_THRESHOLD
    ]

    # ── STEP 3: Batch BERT for candidates only ──────────────────────────────
    bert_scores = [0.0] * len(sources)
    if candidate_indices:
        candidate_texts = [src_contents[i] for i in candidate_indices]
        bert_batch = calculate_bert_similarity_batch(student_text, candidate_texts)
        for idx, bert_s in zip(candidate_indices, bert_batch):
            bert_scores[idx] = bert_s

    # ── STEP 4: Parallel Winnowing + N-gram for candidates ─────────────────
    winnow_matcher = WinnowingMatcher(k=12, w=4)

    ngram_scores:   List[float]     = [0.0] * len(sources)
    winnow_scores:  List[float]     = [0.0] * len(sources)
    winnow_spans:   List[List]      = [[]   for _ in sources]

    def _structural_check(idx: int):
        text = src_contents[idx]
        ng_s  = calculate_ngram_similarity(student_text, text)
        wn_s, spans = winnow_matcher.calculate_similarity(student_text, text)
        return idx, ng_s, wn_s, spans

    if candidate_indices:
        max_workers = min(8, len(candidate_indices))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_structural_check, i): i for i in candidate_indices}
            for fut in as_completed(futures):
                try:
                    idx, ng_s, wn_s, spans = fut.result()
                    ngram_scores[idx]  = ng_s
                    winnow_scores[idx] = wn_s
                    winnow_spans[idx]  = spans
                except Exception as exc:
                    print(f"[WebSearch] Structural check error: {exc}")

    # ── STEP 5: Combine scores and build report ─────────────────────────────
    matches: List[Dict] = []
    highest_score = 0.0

    for i, src in enumerate(sources):
        combined = (
            0.30 * tfidf_scores[i]
            + 0.20 * ngram_scores[i]
            + 0.20 * winnow_scores[i]
            + 0.30 * bert_scores[i]
        )
        pct = round(combined * 100.0, 1)

        if pct >= _REPORT_THRESHOLD:
            if pct > highest_score:
                highest_score = pct

            matches.append({
                "source_id":       str(src["_id"]),
                "title":           src["title"],
                "url":             src["url"],
                "similarity_pct":  pct,
                "scores": {
                    "tfidf":      round(tfidf_scores[i]  * 100.0, 1),
                    "ngram":      round(ngram_scores[i]  * 100.0, 1),
                    "winnowing":  round(winnow_scores[i] * 100.0, 1),
                    "bert":       round(bert_scores[i]   * 100.0, 1),
                },
                "matched_spans": winnow_spans[i][:10],
            })

    matches.sort(key=lambda x: x["similarity_pct"], reverse=True)
    return highest_score, matches
