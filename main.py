from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import os
import shutil
import sqlite3
import json
import re
from datetime import datetime
from typing import Optional, List

from database import get_db_connection, init_db, hash_password, verify_password
from utils.text_extractor import extract_text
from utils.web_search import search_internet_similarity
from algorithms.tfidf_cosine import calculate_tfidf_similarity, calculate_tfidf_similarity_batch
from algorithms.ngram_matching import calculate_ngram_similarity
from algorithms.winnowing import WinnowingMatcher
from algorithms.bert_semantic import calculate_bert_similarity, calculate_bert_similarity_batch, get_bert_model

# Pre-filter: only run expensive algorithms on pairs exceeding this TF-IDF score.
_PREFILTER_THRESHOLD = 0.03   # 3 %
_REPORT_THRESHOLD    = 5.0    # minimum combined % to include in report
_MAX_SCAN_WORKERS    = 8      # max parallel threads for structural analysis

# ─────────────────────────────────────────────────────────────────────────────
# In-memory submission processing status tracker
# Maps submission_id -> 'processing' | 'ready' | 'error'
# ─────────────────────────────────────────────────────────────────────────────
_upload_status: dict = {}

app = FastAPI(title="AI Assignment Plagiarism Analyser", version="1.0.0")

# Enable CORS for convenience
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "css"), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "js"), exist_ok=True)

# Initialize Database on Startup and pre-warm BERT model
@app.on_event("startup")
def startup_event():
    init_db()
    print("Database initialised.")
    # Pre-warm BERT in a background thread so the FIRST plagiarism scan is
    # instant rather than waiting ~40 s for the model to load.
    _thread_pool.submit(_preload_bert)
    print("BERT model warming up in background...")

def _preload_bert():
    """Loads and caches the BERT model at startup (runs in thread pool)."""
    try:
        get_bert_model()
        print("[Startup] BERT model ready.")
    except Exception as exc:
        print(f"[Startup] BERT pre-warm failed: {exc}")

# Shared thread pool for background extraction tasks
_thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="upload-worker")

# Redirect root to static landing page
@app.get("/")
def read_root():
    return RedirectResponse(url="/static/index.html")

# ==================== AUTHENTICATION ENDPOINTS ====================

@app.post("/api/auth/login")
def login(username: str = Form(...), password: str = Form(...)):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, password_hash, role, section, full_name FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    
    if not user or not verify_password(password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
        
    return {
        "status": "success",
        "user_id": user["id"],
        "username": user["username"],
        "role": user["role"],
        "section": user["section"],
        "full_name": user["full_name"]
    }

@app.post("/api/auth/register")
def register(
    username: str = Form(...), 
    password: str = Form(...), 
    full_name: str = Form(...), 
    role: str = Form(...), 
    section: Optional[str] = Form(None)
):
    if role not in ["teacher", "student"]:
        raise HTTPException(status_code=400, detail="Invalid registration role")
        
    # Check section constraints
    if role == "student" and (not section or section not in ["CY2A", "CY2B", "IY2A", "IY2B"]):
        raise HTTPException(status_code=400, detail="Students must specify section: CY2A, CY2B, IY2A, or IY2B")
        
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if username exists
    cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
    if cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="Username already exists")
        
    try:
        cursor.execute(
            "INSERT INTO users (username, password_hash, role, section, full_name) VALUES (?, ?, ?, ?, ?)",
            (username, hash_password(password), role, section if role == "student" else None, full_name)
        )
        conn.commit()
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        
    conn.close()
    return {"status": "success", "message": "User registered successfully"}

# ==================== ADMIN ENDPOINTS ====================

@app.get("/api/admin/users")
def get_users():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, role, section, full_name FROM users WHERE role != 'admin' ORDER BY role, username")
    users = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return users

@app.post("/api/admin/users")
def admin_create_user(
    username: str = Form(...),
    password: str = Form(...),
    full_name: str = Form(...),
    role: str = Form(...),
    section: Optional[str] = Form(None)
):
    if role not in ["teacher", "student"]:
        raise HTTPException(status_code=400, detail="Role must be teacher or student")
        
    if role == "student" and not section:
        raise HTTPException(status_code=400, detail="Student must have a class section")
        
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check duplicate
    cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
    if cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="Username already exists")
        
    cursor.execute(
        "INSERT INTO users (username, password_hash, role, section, full_name) VALUES (?, ?, ?, ?, ?)",
        (username, hash_password(password), role, section if role == "student" else None, full_name)
    )
    conn.commit()
    conn.close()
    return {"status": "success", "message": f"User {username} created successfully"}

@app.delete("/api/admin/users/{user_id}")
def admin_delete_user(user_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check user existence
    cursor.execute("SELECT username, role FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    if not user:
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")
        
    if user["role"] == "admin":
        conn.close()
        raise HTTPException(status_code=400, detail="Cannot delete administrator account")
        
    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    return {"status": "success", "message": f"User {user['username']} deleted successfully"}

# ==================== STUDENT ENDPOINTS ====================

@app.get("/api/student/assignments")
def get_student_assignments(section: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    # Fetch assignments created for this section, joining with teacher details
    cursor.execute("""
        SELECT a.id, a.title, a.description, a.due_date, u.full_name as teacher_name
        FROM assignments a
        LEFT JOIN users u ON a.created_by_teacher_id = u.id
        WHERE a.class_section = ?
        ORDER BY a.id DESC
    """, (section,))
    assignments = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return assignments

@app.get("/api/student/submissions")
def get_student_submissions(student_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.id as submission_id, s.file_name, s.extracted_text, s.submitted_at, s.marks, s.feedback,
               a.id as assignment_id, a.title as assignment_title, a.description as assignment_desc,
               r.peer_similarity_pct, r.internet_similarity_pct, r.overall_plagiarism_pct, r.detailed_report_json
        FROM submissions s
        JOIN assignments a ON s.assignment_id = a.id
        LEFT JOIN plagiarism_reports r ON r.submission_id = s.id
        WHERE s.student_id = ?
        ORDER BY s.submitted_at DESC
    """, (student_id,))
    rows = cursor.fetchall()
    
    submissions = []
    for row in rows:
        d = dict(row)
        if d["detailed_report_json"]:
            d["detailed_report"] = json.loads(d["detailed_report_json"])
        else:
            d["detailed_report"] = None
        del d["detailed_report_json"]
        submissions.append(d)
        
    conn.close()
    return submissions

@app.post("/api/student/upload")
async def student_upload_assignment(
    background_tasks: BackgroundTasks,
    assignment_id: int = Form(...),
    student_id: int = Form(...),
    file: UploadFile = File(...)
):
    """
    Fast upload endpoint — responds to the student immediately after saving
    the file to disk. Text extraction (PDF/DOCX parsing) runs in the
    background so students are never kept waiting.

    Flow:
        1. Validate file type and assignment (fast DB queries).
        2. Read & write file to disk asynchronously (non-blocking).
        3. Create/update the DB record with empty extracted_text.
        4. Return HTTP 200 to the student right away.
        5. Background task extracts text and updates the DB record.
    """
    # ── Validate extension ─────────────────────────────────────────────────
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in [".pdf", ".docx", ".pptx", ".ppt", ".png", ".jpg", ".jpeg"]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Upload PDF, DOCX, PPTX, PNG, or JPG."
        )

    conn = get_db_connection()
    cursor = conn.cursor()

    # ── Verify assignment exists ───────────────────────────────────────────
    cursor.execute("SELECT title, class_section FROM assignments WHERE id = ?", (assignment_id,))
    assignment = cursor.fetchone()
    if not assignment:
        conn.close()
        raise HTTPException(status_code=404, detail="Assignment not found")

    # ── Check for existing submission (re-upload scenario) ─────────────────
    cursor.execute(
        "SELECT id, file_path FROM submissions WHERE assignment_id = ? AND student_id = ?",
        (assignment_id, student_id)
    )
    existing_sub = cursor.fetchone()

    # ── Write file to disk asynchronously (non-blocking) ──────────────────
    timestamp      = datetime.now().strftime("%Y%m%d%H%M%S")
    safe_filename  = f"sub_{student_id}_{assignment_id}_{timestamp}{file_ext}"
    saved_file_path = os.path.join(UPLOAD_DIR, safe_filename)

    # Read entire file content (FastAPI UploadFile supports async read)
    file_content = await file.read()

    # Write to disk in thread pool — avoids blocking the async event loop
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        _thread_pool,
        lambda: open(saved_file_path, "wb").write(file_content)
    )

    # ── Create / update DB record with empty text (fast — no extraction yet)
    now_iso = datetime.now().isoformat()

    if existing_sub:
        # Remove old file
        try:
            if os.path.exists(existing_sub["file_path"]):
                os.remove(existing_sub["file_path"])
        except Exception as fe:
            print(f"[Upload] Could not remove old file: {fe}")

        cursor.execute("""
            UPDATE submissions
            SET file_name = ?, file_path = ?, extracted_text = ?,
                submitted_at = ?, marks = NULL, feedback = NULL
            WHERE id = ?
        """, (file.filename, saved_file_path, "", now_iso, existing_sub["id"]))
        submission_id = existing_sub["id"]
        cursor.execute("DELETE FROM plagiarism_reports WHERE submission_id = ?", (submission_id,))
    else:
        cursor.execute("""
            INSERT INTO submissions
                (assignment_id, student_id, file_name, file_path, extracted_text, submitted_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (assignment_id, student_id, file.filename, saved_file_path, "", now_iso))
        submission_id = cursor.lastrowid

    conn.commit()
    conn.close()

    # ── Mark as processing and queue background extraction ─────────────────
    _upload_status[submission_id] = "processing"
    background_tasks.add_task(
        _extract_and_store, submission_id, saved_file_path
    )

    # ── Respond to student immediately (< 1 second) ────────────────────────
    return {
        "status":        "success",
        "message":       "Assignment submitted! Your file is being processed in the background.",
        "submission_id": submission_id,
        "processing":    True
    }


async def _extract_and_store(submission_id: int, file_path: str):
    """
    Background task: extracts text from the uploaded file and updates the
    submissions table, then triggers the plagiarism scan. Runs after the
    HTTP response is already sent.
    """
    try:
        # Run CPU/IO-bound extraction in thread pool (non-blocking)
        loop = asyncio.get_event_loop()
        extracted_text = await loop.run_in_executor(
            _thread_pool, extract_text, file_path
        )

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE submissions SET extracted_text = ? WHERE id = ?",
            (extracted_text, submission_id)
        )
        conn.commit()
        conn.close()

        print(f"[Upload] Extraction complete for submission {submission_id} "
              f"({len(extracted_text.split())} words). Triggering plagiarism scan...")

        # Run plagiarism scan in executor since it's CPU/IO bound
        await loop.run_in_executor(
            _thread_pool, teacher_scan_submission, submission_id
        )

        _upload_status[submission_id] = "ready"
        print(f"[Upload] Plagiarism scan complete for submission {submission_id}")

    except Exception as exc:
        _upload_status[submission_id] = "error"
        print(f"[Upload] Background processing failed for submission {submission_id}: {exc}")


@app.get("/api/student/upload_status/{submission_id}")
def get_upload_status(submission_id: int):
    """
    Poll this endpoint after uploading to know when text extraction is done.
    Returns: { status: 'processing' | 'ready' | 'error' | 'unknown' }
    """
    status = _upload_status.get(submission_id, "unknown")
    return {"submission_id": submission_id, "status": status}

# ==================== TEACHER ENDPOINTS ====================

@app.post("/api/teacher/assignments")
def teacher_create_assignment(
    title: str = Form(...),
    description: str = Form(...),
    class_section: str = Form(...),
    due_date: str = Form(...),
    teacher_id: int = Form(...)
):
    if class_section not in ["CY2A", "CY2B", "IY2A", "IY2B"]:
        raise HTTPException(status_code=400, detail="Invalid section selection")
        
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO assignments (title, description, class_section, due_date, created_by_teacher_id)
        VALUES (?, ?, ?, ?, ?)
    """, (title, description, class_section, due_date, teacher_id))
    conn.commit()
    conn.close()
    return {"status": "success", "message": "Assignment created successfully."}

@app.get("/api/teacher/assignments")
def teacher_get_assignments(teacher_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, description, class_section, due_date FROM assignments WHERE created_by_teacher_id = ? ORDER BY id DESC", (teacher_id,))
    assignments = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return assignments

@app.get("/api/teacher/submissions")
def teacher_get_submissions(assignment_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.id as submission_id, s.file_name, s.extracted_text, s.submitted_at, s.marks, s.feedback,
               u.full_name as student_name, u.section as student_section,
               r.peer_similarity_pct, r.internet_similarity_pct, r.overall_plagiarism_pct, r.detailed_report_json
        FROM submissions s
        JOIN users u ON s.student_id = u.id
        LEFT JOIN plagiarism_reports r ON r.submission_id = s.id
        WHERE s.assignment_id = ?
        ORDER BY u.full_name ASC
    """, (assignment_id,))
    rows = cursor.fetchall()
    
    submissions = []
    for row in rows:
        d = dict(row)
        if d["detailed_report_json"]:
            d["detailed_report"] = json.loads(d["detailed_report_json"])
        else:
            d["detailed_report"] = None
        del d["detailed_report_json"]
        submissions.append(d)
        
    conn.close()
    return submissions

@app.post("/api/teacher/scan/{submission_id}")
def teacher_scan_submission(submission_id: int):
    """
    High-performance plagiarism scan using:
    - Batch TF-IDF  : one vectoriser fitted on ALL peers at once
    - Pre-filtering : BERT/Winnowing only run on candidates > 3% TF-IDF
    - Batch BERT    : all candidates encoded in a single model.encode() call
    - Parallel N-gram + Winnowing via ThreadPoolExecutor
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # ── 1. Fetch target submission ──────────────────────────────────────────
    cursor.execute("""
        SELECT s.id, s.extracted_text, s.student_id, s.assignment_id, u.full_name, u.section
        FROM submissions s
        JOIN users u ON s.student_id = u.id
        WHERE s.id = ?
    """, (submission_id,))
    target = cursor.fetchone()
    if not target:
        conn.close()
        raise HTTPException(status_code=404, detail="Submission not found")

    target_text = target["extracted_text"]

    # Normalise whitespace once; reuse this clean version for all algorithms.
    target_text_norm = re.sub(r"\s+", " ", target_text).strip()

    # ── Handle empty document ───────────────────────────────────────────────
    if not target_text_norm:
        cursor.execute("DELETE FROM plagiarism_reports WHERE submission_id = ?", (submission_id,))
        cursor.execute("""
            INSERT INTO plagiarism_reports
                (submission_id, peer_similarity_pct, internet_similarity_pct,
                 overall_plagiarism_pct, detailed_report_json)
            VALUES (?, 0.0, 0.0, 0.0, ?)
        """, (submission_id,
               json.dumps({"peer_matches": [], "internet_matches": [],
                           "message": "Document contains no extractable text."})))
        conn.commit()
        conn.close()
        return {"status": "success",
                "message": "Scan completed (Empty document)",
                "results": {"overall_plagiarism": 0}}

    # ── 2. Fetch all peer submissions ───────────────────────────────────────
    cursor.execute("""
        SELECT s.id as peer_sub_id, s.extracted_text, u.full_name as peer_name, u.section as peer_section
        FROM submissions s
        JOIN users u ON s.student_id = u.id
        WHERE s.id != ? AND s.extracted_text != ''
    """, (submission_id,))
    peers = cursor.fetchall()

    peer_matches      = []
    highest_peer_score = 0.0

    if peers:
        peer_texts = [re.sub(r"\s+", " ", p["extracted_text"]).strip() for p in peers]

        # ── STEP A: Batch TF-IDF across ALL peers in one vectoriser pass ────
        tfidf_scores = calculate_tfidf_similarity_batch(target_text_norm, peer_texts)

        # ── STEP B: Pre-filter — skip expensive algorithms for clearly unrelated docs
        candidate_indices  = [i for i, s in enumerate(tfidf_scores) if s >= _PREFILTER_THRESHOLD]
        candidate_texts    = [peer_texts[i] for i in candidate_indices]

        # ── STEP C: Batch BERT — single model.encode() for all candidates ───
        bert_scores = [0.0] * len(peers)
        if candidate_texts:
            bert_batch = calculate_bert_similarity_batch(target_text_norm, candidate_texts)
            for idx, bscore in zip(candidate_indices, bert_batch):
                bert_scores[idx] = bscore

        # ── STEP D: Parallel Winnowing + N-gram for candidates ──────────────
        winnow_matcher   = WinnowingMatcher(k=12, w=4)
        ngram_scores     = [0.0] * len(peers)
        winnow_scores    = [0.0] * len(peers)
        winnow_spans_all = [[]   for _ in peers]

        def _structural(idx: int):
            text = peer_texts[idx]
            ng_s          = calculate_ngram_similarity(target_text_norm, text)
            wn_s, spans   = winnow_matcher.calculate_similarity(target_text_norm, text)
            return idx, ng_s, wn_s, spans

        if candidate_indices:
            n_workers = min(_MAX_SCAN_WORKERS, len(candidate_indices))
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_structural, i): i for i in candidate_indices}
                for fut in as_completed(futures):
                    try:
                        idx, ng_s, wn_s, spans = fut.result()
                        ngram_scores[idx]      = ng_s
                        winnow_scores[idx]     = wn_s
                        winnow_spans_all[idx]  = spans
                    except Exception as exc:
                        print(f"[Scan] Structural analysis error for peer {futures[fut]}: {exc}")

        # ── STEP E: Combine all scores ───────────────────────────────────────
        for i, peer in enumerate(peers):
            combined = (
                0.30 * tfidf_scores[i]
                + 0.20 * ngram_scores[i]
                + 0.20 * winnow_scores[i]
                + 0.30 * bert_scores[i]
            )
            pct = round(combined * 100.0, 1)

            if pct >= _REPORT_THRESHOLD:
                if pct > highest_peer_score:
                    highest_peer_score = pct

                peer_matches.append({
                    "peer_submission_id": peer["peer_sub_id"],
                    "student_name":       peer["peer_name"],
                    "student_section":    peer["peer_section"],
                    "similarity_pct":     pct,
                    "scores": {
                        "tfidf":     round(tfidf_scores[i]  * 100.0, 1),
                        "ngram":     round(ngram_scores[i]  * 100.0, 1),
                        "winnowing": round(winnow_scores[i] * 100.0, 1),
                        "bert":      round(bert_scores[i]   * 100.0, 1),
                    },
                    "matched_spans": winnow_spans_all[i][:10],
                })

        peer_matches.sort(key=lambda x: x["similarity_pct"], reverse=True)

    # ── 3. Internet Plagiarism Scan (also uses batch algorithms internally) ─
    highest_internet_score, internet_matches = search_internet_similarity(target_text_norm)

    # ── 4. Overall score is the higher of peer vs internet ─────────────────
    overall_plagiarism = max(highest_peer_score, highest_internet_score)

    detailed_report = {
        "peer_matches":    peer_matches,
        "internet_matches": internet_matches,
        "metadata": {
            "scanned_at":       datetime.now().isoformat(),
            "target_word_count": len(target_text_norm.split()),
            "peers_scanned":    len(peers),
            "peers_deep_scanned": len(candidate_indices) if peers else 0,
            "algorithms_used": [
                "TF-IDF Cosine (Batch)",
                "N-gram Jaccard (Parallel)",
                "Winnowing Fingerprinting (Parallel)",
                "BERT Semantic Similarity (Batch + Cache)",
            ],
        },
    }

    # ── 5. Persist report ──────────────────────────────────────────────────
    cursor.execute("DELETE FROM plagiarism_reports WHERE submission_id = ?", (submission_id,))
    cursor.execute("""
        INSERT INTO plagiarism_reports
            (submission_id, peer_similarity_pct, internet_similarity_pct,
             overall_plagiarism_pct, detailed_report_json)
        VALUES (?, ?, ?, ?, ?)
    """, (submission_id, highest_peer_score, highest_internet_score,
           overall_plagiarism, json.dumps(detailed_report)))

    conn.commit()
    conn.close()

    return {
        "status":  "success",
        "message": "Scan completed.",
        "results": {
            "peer_similarity":    highest_peer_score,
            "internet_similarity": highest_internet_score,
            "overall_plagiarism":  overall_plagiarism,
            "detailed_report":     detailed_report,
        },
    }

@app.post("/api/teacher/batch_scan")
def teacher_batch_scan_submissions(assignment_id: int):
    """
    Batch-scan all submissions for an assignment.
    Submissions are processed sequentially so that the BERT embedding cache
    is warm after the first scan, making each subsequent scan significantly
    faster (peers' embeddings are already cached).
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM submissions WHERE assignment_id = ?", (assignment_id,))
    subs = cursor.fetchall()
    conn.close()

    scanned_count  = 0
    failed_count   = 0

    for sub in subs:
        try:
            teacher_scan_submission(sub["id"])
            scanned_count += 1
            print(f"[BatchScan] Scanned submission {sub['id']} ({scanned_count}/{len(subs)})")
        except Exception as exc:
            failed_count += 1
            print(f"[BatchScan] Failed to scan submission {sub['id']}: {exc}")

    msg = f"Batch scan complete: {scanned_count} succeeded, {failed_count} failed."
    return {"status": "success", "message": msg,
            "scanned": scanned_count, "failed": failed_count}

@app.post("/api/teacher/grade/{submission_id}")
def teacher_grade_submission(
    submission_id: int,
    marks: int = Form(...),
    feedback: str = Form(...)
):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM submissions WHERE id = ?", (submission_id,))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Submission not found")
        
    cursor.execute("UPDATE submissions SET marks = ?, feedback = ? WHERE id = ?", (marks, feedback, submission_id))
    conn.commit()
    conn.close()
    return {"status": "success", "message": "Grade and feedback saved successfully."}

@app.get("/api/teacher/analytics")
def teacher_get_analytics(teacher_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Fetch average plagiarism and average grade per class section (CY2A, CY2B, IY2A, IY2B)
    sections = ["CY2A", "CY2B", "IY2A", "IY2B"]
    section_data = []
    
    for sect in sections:
        cursor.execute("""
            SELECT AVG(s.marks) as avg_marks, AVG(r.overall_plagiarism_pct) as avg_plagiarism, COUNT(s.id) as sub_count
            FROM submissions s
            JOIN users u ON s.student_id = u.id
            JOIN assignments a ON s.assignment_id = a.id
            LEFT JOIN plagiarism_reports r ON r.submission_id = s.id
            WHERE u.section = ? AND a.created_by_teacher_id = ?
        """, (sect, teacher_id))
        res = cursor.fetchone()
        section_data.append({
            "section": sect,
            "avg_marks": round(res["avg_marks"] or 0, 1),
            "avg_plagiarism": round(res["avg_plagiarism"] or 0, 1),
            "submission_count": res["sub_count"]
        })
        
    # 2. Plagiarism risk levels count
    # Low: < 20%, Medium: 20-50%, High: > 50%
    cursor.execute("""
        SELECT 
            SUM(CASE WHEN r.overall_plagiarism_pct < 20 THEN 1 ELSE 0 END) as low_risk,
            SUM(CASE WHEN r.overall_plagiarism_pct >= 20 AND r.overall_plagiarism_pct <= 50 THEN 1 ELSE 0 END) as med_risk,
            SUM(CASE WHEN r.overall_plagiarism_pct > 50 THEN 1 ELSE 0 END) as high_risk,
            COUNT(s.id) as total_scanned
        FROM submissions s
        JOIN assignments a ON s.assignment_id = a.id
        JOIN plagiarism_reports r ON r.submission_id = s.id
        WHERE a.created_by_teacher_id = ?
    """, (teacher_id,))
    risk_res = cursor.fetchone()
    
    risk_distribution = {
        "low": risk_res["low_risk"] or 0,
        "medium": risk_res["med_risk"] or 0,
        "high": risk_res["high_risk"] or 0,
        "total": risk_res["total_scanned"] or 0
    }
    
    # 3. Marks vs. Plagiarism scatter correlation details
    cursor.execute("""
        SELECT u.full_name as student_name, s.marks, r.overall_plagiarism_pct as plagiarism
        FROM submissions s
        JOIN users u ON s.student_id = u.id
        JOIN assignments a ON s.assignment_id = a.id
        JOIN plagiarism_reports r ON r.submission_id = s.id
        WHERE a.created_by_teacher_id = ? AND s.marks IS NOT NULL
        ORDER BY s.marks ASC
    """, (teacher_id,))
    correlation_rows = cursor.fetchall()
    correlations = [dict(row) for row in correlation_rows]
    
    conn.close()
    
    return {
        "section_data": section_data,
        "risk_distribution": risk_distribution,
        "correlations": correlations
    }

# ==================== STATIC FILES MOUNT ====================

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    # Start the server on host 127.0.0.1 and port 8000
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
