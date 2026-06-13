from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import os
import json
import re
from datetime import datetime
from typing import Optional, List

from database import get_db, init_db, hash_password, verify_password
from bson.objectid import ObjectId

def to_id(id_val):
    if isinstance(id_val, int): return id_val
    if isinstance(id_val, str) and len(id_val) == 24:
        try: return ObjectId(id_val)
        except: pass
    return str(id_val)
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
    
    # Check if BERT model is disabled to conserve memory (e.g., on Render free tier)
    from algorithms.bert_semantic import DISABLE_BERT
    if not DISABLE_BERT:
        # Pre-warm BERT in a background thread so the FIRST plagiarism scan is
        # instant rather than waiting ~40 s for the model to load.
        _thread_pool.submit(_preload_bert)
        print("BERT model warming up in background...")
    else:
        print("[Startup] Lightweight mode: skipping BERT model pre-warming to conserve RAM.")

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
    db = get_db()
    user = db.users.find_one({"username": username})
    
    if not user or not verify_password(password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
        
    return {
        "status": "success",
        "user_id": str(user["_id"]),
        "username": user["username"],
        "role": user["role"],
        "section": user.get("section"),
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
        
    db = get_db()
    
    # Check if username exists
    if db.users.find_one({"username": username}):
        raise HTTPException(status_code=400, detail="Username already exists")
        
    try:
        db.users.insert_one({
            "username": username,
            "password_hash": hash_password(password),
            "role": role,
            "section": section if role == "student" else None,
            "full_name": full_name
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        
    return {"status": "success", "message": "User registered successfully"}

# ==================== ADMIN ENDPOINTS ====================

@app.get("/api/admin/users")
def get_users():
    db = get_db()
    users_cur = db.users.find({"role": {"$ne": "admin"}}).sort([("role", 1), ("username", 1)])
    users = []
    for u in users_cur:
        users.append({
            "id": str(u["_id"]),
            "username": u["username"],
            "role": u["role"],
            "section": u.get("section"),
            "full_name": u["full_name"]
        })
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
        
    db = get_db()
    
    # Check duplicate
    if db.users.find_one({"username": username}):
        raise HTTPException(status_code=400, detail="Username already exists")
        
    db.users.insert_one({
        "username": username,
        "password_hash": hash_password(password),
        "role": role,
        "section": section if role == "student" else None,
        "full_name": full_name
    })
    return {"status": "success", "message": f"User {username} created successfully"}

@app.delete("/api/admin/users/{user_id}")
def admin_delete_user(user_id: str):
    db = get_db()
    
    # Check user existence
    user = db.users.find_one({"_id": to_id(user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    if user["role"] == "admin":
        raise HTTPException(status_code=400, detail="Cannot delete administrator account")
        
    db.users.delete_one({"_id": to_id(user_id)})
    return {"status": "success", "message": f"User {user['username']} deleted successfully"}

# ==================== STUDENT ENDPOINTS ====================

@app.get("/api/student/assignments")
def get_student_assignments(section: str):
    db = get_db()
    pipeline = [
        {"$match": {"class_section": section}},
        {"$lookup": {
            "from": "users",
            "localField": "created_by_teacher_id",
            "foreignField": "_id",
            "as": "teacher"
        }},
        {"$unwind": {"path": "$teacher", "preserveNullAndEmptyArrays": True}},
        {"$sort": {"_id": -1}}
    ]
    assignments_cur = db.assignments.aggregate(pipeline)
    assignments = []
    for a in assignments_cur:
        assignments.append({
            "id": str(a["_id"]),
            "title": a["title"],
            "description": a.get("description"),
            "due_date": a.get("due_date"),
            "teacher_name": a.get("teacher", {}).get("full_name", "Unknown")
        })
    return assignments

@app.get("/api/student/submissions")
def get_student_submissions(student_id: str):
    db = get_db()
    pipeline = [
        {"$match": {"student_id": to_id(student_id)}},
        {"$lookup": {
            "from": "assignments",
            "localField": "assignment_id",
            "foreignField": "_id",
            "as": "assignment"
        }},
        {"$unwind": {"path": "$assignment", "preserveNullAndEmptyArrays": True}},
        {"$lookup": {
            "from": "plagiarism_reports",
            "localField": "_id",
            "foreignField": "submission_id",
            "as": "report"
        }},
        {"$unwind": {"path": "$report", "preserveNullAndEmptyArrays": True}},
        {"$sort": {"submitted_at": -1}}
    ]
    
    rows = db.submissions.aggregate(pipeline)
    submissions = []
    
    for row in rows:
        report = row.get("report", {})
        assignment = row.get("assignment", {})
        
        # In MongoDB, detailed_report is likely already a dict/json object, 
        # but if we store it as string, we load it. Let's assume we store it as dict now.
        detailed_report = report.get("detailed_report_json")
        if isinstance(detailed_report, str):
            detailed_report = json.loads(detailed_report)
            
        submissions.append({
            "submission_id": str(row["_id"]),
            "file_name": row["file_name"],
            "extracted_text": row.get("extracted_text", ""),
            "submitted_at": row["submitted_at"],
            "marks": row.get("marks"),
            "feedback": row.get("feedback"),
            "assignment_id": str(assignment.get("_id", "")),
            "assignment_title": assignment.get("title", ""),
            "assignment_desc": assignment.get("description", ""),
            "peer_similarity_pct": report.get("peer_similarity_pct"),
            "internet_similarity_pct": report.get("internet_similarity_pct"),
            "overall_plagiarism_pct": report.get("overall_plagiarism_pct"),
            "detailed_report": detailed_report
        })
        
    return submissions

@app.post("/api/student/upload")
async def student_upload_assignment(
    background_tasks: BackgroundTasks,
    assignment_id: str = Form(...),
    student_id: str = Form(...),
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

    db = get_db()
    
    # Verify assignment exists
    assignment = db.assignments.find_one({"_id": to_id(assignment_id)})
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found")
        
    # Check for existing submission
    existing_sub = db.submissions.find_one({"assignment_id": to_id(assignment_id), "student_id": to_id(student_id)})
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    safe_filename = f"sub_{student_id}_{assignment_id}_{timestamp}{file_ext}"
    saved_file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    file_content = await file.read()
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_thread_pool, lambda: open(saved_file_path, "wb").write(file_content))
    
    now_iso = datetime.now().isoformat()
    
    if existing_sub:
        try:
            if os.path.exists(existing_sub["file_path"]):
                os.remove(existing_sub["file_path"])
        except Exception as fe:
            print(f"[Upload] Could not remove old file: {fe}")
            
        db.submissions.update_one({"_id": existing_sub["_id"]}, {
            "$set": {
                "file_name": file.filename,
                "file_path": saved_file_path,
                "extracted_text": "",
                "submitted_at": now_iso,
                "marks": None,
                "feedback": None
            }
        })
        submission_id = str(existing_sub["_id"])
        db.plagiarism_reports.delete_many({"submission_id": existing_sub["_id"]})
    else:
        result = db.submissions.insert_one({
            "assignment_id": to_id(assignment_id),
            "student_id": to_id(student_id),
            "file_name": file.filename,
            "file_path": saved_file_path,
            "extracted_text": "",
            "submitted_at": now_iso
        })
        submission_id = str(result.inserted_id)

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


async def _extract_and_store(submission_id: str, file_path: str):
    """
    Background task: extracts text from the uploaded file and updates the
    submissions table, then triggers the plagiarism scan. Runs after the
    HTTP response is already sent.
    """
    try:
        # Run CPU/IO-bound extraction in thread pool (non-blocking)
        loop = asyncio.get_event_loop()
        print(f"[Upload] Starting text extraction for submission {submission_id} from {file_path}")
        
        extracted_text = await loop.run_in_executor(
            _thread_pool, extract_text, file_path
        )

        word_count = len(extracted_text.split()) if extracted_text.strip() else 0
        print(f"[Upload] Extraction result: {word_count} words, {len(extracted_text)} chars for submission {submission_id}")
        
        if not extracted_text.strip():
            print(f"[Upload] WARNING: Zero text extracted from {file_path}. Check OCR logs above.")
            # Still save the empty string — do not abort
        
        db = get_db()
        db.submissions.update_one(
            {"_id": to_id(submission_id)},
            {"$set": {"extracted_text": extracted_text}}
        )
        print(f"[Upload] DB updated for submission {submission_id}.")

        # Always run plagiarism scan even with empty text (will save 0% report)
        print(f"[Upload] Triggering plagiarism scan for submission {submission_id}...")
        await loop.run_in_executor(
            _thread_pool, teacher_scan_submission, submission_id
        )

        _upload_status[submission_id] = "ready"
        print(f"[Upload] Full pipeline complete for submission {submission_id} ({word_count} words scanned).")

    except Exception as exc:
        import traceback
        _upload_status[submission_id] = "error"
        print(f"[Upload] Background processing failed for submission {submission_id}: {exc}")
        print(traceback.format_exc())


@app.get("/api/student/upload_status/{submission_id}")
def get_upload_status(submission_id: str):
    """
    Poll this endpoint after uploading to know when text extraction is done.
    Returns: { status: 'processing' | 'ready' | 'error' | 'unknown' }
    """
    status = _upload_status.get(submission_id, "unknown")
    return {"submission_id": submission_id, "status": status}


@app.post("/api/admin/reextract/{submission_id}")
async def admin_reextract_submission(submission_id: str, background_tasks: BackgroundTasks):
    """
    Force re-extract text and re-scan a specific submission.
    Useful for fixing 0-word extractions without re-uploading.
    """
    db = get_db()
    sub = db.submissions.find_one({"_id": to_id(submission_id)})
    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found")

    file_path = sub.get("file_path", "")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=400, detail=f"Original file not found at: {file_path}")

    _upload_status[submission_id] = "processing"
    background_tasks.add_task(_extract_and_store, submission_id, file_path)
    return {"status": "started", "message": f"Re-extraction started for submission {submission_id}"}


@app.get("/api/admin/debug_submission/{submission_id}")
def admin_debug_submission(submission_id: str):
    """Returns full submission details including extracted text preview — for debugging."""
    db = get_db()
    sub = db.submissions.find_one({"_id": to_id(submission_id)})
    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found")
    report = db.plagiarism_reports.find_one({"submission_id": to_id(submission_id)})
    extracted = sub.get("extracted_text", "")
    return {
        "submission_id": submission_id,
        "file_name": sub.get("file_name"),
        "file_path": sub.get("file_path"),
        "file_exists": os.path.exists(sub.get("file_path", "")),
        "extracted_text_length": len(extracted),
        "extracted_text_words": len(extracted.split()) if extracted.strip() else 0,
        "extracted_text_preview": extracted[:500] if extracted else "[EMPTY]",
        "has_plagiarism_report": report is not None,
        "overall_plagiarism_pct": report.get("overall_plagiarism_pct") if report else None,
    }

# ==================== TEACHER ENDPOINTS ====================

@app.post("/api/teacher/assignments")
def teacher_create_assignment(
    title: str = Form(...),
    description: str = Form(...),
    class_section: str = Form(...),
    due_date: str = Form(...),
    teacher_id: str = Form(...)
):
    if class_section not in ["CY2A", "CY2B", "IY2A", "IY2B"]:
        raise HTTPException(status_code=400, detail="Invalid section selection")
        
    db = get_db()
    db.assignments.insert_one({
        "title": title,
        "description": description,
        "class_section": class_section,
        "due_date": due_date,
        "created_by_teacher_id": to_id(teacher_id)
    })
    return {"status": "success", "message": "Assignment created successfully."}

@app.get("/api/teacher/assignments")
def teacher_get_assignments(teacher_id: str):
    db = get_db()
    cursor = db.assignments.find({"created_by_teacher_id": to_id(teacher_id)}).sort([("_id", -1)])
    assignments = []
    for a in cursor:
        assignments.append({
            "id": str(a["_id"]),
            "title": a["title"],
            "description": a.get("description"),
            "class_section": a.get("class_section"),
            "due_date": a.get("due_date")
        })
    return assignments

@app.get("/api/teacher/submissions")
def teacher_get_submissions(assignment_id: str):
    db = get_db()
    pipeline = [
        {"$match": {"assignment_id": to_id(assignment_id)}},
        {"$lookup": {
            "from": "users",
            "localField": "student_id",
            "foreignField": "_id",
            "as": "student"
        }},
        {"$unwind": {"path": "$student", "preserveNullAndEmptyArrays": True}},
        {"$lookup": {
            "from": "plagiarism_reports",
            "localField": "_id",
            "foreignField": "submission_id",
            "as": "report"
        }},
        {"$unwind": {"path": "$report", "preserveNullAndEmptyArrays": True}},
        {"$sort": {"student.full_name": 1}}
    ]
    
    rows = db.submissions.aggregate(pipeline)
    submissions = []
    
    for row in rows:
        report = row.get("report", {})
        student = row.get("student", {})
        
        detailed_report = report.get("detailed_report_json")
        if isinstance(detailed_report, str):
            detailed_report = json.loads(detailed_report)
            
        submissions.append({
            "submission_id": str(row["_id"]),
            "file_name": row["file_name"],
            "extracted_text": row.get("extracted_text", ""),
            "submitted_at": row["submitted_at"],
            "marks": row.get("marks"),
            "feedback": row.get("feedback"),
            "student_name": student.get("full_name", ""),
            "student_section": student.get("section", ""),
            "peer_similarity_pct": report.get("peer_similarity_pct"),
            "internet_similarity_pct": report.get("internet_similarity_pct"),
            "overall_plagiarism_pct": report.get("overall_plagiarism_pct"),
            "detailed_report": detailed_report
        })
        
    return submissions

@app.post("/api/teacher/scan/{submission_id}")
def teacher_scan_submission(submission_id: str):
    db = get_db()

    # 1. Fetch target submission
    target = db.submissions.find_one({"_id": to_id(submission_id)})
    if not target:
        raise HTTPException(status_code=404, detail="Submission not found")
        
    student = db.users.find_one({"_id": target["student_id"]})

    target_text = target.get("extracted_text", "")
    target_text_norm = re.sub(r"\s+", " ", target_text).strip()

    # Handle empty document
    if not target_text_norm:
        db.plagiarism_reports.delete_many({"submission_id": to_id(submission_id)})
        db.plagiarism_reports.insert_one({
            "submission_id": to_id(submission_id),
            "peer_similarity_pct": 0.0,
            "internet_similarity_pct": 0.0,
            "overall_plagiarism_pct": 0.0,
            "detailed_report_json": {"peer_matches": [], "internet_matches": [], "message": "Document contains no extractable text."}
        })
        return {"status": "success", "message": "Scan completed (Empty document)", "results": {"overall_plagiarism": 0}}

    # 2. Fetch all peer submissions
    peers_cur = db.submissions.find({
        "_id": {"$ne": to_id(submission_id)}, 
        "assignment_id": target["assignment_id"],
        "extracted_text": {"$ne": "", "$exists": True}
    })
    
    peers = []
    for p in peers_cur:
        peer_student = db.users.find_one({"_id": p["student_id"]})
        peers.append({
            "peer_sub_id": str(p["_id"]),
            "extracted_text": p["extracted_text"],
            "peer_name": peer_student["full_name"] if peer_student else "Unknown",
            "peer_section": peer_student.get("section", "") if peer_student else ""
        })

    peer_matches      = []
    highest_peer_score = 0.0

    if peers:
        peer_texts = [re.sub(r"\s+", " ", p["extracted_text"]).strip() for p in peers]

        tfidf_scores = calculate_tfidf_similarity_batch(target_text_norm, peer_texts)
        candidate_indices  = [i for i, s in enumerate(tfidf_scores) if s >= _PREFILTER_THRESHOLD]
        candidate_texts    = [peer_texts[i] for i in candidate_indices]

        bert_scores = [0.0] * len(peers)
        if candidate_texts:
            bert_batch = calculate_bert_similarity_batch(target_text_norm, candidate_texts)
            for idx, bscore in zip(candidate_indices, bert_batch):
                bert_scores[idx] = bscore

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

    # 3. Internet Plagiarism Scan
    highest_internet_score, internet_matches = search_internet_similarity(target_text_norm)

    # 4. Overall score
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

    # 5. Persist report
    db.plagiarism_reports.delete_many({"submission_id": to_id(submission_id)})
    db.plagiarism_reports.insert_one({
        "submission_id": to_id(submission_id),
        "peer_similarity_pct": highest_peer_score,
        "internet_similarity_pct": highest_internet_score,
        "overall_plagiarism_pct": overall_plagiarism,
        "detailed_report_json": detailed_report
    })

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
def teacher_batch_scan_submissions(assignment_id: str):
    db = get_db()
    subs = list(db.submissions.find({"assignment_id": to_id(assignment_id)}))
    scanned_count = 0
    failed_count = 0

    for sub in subs:
        try:
            teacher_scan_submission(str(sub["_id"]))
            scanned_count += 1
            print(f"[BatchScan] Scanned submission {sub['_id']} ({scanned_count}/{len(subs)})")
        except Exception as exc:
            failed_count += 1
            print(f"[BatchScan] Failed to scan submission {sub['_id']}: {exc}")

    msg = f"Batch scan complete: {scanned_count} succeeded, {failed_count} failed."
    return {"status": "success", "message": msg, "scanned": scanned_count, "failed": failed_count}

@app.post("/api/teacher/grade/{submission_id}")
def teacher_grade_submission(submission_id: str, marks: int = Form(...), feedback: str = Form(...)):
    db = get_db()
    if not db.submissions.find_one({"_id": to_id(submission_id)}):
        raise HTTPException(status_code=404, detail="Submission not found")
        
    db.submissions.update_one({"_id": to_id(submission_id)}, {"$set": {"marks": marks, "feedback": feedback}})
    return {"status": "success", "message": "Grade and feedback saved successfully."}

@app.get("/api/teacher/analytics")
def teacher_get_analytics(teacher_id: str):
    db = get_db()
    
    sections = ["CY2A", "CY2B", "IY2A", "IY2B"]
    section_data = []
    
    for sect in sections:
        pipeline = [
            {"$lookup": {"from": "users", "localField": "student_id", "foreignField": "_id", "as": "user"}},
            {"$unwind": "$user"},
            {"$lookup": {"from": "assignments", "localField": "assignment_id", "foreignField": "_id", "as": "assignment"}},
            {"$unwind": "$assignment"},
            {"$match": {"user.section": sect, "assignment.created_by_teacher_id": to_id(teacher_id)}},
            {"$lookup": {"from": "plagiarism_reports", "localField": "_id", "foreignField": "submission_id", "as": "report"}},
            {"$unwind": {"path": "$report", "preserveNullAndEmptyArrays": True}},
            {"$group": {
                "_id": None,
                "avg_marks": {"$avg": "$marks"},
                "avg_plagiarism": {"$avg": "$report.overall_plagiarism_pct"},
                "sub_count": {"$sum": 1}
            }}
        ]
        res = list(db.submissions.aggregate(pipeline))
        if res:
            r = res[0]
            section_data.append({
                "section": sect,
                "avg_marks": round(r.get("avg_marks") or 0, 1),
                "avg_plagiarism": round(r.get("avg_plagiarism") or 0, 1),
                "submission_count": r.get("sub_count") or 0
            })
        else:
            section_data.append({"section": sect, "avg_marks": 0.0, "avg_plagiarism": 0.0, "submission_count": 0})
            
    pipeline_risk = [
        {"$lookup": {"from": "assignments", "localField": "assignment_id", "foreignField": "_id", "as": "assignment"}},
        {"$unwind": "$assignment"},
        {"$match": {"assignment.created_by_teacher_id": to_id(teacher_id)}},
        {"$lookup": {"from": "plagiarism_reports", "localField": "_id", "foreignField": "submission_id", "as": "report"}},
        {"$unwind": "$report"},
        {"$lookup": {"from": "users", "localField": "student_id", "foreignField": "_id", "as": "user"}},
        {"$unwind": {"path": "$user", "preserveNullAndEmptyArrays": True}}
    ]
    all_reports = list(db.submissions.aggregate(pipeline_risk))
    
    low, med, high = 0, 0, 0
    correlations = []
    
    for r in all_reports:
        pct = r["report"].get("overall_plagiarism_pct", 0)
        if pct < 20: low += 1
        elif pct <= 50: med += 1
        else: high += 1
        
        if r.get("marks") is not None:
            correlations.append({
                "student_name": r.get("user", {}).get("full_name", "Unknown"),
                "marks": r["marks"],
                "plagiarism": pct
            })
            
    # Sort correlations by marks to make charts clean
    correlations.sort(key=lambda x: x["marks"])
        
    risk_distribution = {"low": low, "medium": med, "high": high, "total": len(all_reports)}
    
    return {
        "section_data": section_data,
        "risk_distribution": risk_distribution,
        "correlations": correlations
    }

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    from fastapi.responses import Response
    return Response(status_code=204)

# ==================== STATIC FILES MOUNT ====================

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    # Start the server on host 127.0.0.1 and port 8000
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
