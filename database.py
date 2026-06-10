import sqlite3
import os
import hashlib
import json

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plagiarism_analyser.db")

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(password: str) -> str:
    # A simple but secure hash using SHA-256 with a constant salt
    salt = "ai_plagiarism_salt_2026"
    return hashlib.sha256((password + salt).encode('utf-8')).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    return hash_password(password) == hashed

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Users Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT CHECK(role IN ('admin', 'teacher', 'student')) NOT NULL,
        section TEXT, -- e.g., CY2A, CY2B, IY2A, IY2B (null for admin/teacher)
        full_name TEXT NOT NULL
    )
    """)
    
    # 2. Assignments Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS assignments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT,
        class_section TEXT NOT NULL, -- CY2A, CY2B, IY2A, IY2B
        due_date TEXT,
        created_by_teacher_id INTEGER,
        FOREIGN KEY (created_by_teacher_id) REFERENCES users(id)
    )
    """)
    
    # 3. Submissions Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS submissions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        assignment_id INTEGER NOT NULL,
        student_id INTEGER NOT NULL,
        file_name TEXT NOT NULL,
        file_path TEXT NOT NULL,
        extracted_text TEXT NOT NULL,
        submitted_at TEXT NOT NULL,
        marks INTEGER, -- Null initially, set by teacher
        feedback TEXT,  -- Null initially
        FOREIGN KEY (assignment_id) REFERENCES assignments(id),
        FOREIGN KEY (student_id) REFERENCES users(id)
    )
    """)
    
    # 4. Plagiarism Reports Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS plagiarism_reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        submission_id INTEGER UNIQUE NOT NULL,
        peer_similarity_pct REAL DEFAULT 0.0,
        internet_similarity_pct REAL DEFAULT 0.0,
        overall_plagiarism_pct REAL DEFAULT 0.0,
        detailed_report_json TEXT, -- JSON structure of matches
        FOREIGN KEY (submission_id) REFERENCES submissions(id) ON DELETE CASCADE
    )
    """)
    
    # 5. Mock Internet Sources Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS mock_internet_sources (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        url TEXT UNIQUE NOT NULL,
        content TEXT NOT NULL
    )
    """)
    
    conn.commit()
    
    # Seed default users if empty
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        default_users = [
            ("admin", hash_password("admin123"), "admin", None, "System Admin"),
            ("teacher1", hash_password("teacher123"), "teacher", None, "Dr. Sarah Connor"),
            ("teacher2", hash_password("teacher456"), "teacher", None, "Prof. Charles Xavier"),
            ("student1", hash_password("student123"), "student", "CY2A", "Alice Smith"),
            ("student2", hash_password("student234"), "student", "CY2A", "Bob Johnson"),
            ("student3", hash_password("student345"), "student", "CY2B", "Charlie Brown"),
            ("student4", hash_password("student456"), "student", "IY2A", "Diana Prince"),
            ("student5", hash_password("student567"), "student", "IY2B", "Evan Wright")
        ]
        cursor.executemany(
            "INSERT INTO users (username, password_hash, role, section, full_name) VALUES (?, ?, ?, ?, ?)",
            default_users
        )
        conn.commit()
        print("Default users seeded.")
        
    # Seed mock internet articles if empty
    cursor.execute("SELECT COUNT(*) FROM mock_internet_sources")
    if cursor.fetchone()[0] == 0:
        default_articles = [
            (
                "Introduction to Python Programming",
                "https://wikipedia.org/wiki/Python_(programming_language)",
                "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured, object-oriented, and functional programming. Python was created by Guido van Rossum in the late 1980s."
            ),
            (
                "FastAPI Web Framework Guide",
                "https://fastapi.tiangolo.com/features/",
                "FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.8+ based on standard Python type hints. Key features include: Ultra high performance, on par with NodeJS and Go. Fast to code: Increase the speed to develop features by about 200% to 300%. Fewer bugs: Reduce about 40% of human induced errors. Intuitive: Great editor support. Easy: Designed to be easy to use and learn. Robust: Get production-ready code. With automatic interactive documentation."
            ),
            (
                "Machine Learning and Cosine Similarity",
                "https://scikit-learn.org/stable/modules/metrics.html",
                "Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space. It is defined to equal the cosine of the angle between them, which is also the inner product of the vectors normalized to have length 1. In information retrieval and text mining, cosine similarity is widely used to measure document similarity. It computes the dot product of two text vectors divided by the product of their magnitudes."
            ),
            (
                "Winnowing Algorithm for Document Fingerprinting",
                "https://theory.stanford.edu/~aiken/publications/papers/sigmod03.pdf",
                "The Winnowing algorithm is a document fingerprinting technique used for local alignment and copy detection. It works by hashing overlapping substrings of length k (k-grams) and selecting a subset of these hashes (fingerprints) within a sliding window of size w. By storing only a fraction of the hashes, it reduces database storage while guaranteeing that any shared substring of length at least w + k - 1 will be detected. Winnowing is robust to minor modifications like whitespace insertion or text reordering."
            ),
            (
                "Understanding BERT Transformers in NLP",
                "https://huggingface.co/docs/transformers/model_doc/bert",
                "Bidirectional Encoder Representations from Transformers (BERT) is a transformer-based machine learning technique for natural language processing pre-training developed by Google. BERT was created and published in 2018 by Jacob Devlin and his colleagues. BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of NLP tasks."
            )
        ]
        cursor.executemany(
            "INSERT INTO mock_internet_sources (title, url, content) VALUES (?, ?, ?)",
            default_articles
        )
        conn.commit()
        print("Mock internet articles seeded.")
        
    conn.close()

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")
