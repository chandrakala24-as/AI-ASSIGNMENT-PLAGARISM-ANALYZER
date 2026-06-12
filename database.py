import os
import hashlib
import json
from pymongo import MongoClient

# Use local MongoDB for development if MONGO_URI is not set
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

# We use a global client so we don't reconnect on every request
client = MongoClient(MONGO_URI)
db = client["plagcheck_db"]

def get_db():
    """Returns the MongoDB database instance."""
    return db

def hash_password(password: str) -> str:
    # A simple but secure hash using SHA-256 with a constant salt
    salt = "ai_plagiarism_salt_2026"
    return hashlib.sha256((password + salt).encode('utf-8')).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    return hash_password(password) == hashed

def init_db():
    """Seeds the database with default users and mock internet sources if empty."""
    # Seed default users
    if db.users.count_documents({}) == 0:
        default_users = [
            {"_id": "u_admin", "username": "admin", "password_hash": hash_password("admin123"), "role": "admin", "section": None, "full_name": "System Admin"},
            {"_id": "u_teacher1", "username": "teacher1", "password_hash": hash_password("teacher123"), "role": "teacher", "section": None, "full_name": "Dr. Sarah Connor"},
            {"_id": "u_teacher2", "username": "teacher2", "password_hash": hash_password("teacher456"), "role": "teacher", "section": None, "full_name": "Prof. Charles Xavier"},
            {"_id": "u_student1", "username": "student1", "password_hash": hash_password("student123"), "role": "student", "section": "CY2A", "full_name": "Alice Smith"},
            {"_id": "u_student2", "username": "student2", "password_hash": hash_password("student234"), "role": "student", "section": "CY2A", "full_name": "Bob Johnson"},
            {"_id": "u_student3", "username": "student3", "password_hash": hash_password("student345"), "role": "student", "section": "CY2B", "full_name": "Charlie Brown"},
            {"_id": "u_student4", "username": "student4", "password_hash": hash_password("student456"), "role": "student", "section": "IY2A", "full_name": "Diana Prince"},
            {"_id": "u_student5", "username": "student5", "password_hash": hash_password("student567"), "role": "student", "section": "IY2B", "full_name": "Evan Wright"}
        ]
        db.users.insert_many(default_users)
        print("Default users seeded.")
        
    # Seed mock internet articles
    if db.mock_internet_sources.count_documents({}) == 0:
        default_articles = [
            {
                "title": "Introduction to Python Programming",
                "url": "https://wikipedia.org/wiki/Python_(programming_language)",
                "content": "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured, object-oriented, and functional programming. Python was created by Guido van Rossum in the late 1980s."
            },
            {
                "title": "FastAPI Web Framework Guide",
                "url": "https://fastapi.tiangolo.com/features/",
                "content": "FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.8+ based on standard Python type hints. Key features include: Ultra high performance, on par with NodeJS and Go. Fast to code: Increase the speed to develop features by about 200% to 300%. Fewer bugs: Reduce about 40% of human induced errors. Intuitive: Great editor support. Easy: Designed to be easy to use and learn. Robust: Get production-ready code. With automatic interactive documentation."
            },
            {
                "title": "Machine Learning and Cosine Similarity",
                "url": "https://scikit-learn.org/stable/modules/metrics.html",
                "content": "Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space. It is defined to equal the cosine of the angle between them, which is also the inner product of the vectors normalized to have length 1. In information retrieval and text mining, cosine similarity is widely used to measure document similarity. It computes the dot product of two text vectors divided by the product of their magnitudes."
            },
            {
                "title": "Winnowing Algorithm for Document Fingerprinting",
                "url": "https://theory.stanford.edu/~aiken/publications/papers/sigmod03.pdf",
                "content": "The Winnowing algorithm is a document fingerprinting technique used for local alignment and copy detection. It works by hashing overlapping substrings of length k (k-grams) and selecting a subset of these hashes (fingerprints) within a sliding window of size w. By storing only a fraction of the hashes, it reduces database storage while guaranteeing that any shared substring of length at least w + k - 1 will be detected. Winnowing is robust to minor modifications like whitespace insertion or text reordering."
            },
            {
                "title": "Understanding BERT Transformers in NLP",
                "url": "https://huggingface.co/docs/transformers/model_doc/bert",
                "content": "Bidirectional Encoder Representations from Transformers (BERT) is a transformer-based machine learning technique for natural language processing pre-training developed by Google. BERT was created and published in 2018 by Jacob Devlin and his colleagues. BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of NLP tasks."
            }
        ]
        db.mock_internet_sources.insert_many(default_articles)
        print("Mock internet articles seeded.")

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")
