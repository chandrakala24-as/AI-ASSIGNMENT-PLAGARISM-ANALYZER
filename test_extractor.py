from utils.text_extractor import extract_text
import os

uploads_dir = "uploads"
files = os.listdir(uploads_dir)
for f in files:
    path = os.path.join(uploads_dir, f)
    text = extract_text(path)
    print(f"File: {f}, Extracted: {len(text)} chars")
