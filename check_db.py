from database import get_db

db = get_db()
subs = list(db.submissions.find())
print(f"Found {len(subs)} submissions.")
for s in subs[:10]:
    file_name = s.get("file_name", "Unknown")
    extracted = s.get("extracted_text", "")
    print(f"File: {file_name}, Extracted text length: {len(extracted)} chars")
