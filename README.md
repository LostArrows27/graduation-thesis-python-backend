## Python image labeling server

### 0. Introduction 📜
- FastAPI python server to labeling images using OpenAI's Open-clip model.
- Trigger whenever new record in ```image``` table in Supabase.

### 1. Tech stack 🚀
- Python
- Supabase (PostgreSQL string connection)
- FastAPI
- open-clip (OpenAI)

### 2. Install 🛠️
```bash
pip install pydantic pydantic_settings fastapi dotenv uvicorn open_clip_torch redis supabase psycopg2
```

### 3. Run 🚀
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8080 --reload
```
