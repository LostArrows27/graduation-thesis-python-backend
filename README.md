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
python -m venv env
pip install -r requirements.txt
```

### 3. Run 🚀
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8080 --reload
```
