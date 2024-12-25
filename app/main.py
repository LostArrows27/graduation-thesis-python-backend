from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.tasks.db_listener import start_listener, stop_listener

@asynccontextmanager
async def lifespan(app: FastAPI):
    start_listener()
    yield
    stop_listener()

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def read_root():
    return {"message": "FastAPI server is running with background tasks"}
