from fastapi import FastAPI, UploadFile
from contextlib import asynccontextmanager
from app.tasks.db_listener import start_listener, stop_listener
from app.models.model import AIModel
from app.models.inference import AIInferenceService

ai_model = None
ai_inference_service = None


def initialize_ai_service():
    global ai_model, ai_inference_service
    print("Initializing AI Model...")
    ai_model = AIModel()
    ai_inference_service = AIInferenceService(ai_model)
    print("AI Model Initialized")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # AI Model
    initialize_ai_service()
    # pg_notify
    start_listener()

    yield

    # stop pg_notify
    stop_listener()

app = FastAPI(lifespan=lifespan)


@app.get("/")
async def read_root():
    return {"message": "FastAPI server is running with background tasks and AI service"}
