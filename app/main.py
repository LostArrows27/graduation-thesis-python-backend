from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.services.redis_service import RedisService
from app.tasks.check_db_on_startup import start_background_processor
from app.tasks.db_listener import start_listener, stop_listener
from app.services.ai_services import AIService, get_ai_service
from app.services.supabase_service import SupabaseService
import threading

from app.tasks.redis_processor import start_stream_processors, stop_stream_processors

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# dependency injection -> AI labeling service + Supabase service into -> bg_tasks
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.supabase_service = SupabaseService()
    app.state.ai_service = AIService(app.state.supabase_service)
    app.state.redis_service = RedisService()

    # init consumer group
    app.state.redis_service.create_consumer_group(
        'image_label_stream', 'image_label_group')

    # start db not processed image processor
    start_background_processor(
        app.state.ai_service, app.state.supabase_service)
    # start db change listener
    start_listener(app.state.redis_service)
    # start redis stream processors
    start_stream_processors(app.state.ai_service, app.state.redis_service)

    yield
    stop_listener()
    stop_stream_processors()

app = FastAPI(lifespan=lifespan)


def get_ai_service(request: Request) -> AIService:
    return request.app.state.ai_service


def get_supabase_service(request: Request) -> SupabaseService:
    return request.app.state.supabase_service


@app.post("/upload")
def upload_image(filename: str, service: AIService = Depends(get_ai_service)):
    return service.classify_image(filename)
