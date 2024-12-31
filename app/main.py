from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.services.redis_service import RedisService
from app.tasks.db_listener import start_listener, stop_listener
from app.services.ai_services import AIService, get_ai_service
from app.services.supabase_service import SupabaseService
import threading

from app.tasks.redis_processor import process_label_job, process_pending_label_job

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# dependency injection -> AI labeling service + Supabase service into -> bg_task
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.supabase_service = SupabaseService()
    app.state.ai_service = AIService(app.state.supabase_service)
    app.state.redis_service = RedisService()

    # init consumer group
    app.state.redis_service.create_consumer_group(
        'image_label_stream', 'image_label_group')

    start_listener(app.state.redis_service)

    # 0 -> old stream
    old_stream_redis_thread = threading.Thread(target=process_pending_label_job, args=(
        app.state.ai_service, app.state.redis_service))

    # > -> new stream
    new_stream_redis_thread = threading.Thread(target=process_label_job, args=(
        app.state.ai_service, app.state.redis_service))

    old_stream_redis_thread.start()
    new_stream_redis_thread.start()

    yield
    stop_listener()
    old_stream_redis_thread.join()
    new_stream_redis_thread.join()

app = FastAPI(lifespan=lifespan)


def get_ai_service(request: Request) -> AIService:
    return request.app.state.ai_service


def get_supabase_service(request: Request) -> SupabaseService:
    return request.app.state.supabase_service


@app.post("/upload")
def upload_image(filename: str, service: AIService = Depends(get_ai_service)):
    return service.classify_image(filename)
