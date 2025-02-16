from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.services.redis_service import RedisService
from app.tasks.check_db_on_startup import start_background_processor
from app.tasks.db_listener import start_listener, stop_listener
from app.services.ai_services import AIService, get_ai_service
from app.services.supabase_service import SupabaseService
from app.tasks.redis_processor import start_stream_processors, stop_stream_processors
from dotenv import load_dotenv
from pydantic import BaseModel


def reload_env():
    """Reload environment variables"""
    load_dotenv(override=True)


reload_env()

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


class ImageRequest(BaseModel):
    image_bucket_id: str
    image_name: str


@app.post("/classify-image")
def classify_image(request: ImageRequest, service: AIService = Depends(get_ai_service)):
    try:
        results, image_features = service.classify_image(
            request.image_bucket_id, request.image_name, image_id=None)

        supabase_service: SupabaseService = service.inference_service.supabase_service
        image_row = supabase_service.save_image_features_and_labels(
            request.image_bucket_id, request.image_name, results, image_features.squeeze(0).tolist())

        # remove image_features from response
        image_row.pop('image_features')

        return {"status": "success", "data": image_row}
    except Exception as e:
        return {"status": "error", "message": str(e)}
