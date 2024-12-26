from fastapi import FastAPI, Request, Depends
from contextlib import asynccontextmanager
from app.tasks.db_listener import start_listener, stop_listener
from app.services.ai_services import AIService, get_ai_service
from app.services.supabase_service import SupabaseService

app = FastAPI()


# dependency injection -> AI labeling service + Supabase service into -> bg_task
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.supabase_service = SupabaseService()
    app.state.ai_service = AIService(app.state.supabase_service)
    start_listener(app.state.ai_service)
    yield
    stop_listener()

app = FastAPI(lifespan=lifespan)


def get_ai_service(request: Request) -> AIService:
    return request.app.state.ai_service


def get_supabase_service(request: Request) -> SupabaseService:
    return request.app.state.supabase_service


@app.post("/upload")
def upload_image(filename: str, service: AIService = Depends(get_ai_service)):
    return service.classify_image(filename)
