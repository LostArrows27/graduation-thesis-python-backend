from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    db_name: str = os.getenv("DB_NAME")
    db_user: str = os.getenv("DB_USER")
    db_password: str = os.getenv("DB_PASSWORD")
    db_host: str = os.getenv("DB_HOST")
    db_port: int = os.getenv("DB_PORT")
    
    supabase_url: str = os.getenv("SUPABASE_URL")
    supabase_key: str = os.getenv("SUPABASE_KEY")

    class Config:
        env_file = ".env"  
        
settings = Settings()