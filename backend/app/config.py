from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    db_url: str
    chroma_path: str = "./chroma"
    glm_api_key: str | None = None
    ernie_api_key: str | None = None
    model_provider: str = "glm"
    doc_chunk_size: int = 300
    doc_chunk_overlap: int = 80
    embedding_model_name: str = "BAAI/bge-small-zh-v1.5"
    embedding_fallback_model: str = "all-MiniLM-L6-v2"
    enable_small_qa_assist: bool = False
    jwt_secret: str = "change_me"
    jwt_algorithm: str = "HS256"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "protected_namespaces": ("settings_",),
    }

settings = Settings()
