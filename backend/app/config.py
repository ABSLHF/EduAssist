from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    db_url: str
    chroma_path: str = "./chroma"
    material_storage_mode: str = "local"
    oss_endpoint: str | None = None
    oss_bucket: str | None = None
    oss_access_key_id: str | None = None
    oss_access_key_secret: str | None = None
    oss_secure: bool = True
    glm_api_key: str | None = None
    ernie_api_key: str | None = None
    dify_api_key: str | None = None
    dify_base_url: str = "https://dify.aipfuture.com/v1"
    model_provider: str = "glm"
    llm_timeout_seconds: int = 70
    doc_chunk_size: int = 300
    doc_chunk_overlap: int = 80
    embedding_model_name: str = "BAAI/bge-small-zh-v1.5"
    embedding_fallback_model: str = "all-MiniLM-L6-v2"
    enable_small_qa_assist: bool = False
    enable_finetuned_qa_in_rag: bool = True
    finetuned_qa_model_path: str | None = None
    finetuned_qa_min_conf: float = 0.45
    finetuned_qa_top_chunks: int = 4
    finetuned_qa_max_evidence: int = 2
    qa_domain_route_enabled: bool = True
    qa_domain_mismatch_guard: bool = True
    enable_assignment_relevance_model: bool = False
    assignment_relevance_model_path: str | None = None
    assignment_relevance_threshold_hi: float = 0.70
    assignment_relevance_threshold_lo: float = 0.25
    assignment_relevance_use_reranker: bool = False
    assignment_relevance_reranker_model: str = "BAAI/bge-reranker-v2-m3"
    assignment_relevance_reranker_weight: float = 0.25
    assignment_feedback_mode: str = "legacy"
    assignment_feedback_shadow_log_path: str = "logs/assignment_feedback_shadow.jsonl"
    enable_assignment_feedback_model: bool = False
    assignment_feedback_model_path: str | None = None
    kg_extractor: str = "hybrid"
    kg_top_k: int = 12
    kg_min_term_len: int = 2
    kg_enable_noise_filter: bool = True
    kg_stopwords_path: str = "training/data/kg_stopwords.txt"
    kg_domain_lexicon_path: str = "training/data/kg_domain_lexicon.txt"
    kg_uie_model_path: str | None = None
    kg_hanlp_model: str | None = None
    kg_deepke_endpoint: str | None = None
    kg_deepke_timeout_seconds: int = 20
    kg_deepke_api_key: str | None = None
    kg_min_cooccur_count: int = 2
    kg_edge_min_score: float = 1.0
    kg_weak_edge_top_n: int = 3
    kg_weak_edge_min_score: float = 0.2
    kg_min_edges_per_material: int = 2
    kg_min_degree_per_term: int = 1
    kg_enable_order_fallback_pairs: bool = True
    kg_max_term_distance: int = 48
    kg_cross_sentence_penalty: float = 0.2
    cors_allow_origins: str = "http://127.0.0.1:5999,http://localhost:5999,http://127.0.0.1:5173,http://localhost:5173"
    cors_allow_origin_regex: str = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"
    jwt_secret: str = "change_me"
    jwt_algorithm: str = "HS256"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "protected_namespaces": ("settings_",),
    }

settings = Settings()
