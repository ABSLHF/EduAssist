from sqlalchemy import inspect, text

from app.db import Base, engine
import app.models.models  # noqa: F401


def _ensure_column(table: str, column: str, ddl: str) -> None:
    inspector = inspect(engine)
    cols = {c["name"] for c in inspector.get_columns(table)}
    if column in cols:
        return
    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))


def _ensure_compat_columns() -> None:
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    if "materials" in tables:
        _ensure_column("materials", "parse_status", "VARCHAR(20) NOT NULL DEFAULT 'pending'")
        _ensure_column("materials", "parse_error", "TEXT")
        _ensure_column("materials", "extracted_chars", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column("materials", "parsed_at", "DATETIME")


def main() -> None:
    Base.metadata.create_all(bind=engine)
    _ensure_compat_columns()
    print("Tables created/updated")


if __name__ == "__main__":
    main()
