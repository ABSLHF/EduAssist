from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

CURRENT_FILE = Path(__file__).resolve()
BACKEND_ROOT = CURRENT_FILE.parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
# Ensure pydantic settings reads backend/.env regardless of current shell path.
os.chdir(BACKEND_ROOT)

from app.config import settings
from app.services.keywords import extract_keywords_with_meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare KG keyword extraction results across hybrid/hanlp/deepke.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="",
        help="Input text. If omitted, --file will be used.",
    )
    parser.add_argument(
        "--file",
        type=str,
        default="",
        help="Path to a UTF-8 text file as input.",
    )
    parser.add_argument("--top-k", type=int, default=12, help="Top K candidates.")
    parser.add_argument(
        "--hanlp-model",
        type=str,
        default="",
        help="HanLP model id/path for KG_HANLP_MODEL.",
    )
    parser.add_argument(
        "--deepke-endpoint",
        type=str,
        default="",
        help="DeepKE HTTP endpoint for KG_DEEPKE_ENDPOINT.",
    )
    parser.add_argument(
        "--deepke-api-key",
        type=str,
        default="",
        help="Optional DeepKE API key.",
    )
    parser.add_argument(
        "--deepke-timeout",
        type=int,
        default=20,
        help="DeepKE timeout seconds.",
    )
    return parser.parse_args()


def load_text(args: argparse.Namespace) -> str:
    if args.text.strip():
        return args.text.strip()
    if args.file.strip():
        path = Path(args.file.strip())
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return path.read_text(encoding="utf-8").strip()
    raise ValueError("Please provide --text or --file.")


def run_once(extractor: str, text: str, top_k: int):
    settings.kg_extractor = extractor
    result = extract_keywords_with_meta(text=text, top_k=top_k)
    print(f"\n=== extractor={extractor} ===")
    print(f"actual_extractor: {result.extractor}")
    print(f"fallback_used:   {result.fallback_used}")
    print(f"filtered_noise:  {result.filtered_noise}")
    print(f"candidates({len(result.candidates)}): {', '.join(result.candidates)}")


def main():
    args = parse_args()
    text = load_text(args)
    if not text:
        raise ValueError("Input text is empty.")

    if args.hanlp_model.strip():
        settings.kg_hanlp_model = args.hanlp_model.strip()
    if args.deepke_endpoint.strip():
        settings.kg_deepke_endpoint = args.deepke_endpoint.strip()
    if args.deepke_api_key.strip():
        settings.kg_deepke_api_key = args.deepke_api_key.strip()
    settings.kg_deepke_timeout_seconds = max(3, args.deepke_timeout)

    print("Input preview:")
    print(text[:200] + ("..." if len(text) > 200 else ""))
    print(f"top_k={max(1, args.top_k)}")

    for extractor in ("hybrid", "hanlp", "deepke"):
        run_once(extractor, text, max(1, args.top_k))


if __name__ == "__main__":
    main()
