#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import contextlib
import io
import logging
import os
import sys
import warnings
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"

warnings.filterwarnings("ignore")

try:
    from transformers.utils import logging as hf_transformers_logging
    hf_transformers_logging.set_verbosity_error()
except Exception:
    pass

try:
    from huggingface_hub.utils import logging as hf_hub_logging
    hf_hub_logging.set_verbosity_error()
except Exception:
    pass

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer
from tools.real_embedding import (
    DEFAULT_MODEL_NAME,
    DEFAULT_CHUNKS_PATH,
    DEFAULT_EMBEDDINGS_PATH,
    load_chunks,
    load_embeddings,
    build_query_frame,
    maybe_use_llm_parser,
    semantic_search,
    execute_query,
)

QUERIES = [
    "In welchem Package liegt der Akku",
    "In welchem Diagramm erscheint der E-Motor",
    "Zu welchem System gehört die Energieversorgung",
    "Welche Elemente enthält das bdd_Blocks Package",
    "Was ist der Strom des Akku",
    "Was ist die Spannung des Akku",
    "Wie hoch ist die Beschleunigung",
    "Wie hoch ist die Beschleunigungszeit",
    "Wie hoch ist die Gesamtmasse_Fahrrad",
]


def load_model_quietly(model_name: str):
    class _SilentStderr(io.StringIO):
        def write(self, s):
            blocked_keywords = [
                "Warning: You are sending unauthenticated requests to the HF Hub",
                "BertModel LOAD REPORT",
                "embeddings.position_ids",
                "UNEXPECTED",
                "Loading weights:",
                "Materializing param=",
                "can be ignored when loading from different task/architecture",
            ]
            if any(k in s for k in blocked_keywords):
                return len(s)
            return super().write(s)

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(_SilentStderr()):
        model = SentenceTransformer(model_name)
    return model


def main():
    model = load_model_quietly(DEFAULT_MODEL_NAME)
    chunks = load_chunks(DEFAULT_CHUNKS_PATH)
    embeddings = load_embeddings(DEFAULT_EMBEDDINGS_PATH)

    for q in QUERIES:
        frame = build_query_frame(q, chunks)
        frame = maybe_use_llm_parser(frame, use_llm_parser=False)
        semantic_hits = semantic_search(
            model=model,
            frame=frame,
            chunks=chunks,
            embeddings=embeddings,
            top_k=10,
        )
        bundle = execute_query(frame, chunks, semantic_hits)

        print("=" * 80)
        print("QUERY:")
        print(q)
        print("-" * 80)
        print("ANSWER:")
        print(bundle.final_answer)
        print()


if __name__ == "__main__":
    main()