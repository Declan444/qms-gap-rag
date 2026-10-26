"""Settings for qms-gap-rag. Change one thing at a time, then rerun score.py."""
from pathlib import Path

ROOT = Path(__file__).parent
PROCEDURES_DIR = ROOT / "data" / "procedures"
CHECKLIST_FILE = ROOT / "data" / "checklist" / "mdr_article_10_9_checklist.md"
EXPECTED_FILE = ROOT / "data" / "answer_key" / "expected.json"
INDEX_DIR = ROOT / "output" / "index"
REPORT_DIR = ROOT / "output"

# Embedding model (the lecture's search model)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Local LLM via Ollama (https://ollama.com). Pull it first: ollama pull qwen2.5:7b-instruct
OLLAMA_URL = "http://localhost:11434/api/chat"
LLM_MODEL = "qwen2.5:7b-instruct"

# Retrieval settings
TOP_K = 4                 # chunks given to the model per checklist item
MAX_CHUNK_WORDS = 120     # long sections are split again (the lecture used 120)
MIN_SIMILARITY = 0.30     # below this, no chunk counts as relevant -> MISSING without asking the LLM
