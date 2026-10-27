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

# ---- Improvement experiments: switch on ONE at a time, then rescore ----
# Experiment 1 - retrieval. "chunks": the model reads the TOP_K closest sections (baseline).
#                "full_sop": find the most relevant procedures, then the model reads them in full.
CONTEXT_MODE = "chunks"
FULL_SOP_COUNT = 2        # how many whole procedures the model reads in full_sop mode

# Experiment 2 - prompt. "basic" (baseline) or "strict": check each element of the requirement,
#                and a mention, title or purpose statement does not count as coverage.
PROMPT_STYLE = "basic"

# Experiment 3 - give the model the rules themselves (data/regulation/), not just the company's procedures.
USE_REGULATION = False
REGULATION_FILE = ROOT / "data" / "regulation" / "mdr_reference_notes.md"
REGULATION_TOP_K = 3
