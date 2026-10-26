# qms-gap-rag

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Declan444/qms-gap-rag/blob/main/qms_gap_rag_colab.ipynb)

Reverse RAG for a quality management system: each regulatory requirement becomes the "question", the system retrieves the closest procedure sections, and a small local model judges COVERED, WEAK or MISSING, with citations. A person reviews every verdict.

Stage 2 of the project. Test data: the fictional Slieve Bloom Medical Ltd QMS, checked against MDR Article 10(9).

## Run it in your browser

Click the Open in Colab badge, set Runtime › Change runtime type › T4 GPU, and run the cells in order. No installation needed.

## Run it on your own PC (setup once)

1. Python 3.10+, then:
   ```
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Install Ollama from https://ollama.com, then:
   ```
   ollama pull qwen2.5:7b-instruct
   ```
   Needs roughly 5 GB of disk and 8 GB+ of RAM. On a smaller machine, set `LLM_MODEL = "qwen2.5:3b-instruct"` in config.py and compare.

## Run

```
python build_index.py     # chunk and embed the SOPs (rerun only when SOPs or chunk settings change)
python gap_report.py      # one verdict per checklist item -> output/gap_report_<time>.md
python score.py           # compare against the answer key
```

## Folders

- `data/procedures/` - the documents being checked (indexed)
- `data/checklist/` - the requirements (queries)
- `data/answer_key/` - the test set. Never indexed.
- `output/` - index and reports

## Experiments worth one run each (change one setting, rebuild if needed, rescore)

- `MAX_CHUNK_WORDS` 120 -> 60 (rebuild index). The lecture's flight question flipped on this.
- `TOP_K` 4 -> 2 or 8. More context is not always better ("lost in the middle").
- `MIN_SIMILARITY` 0.30 -> 0.40. Does UDI (h) get caught as MISSING without the model being asked?
- `LLM_MODEL` 3b vs 7b. Same test set, different brains.

Keep a log of each run's score. That table is the blog post.

## Known limits

- The model only sees retrieved extracts, not the law itself. It will struggle to know that 30 days is wrong (MDR Art. 87) or that Class IIa needs a PSUR. Stage 3 adds the regulation text as a second source.
- The SOP-to-SOP contradiction (G02) is outside the checklist; it needs a different check.
