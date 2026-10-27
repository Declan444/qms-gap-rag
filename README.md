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

## Improvement experiments

Baseline (all off): 6 of 13, every error too generous. Switch on one at a time in `config.py` (or step 5 of the notebook), rerun `gap_report.py` and `score.py`. Each run is appended to `output/run_log.csv`.

- `CONTEXT_MODE = "full_sop"`: find the most relevant procedures, then the model reads them whole. Fixes the search handing over Purpose/Scope sections instead of the section with the problem.
- `PROMPT_STYLE = "strict"`: the model checks each element of the requirement; a title or passing mention is not coverage.
- `USE_REGULATION = True`: the model also reads the relevant MDR rules (`data/regulation/`), so it can spot wrong deadlines and outdated standards.

Scoring gives two numbers: verdict (is the verdict acceptable?) and reason (does the reason name the planted problem?). The reason score is the honest one: a model that says WEAK to everything scores well on verdicts alone. `python score.py --all` rescores every saved report.

Then combine them. Honest caveat: the fixes were designed after seeing the baseline failures on this test set. A fair final check is a second mock QMS with different planted gaps.

## Known limits

- With `USE_REGULATION` off, the model only sees the company's procedures, not the law, so it cannot know that 30 days is wrong (MDR Art. 87) or that Class IIa needs a PSUR.
- `data/regulation/` is paraphrased for testing; check it against EUR-Lex.
- The SOP-to-SOP contradiction (G02) is outside the checklist; it needs a different check.
