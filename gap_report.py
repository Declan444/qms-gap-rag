"""Steps 2 and 3 in reverse: each checklist requirement is the 'question'.
Retrieve the closest SOP chunks, then ask a local model: COVERED, WEAK or MISSING?"""
import json
from datetime import datetime

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

import config
from chunking import parse_checklist

PROMPT = """You are a medical device quality auditor checking a company's procedures against one EU MDR requirement.

REQUIREMENT ({id}) {title}: {text}

Use only the numbered procedure extracts below. Do not assume anything they do not say.

{sources}

Decide one verdict:
- COVERED: the extracts clearly address the requirement.
- WEAK: the extracts address it partly, vaguely, with an outdated reference, or with a rule you believe is wrong.
- MISSING: the extracts do not address it.

Answer in JSON only, with these keys:
{{"verdict": "COVERED|WEAK|MISSING", "reason": "one or two sentences", "citations": [list of source numbers you relied on], "suggested_fix": "one sentence, or empty if COVERED"}}"""


def ask_llm(prompt: str) -> dict:
    r = requests.post(config.OLLAMA_URL, json={
        "model": config.LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0},
    }, timeout=300)
    r.raise_for_status()
    try:
        return json.loads(r.json()["message"]["content"])
    except (json.JSONDecodeError, KeyError):
        return {"verdict": "ERROR", "reason": "Model did not return valid JSON.", "citations": [], "suggested_fix": ""}


def main():
    chunks = json.loads((config.INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    vectors = np.load(config.INDEX_DIR / "vectors.npy")
    model = SentenceTransformer(config.EMBED_MODEL)
    items = parse_checklist(config.CHECKLIST_FILE)

    results = []
    for item in items:
        q = model.encode([f"{item['title']}: {item['text']}"], normalize_embeddings=True)[0]
        scores = vectors @ q                       # cosine similarity (vectors are normalised)
        top = np.argsort(-scores)[: config.TOP_K]
        retrieved = [{**chunks[i], "score": float(scores[i])} for i in top]

        if retrieved[0]["score"] < config.MIN_SIMILARITY:
            answer = {"verdict": "MISSING",
                      "reason": f"No procedure text above the similarity threshold ({config.MIN_SIMILARITY}).",
                      "citations": [], "suggested_fix": "Write a procedure for this requirement."}
        else:
            sources = "\n\n".join(
                f"[{n}] {c['doc_title']} > {c['section']}\n{c['text']}" for n, c in enumerate(retrieved, 1))
            answer = ask_llm(PROMPT.format(**item, sources=sources))

        # Turn citation numbers back into SOP references a person can open
        cited = []
        for n in answer.get("citations", []):
            if isinstance(n, int) and 1 <= n <= len(retrieved):
                c = retrieved[n - 1]
                cited.append(f"{c['doc']} > {c['section']}")
            else:
                cited.append(f"[{n}] (source does not exist)")
        results.append({**item, **answer, "cited": cited,
                        "retrieved": [{"ref": f"{c['doc']} > {c['section']}", "score": round(c["score"], 3)}
                                      for c in retrieved]})
        print(f"({item['id']}) {item['title']}: {answer.get('verdict')}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    (config.REPORT_DIR / f"gap_report_{stamp}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (config.REPORT_DIR / "gap_report_latest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_markdown(results, stamp)
    print(f"\nReport saved to output/gap_report_{stamp}.md")


def write_markdown(results, stamp):
    lines = [f"# QMS gap report - {stamp}",
             f"Model: {config.LLM_MODEL} | Embeddings: {config.EMBED_MODEL} | "
             f"top_k={config.TOP_K}, chunk={config.MAX_CHUNK_WORDS} words, threshold={config.MIN_SIMILARITY}",
             "", "Draft for review by a qualified person. Not a compliance decision.", ""]
    for r in results:
        lines += [f"## ({r['id']}) {r['title']}: {r.get('verdict')}",
                  r.get("reason", ""), ""]
        if r.get("suggested_fix"):
            lines += [f"Suggested fix: {r['suggested_fix']}", ""]
        if r["cited"]:
            lines += ["Cited: " + "; ".join(r["cited"]), ""]
        lines += ["Retrieved: " + "; ".join(f"{x['ref']} ({x['score']})" for x in r["retrieved"]), ""]
    (config.REPORT_DIR / f"gap_report_{stamp}.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
