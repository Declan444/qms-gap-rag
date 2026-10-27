"""Reverse RAG: each checklist requirement is the 'question'.
Retrieve the closest procedure text, optionally the relevant regulation text,
then ask a local model: COVERED, WEAK or MISSING?"""
import json
import re
from datetime import datetime

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

import config
from chunking import parse_checklist, split_markdown

BASIC = """You are a medical device quality auditor checking a company's procedures against one EU MDR requirement.

REQUIREMENT ({id}) {title}: {text}
{reference}
Use only the numbered procedure extracts below. Do not assume anything they do not say.

{sources}

Decide one verdict:
- COVERED: the extracts clearly address the requirement.
- WEAK: the extracts address it partly, vaguely, with an outdated reference, or with a rule you believe is wrong.
- MISSING: the extracts do not address it.

Answer in JSON only, with these keys:
{{"verdict": "COVERED|WEAK|MISSING", "reason": "one or two sentences", "citations": [list of source numbers you relied on], "suggested_fix": "one sentence, or empty if COVERED"}}"""

STRICT = """You are a strict medical device quality auditor checking a company's procedures against one EU MDR requirement. Your job is to find gaps, not to reassure.

REQUIREMENT ({id}) {title}: {text}
{reference}
COMPANY PROCEDURES (numbered sources):

{sources}

Work in this order:
1. Break the requirement into the separate things it asks for.
2. For each one, find procedure text that DESCRIBES A PROCESS for it: who does what, when and how. A title, a purpose or scope statement, a list of responsibilities, or a passing mention does NOT count.
3. Compare that process with the reference rules, if any are given. A wrong deadline, a wrong type of report, or an outdated regulation or standard means the element is not properly met.

Verdict:
- COVERED: every element has a described process, and nothing conflicts with the reference rules.
- MISSING: no procedure describes a process for the requirement at all.
- WEAK: anything in between.

Answer in JSON only, with these keys:
{{"elements": [{{"element": "...", "met": true or false, "evidence": "procedure section, or 'none'"}}], "verdict": "COVERED|WEAK|MISSING", "reason": "one or two sentences naming the specific problem", "citations": [list of source numbers you relied on], "suggested_fix": "one sentence, or empty if COVERED"}}"""

REFERENCE_BLOCK = """
REFERENCE RULES (from the MDR and current harmonised standards):
{rules}
"""


def ask_llm(prompt: str) -> dict:
    r = requests.post(config.OLLAMA_URL, json={
        "model": config.LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0},
    }, timeout=600)
    r.raise_for_status()
    try:
        return json.loads(r.json()["message"]["content"])
    except (json.JSONDecodeError, KeyError):
        return {"verdict": "ERROR", "reason": "Model did not return valid JSON.", "citations": [], "suggested_fix": ""}


def full_sop_text(doc: str) -> str:
    """Whole procedure, minus the company/revision header line."""
    text = (config.PROCEDURES_DIR / f"{doc}.md").read_text(encoding="utf-8")
    return "\n".join(l for l in text.splitlines() if not l.startswith("Slieve Bloom Medical Ltd |")).strip()


def load_regulation(model):
    if not config.USE_REGULATION:
        return None, None
    sections = split_markdown(config.REGULATION_FILE, max_words=400)
    sections = [s for s in sections if s["section"] != "(intro)"]
    vecs = model.encode([f"{s['section']}: {s['text']}" for s in sections], normalize_embeddings=True)
    return sections, vecs


def main():
    chunks = json.loads((config.INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    vectors = np.load(config.INDEX_DIR / "vectors.npy")
    model = SentenceTransformer(config.EMBED_MODEL)
    items = parse_checklist(config.CHECKLIST_FILE)
    reg_sections, reg_vecs = load_regulation(model)
    template = STRICT if config.PROMPT_STYLE == "strict" else BASIC

    results = []
    for item in items:
        q = model.encode([f"{item['title']}: {item['text']}"], normalize_embeddings=True)[0]
        scores = vectors @ q
        if config.SKIP_BOILERPLATE:
            scores = np.where([c["section"].startswith(config.BOILERPLATE_SECTIONS) for c in chunks], -1.0, scores)
        order = np.argsort(-scores)
        top_chunks = [{**chunks[i], "score": float(scores[i])} for i in order[: config.TOP_K]]

        # What the model reads: the closest sections, or the closest whole procedures
        if config.CONTEXT_MODE == "full_sop":
            docs = []
            for i in order:
                if chunks[i]["doc"] not in docs:
                    docs.append(chunks[i]["doc"])
                if len(docs) == config.FULL_SOP_COUNT:
                    break
            sources_list = [{"ref": d, "text": full_sop_text(d)} for d in docs]
        else:
            sources_list = [{"ref": f"{c['doc']} > {c['section']}",
                             "text": f"{c['doc_title']} > {c['section']}\n{c['text']}"} for c in top_chunks]

        reference = ""
        if reg_sections is not None:
            rs = reg_vecs @ q
            rules = [reg_sections[i] for i in np.argsort(-rs)[: config.REGULATION_TOP_K]]
            reference = REFERENCE_BLOCK.format(rules="\n".join(f"- {r['section']}: {r['text']}" for r in rules))

        if top_chunks[0]["score"] < config.MIN_SIMILARITY:
            answer = {"verdict": "MISSING",
                      "reason": f"No procedure text above the similarity threshold ({config.MIN_SIMILARITY}).",
                      "citations": [], "suggested_fix": "Write a procedure for this requirement."}
        else:
            sources = "\n\n".join(f"[{n}] {s['text']}" for n, s in enumerate(sources_list, 1))
            answer = ask_llm(template.format(**item, reference=reference, sources=sources))

        cited = []
        for n in answer.get("citations", []) or []:
            try:
                n = int(re.sub(r"\D", "", str(n)))
            except ValueError:
                n = 0
            cited.append(sources_list[n - 1]["ref"] if 1 <= n <= len(sources_list)
                         else f"[{n}] (source does not exist)")
        results.append({**item, **answer, "cited": cited,
                        "read": [s["ref"] for s in sources_list],
                        "retrieved": [{"ref": f"{c['doc']} > {c['section']}", "score": round(c["score"], 3)}
                                      for c in top_chunks]})
        print(f"({item['id']}) {item['title']}: {answer.get('verdict')}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    for name in (f"gap_report_{stamp}.json", "gap_report_latest.json"):
        (config.REPORT_DIR / name).write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_markdown(results, stamp)
    print(f"\nReport saved to output/gap_report_{stamp}.md")


def settings_line():
    return (f"Model: {config.LLM_MODEL} | context={config.CONTEXT_MODE} | prompt={config.PROMPT_STYLE} | "
            f"regulation={'on' if config.USE_REGULATION else 'off'} | "
            f"skip_boilerplate={config.SKIP_BOILERPLATE} | top_k={config.TOP_K}, "
            f"chunk={config.MAX_CHUNK_WORDS} words, threshold={config.MIN_SIMILARITY}")


def write_markdown(results, stamp):
    lines = [f"# QMS gap report - {stamp}", settings_line(), "",
             "Draft for review by a qualified person. Not a compliance decision.", ""]
    for r in results:
        lines += [f"## ({r['id']}) {r['title']}: {r.get('verdict')}", r.get("reason", ""), ""]
        for e in r.get("elements", []) or []:
            if isinstance(e, dict):
                mark = "met" if e.get("met") else "NOT met"
                lines.append(f"- {e.get('element', '')}: {mark} ({e.get('evidence', '')})")
        if r.get("elements"):
            lines.append("")
        if r.get("suggested_fix"):
            lines += [f"Suggested fix: {r['suggested_fix']}", ""]
        if r["cited"]:
            lines += ["Cited: " + "; ".join(r["cited"]), ""]
        lines += ["Model read: " + "; ".join(r["read"]), ""]
    (config.REPORT_DIR / f"gap_report_{stamp}.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
