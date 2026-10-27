"""Passes 2 and 3: read the procedures the way an auditor proofreads them.

Pass 2 - section vs rules:   every procedure section is compared with the closest regulation rules.
                             Catches WRONG things: deadlines, report types, outdated references.
Pass 3 - section vs section: the most similar sections from different procedures are compared.
                             Catches CONFLICTS between procedures.

Each check is one short yes/no question, which suits small models better than one long prompt.
"""
import json
from datetime import datetime

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

import config
from chunking import split_markdown

SECTION_PROMPT = """You are checking one section of a company's medical device procedure against regulation rules.

COMPANY: {company}

PROCEDURE SECTION ({ref}):
{text}

RULES:
{rules}

Question: does this procedure section state something that CONTRADICTS one of these rules? For example: a different deadline, a different type of report than the rule requires for this device, or a reference to a replaced regulation or an outdated edition of a standard.
Something that is simply not mentioned is NOT a contradiction. Only flag a clear conflict.

Answer in JSON only:
{{"contradiction": true or false, "rule": "rule title, or empty", "explanation": "one sentence quoting the conflicting words", "suggested_fix": "one sentence, or empty"}}"""

PAIR_PROMPT = """Two sections from different procedures of the same company:

SECTION A ({ref_a}):
{text_a}

SECTION B ({ref_b}):
{text_b}

Question: do these two sections give CONFLICTING instructions about the same matter? For example: different timing, a different responsible person, or opposite requirements.
Different wording, or one section being more detailed than the other, is NOT a conflict.

Answer in JSON only:
{{"conflict": true or false, "explanation": "one sentence quoting the conflicting words", "suggested_fix": "one sentence, or empty"}}"""


def ask_llm(prompt: str) -> dict:
    r = requests.post(config.OLLAMA_URL, json={
        "model": config.LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "format": "json", "stream": False, "options": {"temperature": 0},
    }, timeout=600)
    r.raise_for_status()
    try:
        return json.loads(r.json()["message"]["content"])
    except (json.JSONDecodeError, KeyError):
        return {}


def is_true(v):
    return v is True or str(v).strip().lower() == "true"


def main():
    chunks = json.loads((config.INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    vectors = np.load(config.INDEX_DIR / "vectors.npy")
    keep = [i for i, c in enumerate(chunks) if not c["section"].startswith(config.BOILERPLATE_SECTIONS)]
    model = SentenceTransformer(config.EMBED_MODEL)

    rules = [s for s in split_markdown(config.REGULATION_FILE, max_words=400) if s["section"] != "(intro)"]
    rule_vecs = model.encode([f"{r['section']}: {r['text']}" for r in rules], normalize_embeddings=True)

    findings = []

    # Pass 2: each section against its closest rules
    print(f"Pass 2: checking {len(keep)} procedure sections against the rules...")
    for i in keep:
        c = chunks[i]
        ref = f"{c['doc']} > {c['section']}"
        top = np.argsort(-(rule_vecs @ vectors[i]))[: config.SECTION_RULES_TOP_K]
        rule_text = "\n".join(f"- {rules[j]['section']}: {rules[j]['text']}" for j in top)
        a = ask_llm(SECTION_PROMPT.format(company=config.COMPANY_CONTEXT, ref=ref, text=c["text"], rules=rule_text))
        if is_true(a.get("contradiction")):
            findings.append({"pass": "section vs rules", "docs": [c["doc"]], "where": ref,
                             "rule": a.get("rule", ""), "explanation": a.get("explanation", ""),
                             "suggested_fix": a.get("suggested_fix", "")})
            print(f"  FLAG {ref}: {a.get('explanation', '')}")

    # Pass 3: most similar section pairs from different procedures
    sub = vectors[keep]
    sims = sub @ sub.T
    pairs = []
    for x in range(len(keep)):
        for y in range(x + 1, len(keep)):
            if chunks[keep[x]]["doc"] != chunks[keep[y]]["doc"]:
                pairs.append((float(sims[x, y]), keep[x], keep[y]))
    pairs = sorted(pairs, reverse=True)[: config.CONFLICT_PAIRS]
    print(f"\nPass 3: comparing the {len(pairs)} most similar section pairs from different procedures...")
    for sim, i, j in pairs:
        a_, b_ = chunks[i], chunks[j]
        ref_a, ref_b = f"{a_['doc']} > {a_['section']}", f"{b_['doc']} > {b_['section']}"
        a = ask_llm(PAIR_PROMPT.format(ref_a=ref_a, text_a=a_["text"], ref_b=ref_b, text_b=b_["text"]))
        if is_true(a.get("conflict")):
            findings.append({"pass": "section vs section", "docs": [a_["doc"], b_["doc"]],
                             "where": f"{ref_a}  <->  {ref_b}", "similarity": round(sim, 3),
                             "explanation": a.get("explanation", ""), "suggested_fix": a.get("suggested_fix", "")})
            print(f"  FLAG {ref_a} <-> {ref_b}: {a.get('explanation', '')}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    for name in (f"review_{stamp}.json", "review_latest.json"):
        (config.REPORT_DIR / name).write_text(json.dumps(findings, indent=2), encoding="utf-8")
    lines = [f"# Procedure review - {stamp}", f"Model: {config.LLM_MODEL}", "",
             "Draft for review by a qualified person. Not a compliance decision.", ""]
    for p in ("section vs rules", "section vs section"):
        lines += [f"## {p.capitalize()}", ""]
        for f in [f for f in findings if f["pass"] == p]:
            lines += [f"### {f['where']}", f.get("explanation", "")]
            if f.get("rule"):
                lines.append(f"Rule: {f['rule']}")
            if f.get("suggested_fix"):
                lines.append(f"Suggested fix: {f['suggested_fix']}")
            lines.append("")
    (config.REPORT_DIR / f"review_{stamp}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{len(findings)} flags. Report saved to output/review_{stamp}.md")


if __name__ == "__main__":
    main()
