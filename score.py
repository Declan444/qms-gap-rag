"""Score a gap report against the answer key (the test set).

Two scores:
  verdict - is the verdict acceptable?
  reason  - is the verdict acceptable AND does the reason name the planted problem?
            (keyword check; for items with no planted problem the verdict alone counts)

Usage:
  python score.py              score the latest report and log it to output/run_log.csv
  python score.py --all        rescore every saved report (no logging)
  python score.py --total      how many of the 15 hidden mistakes did all passes find, between them?
"""
import csv
import json
import sys
from datetime import datetime

import config


def score_results(results, expected, verbose=True):
    results = {r["id"]: r for r in results}
    v_right = r_right = 0
    if verbose:
        print(f"{'Item':<6}{'Got':<10}{'Expected':<16}{'Verdict':<9}Reason")
    for item_id, exp in expected["checklist"].items():
        r = results.get(item_id, {})
        got = r.get("verdict", "NONE")
        ok = got in exp["acceptable"]
        text = " ".join(str(r.get(k, "")) for k in ("reason", "suggested_fix", "elements")).lower()
        kws = exp.get("reason_keywords")
        reason_ok = ok and (kws is None or any(k in text for k in kws))
        v_right += ok
        r_right += reason_ok
        if verbose:
            note = "" if reason_ok else f"({exp['gaps']})"
            if any("does not exist" in c for c in r.get("cited", [])):
                note += "  + cites a source that does not exist"
            print(f"({item_id})   {got:<10}{'/'.join(exp['acceptable']):<16}"
                  f"{'ok' if ok else 'WRONG':<9}{'ok' if reason_ok else 'WRONG'}  {note}")
    all_text = json.dumps(list(results.values())).lower()
    bonus = [name for name, kws in expected.get("bonus", {}).items() if any(k in all_text for k in kws)]
    return v_right, r_right, bonus


def log_run(v, r, total, bonus):
    """Append this run to output/run_log.csv: the table for comparing experiments."""
    log = config.REPORT_DIR / "run_log.csv"
    header = ["time", "model", "context", "prompt", "regulation", "top_k", "chunk_words", "threshold",
              "verdict_score", "reason_score", "bonus"]
    if log.exists() and log.read_text(encoding="utf-8").splitlines()[0].split(",") != header:
        log.rename(log.with_name("run_log_old.csv"))   # older format: keep it, start a new log
    new = not log.exists()
    with log.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M"), config.LLM_MODEL,
                    config.CONTEXT_MODE + ("+clean" if config.SKIP_BOILERPLATE else ""),
                    config.PROMPT_STYLE, "on" if config.USE_REGULATION else "off", config.TOP_K,
                    config.MAX_CHUNK_WORDS, config.MIN_SIMILARITY, f"{v}/{total}", f"{r}/{total}",
                    "; ".join(bonus)])
    print(f"Logged to {log.name}")


def found_by(spec, gap, review):
    """Which pass found this hidden mistake, if any."""
    kws = spec.get("keywords", [])
    has = lambda text: any(k in text.lower() for k in kws)
    for item in spec.get("gap_missing", []):
        if gap.get(item, {}).get("verdict") == "MISSING":
            return "requirement check"
    for item in spec.get("gap", []):
        r = gap.get(item, {})
        text = " ".join(str(r.get(k, "")) for k in ("reason", "suggested_fix", "elements"))
        if r.get("verdict") in ("WEAK", "MISSING") and has(text):
            return "requirement check"
    for f in review:
        text = f.get("explanation", "") + " " + f.get("suggested_fix", "") + " " + f.get("rule", "")
        if spec.get("review_docs") and any(d in f["docs"] for d in spec["review_docs"]) and has(text):
            return f["pass"]
        if spec.get("review_pair") and all(d in f["docs"] for d in spec["review_pair"]):
            return f["pass"]
    if spec.get("anywhere"):
        all_text = json.dumps([list(gap.values()), review]).lower()
        if any(k in all_text for k in spec["anywhere"]):
            return "anywhere (bonus)"
    return None


def total_found(expected):
    gap_file = config.REPORT_DIR / "gap_report_latest.json"
    rev_file = config.REPORT_DIR / "review_latest.json"
    gap = {r["id"]: r for r in json.loads(gap_file.read_text(encoding="utf-8"))} if gap_file.exists() else {}
    review = json.loads(rev_file.read_text(encoding="utf-8")) if rev_file.exists() else []
    if not rev_file.exists():
        print("(No review yet: run review.py for passes 2 and 3.)\n")
    found = 0
    for name, spec in expected["hidden_mistakes"].items():
        how = found_by(spec, gap, review)
        found += bool(how)
        print(f"{'FOUND ' if how else 'missed'}  {name:<62}{how or ''}")
    matched_flags = 0
    for f in review:
        text = (f.get("explanation", "") + " " + f.get("suggested_fix", "") + " " + f.get("rule", "")).lower()
        if any((spec.get("review_docs") and any(d in f["docs"] for d in spec["review_docs"])
                and any(k in text for k in spec.get("keywords", [])))
               or (spec.get("review_pair") and all(d in f["docs"] for d in spec["review_pair"]))
               for spec in expected["hidden_mistakes"].values()):
            matched_flags += 1
    print(f"\nHidden mistakes found: {found} of {len(expected['hidden_mistakes'])}")
    if review:
        print(f"Review flags: {len(review)}, of which {len(review) - matched_flags} match no planted mistake "
              f"(false alarms, or real issues I didn't plant: check them by hand)")


def main():
    expected = json.loads(config.EXPECTED_FILE.read_text(encoding="utf-8"))
    total = len(expected["checklist"])

    if "--total" in sys.argv:
        total_found(expected)
        return

    if "--all" in sys.argv:
        print(f"{'Report':<34}{'Verdict':<9}{'Reason':<8}Settings")
        for f in sorted(config.REPORT_DIR.glob("gap_report_2*.json")):
            v, r, bonus = score_results(json.loads(f.read_text(encoding="utf-8")), expected, verbose=False)
            md = f.with_suffix(".md")
            settings = md.read_text(encoding="utf-8").splitlines()[1] if md.exists() else ""
            print(f"{f.name:<34}{f'{v}/{total}':<9}{f'{r}/{total}':<8}{settings}")
        return

    results = json.loads((config.REPORT_DIR / "gap_report_latest.json").read_text(encoding="utf-8"))
    v, r, bonus = score_results(results, expected)
    print(f"\nVerdict score: {v} of {total}")
    print(f"Reason score:  {r} of {total}   (verdict right AND the reason names the planted problem)")
    if bonus:
        print("Bonus finds outside the checklist: " + "; ".join(bonus))
    log_run(v, r, total, bonus)
    print("\nStill check by hand: open the citations. Keywords can be fooled too.")


if __name__ == "__main__":
    main()
