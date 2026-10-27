"""Score a gap report against the answer key (the test set).

Two scores:
  verdict - is the verdict acceptable?
  reason  - is the verdict acceptable AND does the reason name the planted problem?
            (keyword check; for items with no planted problem the verdict alone counts)

Usage:
  python score.py              score the latest report and log it to output/run_log.csv
  python score.py --all        rescore every saved report (no logging)
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
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M"), config.LLM_MODEL, config.CONTEXT_MODE,
                    config.PROMPT_STYLE, "on" if config.USE_REGULATION else "off", config.TOP_K,
                    config.MAX_CHUNK_WORDS, config.MIN_SIMILARITY, f"{v}/{total}", f"{r}/{total}",
                    "; ".join(bonus)])
    print(f"Logged to {log.name}")


def main():
    expected = json.loads(config.EXPECTED_FILE.read_text(encoding="utf-8"))
    total = len(expected["checklist"])

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
