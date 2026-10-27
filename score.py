"""Score the latest gap report against the answer key (the test set)."""
import csv
import json
from datetime import datetime

import config


def main():
    expected = json.loads(config.EXPECTED_FILE.read_text(encoding="utf-8"))
    results = {r["id"]: r for r in json.loads((config.REPORT_DIR / "gap_report_latest.json").read_text())}

    right = 0
    print(f"{'Item':<6}{'Got':<10}{'Expected':<20}Result")
    for item_id, exp in expected["checklist"].items():
        got = results.get(item_id, {}).get("verdict", "NONE")
        ok = got in exp["acceptable"]
        right += ok
        bad_cite = any("does not exist" in c for c in results.get(item_id, {}).get("cited", []))
        note = "ok" if ok else f"WRONG  ({exp['gaps']})"
        if bad_cite:
            note += "  + cites a source that does not exist"
        print(f"({item_id})   {got:<10}{'/'.join(exp['acceptable']):<20}{note}")

    total = len(expected["checklist"])
    print(f"\nVerdicts acceptable: {right} of {total}")
    log_run(right, total)
    print("\nCheck by hand (outside the checklist):")
    for g in expected["manual_checks"]:
        print(f"- {g}")
    print("\nAlso open each citation: right verdict with the wrong reason still counts as partly found.")


def log_run(right, total):
    """Append this run to output/run_log.csv: the table for comparing experiments."""
    log = config.REPORT_DIR / "run_log.csv"
    new = not log.exists()
    with log.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "model", "context", "prompt", "regulation", "top_k", "chunk_words",
                        "threshold", "score"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M"), config.LLM_MODEL, config.CONTEXT_MODE,
                    config.PROMPT_STYLE, "on" if config.USE_REGULATION else "off", config.TOP_K,
                    config.MAX_CHUNK_WORDS, config.MIN_SIMILARITY, f"{right}/{total}"])
    print(f"Logged to {log.name}")


if __name__ == "__main__":
    main()
