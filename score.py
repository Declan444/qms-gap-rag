"""Score the latest gap report against the answer key (the test set)."""
import json
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
    print("\nCheck by hand (outside the checklist):")
    for g in expected["manual_checks"]:
        print(f"- {g}")
    print("\nAlso open each citation: right verdict with the wrong reason still counts as partly found.")


if __name__ == "__main__":
    main()
