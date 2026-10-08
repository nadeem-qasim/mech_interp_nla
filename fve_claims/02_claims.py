"""Prepare score-blind atomization tasks, or validate/merge completed JSONL annotations.

python fve_claims/02_claims.py --batch 0 --start 0 --end 25
python fve_claims/02_claims.py --labels tasks/02_atoms_7b_batch*.jsonl

Sentence boundaries are context only. Semantic atomization/deduplication is done by annotators.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from atomic import PROTOCOL, SUBTYPES, sentence_spans, validate_claims

HERE = Path(__file__).resolve().parent


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def build_tasks():
    rows = {r["pilot_id"]: r for r in read_jsonl(HERE.parent / "data/redocred_pilot/pilot.jsonl")}
    tasks = {}
    for g in read_jsonl(HERE / "01_av_7b.jsonl"):
        r = rows[g["pilot_id"]]
        z = g["explanation"] if g["parse_ok"] else g["raw_generation"]
        tasks[g["pilot_id"]] = {
            "protocol": PROTOCOL, "pilot_id": g["pilot_id"], "split": g["split"],
            "prefix_text": r["prefix_text"], "explanation": z, "sentences": sentence_spans(z),
            "annotation_guide": (HERE / "ANNOTATION_GUIDE.md").read_text(),
            "subtypes": {k: sorted(v) for k, v in SUBTYPES.items()},
        }
    return tasks


def count_table(claims):
    def cls(c):
        return c["truth"] if c["truth"] != "false" else f"false_{c.get('related', 'NA')}"
    counts = Counter((c["type"], cls(c)) for c in claims)
    cols = ("true", "false_related", "false_unrelated", "irrelevant")
    lines = ["| Claim type | True | False (related) | False (unrelated) | Irrelevant | Total |", "|---|---:|---:|---:|---:|---:|"]
    totals = [0] * len(cols)
    for kind in ("entity", "detail", "theme", "forecast"):
        values = [counts[kind, k] for k in cols]
        totals = [a + b for a, b in zip(totals, values)]
        lines.append(f"| {kind.title()} | " + " | ".join(map(str, values + [sum(values)])) + " |")
    lines.append("| **Total** | " + " | ".join(map(str, totals + [sum(totals)])) + " |")
    return "\n".join(lines)


def write_reports(claims, tasks, output):
    ids = sorted({c["pilot_id"] for c in claims})
    first10 = [c for c in claims if 0 <= c["pilot_id"] < 10]
    report = ("# Atomic annotation counts (provisional agent labels)\n\n"
              f"Annotated documents: {ids}. Total distinct propositions: {len(claims)}.\n\n"
              "## IDs 0–9\n\n" + count_table(first10) + "\n\n## All annotated documents\n\n" + count_table(claims) + "\n")
    (output.parent / "02_annotation_summary.md").write_text(report)
    with (output.parent / "02_review_blind_7b.csv").open("w", newline="") as f:
        fields = ["pilot_id", "claim_id", "proposition", "av_spans", "type", "subtype", "truth", "related", "prefix_evidence", "rationale", "review_flag", "prefix_text", "explanation"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c in claims:
            row = {k: c[k] for k in fields if k in c}
            for k in ("av_spans", "prefix_evidence"):
                row[k] = json.dumps(row[k], ensure_ascii=False)
            row.update(prefix_text=tasks[c["pilot_id"]]["prefix_text"], explanation=tasks[c["pilot_id"]]["explanation"])
            writer.writerow(row)
    inputs = [HERE / "01_av_7b.jsonl", HERE.parent / "data/redocred_pilot/pilot.jsonl", HERE / "ANNOTATION_GUIDE.md"]
    settings = {"protocol": PROTOCOL, "wall_end": datetime.now(timezone.utc).isoformat(), "annotated_ids": ids,
                "n_claims": len(claims), "n_review_flags": sum(c["review_flag"] for c in claims),
                "missing_ids": sorted(set(tasks) - set(ids)), "score_blind": True,
                "input_sha256": {str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_name("02_settings.json").write_text(json.dumps(settings, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int)
    ap.add_argument("--start", type=int)
    ap.add_argument("--end", type=int)
    ap.add_argument("--labels", nargs="+", type=Path)
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    tasks = build_tasks()
    if a.labels:
        claims = [c for path in a.labels for c in read_jsonl(path)]
        if not claims: ap.error("No atomic annotations supplied")
        validate_claims(claims, tasks)
        output = a.output or HERE / "out/02_atoms_7b.jsonl"
        output.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in claims))
        write_reports(claims, tasks, output)
        missing = sorted(set(tasks) - {c["pilot_id"] for c in claims})
        print(f"Validated {len(claims)} atoms; documents without annotations: {missing}")
    else:
        if a.batch is None or a.start is None or a.end is None or not 0 <= a.start < a.end:
            ap.error("Supply --batch, --start and --end, or --labels")
        selected = [tasks[i] for i in range(a.start, a.end)]
        output = a.output or HERE / "tasks" / f"02_atomic_tasks_7b_batch{a.batch}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(selected, indent=2, ensure_ascii=False))
        print(f"Wrote {len(selected)} score-blind tasks to {output}")


if __name__ == "__main__":
    main()
