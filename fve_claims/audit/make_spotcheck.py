"""Blind label spot-check sheet: N_RANDOM Claude-true + N_RANDOM Claude-false random detail claims per model, plus the
N_TAIL largest 27B detail deletion drops, shuffled. Writes the BLIND sheet and a separate KEY. Run from the repo root."""
import csv, glob, json, random, pandas as pd

BLIND, KEY, SEED, N_RANDOM, N_TAIL = "fve_claims/audit/out/claim_spotcheck_BLIND.md", "fve_claims/audit/out/claim_spotcheck_KEY.csv", 11, 3, 5
P = {r["pilot_id"]: r for r in map(json.loads, open("data/redocred_pilot/pilot.jsonl"))}


def scored(m):
    pats = ["fve_claims/tasks/02_atoms_27b_doc*.jsonl"] if m == "27B" else ["fve_claims/tasks/02_atoms_7b_doc*.jsonl"]
    recs = {}
    for pat in pats:
        for f in sorted(glob.glob(pat)):
            for l in open(f):
                r = json.loads(l); recs[(r["pilot_id"], r["claim_id"])] = r
    a = pd.DataFrame(recs.values())
    if m == "27B":
        s = pd.read_csv("fve_claims/04_06_scores_27b_full.csv").rename(columns={"P_orig": "drop"})
    else:
        s = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("fve_claims/04_scores_7b_b*.csv"))]).rename(columns={"fve_drop_local": "drop"})
    a = a.merge(s[["pilot_id", "claim_id", "drop"]], on=["pilot_id", "claim_id"])
    return a[(a.type == "detail") & a.truth.isin(["true", "false"])]  # the completed 2026-09-11 sheet used detail claims


rng, items = random.Random(SEED), []
for m in ["27B", "7B"]:
    d = scored(m).sort_values(["pilot_id", "claim_id"])
    for t in ["true", "false"]:
        pool = d[d.truth == t]
        items += [("random", m, r) for r in pool.iloc[sorted(rng.sample(range(len(pool)), N_RANDOM))].itertuples()]
tail = scored("27B").sort_values("drop", ascending=False).head(N_TAIL)
items += [("tail", "27B", r) for r in tail.itertuples()
          if not any(i[1] == "27B" and i[2].pilot_id == r.pilot_id and i[2].claim_id == r.claim_id for i in items)]
rng.shuffle(items)

B = ["# Claim spot-check — BLIND (label every item before opening the KEY)", "",
     "For each claim write `human_label: true`, `false` or `irrelevant`, using only the passage and fve_claims/ANNOTATION_GUIDE.md:",
     "true = stated or entailed by the passage; false = contradicted, or a specific the passage does not give in a slot the passage fills;",
     "irrelevant = forecasts / what comes next. `note:` is optional. Random and largest-drop items are mixed on purpose.", ""]
K = []
for i, (kind, m, r) in enumerate(items, 1):
    iid = f"S{i:02d}"; span = " … ".join(s["text"] for s in r.av_spans)
    B += [f"## {i}. {iid}", "", f"Model {m} · document {r.pilot_id}", "", f"**Passage:** {P[r.pilot_id]['prefix_text']}", "",
          f"**AV wrote:** \"{span}\"", "", f"**Claim:** {r.proposition}", "", "human_label: ", "note: ", ""]
    K.append({"item": iid, "sample": kind, "model": m, "pilot_id": r.pilot_id, "claim_id": r.claim_id, "subtype": r.subtype,
              "truth": r.truth, "fve_drop_pp": round(r.drop * 100, 3), "rationale": r.rationale})
import os, sys
if os.path.exists(BLIND) and "--force" not in sys.argv:
    sys.exit(f"{BLIND} exists and holds hand-entered labels; pass --force to overwrite it.")
open(BLIND, "w").write("\n".join(B))
with open(KEY, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(K[0])); w.writeheader(); w.writerows(K)
print(f"wrote {BLIND}: {sum(k['sample'] == 'random' for k in K)} random + {sum(k['sample'] == 'tail' for k in K)} largest-drop items; "
      f"KEY in {KEY} (do not open until done)")
