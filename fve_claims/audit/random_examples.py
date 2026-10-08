"""Randomly selected examples for the write-up: N_DOCS random evaluation documents, both models, full passage, the
explanation, and every claim with its label and deletion drop. Writes markdown ready to paste. Run from the repo root."""
import glob, json, random, pandas as pd

OUT, SEED, N_DOCS = "fve_claims/audit/out/claim_random_examples.md", 2026, 2
P = {r["pilot_id"]: r for r in map(json.loads, open("data/redocred_pilot/pilot.jsonl"))}


def atoms(m):
    pats = ["fve_claims/tasks/02_atoms_27b_doc*.jsonl"] if m == "27B" else ["fve_claims/tasks/02_atoms_7b_doc*.jsonl"]
    recs = {}
    for pat in pats:
        for f in sorted(glob.glob(pat)):
            for l in open(f):
                r = json.loads(l); recs[(r["pilot_id"], r["claim_id"])] = r
    return pd.DataFrame(recs.values())


def expl(m):
    files = ["fve_claims/01_av_27b_25.jsonl", "fve_claims/01_av_27b_25_50.jsonl", "fve_claims/01_av_27b_50_100.jsonl"] if m == "27B" else ["fve_claims/01_av_7b.jsonl"]
    return {g["pilot_id"]: (g["explanation"] if g["parse_ok"] else g["raw_generation"]) for f in files for g in map(json.loads, open(f))}


def drops(m):
    if m == "27B":
        s = pd.read_csv("fve_claims/04_06_scores_27b_full.csv").rename(columns={"P_orig": "drop"})
    else:
        s = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("fve_claims/04_scores_7b_b*.csv"))]).rename(columns={"fve_drop_local": "drop"})
    return {(r.pilot_id, r.claim_id): r.drop * 100 for r in s.itertuples()}


docs = random.Random(SEED).sample(sorted(p for p in P if p >= 20), N_DOCS)  # evaluation split only
L = [f"# Randomly selected examples (seed {SEED}, eval documents {docs}; not cherry-picked)", "",
     "Truth labels are Claude's (provisional). A blank FVE drop means the claim was not scored: deleting its text would also have "
     "removed part of another claim.", ""]
for pid in docs:
    L += [f"## Document {pid}", "", f"**Passage (the activation is read at its final word):** {P[pid]['prefix_text']}", ""]
    for m in ["27B", "7B"]:
        a, e, dr = atoms(m), expl(m), drops(m)
        L += [f"### {m} explanation", "", e[pid].strip(), "", "| claim | type | truth | FVE drop when deleted (pp) |", "|---|---|---|---|"]
        for r in a[a.pilot_id == pid].sort_values("claim_id").itertuples():
            v = dr.get((pid, r.claim_id))
            L.append(f"| {r.proposition} | {r.type} | {r.truth} | {'' if v is None else f'{v:.2f}'} |")
        L.append("")
open(OUT, "w").write("\n".join(L))
print(f"wrote {OUT}: documents {docs}, {len(L)} lines")
