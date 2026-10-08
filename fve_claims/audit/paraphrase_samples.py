"""Random heavy paraphrases for a by-eye check: N scored true/false claims per model, showing the claim and the part of the
explanation that changed under the heavy rewrite (before and after). Scores and labels are not shown. Run from the repo root."""
import difflib, glob, json, random
import pandas as pd

OUT, SEED, N = "fve_claims/audit/out/paraphrase_samples.md", 7, 5


def truth(patterns):
    recs = {}
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            for l in open(f):
                r = json.loads(l); recs[(r["pilot_id"], int(r["claim_id"]))] = r["truth"]
    return recs


MODELS = {
    "27B": (["fve_claims/tasks/02_atoms_27b_doc*.jsonl"], ["fve_claims/04_06_scores_27b_full.csv"], "fve_claims/tasks/04_rewrites_27b_doc*.jsonl"),
    "7B": (["fve_claims/tasks/02_atoms_7b_doc*.jsonl"], sorted(glob.glob("fve_claims/04_scores_7b_b*.csv")), "fve_claims/tasks/04_rewrites_7b_doc*.jsonl"),
}


def changed(a, b, context=6):
    a, b = a.split(), b.split()
    ops = [o for o in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if o[0] != "equal"]
    if not ops: return "(no change)", "(no change)"
    i1, i2, j1, j2 = ops[0][1], ops[-1][2], ops[0][3], ops[-1][4]
    return " ".join(a[max(0, i1 - context): i2 + context]), " ".join(b[max(0, j1 - context): j2 + context])


rng = random.Random(SEED)
L = [f"# Heavy paraphrase samples (seed {SEED}; {N} random scored true/false claims per model)", "",
     "For each: does the rewrite keep the same meaning and the same specifics? Write `check: same` or `check: changed — <what>`.", ""]
i = 0
for m, (label_pats, score_files, rewrite_pat) in MODELS.items():
    tr = truth(label_pats)
    scored = sorted({(r.pilot_id, int(r.claim_id)) for f in score_files for r in pd.read_csv(f).itertuples()})
    keys = [k for k in scored if tr.get(k) in ("true", "false")]
    rw = {(r["pilot_id"], int(r["claim_id"])): r for f in sorted(glob.glob(rewrite_pat)) for r in map(json.loads, open(f))}
    for k in sorted(rng.sample(keys, N)):
        i += 1; r = rw[k]; before, after = changed(r["original_explanation"], r["heavy_text"])
        L += [f"## {i}. {m} · document {k[0]} · claim {k[1]}", "", f"**Claim:** {r['proposition']}", "",
              f"**Before:** …{before}…", "", f"**After heavy paraphrase:** …{after}…", "", "check: ", ""]
import os, sys
if os.path.exists(OUT) and "--force" not in sys.argv:
    sys.exit(f"{OUT} exists and holds hand-entered labels; pass --force to overwrite it.")
open(OUT, "w").write("\n".join(L))
print(f"wrote {OUT}: {i} samples")
