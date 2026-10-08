# fve_claims — atomic claim annotation and deletion FVE

Current protocol: **atomic-v1**.

## Annotation
Read [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md). One record describes one independently
checkable proposition, deduplicated within an explanation with all AV occurrence spans.
Truth uses only the exact visible prefix. Types are entity, detail, theme, and forecast.
Tasks are score-blind; sentences supply context IDs, never claim boundaries.

```sh
python3 fve_claims/02_claims.py --batch 0 --start 0 --end 25
python3 fve_claims/02_claims.py --batch 1 --start 25 --end 50
python3 fve_claims/02_claims.py --batch 2 --start 50 --end 75
python3 fve_claims/02_claims.py --batch 3 --start 75 --end 100
# After annotators write the atomic JSONL records:
python3 fve_claims/02_claims.py --labels fve_claims/tasks/02_atoms_7b_batch*.jsonl
```

Task files: `tasks/02_atomic_tasks_7b_batchN.json`. Merged annotations: `out/02_atoms_7b.jsonl`.

**Note.** The per-document files `tasks/02_atoms_{7b,27b}_doc*.jsonl` and `tasks/04_rewrites_*_doc*.jsonl` are the labels and
edits the reported results use.
`--output` overrides either destination. The merger validates IDs, labels, exact source
spans, evidence, and rationale format and reports documents without annotations.
Annotators must check semantic atomicity, equivalence, complete occurrence coverage, and
truth; code cannot establish these from offsets. All labels remain provisional for review.

## Deletion scoring
`gen_deletions.py` (7B) and `gen_deletions_27b.py` remove each claim's exact `av_spans` from the
explanation (string operation, no LLM) and write `tasks/04_rewrites_{7b,27b}_doc*.jsonl`. A claim
whose spans overlap another claim's is flagged and not scored. The scorer rejects unchanged, empty,
or stale counterfactuals before loading the model.

```sh
python3 fve_claims/04_score.py
python3 fve_claims/05_analyze.py
# Human labels can be substituted, keeping claim IDs/propositions fixed:
python3 fve_claims/05_analyze.py --labels path/to/reviewed_atoms.jsonl --split eval
```

Outputs: `04_scores_7b_b*.csv` and `04_expl_7b_b*.csv` (one pair per document range), `05_summary_*.md`,
and figures under `fig/` (regenerated on each run).
Analysis joins on document and claim IDs, rejects unmatched scores and duplicate keys,
and reports all four types × all three truth labels, with document-cluster bootstrap CIs
(1000 draws, seed 0). Means weight claims equally. Relatedness is not collected in this
revision. Unscored claims are not included in deletion statistics; counts show coverage.

Score remains `cos = cos(h, AR(z))`, `FVE = 1 - 2(1-cos)/D` and
`drop = FVE(z) - FVE(edited z)`, reported in percentage points. Denominators are the
released 7B value 0.7335 and local variance of sqrt(d)-normalized pilot activations.
Changing D rescales drops and CI endpoints. It does not change ranks.

The 100 Re-DocRED prefixes, dev IDs 0–19 / eval IDs 20–99, and original 7B generation setup
are documented in `data/redocred_pilot/README.md`. Scripts 00/01 produce the activations and greedy AV
explanations; the 7B model classes come from `src/nla_lib.py`. The 27B pipeline (`common_27b.py` and the
`*27b*` scripts) ran on a rented GPU; `04_06_score_27b_full.py` scores all 100 documents, deletions and
heavy paraphrases. 7B paraphrase scoring is `06_score_paraphrase.py`. `07_tag_final_sentence.py` adds the
per-claim `final_sentence` tag.

## Tests
```sh
.venv/bin/python -m unittest discover -s fve_claims -p 'test_*.py' -v
```
Tests cover provenance, deduplication keys, conflicting claims, forecast/token restrictions,
reviewed-deletion requirements, real task preparation, and scoring/analysis with a fake AR.
No model inference is needed for these checks.
