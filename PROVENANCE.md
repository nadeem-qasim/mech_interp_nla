# Human log

## RunPod rent + 27B smoke test — 2026-09-09 (~2 hours)

- Rented Pod with H100 SXM, smoke test on 27B NLA (verbalizations and reconstructions on the 27B NLA's own training data, FineFineWe, included). Output: `fve_claims/logs_27b_pod/nla27b_smoke_out.json` (run by Claude; not re-checked by me).
- Claude Sonnet got 163 claims and labelled them against source text prefix: `notes/nla_setup/nla27b_smoke_claims_annotated.csv`
- Reviewed ANNOTATION_GUIDE.md; reduced verbosity, over complications

## 2026-10-10 

- Brainstormed datasets w/ claude/gpt; settled on Re-DocRed (Wiki intros), picked out 100 random for iteration speed
- Verbalizations on fixed length prefixes (7B on mb pro, 27b on RunPod)
- Claim extraction and labeling and deletion version and paraphrased version (all local, pulled results from RunPod)
- AR scoring on prepared data (7B on mb pro, 27b on RunPod)
- All raw cosine scores pulled locally
- FVE drop computations and breakdowns
- Claim re-label review with Claude Fable 5.1 XHigh (1 entity and 5 detail labels changed only; text was easy enough for Sonnet)
- Deletion-rewrite agent spawned 4 sub-agents and ate token budget. Killed & explicit instructions to not spawn; deletions made mechanical text-span removal, no LLM (27B all; 7B for remaining 441 of 1600)
- Pod ran out of credit mid-scoring. Data survived in network volume. Resumed from there later



## 2026-10-11

- Blind annotated (T/F) 17 random claims; `uv run python fve_claims/audit/score_spotcheck.py`
Disagreement on 3/17 claims; Two of those I was wrong; *Claude only mislabeled one claim, S13*
- Random 10 Heavy paraphrase check; `uv run python fve_claims/audit/paraphrase_samples.py`;
*All correct* 
- Raw FVE scores CSV format and sanity check
- FVE script aggregate metrics calculations check
- Found paraphrases are per *sentence*, not per claim (1308 claims on 571 rewrites, 27B). Biggest limitation; found late
- Form application answers typed by hand, repo cleaned
- Exec summary and write-up





