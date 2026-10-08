# CLAUDE.md — mech_interp_nla

Claim-deletion tests of natural language autoencoder (NLA) reconstructors on two open NLAs
(Qwen2.5-7B layer 20, `kitft/nla-qwen2.5-7b-L20-{av,ar}`; Qwen3.6-27B layer 42,
`ceselder/qwen3.6-27b-nla-rl`). `README.md` states the question, findings and caveats.

## Layout
- `fve_claims/` — the experiment: steps `00`–`07`, labels and edits in `tasks/`, scores, `analysis_27b/`
  (one script per reported 27B number), `audit/` (re-derivations and the blind label check), `figures/`.
- `src/nla_lib.py` (7B target/AV/AR), `src/model_utils.py` (the only model loader; import it before `torch`,
  it sets the MPS env from `scripts/env.sh`). 27B machinery is `fve_claims/common_27b.py` (CUDA).
- `data/redocred_pilot/` — the 100 Re-DocRED prefixes; `scripts/redocred_pilot.py` rebuilds them.
- `PROVENANCE.md` — the human's log of what was decided and checked by hand.

## Conventions
- `uv` project: `uv run python …` from the repo root. Tests: `uv run python -m unittest discover -s fve_claims`.
- Agents run analyses and report numbers. The human writes all prose meant for readers (README findings,
  the blog post); never draft or polish that text.
- Every reported number must come from a script in the repo; add one under `fve_claims/audit/` rather than
  quoting a one-off computation.
- Do not overwrite `fve_claims/audit/out/claim_spotcheck_BLIND.md`: it holds the human's blind labels.
