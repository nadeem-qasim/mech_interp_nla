# Re-DocRED pilot set (100 docs) for the NLA experiment

Built 2026-09-09 by `scripts/redocred_pilot.py` (deterministic; re-running reproduces
`pilot.jsonl` byte-for-byte — verified with `cmp` after two consecutive runs). Version 2: adds the single-quote pairing rule, the unclosed-quote/
bracket cut rule, the DocRED-hole exclusions, `entities_by_type` and per-relation `stated`.

## Files here
- `pilot.jsonl` — one JSON object per pilot document (100 lines). Fields:
  `pilot_id` (0–99), `split` (`dev` = pilot_id 0–19, `eval` = 20–99), `source_split` (`dev`/`test`
  of Re-DocRED), `source_index` (index in that file), `title`, `prefix_text`, `position`
  (= `n_tokens_prefix` − 1, the extraction index under the 7B tokenizer), `final_word`,
  `dropped_tail` (the characters after the cut up to the end of the last kept sentence, for audit),
  `cut_moved` (bool) and `cut_reason` (why the cut moved back, or null), `n_tokens_prefix`
  (Qwen2.5-7B-Instruct), `n_tokens_prefix_27b` (Qwen3.6-27B NLA av_base tokenizer), `n_tokens_full`,
  `n_sents_kept`, `n_sents_kept_before_cut_move`, `n_sents_total`, `truncated` (token-budget
  truncation only), `entities`, `relations`, `n_entities`, `n_relations`, `n_relations_stated`,
  `by_type` (counts), `entities_by_type` (type → list of entity names).
  - `entities[]`: `{id (vertexSet index), name (first in-prefix mention in document order), type,
    aliases (distinct in-prefix mention strings), in_prefix_mentions [{sent_id, pos, name,
    char_start, char_end, surface}]}` — `char_*` are offsets into `prefix_text`; `surface` is
    `prefix_text[char_start:char_end]`. Only entities with ≥1 mention in a kept sentence (before
    the cut; a mention in the dropped tail of the final sentence still counts as in-prefix — see note §5).
  - `relations[]`: `{h_id, t_id, h_name, t_name, r_pid, r_name, evidence, stated}` — only relations
    whose head and tail both have an in-prefix mention and whose evidence sentence ids (if any) are
    all kept. `stated` = true iff `evidence` is non-empty and all evidence sentences are kept; 1851
    of 3348 pilot relations (55%) have an empty evidence list in Re-DocRED and are `stated: false`.
- `pilot_summary.csv` — one row per doc: pilot_id, split, title, n_tokens_prefix,
  n_tokens_prefix_27b, sents_kept/total, truncated, cut_moved, n_entities, n_relations,
  n_relations_stated, entity-type histogram, final_word.
- `stats.json` — funnel counts (with the lists of hole-pattern, fragment and cut-moved pool docs),
  distributions (pilot, pool, qualifying), detokenization examples, mention-surface mismatches, hashes.

## Source files (raw copies are NOT in the repo)
Default raw dir: `data/redocred_raw/` (gitignored; download the files below into it,
or pass `--raw-dir` to the script to point elsewhere; the script checks the SHA256s below and warns on mismatch).

| file | source | sha256 |
|---|---|---|
| `dev_revised.json` (3,245,588 B) | `https://raw.githubusercontent.com/tonytan48/Re-DocRED/main/data/dev_revised.json`, repo HEAD `ccfb54f5ddf5836027c87badda10f6dfc56efaac` (2023-08-21); data files last changed in commit `0b1584e7` (2022-05-23) | `051ee1d057204a5d08ef5502beacdadf191245b5eaf0e29ec2c607cf002c016f` |
| `test_revised.json` (3,207,228 B) | same repo/commit, `data/test_revised.json` | `ea673b7ef91e16510d7eb88863d3903e47318dc5653ef361266f27ae3b84723c` |
| `train_revised.json` (18,602,933 B) | same repo/commit — downloaded, **not used** (sampling pool is dev+test only) | `c40a137a1c57f3e2daaf5ebf511eaa1518f9305ffaa3052d46271921a91826f2` |
| `rel_info.json` (96 P-ids) | Not in the Re-DocRED repo. Taken from the HF dataset mirror of DocRED: `https://huggingface.co/datasets/thunlp/docred/resolve/7985b4e0371e6c61a756feb41b7b27becf71c666/data/rel_info.json.gz` (gunzipped) | `5ecf4e5e55c179fc83a3a3d19baa01efffecb26ba5edc0b4ac5a54ddf61fe3de` |

Re-DocRED: Tan, Zhou, Wang, Bing & Joty, "Revisiting DocRED — Addressing the False Negative Problem
in Relation Extraction", EMNLP 2022. Dev and test are the fully re-annotated splits; 500 docs each,
pool = 1000.

License: Re-DocRED and DocRED are released under the MIT license (GitHub `tonytan48/Re-DocRED`,
`thunlp/DocRED`). The document text is from English Wikipedia (CC BY-SA).

Tokenizers (from the local HF cache, `local_files_only`):
- 7B: `Qwen/Qwen2.5-7B-Instruct`, snapshot `a09a35458c702b33eeacc393d103063234e8bc28`.
- 27B: `ceselder/qwen3.6-27b-nla-rl`, snapshot `5a13b7ec21a69fcdd0fb24d5edbe96a92aef4b9f`, subdir `av_base/`.
  Loaded fine, so `n_tokens_prefix_27b` is filled for all 100 docs (7B − 27B count ranges −1…+15
  tokens; identical for 35 docs).

## Rules (applied identically to every doc)
**Detokenization.** Tokens of all sentences are joined with single spaces, then spaces are deleted by
rule; every rule is a deletion, so token→character offsets are exact. From the brief: no space before
`. , ; : ! ? ) ] %`; no space after `( [`; straight `"` handled as open/close pairs (parity resets at
each DocRED sentence; opening quote loses the space after it, closing quote the space before);
runs of spaces collapsed. Rules the brief did not settle (decisions; the single-quote rule was
specified by the coordinator in the v2 request):
1. **single quotes**: tokens starting with `'` (`'s`, `'m`, `'ll`, `'grom`) attach to the preceding
   word; a *standalone* `'` token is an opening quotation mark when it can be paired with a later
   standalone `'` in the same sentence (open/close alternate, and the opener must be followed by a
   word), otherwise it is a closing quote or plural possessive and attaches to the preceding word.
   `known as ' kalamaki '` → `known as 'kalamaki'`; `Illinois ' 12th` → `Illinois' 12th`. (The
   coordinator's phrasing "opening when it follows whitespace" is undefined on space-joined DocRED
   tokens, where every token follows whitespace; pairing is the operational version.)
2. curly quotes: `“ ‘` behave like an opening quote, `” ’` like a closing quote/apostrophe;
3. a standalone `-` (or `.-`) token between two word tokens is joined tightly (`ill - gotten` → `ill-gotten`);
4. `$ £` take no space after them;
5. the DocRED contraction token `n't` attaches to the preceding word (`Ai n't` → `Ain't`);
6. whitespace-only tokens (`\xa0`, ` `) are dropped; `\xa0` inside a token becomes a space;
7. en/em dashes, `/`, `&`, `…` keep their spaces as in DocRED (so date ranges read `1845 – 1923`).

**Prefix.** Tokenize the full detokenized document with the 7B tokenizer (`add_special_tokens=False`).
If ≤ 512 tokens, keep all sentences; otherwise keep the longest prefix of complete DocRED sentences
whose text (including that sentence's final punctuation) is ≤ 512 tokens. The prefix text is then cut
immediately after the last non-punctuation token (token containing an alphanumeric character) of the
final kept sentence, so it ends on a word.

**Cut move (v2).** If that final word sits inside an unclosed quote or bracket — an opener in the kept
text with no closer before the cut — the cut moves back to the last word before that opener; if the
opener's sentence has no word before it, that sentence (and any later ones) is dropped and the check
repeats. Openers/closers: `(`/`)`, `[`/`]`, `“`/`”`, standalone `‘`/`’`, tracked with a
document-level stack per kind; straight `"` by document-level parity (odd count = open); standalone
`'` by the per-sentence pairing of rule 1. Recorded as `cut_moved`, `cut_reason`,
`n_sents_kept_before_cut_move`. `position = n_tokens_prefix − 1`, `n_tokens_prefix` = 7B token count
of the cut text.

**Exclusions (v2).** A doc is excluded if its cut prefix text matches the DocRED template-hole regex
`\b(about|approximately|exceeds|measures|of|is|are)\s+(in (width|length|height|diameter)|(east|west|north|south) of)\b`
(e.g. "located about east of Madagascar", where `{{convert}}` distances were stripped) or contains
the `(;` fragment (a stripped non-Latin name/IPA parenthetical).

**Filter.** In order: `n_tokens_prefix ≥ 50` → hole-regex exclusion → `(;` exclusion → `≥ 3`
in-prefix entities → `≥ 3` in-prefix relations (definitions above).

**Sample.** `random.Random(20260909).sample(qualifying_sorted_by_(source_split, source_index), 100)`;
the first 20 in sampled order are `dev`, the remaining 80 `eval`.

## Funnel (v2)
| step | docs | removed |
|---|---|---|
| pool (Re-DocRED dev 500 + test 500) | 1000 | |
| `n_tokens_prefix ≥ 50` | 996 | 4 docs emptied by the cut-move rule: dev#107 *Ali Kuli Khan Khattak* (unclosed `(` in sentence 0 → 7 tokens), dev#419 *Li Jiancheng* (3 tokens), test#319 *Klassics with a "K"* (49), test#337 *George Nostrand* (38) |
| hole-regex exclusion | 987 | 9: dev#92 *Regal Mountain*, dev#248 *Piton des Neiges*, dev#290 *Roald*, test#65 *Cloughton*, test#66 *Beaverton, Oregon*, test#67 *Gambier Island*, test#163 *Malolos*, test#181, test#393 *Shire of Murray* (list with matched snippets in `stats.json`) |
| `(;` fragment exclusion | 955 | 32 (no overlap with the regex hits) |
| `≥ 3` in-prefix entities | 954 | test#474 *Gromshin Heights* (2 entities) |
| `≥ 3` in-prefix relations | **951** | dev#455 *Rachel Perry (artist)* (cut move dropped 2 of 4 sentences → 2 relations), dev#492 *Scimitar oryx* (0 labels), test#300 *Fugue* (all 22 relations outside the 12/15 kept sentences) |
| qualifying by split | dev 475 / test 476 | |
| sampled | 100 (pilot-dev 20 / pilot-eval 80; source dev 47 / test 53) | |

Cut-move rule in the pool: fires in 87 of 1000 docs; 68 of those lose no sentence (the cut moves
within the final sentence), 5 lose one, 14 lose two or more (worst: dev#457 *Miguel Hidalgo y
Costilla* 8 → 1 sentences from an unclosed `(` in sentence 0; the 8 docs with an odd total `"` count
are the main victims of document-level quote parity). Token-budget truncation: 10 pool docs, 2 in the
pilot (pilot 25 *Faliero coup* 13/15 sentences, pilot 27 *Clandestine literature* 13/14).

Overlap with the v1 sample (before these rules): 15 of 100 docs.

## Regenerate
```
uv run python scripts/redocred_pilot.py            # ~10 s; needs the two tokenizers in the HF cache
```
