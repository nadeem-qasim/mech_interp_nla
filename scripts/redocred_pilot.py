#!/usr/bin/env python
"""Build the 100-document Re-DocRED pilot set for the NLA experiment.

Deterministic: re-running on the same raw files reproduces the same sample.

Inputs (raw, not in the repo; see data/redocred_pilot/README.md for URLs and hashes):
    <raw_dir>/dev_revised.json, <raw_dir>/test_revised.json   (Re-DocRED, Tan et al. 2022)
    <raw_dir>/rel_info.json                                    (DocRED P-id -> relation name)
Outputs (in the repo):
    data/redocred_pilot/pilot.jsonl          one object per pilot doc
    data/redocred_pilot/pilot_summary.csv    one row per pilot doc
    data/redocred_pilot/stats.json           funnel counts, distributions, detok examples

Rules (pre-registered in the task brief; every doc is processed identically):
  Detokenize: join DocRED tokens with single spaces, then delete spaces with regex-style rules
    (no space before . , ; : ! ? ' ) ] % ; none after ( [ ; " handled as open/close pairs per
    sentence; double spaces collapsed).  Extra rules beyond the brief are listed in EXTRA_RULES.
    Every rule is a *deletion* of characters from the space-joined string, so the token -> char
    offset map is exact.
  Prefix: tokenize the full detokenized text with Qwen/Qwen2.5-7B-Instruct (add_special_tokens=
    False). If <= MAX_TOKENS keep whole; else keep the longest prefix of complete DocRED sentences
    whose tokenized text is <= MAX_TOKENS.  The prefix text is cut immediately after the last
    non-punctuation token (token containing an alphanumeric char) of the final kept sentence.
    Extraction position = n_tokens_prefix - 1.
  Filter: n_tokens_prefix >= MIN_TOKENS; >= MIN_ENTITIES entities with a mention in kept
    sentences; >= MIN_RELATIONS relations whose head and tail both have an in-prefix mention and
    whose evidence sentences (if any) are all kept.
    Cut move: if the final word sits inside an unclosed quote/bracket (an opener in the kept text
    with no closer before the cut), the cut moves back to the last word before that opener; if that
    empties the sentence, the sentence is dropped and the check repeats (cut_moved / cut_reason).
  Exclusions: docs whose prefix text matches HOLE_RE (DocRED {{convert}}-template holes such as
    "about east of") or contains the "(;" fragment are excluded.
  Sample: random.Random(SEED).sample(qualifying sorted by (split, index), N_PILOT);
    first 20 in sampled order -> "dev", rest -> "eval".
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import random
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RAW = REPO / "data" / "redocred_raw"  # raw Re-DocRED JSON files (not in the repo; see data/redocred_pilot/README.md)
OUT_DIR = REPO / "data" / "redocred_pilot"

SEED = 20260909
N_PILOT = 100
N_DEV = 20
MAX_TOKENS = 512
MIN_TOKENS = 50
MIN_ENTITIES = 3
MIN_RELATIONS = 3
TOK_7B = "Qwen/Qwen2.5-7B-Instruct"
TOK_27B_REPO = "ceselder/qwen3.6-27b-nla-rl"  # tokenizer lives in av_base/ inside the snapshot

EXPECTED_SHA256 = {
    "dev_revised.json": "051ee1d057204a5d08ef5502beacdadf191245b5eaf0e29ec2c607cf002c016f",
    "test_revised.json": "ea673b7ef91e16510d7eb88863d3903e47318dc5653ef361266f27ae3b84723c",
    "rel_info.json": "5ecf4e5e55c179fc83a3a3d19baa01efffecb26ba5edc0b4ac5a54ddf61fe3de",
}

# ----------------------------------------------------------------------------- detokenization
# Rules from the brief.
NO_SPACE_BEFORE = set(".,;:!?)]%")  # ' handled separately (contractions attach; standalone ' paired)
NO_SPACE_AFTER = set("([")
# Extra rules (decisions not settled by the brief; see README):
EXTRA_RULES = [
    "curly quotes: “ ‘ behave like an opening \", ” ’ like a closing quote / apostrophe",
    "standalone '-' (and '.-') token between two word tokens is joined tightly: 'ill - gotten' -> 'ill-gotten'",
    "currency signs $ £ take no space after them: '$ 7 billion' -> '$7 billion'",
    "the DocRED contraction token n't attaches to the preceding word: 'Ai n't' -> 'Ain't'",
    "tokens starting with ' ('s, 'm, 'grom) attach to the preceding word; a standalone ' token is an opening "
    "quotation mark when it is paired with a later standalone ' in the same sentence (open/close alternate), "
    "otherwise it is a closing quote / plural possessive and attaches to the preceding word: "
    "\"known as ' kalamaki '\" -> \"known as 'kalamaki'\", \"Illinois ' 12th\" -> \"Illinois' 12th\"",
    "tokens that are only whitespace (NBSP '\\xa0', ' ') are dropped; '\\xa0' inside a token becomes a space",
    "en/em dashes (– —), '/', '&', '…' keep their spaces (left as in DocRED)",
]
NO_SPACE_BEFORE |= set("”’")
NO_SPACE_AFTER |= set("“‘$£")


def is_word(tok: str) -> bool:
    """Non-punctuation token = contains at least one alphanumeric character."""
    return any(ch.isalnum() for ch in tok)


def detokenize(sents: list[list[str]]):
    """Return (text, tok_char: dict (sent_id, tok_id) -> (start, end), sent_char: list of (start, end)).

    Implementation: build the space-joined string, decide a set of character indices to delete,
    delete them, and map every original index through the deletion mask.  All rules are
    deletions, so offsets stay exact.
    """
    toks = []  # (sent_id, tok_id, token text after per-token cleanup)
    for si, s in enumerate(sents):
        for ti, t in enumerate(s):
            toks.append((si, ti, t.replace("\xa0", " ")))
    joined = " ".join(t for _, _, t in toks)
    n = len(joined)
    delete = [False] * n
    spans = {}
    pos = 0
    for si, ti, t in toks:
        spans[(si, ti)] = (pos, pos + len(t))
        pos += len(t) + 1

    # whitespace-only tokens are dropped entirely
    for (si, ti), (a, b) in spans.items():
        if not toks_text(sents, si, ti).strip():
            for i in range(a, b):
                delete[i] = True

    # rule: no space before X
    for m in re.finditer(r" +(?=[" + re.escape("".join(sorted(NO_SPACE_BEFORE))) + "])", joined):
        for i in range(m.start(), m.end()):
            delete[i] = True
    # rule: tokens beginning with an apostrophe attach to the preceding word; standalone ' is paired
    for si, s in enumerate(sents):
        for ti, t in enumerate(s):
            if t.startswith("'") and t != "'":
                a, _ = spans[(si, ti)]
                i = a - 1
                while i >= 0 and joined[i] == " ":
                    delete[i] = True
                    i -= 1
    for si, s in enumerate(sents):
        for ti, role in single_quote_roles(s).items():
            a, b = spans[(si, ti)]
            if role == "open":
                i = b
                while i < n and joined[i] == " ":
                    delete[i] = True
                    i += 1
            else:
                i = a - 1
                while i >= 0 and joined[i] == " ":
                    delete[i] = True
                    i -= 1
    # rule: no space before the contraction token n't
    for m in re.finditer(r" +(?=n't(?: |$))", joined):
        for i in range(m.start(), m.end()):
            delete[i] = True
    # rule: no space after X
    for m in re.finditer(r"(?<=[" + re.escape("".join(sorted(NO_SPACE_AFTER))) + "]) +", joined):
        for i in range(m.start(), m.end()):
            delete[i] = True
    # rule: straight double quotes in open/close pairs, parity reset at each sentence
    for si, s in enumerate(sents):
        open_next = True
        for ti, t in enumerate(s):
            if t == '"':
                a, b = spans[(si, ti)]
                if open_next:
                    i = b
                    while i < n and joined[i] == " ":
                        delete[i] = True
                        i += 1
                else:
                    i = a - 1
                    while i >= 0 and joined[i] == " ":
                        delete[i] = True
                        i -= 1
                open_next = not open_next
    # rule: standalone hyphen between two word tokens joins tightly
    flat = [(si, ti, t) for si, ti, t in toks]
    for k, (si, ti, t) in enumerate(flat):
        if t in ("-", ".-") and 0 < k < len(flat) - 1 and is_word(flat[k - 1][2]) and is_word(flat[k + 1][2]):
            a, b = spans[(si, ti)]
            i = a - 1
            while i >= 0 and joined[i] == " ":
                delete[i] = True
                i -= 1
            i = b
            while i < n and joined[i] == " ":
                delete[i] = True
                i += 1
    # collapse runs of surviving spaces to one; strip leading/trailing
    prev_space = True  # so leading spaces are deleted
    for i, ch in enumerate(joined):
        if delete[i]:
            continue
        if ch == " ":
            if prev_space:
                delete[i] = True
            prev_space = True
        else:
            prev_space = False
    i = n - 1
    while i >= 0 and (delete[i] or joined[i] == " "):
        delete[i] = True
        i -= 1

    # build the new string and the index map
    newpos = [0] * (n + 1)
    out = []
    for i, ch in enumerate(joined):
        newpos[i] = len(out)
        if not delete[i]:
            out.append(ch)
    newpos[n] = len(out)
    text = "".join(out)

    tok_char = {}
    for key, (a, b) in spans.items():
        # skip deleted leading chars of the token (only whitespace tokens are deleted)
        aa = a
        while aa < b and delete[aa]:
            aa += 1
        bb = b
        while bb > aa and delete[bb - 1]:
            bb -= 1
        tok_char[key] = (newpos[aa], newpos[bb]) if aa < bb else (newpos[a], newpos[a])
    sent_char = []
    for si, s in enumerate(sents):
        starts = [tok_char[(si, ti)][0] for ti in range(len(s)) if tok_char[(si, ti)][1] > tok_char[(si, ti)][0]]
        ends = [tok_char[(si, ti)][1] for ti in range(len(s)) if tok_char[(si, ti)][1] > tok_char[(si, ti)][0]]
        sent_char.append((min(starts), max(ends)) if starts else (0, 0))
    return text, tok_char, sent_char


def single_quote_roles(sent: list[str]) -> dict[int, str]:
    """Role of each standalone ' token in a sentence: 'open' / 'close' (paired, alternating) or
    'close' for an unpaired leftover (plural possessive / closing quote)."""
    idx = [ti for ti, t in enumerate(sent) if t == "'"]
    roles = {}
    k = 0
    while k < len(idx):
        if k + 1 < len(idx) and idx[k] + 1 < len(sent) and is_word(sent[idx[k] + 1]):
            roles[idx[k]] = "open"
            roles[idx[k + 1]] = "close"
            k += 2
        else:
            roles[idx[k]] = "close"
            k += 1
    return roles


# ----------------------------------------------------------------------------- cut rule helpers
OPENERS = {"(": ")", "[": "]", "“": "”", "‘": "’"}
HOLE_RE = re.compile(r"\b(about|approximately|exceeds|measures|of|is|are)\s+(in (width|length|height|diameter)|(east|west|north|south) of)\b")
FRAGMENT = "(;"


def unclosed_openers(sents, n_kept, last_word_ti):
    """Openers in the kept text (sentences < n_kept, plus tokens <= last_word_ti of the final one)
    that have no closer before the cut.  Brackets and curly quotes use a document-level stack per
    kind; straight \" uses document-level parity (odd = open); standalone ' uses the per-sentence
    pairing of the detokenizer.  Returns a list of (sent_id, tok_id, opener) sorted by position."""
    stacks = collections.defaultdict(list)
    dq_open = None
    for si in range(n_kept):
        sent = sents[si]
        roles = single_quote_roles(sent)
        limit = len(sent) if si < n_kept - 1 else last_word_ti + 1
        for ti in range(limit):
            t = sent[ti]
            if t == "'":
                if roles[ti] == "open":
                    stacks["'"].append((si, ti))
                elif stacks["'"]:
                    stacks["'"].pop()
                continue
            for ch in t:
                if ch in OPENERS:
                    stacks[ch].append((si, ti))
                elif ch in OPENERS.values():
                    k = next((o for o, c in OPENERS.items() if c == ch), None)
                    if k == "’" and t != "’":  # apostrophe inside a word, not a closer
                        continue
                    if stacks[k]:
                        stacks[k].pop()
                elif ch == '"':
                    if dq_open is None:
                        dq_open = (si, ti)
                    else:
                        dq_open = None
    out = [(si, ti, k) for k, st in stacks.items() for si, ti in st]
    if dq_open is not None:
        out.append((dq_open[0], dq_open[1], '"'))
    return sorted(out)


def toks_text(sents, si, ti):
    return sents[si][ti].replace("\xa0", " ")


# ----------------------------------------------------------------------------- helpers
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_tokenizers():
    from transformers import AutoTokenizer

    tok7 = AutoTokenizer.from_pretrained(TOK_7B)
    tok27 = None
    try:
        from huggingface_hub import snapshot_download

        snap = snapshot_download(TOK_27B_REPO, allow_patterns=["av_base/tokenizer*", "av_base/*.json"], local_files_only=True)
        tok27 = AutoTokenizer.from_pretrained(str(Path(snap) / "av_base"))
    except Exception as e:  # not cached -> skip
        print(f"[warn] 27B tokenizer not available ({type(e).__name__}: {e}); n_tokens_prefix_27b will be null", file=sys.stderr)
    return tok7, tok27


def ntok(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def process_doc(doc, split, idx, tok7, tok27, rel_info):
    sents = doc["sents"]
    text, tok_char, sent_char = detokenize(sents)
    n_full = ntok(tok7, text)
    n_sents = len(sents)
    if n_full <= MAX_TOKENS:
        n_kept = n_sents
    else:
        n_kept = 0
        for k in range(1, n_sents + 1):
            if ntok(tok7, text[: sent_char[k - 1][1]]) <= MAX_TOKENS:
                n_kept = k
            else:
                break
    rec = {
        "source_split": split,
        "source_index": idx,
        "title": doc["title"],
        "n_sents_total": n_sents,
        "n_sents_kept": n_kept,
        "n_tokens_full": n_full,
        "truncated": n_kept < n_sents,
    }
    if n_kept == 0:
        rec.update(prefix_text="", position=-1, final_word=None, n_tokens_prefix=0, n_tokens_prefix_27b=None,
                   entities=[], relations=[], n_entities=0, n_relations=0, by_type={}, entities_by_type={},
                   dropped_tail="", cut_moved=False, cut_reason=None, hole_pattern=None, has_fragment=False)
        return rec
    # cut after the last word of the final kept sentence; move back if it sits inside an unclosed
    # quote / bracket (opener in the kept text with no closer before the cut)
    cut_moved = False
    cut_reasons = []
    n_kept_before_move = n_kept
    wt = None  # token index of the final word in the final kept sentence (None = last word of it)
    while n_kept > 0:
        last = n_kept - 1
        if wt is None:
            word_ids = [ti for ti, t in enumerate(sents[last]) if is_word(t)]
            if not word_ids:  # sentence with no word token at all: drop it
                n_kept -= 1
                cut_moved = True
                cut_reasons.append(f"s{last} has no word token")
                continue
            wt = word_ids[-1]
        unc = unclosed_openers(sents, n_kept, wt)  # pairing/parity always on the full sentences
        if not unc:
            break
        si, ti, opener = unc[0]  # earliest unclosed opener
        cut_moved = True
        cut_reasons.append(f"unclosed {opener!r} opened at s{si} t{ti} ({sents[si][ti]!r}); cut was after s{last} t{wt} ({sents[last][wt]!r})")
        before = [x for x in range(ti) if is_word(sents[si][x])]
        if before:  # cut after the last word before the opener (drops any later sentences)
            n_kept, wt = si + 1, before[-1]
        else:  # empties sentence si: drop it and repeat from the previous sentence's last word
            n_kept, wt = si, None
    if n_kept == 0:
        rec.update(prefix_text="", position=-1, final_word=None, n_tokens_prefix=0, n_tokens_prefix_27b=None,
                   entities=[], relations=[], n_entities=0, n_relations=0, by_type={}, entities_by_type={},
                   dropped_tail="", cut_moved=True, cut_reason="; ".join(cut_reasons), hole_pattern=None,
                   has_fragment=False, n_sents_kept=0, n_relations_stated=0)
        return rec
    cut = tok_char[(last, wt)][1]
    final_word = sents[last][wt]
    rec["n_sents_kept"] = n_kept
    rec["n_sents_kept_before_cut_move"] = n_kept_before_move
    rec["truncated"] = n_kept_before_move < n_sents  # token-budget truncation only; cut moves are separate
    rec["cut_moved"] = cut_moved
    rec["cut_reason"] = "; ".join(cut_reasons) if cut_reasons else None
    prefix = text[:cut]
    m = HOLE_RE.search(prefix)
    rec["hole_pattern"] = prefix[max(0, m.start() - 30): m.end() + 10] if m else None
    rec["has_fragment"] = FRAGMENT in prefix
    n_prefix = ntok(tok7, prefix)
    rec["prefix_text"] = prefix
    rec["final_word"] = final_word
    rec["dropped_tail"] = text[cut: sent_char[last][1]]  # trailing punctuation removed by the cut
    rec["n_tokens_prefix"] = n_prefix
    rec["position"] = n_prefix - 1
    rec["n_tokens_prefix_27b"] = ntok(tok27, prefix) if tok27 is not None else None

    kept = set(range(n_kept))
    entities = []
    ent_in = set()
    for eid, ment in enumerate(doc["vertexSet"]):
        inm = sorted((m for m in ment if m["sent_id"] in kept), key=lambda m: (m["sent_id"], m["pos"][0]))
        if not inm:
            continue
        ent_in.add(eid)
        mentions = []
        for m in inm:
            a = tok_char[(m["sent_id"], m["pos"][0])][0]
            b = tok_char[(m["sent_id"], m["pos"][1] - 1)][1]
            mentions.append({"sent_id": m["sent_id"], "pos": m["pos"], "name": m["name"],
                             "char_start": a, "char_end": b, "surface": text[a:b]})
        entities.append({
            "id": eid,
            "name": inm[0]["name"],
            "type": collections.Counter(m["type"] for m in inm).most_common(1)[0][0],
            "aliases": sorted(set(m["name"] for m in inm)),
            "in_prefix_mentions": mentions,
        })
    relations = []
    for lab in doc["labels"]:
        if lab["h"] in ent_in and lab["t"] in ent_in and all(e in kept for e in lab.get("evidence", [])):
            ev = lab.get("evidence", [])
            relations.append({
                "h_id": lab["h"], "t_id": lab["t"],
                "h_name": next(e["name"] for e in entities if e["id"] == lab["h"]),
                "t_name": next(e["name"] for e in entities if e["id"] == lab["t"]),
                "r_pid": lab["r"], "r_name": rel_info.get(lab["r"], lab["r"]),
                "evidence": ev,
                "stated": bool(ev) and all(e in kept for e in ev),
            })
    rec["entities"] = entities
    rec["relations"] = relations
    rec["n_entities"] = len(entities)
    rec["n_relations"] = len(relations)
    rec["by_type"] = dict(sorted(collections.Counter(e["type"] for e in entities).items()))
    ebt = collections.defaultdict(list)
    for e in entities:
        ebt[e["type"]].append(e["name"])
    rec["entities_by_type"] = dict(sorted(ebt.items()))
    rec["n_relations_stated"] = sum(r["stated"] for r in relations)
    return rec


def dist(xs):
    xs = sorted(xs)
    if not xs:
        return {}
    q = lambda p: xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]
    return {"n": len(xs), "min": xs[0], "p10": q(0.1), "p25": q(0.25), "median": q(0.5), "p75": q(0.75),
            "p90": q(0.9), "max": xs[-1], "mean": round(statistics.mean(xs), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    hashes = {}
    for fn, exp in EXPECTED_SHA256.items():
        h = sha256(args.raw_dir / fn)
        hashes[fn] = h
        if h != exp:
            print(f"[warn] {fn}: sha256 {h} != expected {exp}", file=sys.stderr)
    rel_info = json.load(open(args.raw_dir / "rel_info.json"))
    pool = []
    for split in ("dev", "test"):
        docs = json.load(open(args.raw_dir / f"{split}_revised.json"))
        pool += [(split, i, d) for i, d in enumerate(docs)]
    print(f"pool: {len(pool)} docs", file=sys.stderr)

    tok7, tok27 = load_tokenizers()
    recs = [process_doc(d, split, i, tok7, tok27, rel_info) for split, i, d in pool]

    # ---- funnel (sequential) and per-criterion counts
    f_tok = [r for r in recs if r["n_tokens_prefix"] >= MIN_TOKENS]
    f_hole = [r for r in f_tok if r["hole_pattern"] is None]
    f_frag = [r for r in f_hole if not r["has_fragment"]]
    f_ent = [r for r in f_frag if r["n_entities"] >= MIN_ENTITIES]
    f_rel = [r for r in f_ent if r["n_relations"] >= MIN_RELATIONS]
    funnel = {
        "pool": len(recs),
        "pool_by_split": dict(collections.Counter(r["source_split"] for r in recs)),
        "after_min_tokens": len(f_tok),
        "after_hole_pattern_exclusion": len(f_hole),
        "after_fragment_exclusion": len(f_frag),
        "after_min_entities": len(f_ent),
        "after_min_relations": len(f_rel),
        "fail_min_tokens_alone": sum(r["n_tokens_prefix"] < MIN_TOKENS for r in recs),
        "hole_pattern_alone": sum(r["hole_pattern"] is not None for r in recs),
        "fragment_alone": sum(r["has_fragment"] for r in recs),
        "hole_or_fragment": sum(r["hole_pattern"] is not None or r["has_fragment"] for r in recs),
        "hole_pattern_docs": [(r["source_split"], r["source_index"], r["title"], r["hole_pattern"]) for r in recs if r["hole_pattern"] is not None],
        "fragment_docs": [(r["source_split"], r["source_index"], r["title"]) for r in recs if r["has_fragment"]],
        "cut_moved_in_pool": sum(r.get("cut_moved", False) for r in recs),
        "cut_moved_docs": [(r["source_split"], r["source_index"], r["title"], r.get("cut_reason")) for r in recs if r.get("cut_moved")],
        "fail_min_entities_alone": sum(r["n_entities"] < MIN_ENTITIES for r in recs),
        "fail_min_relations_alone": sum(r["n_relations"] < MIN_RELATIONS for r in recs),
        "truncated_in_pool": sum(r["truncated"] for r in recs),
        "qualifying_by_split": dict(collections.Counter(r["source_split"] for r in f_rel)),
    }
    qualifying = sorted(f_rel, key=lambda r: (r["source_split"], r["source_index"]))
    sample = random.Random(SEED).sample(qualifying, N_PILOT)

    key_order = ["pilot_id", "split", "source_split", "source_index", "title", "prefix_text", "position", "final_word",
                 "dropped_tail", "cut_moved", "cut_reason", "n_tokens_prefix", "n_tokens_prefix_27b", "n_tokens_full",
                 "n_sents_kept", "n_sents_kept_before_cut_move", "n_sents_total", "truncated", "entities", "relations",
                 "n_entities", "n_relations", "n_relations_stated", "by_type", "entities_by_type"]
    out = []
    for pid, r in enumerate(sample):
        o = {"pilot_id": pid, "split": "dev" if pid < N_DEV else "eval"}
        o.update({k: r[k] for k in key_order if k in r})
        out.append({k: o[k] for k in key_order})
    with open(args.out_dir / "pilot.jsonl", "w") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    with open(args.out_dir / "pilot_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pilot_id", "split", "title", "n_tokens_prefix", "n_tokens_prefix_27b", "sents_kept/total",
                    "truncated", "cut_moved", "n_entities", "n_relations", "n_relations_stated", "entity_types", "final_word"])
        for o in out:
            w.writerow([o["pilot_id"], o["split"], o["title"], o["n_tokens_prefix"], o["n_tokens_prefix_27b"],
                        f"{o['n_sents_kept']}/{o['n_sents_total']}", int(o["truncated"]), int(o["cut_moved"]), o["n_entities"],
                        o["n_relations"], o["n_relations_stated"], " ".join(f"{k}:{v}" for k, v in o["by_type"].items()), o["final_word"]])

    # ---- stats for the review note
    def dists(rs):
        return {
            "n_tokens_prefix": dist([r["n_tokens_prefix"] for r in rs]),
            "n_tokens_prefix_27b": dist([r["n_tokens_prefix_27b"] for r in rs if r["n_tokens_prefix_27b"] is not None]),
            "n_tokens_full": dist([r["n_tokens_full"] for r in rs]),
            "n_sents_kept": dist([r["n_sents_kept"] for r in rs]),
            "n_sents_total": dist([r["n_sents_total"] for r in rs]),
            "n_entities": dist([r["n_entities"] for r in rs]),
            "n_relations": dist([r["n_relations"] for r in rs]),
            "n_truncated": sum(r["truncated"] for r in rs),
            "n_cut_moved": sum(r.get("cut_moved", False) for r in rs),
            "n_relations_stated": dist([r.get("n_relations_stated", 0) for r in rs]),
            "n_whole": sum(not r["truncated"] for r in rs),
            "entity_types_total": dict(sorted(collections.Counter(e["type"] for r in rs for e in r["entities"]).items())),
            "relation_names_top": collections.Counter(x["r_name"] for r in rs for x in r["relations"]).most_common(15),
            "dropped_tail_counts": dict(collections.Counter(r["dropped_tail"] for r in rs).most_common()),
            "final_word_ends_with_period": sum(bool(r["final_word"]) and r["final_word"].endswith(".") for r in rs),
            "relations_with_empty_evidence": sum(not x["evidence"] for r in rs for x in r["relations"]),
            "relations_total": sum(len(r["relations"]) for r in rs),
        }

    # detokenization examples: pick 3 pilot docs whose first sentence exercises the rules
    def score(sent):
        return sum(t in ('"', "(", ")", "'s", "-", ",", ";", ":") for t in sent)
    cands = sorted(out, key=lambda o: -score(pool_lookup(pool, o)["sents"][0]))[:3]
    detok_examples = []
    for o in cands:
        d = pool_lookup(pool, o)
        s0 = d["sents"][0]
        text, tok_char, sent_char = detokenize(d["sents"])
        detok_examples.append({"pilot_id": o["pilot_id"], "title": o["title"], "tokens": s0,
                               "space_joined": " ".join(s0), "detokenized": text[sent_char[0][0]: sent_char[0][1]]})

    # sanity: every in-prefix mention surface must match its name up to spacing
    mism = []
    for o in out:
        for e in o["entities"]:
            for m in e["in_prefix_mentions"]:
                if re.sub(r"\s+", "", m["surface"]) != re.sub(r"\s+", "", m["name"]):
                    mism.append((o["pilot_id"], e["id"], m["name"], m["surface"]))
    stats = {
        "hashes": hashes, "seed": SEED, "max_tokens": MAX_TOKENS, "min_tokens": MIN_TOKENS,
        "extra_detok_rules": EXTRA_RULES, "funnel": funnel,
        "pilot": dists(out), "pilot_by_split": {s: dists([o for o in out if o["split"] == s]) for s in ("dev", "eval")},
        "pool": dists(recs), "qualifying": dists(qualifying),
        "detok_examples": detok_examples, "mention_surface_mismatches": mism,
        "odd_quote_sentences_in_pool": sum(s.count('"') % 2 for _, _, d in pool for s in d["sents"]),
        "tok27_available": tok27 is not None,
    }
    json.dump(stats, open(args.out_dir / "stats.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(funnel, indent=1), file=sys.stderr)
    print(f"wrote {len(out)} docs to {args.out_dir}", file=sys.stderr)


def pool_lookup(pool, o):
    for split, i, d in pool:
        if split == o["source_split"] and i == o["source_index"]:
            return d
    raise KeyError


if __name__ == "__main__":
    main()
