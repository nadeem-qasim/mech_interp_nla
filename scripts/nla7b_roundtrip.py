"""Round-trip smoke test for the kitft Qwen2.5-7B layer-20 NLA pair on MPS.

    uv run python scripts/nla7b_roundtrip.py [--positions 34 35 38 40] [--n-new 200]

Three phases, loaded one model at a time so the 48 GB box never holds two 7B
models plus the 15 GB target:

  1. target   Qwen/Qwen2.5-7B-Instruct: greedy reply to "What are you hiding?",
              collect layer-20 residual (= hidden_states[21], output of block 20)
              at chosen sequence positions.  Mirrors
              ~/tools/natural_language_autoencoders/examples/qwen7b_layer20_step4200.txt
              so decodes can be compared line-for-line.
  2. AV       kitft/nla-qwen2.5-7b-L20-av: inject the vector (rescaled to
              injection_scale from nla_meta.yaml) in place of the marker token's
              *embedding*, greedy-generate the <explanation>.
  3. AR       kitft/nla-qwen2.5-7b-L20-ar: truncated 21-block backbone, final
              norm -> Identity, value head; score MSE = 2(1-cos) and fve_nrm
              against the example's training denominator 0.7335.

Every NLA constant (template, token ids, scales) is read from each checkpoint's
nla_meta.yaml and asserted against the live tokenizer, per docs/inference.md.
Nothing is hardcoded from the docs' orientation table.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model_utils import free, load_model, rss_mb, to_cpu_f32  # noqa: E402  (sets MPS env first)

import torch  # noqa: E402
import yaml  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

TARGET = "Qwen/Qwen2.5-7B-Instruct"
AV_REPO = "kitft/nla-qwen2.5-7b-L20-av"
AR_REPO = "kitft/nla-qwen2.5-7b-L20-ar"
USER_MSG = "What are you hiding?"
FVE_DENOM_FROM_EXAMPLE = 0.7335  # Var(v_nrm) of the training set, per examples/*.txt
EXPL_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} rss={rss_mb() / 1024:.1f}G] {msg}", flush=True)


# ----------------------------------------------------------------------------- phase 1
def extract_target(positions: list[int], layer_index: int) -> dict:
    model, tok = load_model(TARGET)
    ids = tok.apply_chat_template([{"role": "user", "content": USER_MSG}],
                                  add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"].to("mps")
    n_prompt = ids.shape[1]
    with torch.inference_mode():
        out = model.generate(ids, max_new_tokens=120, do_sample=False)
    full = out[0:1]
    reply = tok.decode(full[0, n_prompt:], skip_special_tokens=True)
    with torch.inference_mode():
        hs = model(full, output_hidden_states=True).hidden_states
    assert len(hs) == model.config.num_hidden_layers + 1
    resid = hs[layer_index + 1][0]  # output of block `layer_index`
    toks = [tok.decode([t]) for t in full[0].tolist()]
    got = {}
    for p in positions:
        assert p < full.shape[1], f"position {p} beyond sequence length {full.shape[1]}"
        v = to_cpu_f32(resid[p])
        got[p] = {"token": toks[p], "role": "PROMPT" if p < n_prompt else "REPLY",
                  "norm": float(v.norm()), "vec": v}
    log(f"target: prompt {n_prompt} tok, full {full.shape[1]} tok; reply={reply!r}")
    free(model)
    return {"reply": reply, "n_prompt": n_prompt, "seq_len": full.shape[1], "positions": got}


# ----------------------------------------------------------------------------- phase 2
def load_meta(ckpt_dir: Path) -> dict:
    return yaml.safe_load((ckpt_dir / "nla_meta.yaml").read_text())


def av_prompt_ids(tok, meta: dict) -> tuple[torch.Tensor, int]:
    t = meta["tokens"]
    content = meta["prompt_templates"]["av"].format(injection_char=t["injection_char"])
    ids = tok.apply_chat_template([{"role": "user", "content": content}],
                                  tokenize=True, add_generation_prompt=True)
    if not isinstance(ids, list):  # transformers>=5 may return BatchEncoding
        ids = ids["input_ids"] if "input_ids" in ids else list(ids)
    live = tok.encode(t["injection_char"], add_special_tokens=False)
    assert live == [t["injection_token_id"]], f"tokenizer drift: {live}"
    hits = [i for i, x in enumerate(ids) if x == t["injection_token_id"]]
    assert len(hits) == 1, hits
    p = hits[0]
    assert ids[p - 1] == t["injection_left_neighbor_id"], (ids[p - 1], t)
    assert ids[p + 1] == t["injection_right_neighbor_id"], (ids[p + 1], t)
    return torch.tensor([ids]), p


def verbalize(vecs: dict, n_new: int) -> dict:
    ckpt = Path(snapshot_download(AV_REPO))
    meta = load_meta(ckpt)
    assert meta["role"] == "av" and meta["extraction_layer_index"] is not None
    scale = float(meta["extraction"]["injection_scale"])
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16, device_map="mps").eval()
    ids, inj_pos = av_prompt_ids(tok, meta)
    log(f"AV loaded; prompt {ids.shape[1]} tok, marker at {inj_pos}, injection_scale={scale}")
    ids = ids.to("mps")
    embed = model.get_input_embeddings()
    with torch.inference_mode():
        base_embeds = embed(ids)  # [1, T, d]
    results = {}
    for p, rec in vecs.items():
        v = rec["vec"]
        v_scaled = (v / v.norm().clamp_min(1e-12) * scale).to("mps", torch.bfloat16)
        embeds = base_embeds.clone()
        embeds[0, inj_pos] = v_scaled
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(inputs_embeds=embeds, attention_mask=torch.ones_like(ids),
                                 max_new_tokens=n_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        text = tok.decode(out[0], skip_special_tokens=True)  # inputs_embeds -> only new tokens
        m = EXPL_RE.search(text)
        results[p] = {"raw": text, "explanation": m.group(1) if m else None,
                      "gen_s": time.time() - t0, "n_gen": int(out.shape[1])}
        log(f"AV pos {p} {rec['token']!r} ({results[p]['n_gen']} tok, {results[p]['gen_s']:.0f}s)")
    free(model)
    return {"meta": {"injection_scale": scale, "marker_pos": inj_pos, "prompt_len": int(ids.shape[1])},
            "results": results}


# ----------------------------------------------------------------------------- phase 3
def reconstruct_and_score(vecs: dict, expl: dict) -> dict:
    ckpt = Path(snapshot_download(AR_REPO))
    meta = load_meta(ckpt)
    assert meta["role"] == "ar"
    mse_scale = float(meta["extraction"]["mse_scale"])
    template = meta["prompt_templates"]["ar"]
    tok = AutoTokenizer.from_pretrained(ckpt)
    backbone = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16, device_map="mps").eval()
    n_layers = backbone.config.num_hidden_layers
    assert n_layers == meta["critic"]["extraction_layer_index"] + 1, n_layers
    backbone.lm_head = torch.nn.Identity()
    backbone.model.norm = torch.nn.Identity()
    d = backbone.config.hidden_size
    head = torch.nn.Linear(d, d, bias=False, dtype=torch.bfloat16)
    head.load_state_dict(load_file(str(ckpt / "value_head.safetensors")))
    head = head.to("mps").eval()
    suffix = meta["tokens"]["critic_suffix_ids"]
    log(f"AR loaded: {n_layers} layers, mse_scale={mse_scale:.3f}")
    scores = {}
    for p, rec in expl.items():
        text = rec["explanation"] if rec["explanation"] is not None else rec["raw"]
        ids = tok(template.format(explanation=text), return_tensors="pt", add_special_tokens=True)["input_ids"]
        assert ids[0, -len(suffix):].tolist() == suffix, ids[0, -len(suffix):].tolist()
        with torch.inference_mode():
            h = backbone.model(ids.to("mps"), use_cache=False).last_hidden_state[:, -1]
            pred = to_cpu_f32(head(h)[0])
        gold = vecs[p]["vec"]
        pred_n = pred / pred.norm() * mse_scale
        gold_n = gold / gold.norm() * mse_scale
        mse = float(((pred_n - gold_n) ** 2).mean())
        cos = float(pred_n @ gold_n / (pred_n.norm() * gold_n.norm()))
        scores[p] = {"mse_nrm": mse, "cos": cos, "fve_nrm": 1 - mse / FVE_DENOM_FROM_EXAMPLE,
                     "pred_norm_raw": float(pred.norm()), "used_raw_text": rec["explanation"] is None}
        log(f"AR pos {p}: mse_nrm={mse:.3f} cos={cos:.3f} fve_nrm={scores[p]['fve_nrm']:.3f}")
    free(backbone)
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, nargs="+", default=[34, 35, 38, 40])
    ap.add_argument("--n-new", type=int, default=200)
    ap.add_argument("--out", default="fve_claims/out/nla7b_roundtrip_out.json")
    args = ap.parse_args()

    log("phase 1: target extraction")
    tgt = extract_target(args.positions, layer_index=20)
    log("phase 2: AV verbalization")
    av = verbalize(tgt["positions"], args.n_new)
    log("phase 3: AR reconstruction")
    sc = reconstruct_and_score(tgt["positions"], av["results"])

    report = {"target": {k: v for k, v in tgt.items() if k != "positions"},
              "positions": {p: {"token": r["token"], "role": r["role"], "norm": r["norm"],
                                "explanation": av["results"][p]["explanation"],
                                "raw": av["results"][p]["raw"], **sc[p]}
                            for p, r in tgt["positions"].items()},
              "av": av["meta"]}
    out = Path(args.out).resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("\n" + "=" * 78)
    for p, r in report["positions"].items():
        print(f"[{p}] {r['role']} token={r['token']!r} ||v||={r['norm']:.1f} "
              f"mse_nrm={r['mse_nrm']:.3f} cos={r['cos']:.3f} fve_nrm={r['fve_nrm']:.3f}")
        print("    " + (r["explanation"] or f"(no tags) {r['raw']}").replace("\n", "\n    "))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
