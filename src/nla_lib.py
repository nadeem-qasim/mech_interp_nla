"""Shared NLA round-trip library for the Qwen2.5-7B layer-20 pair (target model, AV, AR).

Vendored from scripts/nla7b_roundtrip.py (read-only reference, verified 2026-09-06 against the
released example transcript). Every assert from the reference is kept. Import this module
before touching torch: it imports src.model_utils first, which sets the MPS env vars.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src.model_utils import free, load_model, rss_mb, to_cpu_f32  # noqa: E402,F401  (sets MPS env first)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

TARGET = "Qwen/Qwen2.5-7B-Instruct"
AV_REPO = "kitft/nla-qwen2.5-7b-L20-av"
AR_REPO = "kitft/nla-qwen2.5-7b-L20-ar"
LAYER = 20  # block index; activation = hidden_states[LAYER + 1]

EXPL_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)
CJK_RE = re.compile(r"[　-鿿가-힯]")
DEVICE = "mps"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} rss={rss_mb() / 1024:.1f}G] {msg}", flush=True)


def git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=REPO_ROOT).stdout.strip()
    except Exception:
        return "unknown"


def snapshot(repo_id: str) -> Path:
    return Path(snapshot_download(repo_id, local_files_only=True))


def load_meta(ckpt_dir: Path) -> dict:
    return yaml.safe_load((ckpt_dir / "nla_meta.yaml").read_text())


def has_cjk(s: str) -> bool:
    return bool(CJK_RE.search(s or ""))


def parse_explanation(text: str) -> tuple[str, bool]:
    m = EXPL_RE.search(text)
    if m:
        return m.group(1), True
    return text.strip(), False


# ----------------------------------------------------------------------------- target
class Target:
    def __init__(self):
        self.model, self.tok = load_model(TARGET)
        assert self.model.config.num_hidden_layers == 28
        log("TARGET loaded")

    def hidden_states(self, ids: torch.Tensor):
        """ids [1, T] (no padding). Returns tuple of hidden_states (len 29)."""
        with torch.inference_mode():
            hs = self.model(ids.to(DEVICE), output_hidden_states=True, use_cache=False).hidden_states
        assert len(hs) == self.model.config.num_hidden_layers + 1
        return hs

    def chat_generate(self, user_msg: str, max_new_tokens: int) -> str:
        text = self.tok.apply_chat_template([{"role": "user", "content": user_msg}],
                                            tokenize=False, add_generation_prompt=True)
        ids = self.tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEVICE)
        with torch.inference_mode():
            out = self.model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    def free(self):
        free(self.model)
        self.model = None


# ----------------------------------------------------------------------------- AV
def av_prompt_ids(tok, meta: dict, content: str | None = None) -> tuple[torch.Tensor, int]:
    t = meta["tokens"]
    if content is None:
        content = meta["prompt_templates"]["av"].format(injection_char=t["injection_char"])
    text = tok.apply_chat_template([{"role": "user", "content": content}],
                                   tokenize=False, add_generation_prompt=True)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    live = tok.encode(t["injection_char"], add_special_tokens=False)
    assert live == [t["injection_token_id"]], f"tokenizer drift: {live}"
    hits = [i for i, x in enumerate(ids) if x == t["injection_token_id"]]
    assert len(hits) == 1, hits
    p = hits[0]
    assert ids[p - 1] == t["injection_left_neighbor_id"], (ids[p - 1], t)
    assert ids[p + 1] == t["injection_right_neighbor_id"], (ids[p + 1], t)
    return torch.tensor([ids]), p


class AV:
    def __init__(self):
        ckpt = snapshot(AV_REPO)
        self.meta = load_meta(ckpt)
        assert self.meta["role"] == "av" and self.meta["extraction_layer_index"] == LAYER
        self.scale = float(self.meta["extraction"]["injection_scale"])
        self.tok = AutoTokenizer.from_pretrained(ckpt)
        self.model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16, device_map=DEVICE).eval()
        self.embed = self.model.get_input_embeddings()
        self.default_prompt = self.meta["prompt_templates"]["av"].format(injection_char=self.meta["tokens"]["injection_char"])
        self._cache: dict[str, tuple[torch.Tensor, int, torch.Tensor]] = {}
        ids, p, _ = self.prompt(None)
        log(f"AV loaded; prompt {ids.shape[1]} tok, marker at {p}, injection_scale={self.scale}")

    def prompt(self, content: str | None):
        key = content if content is not None else "__default__"
        if key not in self._cache:
            ids, p = av_prompt_ids(self.tok, self.meta, content)
            ids = ids.to(DEVICE)
            with torch.inference_mode():
                base = self.embed(ids)
            self._cache[key] = (ids, p, base)
        return self._cache[key]

    def verbalize(self, vec: torch.Tensor, content: str | None = None, max_new_tokens: int = 200) -> dict:
        """vec: fp32 CPU [d]. Returns dict(raw, explanation, parse_ok, cjk, n_tokens, gen_s)."""
        ids, inj_pos, base = self.prompt(content)
        v = vec.float()
        v_scaled = (v / v.norm().clamp_min(1e-12) * self.scale).to(DEVICE, torch.bfloat16)
        embeds = base.clone()
        embeds[0, inj_pos] = v_scaled
        t0 = time.time()
        with torch.inference_mode():
            out = self.model.generate(inputs_embeds=embeds, attention_mask=torch.ones_like(ids),
                                      max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        raw = self.tok.decode(out[0], skip_special_tokens=True)  # inputs_embeds -> only new tokens
        expl, ok = parse_explanation(raw)
        return {"raw_generation": raw, "explanation": expl, "parse_ok": ok, "cjk": has_cjk(raw),
                "n_tokens": int(out.shape[1]), "gen_s": time.time() - t0}

    def free(self):
        free(self.model)
        self.model = None


# ----------------------------------------------------------------------------- AR
class AR:
    def __init__(self):
        ckpt = snapshot(AR_REPO)
        self.meta = load_meta(ckpt)
        assert self.meta["role"] == "ar"
        self.mse_scale = float(self.meta["extraction"]["mse_scale"])
        self.template = self.meta["prompt_templates"]["ar"]
        self.tok = AutoTokenizer.from_pretrained(ckpt)
        self.backbone = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16, device_map=DEVICE).eval()
        n_layers = self.backbone.config.num_hidden_layers
        assert n_layers == self.meta["critic"]["extraction_layer_index"] + 1, n_layers
        self.backbone.lm_head = torch.nn.Identity()
        self.backbone.model.norm = torch.nn.Identity()
        d = self.backbone.config.hidden_size
        head = torch.nn.Linear(d, d, bias=False, dtype=torch.bfloat16)
        head.load_state_dict(load_file(str(ckpt / "value_head.safetensors")))
        self.head = head.to(DEVICE).eval()
        self.suffix = self.meta["tokens"]["critic_suffix_ids"]
        self.n_forward = 0
        log(f"AR loaded: {n_layers} layers, mse_scale={self.mse_scale:.3f}")

    def predict(self, explanation: str) -> torch.Tensor:
        """Reconstructed activation (fp32 CPU [d], raw scale) for an explanation string."""
        ids = self.tok(self.template.format(explanation=explanation), return_tensors="pt",
                       add_special_tokens=True)["input_ids"]
        assert ids[0, -len(self.suffix):].tolist() == self.suffix, ids[0, -len(self.suffix):].tolist()
        with torch.inference_mode():
            h = self.backbone.model(ids.to(DEVICE), use_cache=False).last_hidden_state[:, -1]
            pred = to_cpu_f32(self.head(h)[0])
        self.n_forward += 1
        return pred

    def free(self):
        free(self.backbone)
        self.backbone = None


def cos(a: torch.Tensor | np.ndarray, b: torch.Tensor | np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def mse_from_cos(c: float) -> float:
    return 2.0 * (1.0 - c)
