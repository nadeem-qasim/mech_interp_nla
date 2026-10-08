"""fve_claims machinery for ceselder/qwen3.6-27b-nla-rl on CUDA. Mirrors common.py's interface (Settings,
log, cos) but is built directly from scripts/nla27b_smoke.py (proven working: 8/8
examples, cos_own 0.9657, reproduced on the pod). Does NOT use src/nla_lib.py -- that pipeline is
7B/MPS-specific and uses a different injection mechanism (embedding replacement) than this model's
(additive norm-matched at block 1).

TARGET = av_base (decided, not investigated further): av_base is Qwen3.6-27B with the warmstart AV LoRA
(r128, all-module) already merged in, per the model card -- it is NOT a pristine base model the way the
7B arm's Qwen2.5-7B-Instruct is. Extracting from it means testing whether the NLA can describe a network
that was itself partly trained toward being describable by it. This is a real, deliberate deviation from
the 7B arm's methodology and must be stated prominently wherever 27B numbers are reported.

No FVE denominator is shipped in the sidecar (unlike the 7B pair's released-example 0.7335). Score
primarily by cosine. MSE_SCALE (sqrt(d_model)) is EasyNLA's internal AR normalization convention, not an
FVE denominator -- do not confuse the two. A locally-computed FVE denominator (variance of the extracted
activation set, matching fve_claims/common.py's D_local convention for the 7B) can be added once a full
activation set exists.
"""
import json, re, subprocess, sys, time, gc
from pathlib import Path
import torch, yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

REPO = next(Path("/workspace/hf/hub/models--ceselder--qwen3.6-27b-nla-rl/snapshots").iterdir())
META = yaml.safe_load((REPO / "nla_meta.yaml").read_text())
INJ_CHAR = META["tokens"]["injection_char"]
INJ_ID = META["tokens"]["injection_token_id"]
LEFT_ID = META["tokens"]["injection_left_neighbor_id"]
RIGHT_ID = META["tokens"]["injection_right_neighbor_id"]
ACTOR_TEMPLATE = META["prompt_templates"]["actor"]
CRITIC_TEMPLATE = META["prompt_templates"]["critic"]
D_MODEL = META["extraction"]["d_model"]
MSE_SCALE = D_MODEL ** 0.5  # EasyNLA default (sidecar ships no mse_scale); AR's internal normalization only
LAYER = META["extraction"]["layer_index"]  # 42; activation = output of decoder block LAYER = hidden_states[LAYER+1]
DEVICE = "cuda"


def git_hash() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def log(msg: str) -> None:
    g = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    print(f"[{time.strftime('%H:%M:%S')} gpu={g:.1f}G] {msg}", flush=True)


def settings(step: str, **kw) -> dict:
    d = {"step": step, "git": git_hash(), "argv": sys.argv, "repo": str(REPO), "device": DEVICE,
         "target_note": "TARGET=av_base, has warmstart LoRA merged in -- see module docstring",
         "wall_start": time.strftime("%Y-%m-%dT%H:%M:%S")}
    d.update(kw)
    (HERE / f"{step}_settings.json").write_text(json.dumps(d, indent=2, default=str))
    return d


def finish(step: str, d: dict, **kw) -> None:
    d.update(kw)
    d["wall_end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (HERE / f"{step}_settings.json").write_text(json.dumps(d, indent=2, default=str))


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.reshape(1, -1).double(), b.reshape(1, -1).double()).item()


class Target:
    """av_base used as the extraction model. Extracts the OUTPUT of decoder block LAYER (42) at the last
    token of raw text (no chat template -- matches the 7B convention)."""

    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(REPO / "av_base")
        self.model = AutoModelForCausalLM.from_pretrained(
            REPO / "av_base", dtype=torch.bfloat16, device_map=DEVICE, attn_implementation="sdpa"
        ).eval()
        log(f"Target(av_base) loaded, {type(self.model).__name__}")

    def activation(self, text: str) -> torch.Tensor:
        ids = self.tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEVICE)
        with torch.no_grad():
            hs = self.model(ids, output_hidden_states=True, use_cache=False).hidden_states
        return hs[LAYER + 1][0, -1].float().cpu()

    def n_tokens(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def free(self):
        del self.model
        gc.collect(); torch.cuda.empty_cache()
        self.model = None


class AV:
    """av_base + one RL LoRA adapter. Additive norm-matched injection on the OUTPUT of decoder block 1
    (EasyNLA nla/injection.py convention) -- NOT embedding replacement."""

    def __init__(self, adapter: str = "iter_000600"):
        self.tok = AutoTokenizer.from_pretrained(REPO / "av_base")
        base = AutoModelForCausalLM.from_pretrained(
            REPO / "av_base", dtype=torch.bfloat16, device_map=DEVICE, attn_implementation="sdpa"
        )
        inner = base.model
        if hasattr(inner, "language_model"):
            inner = inner.language_model
        self.layers = inner.layers
        self.model = PeftModel.from_pretrained(base, REPO / "av_rl_adapters" / adapter).eval()
        self._vref = {"v": None, "ids": None, "hits": 0}

        def embed_hook(m, a, kw, out):
            self._vref["ids"] = kw.get("input") if kw and kw.get("input") is not None else (a[0] if a else None)
            return out

        def layer_hook(m, a, out):
            resid = out[0] if isinstance(out, tuple) else out
            ids = self._vref["ids"]
            if self._vref["v"] is None or ids is None or resid.shape[1] < 2:
                return out
            ids = ids.to(resid.device)
            new = resid.clone()
            hits = 0
            for b, pos in (ids == INJ_ID).nonzero().tolist():
                if pos == 0 or pos == ids.shape[1] - 1:
                    continue
                if ids[b, pos - 1] != LEFT_ID or ids[b, pos + 1] != RIGHT_ID:
                    continue
                h = new[b, pos].clone()
                v = self._vref["v"].to(new.device, new.dtype)
                new[b, pos] = h + h.norm() * (v / (v.norm() + 1e-9))
                hits += 1
            self._vref["hits"] = hits
            return (new, *out[1:]) if isinstance(out, tuple) else new

        self.model.get_input_embeddings().register_forward_hook(embed_hook, with_kwargs=True)
        self.layers[1].register_forward_hook(layer_hook)

        msgs = [{"role": "user", "content": ACTOR_TEMPLATE.format(injection_char=INJ_CHAR)}]
        ptxt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = self.tok.encode(ptxt, add_special_tokens=False)
        pos = [i for i, t in enumerate(ids) if t == INJ_ID]
        assert len(pos) == 1 and ids[pos[0] - 1] == LEFT_ID and ids[pos[0] + 1] == RIGHT_ID
        self._prompt_ids = torch.tensor([ids], device=DEVICE)
        self.marker_pos = pos[0]
        self.prompt_tokens = len(ids)
        log(f"AV(av_base+{adapter}) loaded; prompt {self.prompt_tokens} tok, marker at {self.marker_pos}")

    def verbalize(self, vec: torch.Tensor, max_new_tokens: int = 200) -> dict:
        self._vref["v"] = vec.reshape(-1)
        self._vref["hits"] = 0
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(
                input_ids=self._prompt_ids, attention_mask=torch.ones_like(self._prompt_ids),
                max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tok.eos_token_id,
            )
        self._vref["v"] = None
        raw = self.tok.decode(out[0, self._prompt_ids.shape[1]:], skip_special_tokens=True)
        m = re.search(r"<explanation>(.*?)</explanation>", raw, re.DOTALL)
        has_cjk = any("一" <= c <= "鿿" or "぀" <= c <= "ヿ" for c in raw)
        return {
            "raw_generation": raw, "explanation": m.group(1).strip() if m else None, "parse_ok": m is not None,
            "cjk": has_cjk, "n_tokens": int(out.shape[1] - self._prompt_ids.shape[1]),
            "gen_s": time.time() - t0, "inject_hits": self._vref["hits"],
        }

    def free(self):
        del self.model, self.layers
        gc.collect(); torch.cuda.empty_cache()
        self.model = None


class AR:
    """ar_reconstructor: 43-block truncated backbone, final norm -> Identity, + a separately-shipped fp32
    value_head. Never co-resident with AV -- 52GB + 35GB does not fit in 80GB."""

    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(REPO / "av_base")
        self.model = AutoModelForCausalLM.from_pretrained(
            REPO / "ar_reconstructor", dtype=torch.bfloat16, device_map=DEVICE, attn_implementation="sdpa"
        ).eval()
        inner = self.model.model
        if hasattr(inner, "language_model"):
            inner = inner.language_model
        inner.norm = torch.nn.Identity()
        self.inner = inner
        self.value_head = torch.nn.Linear(D_MODEL, D_MODEL, bias=False, dtype=torch.float32).to(DEVICE)
        self.value_head.load_state_dict(load_file(REPO / "ar_reconstructor/value_head.safetensors"))
        self.n_forward = 0
        log(f"AR(ar_reconstructor) loaded, {len(inner.layers)} blocks, final norm -> Identity")

    def predict(self, explanation: str) -> torch.Tensor:
        cids = self.tok.encode(CRITIC_TEMPLATE.format(explanation=explanation), add_special_tokens=False)
        x = torch.tensor([cids], device=DEVICE)
        with torch.no_grad():
            h = self.inner(input_ids=x, use_cache=False).last_hidden_state[0, -1].float()
            h = h / (h.norm() + 1e-9) * MSE_SCALE
            pred = self.value_head(h)
        self.n_forward += 1
        return pred.cpu()

    def free(self):
        del self.model, self.inner
        gc.collect(); torch.cuda.empty_cache()
        self.model = None
