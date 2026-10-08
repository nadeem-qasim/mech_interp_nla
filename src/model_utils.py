"""Shared loader + hook utilities for local mech-interp work.

Import this from anywhere in the repo as `from src.model_utils import load_model`,
running from the repo root via `uv run`.  Everything -- the smoke test included --
goes through this module; a second loader elsewhere would drift on dtype, chat
template and attention implementation.

MPS env vars are set *in-process, above `import torch`* (see `_load_env_sh`).
torch reads them at import and at first MPS use, so setting them later is a no-op
that looks like it worked.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENV_SH = _REPO_ROOT / "scripts" / "env.sh"

_EXPORT_RE = re.compile(r"^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def _load_env_sh(path: Path = _ENV_SH) -> dict[str, str]:
    """Parse `export K=V` lines out of scripts/env.sh and apply them to os.environ.

    Values already present in the environment win (the human's shell overrides the
    file).  We parse rather than `source`: a subprocess `source` dies with the child
    and sets nothing in the parent.
    """
    applied: dict[str, str] = {}
    fallback = {
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.8",
        "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.7",  # must be <= the high ratio
    }
    if path.exists():
        for line in path.read_text().splitlines():
            m = _EXPORT_RE.match(line)
            if not m:
                continue
            key, raw = m.group(1), m.group(2)
            val = raw.split("#", 1)[0].strip().strip("'\"")
            if val:
                applied[key] = val
    else:  # duplicate the exports so the module still works standalone
        applied = dict(fallback)

    for key, val in applied.items():
        os.environ.setdefault(key, val)
    return {k: os.environ[k] for k in applied}


MPS_ENV = _load_env_sh()

import torch  # noqa: E402  -- must follow _load_env_sh()
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

DEFAULT_REPO_ID = "Qwen/Qwen3-4B-Instruct-2507"


# --------------------------------------------------------------------------- #
# process / device helpers
# --------------------------------------------------------------------------- #
def rss_mb() -> float:
    """Resident set size of this process in MiB.

    Process RSS, not `torch.mps.current_allocated_memory()` -- the latter undercounts
    and misses CPU-side float32 caches.
    """
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True
    ).stdout.strip()
    return int(out) / 1024.0


def get_blocks(model) -> torch.nn.ModuleList:
    """The decoder blocks, resolved dynamically (Qwen: model.model.layers)."""
    for path in ("model.layers", "model.model.layers", "transformer.h", "layers"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and len(obj) > 0:
            return obj
    raise AttributeError(f"could not locate decoder blocks on {type(model).__name__}")


def get_ln_f(model) -> torch.nn.Module:
    """The final norm applied after the last decoder block, before lm_head."""
    for path in ("model.norm", "model.model.norm", "transformer.ln_f"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise AttributeError(f"could not locate final norm on {type(model).__name__}")


def _block_output(out):
    """Decoder blocks may return a tensor or a tuple whose [0] is the hidden state."""
    return out[0] if isinstance(out, tuple) else out



def to_cpu_f32(t: "torch.Tensor") -> "torch.Tensor":
    """Move a tensor to CPU as float32 -- casting on-device FIRST, deliberately.

    Do not "simplify" this to `t.to("cpu", torch.float32)`.  In torch 2.13.0 a
    combined device+dtype copy off MPS reads from the wrong storage offset when the
    source dtype is 2 bytes wide (bf16/fp16): the offset is applied in the
    *destination* element size.  A sliced-out row such as `logits[0, -1]` or
    `h[:, -1]` therefore comes back holding neighbouring data, silently.

        a = torch.arange(20, dtype=torch.bfloat16, device="mps").reshape(2, 10)
        a[1].to("cpu", torch.float32)   # [15,16,17,18,19, 0,0,0,0,0]  <- WRONG
        a[1].float().cpu()              # [10,...,19]                  <- correct

    `contiguous()` does not help (the view is already contiguous, merely offset);
    `clone()` does, but casting first is cheaper and clearer.  fp32 sources are
    unaffected.
    """
    return t.detach().float().cpu()


# --------------------------------------------------------------------------- #
# load / free
# --------------------------------------------------------------------------- #
def load_model(
    repo_id: str = DEFAULT_REPO_ID,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "mps",
    attn_implementation: str = "sdpa",
):
    """Load model + tokenizer, ready for inference.

    `device_map=device` maps the safetensors straight onto the target device; a CPU
    load followed by `.to("mps")` can peak near 2x the weight size during the copy.

    Eager attention is only needed for `output_attentions=True` and is a real speed
    tax on MPS, so it stays a per-call argument rather than a global default.
    """
    tok = AutoTokenizer.from_pretrained(repo_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token  # Qwen often reuses EOS

    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        dtype=dtype,
        attn_implementation=attn_implementation,
        device_map=device,
    )
    model.eval()
    return model, tok


def free(model) -> float:
    """Drop a model and try to return its memory.  Returns RSS (MiB) afterwards.

    On unified memory macOS often does not hand the pages back.  If RSS has not
    dropped, restart the kernel rather than loading anything else.
    """
    import gc

    before = rss_mb()
    try:
        model.to("meta")
    except Exception:
        pass
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()
    gc.collect()
    after = rss_mb()
    print(f"[free] RSS {before:.0f} MiB -> {after:.0f} MiB (released {before - after:.0f} MiB)")
    if after > 0.8 * before:
        print("[free] WARNING: RSS did not drop meaningfully; restart the kernel "
              "before loading anything else.")
    return after


# --------------------------------------------------------------------------- #
# prompting
# --------------------------------------------------------------------------- #
_THINKING_KWARG_OK: bool | None = None


def build_prompt(tok, prompt: str) -> str:
    """Apply the chat template with add_generation_prompt=True.

    2507 is instruct-only, so `enable_thinking=False` may be unsupported; we try it
    once and drop the kwarg permanently if the tokenizer objects.
    """
    global _THINKING_KWARG_OK
    messages = [{"role": "user", "content": prompt}]
    if _THINKING_KWARG_OK is not False:
        try:
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            _THINKING_KWARG_OK = True
            return text
        except Exception:
            _THINKING_KWARG_OK = False
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_prompts(tok, prompts: list[str]) -> list[str]:
    return [build_prompt(tok, p) for p in prompts]


def tokenize(tok, prompts: list[str], max_length: int = 512, device: str = "cpu"):
    """Chat-template + left-pad a batch.  `max_length` is a required cap: uncapped
    batching OOMs."""
    texts = build_prompts(tok, prompts)
    enc = tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,  # the template already inserted them
    )
    return {k: v.to(device) for k, v in enc.items()}


# --------------------------------------------------------------------------- #
# activations
# --------------------------------------------------------------------------- #
def cache_residual(
    model,
    tok,
    prompts: list[str],
    layers: int | list[int],
    max_length: int = 512,
    batch_size: int = 8,
) -> dict[int, torch.Tensor]:
    """Residual stream at the **last non-pad position only**.

    Returns {layer: tensor[n_prompts, hidden]}, float32, on CPU.  Not the full
    [n, seq, hidden] cube -- ask for that explicitly if you ever need it.

    Hooks the decoder block (its output *is* the residual stream after the block).
    """
    if isinstance(layers, int):
        layers = [layers]
    blocks = get_blocks(model)
    device = next(model.parameters()).device
    out: dict[int, list[torch.Tensor]] = {ell: [] for ell in layers}

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        enc = tokenize(tok, batch, max_length=max_length, device=str(device))
        mask = enc["attention_mask"]
        last_idx = mask.sum(dim=1) - 1  # left padding -> this is seq_len-1, but be exact
        if tok.padding_side == "left":
            last_idx = torch.full_like(last_idx, mask.shape[1] - 1)

        captured: dict[int, torch.Tensor] = {}

        def make_hook(ell):
            def hook(_mod, _inp, output):
                h = _block_output(output)
                sel = h[torch.arange(h.shape[0], device=h.device), last_idx]
                captured[ell] = to_cpu_f32(sel)
            return hook

        handles = []
        try:
            for ell in layers:
                handles.append(blocks[ell].register_forward_hook(make_hook(ell)))
            with torch.inference_mode():
                model(**enc)
        finally:
            for h in handles:
                h.remove()

        for ell in layers:
            out[ell].append(captured[ell])

    return {ell: torch.cat(chunks, dim=0) for ell, chunks in out.items()}


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
def generate(
    model,
    tok,
    prompts: list[str],
    max_new_tokens: int = 64,
    max_length: int = 512,
) -> list[str]:
    """Greedy decode.  Returns only the newly generated text, one string per prompt.

    Greedy on purpose: MPS sampling plus seeded reproducibility is a separate mess.
    """
    device = next(model.parameters()).device
    enc = tokenize(tok, prompts, max_length=max_length, device=str(device))
    with torch.inference_mode():
        ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    new = ids[:, enc["input_ids"].shape[1] :]
    return tok.batch_decode(new, skip_special_tokens=True)
