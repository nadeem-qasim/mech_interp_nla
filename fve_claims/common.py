"""Shared pieces for fve_claims. Imports the 7B model classes from src/nla_lib.py so the folder stays small;
everything else (paths, settings, FVE, sentence handling) is local to this folder."""
import json, re, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))
import nla_lib as L  # noqa: E402  (Target, AV, AR, cos, to_cpu_f32, parse_explanation, has_cjk)

PILOT = ROOT / "data/redocred_pilot/pilot.jsonl"
FVE_DENOM_RELEASED_7B = 0.7335  # Var of the sqrt(d)-normalised activation over the released 7B training set (examples/qwen7b_layer20_step4200.txt)

SENT_RE = re.compile(r"(?<=[.!?])\s+")
MARKER_RE = re.compile(r"^\s*(\d+[.)]|[-*•])\s*")
MIN_WORDS = 3


def git_hash() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def settings(step: str, **kw) -> dict:
    d = {"step": step, "git": git_hash(), "argv": sys.argv, "wall_start": time.strftime("%Y-%m-%dT%H:%M:%S")}
    d.update(kw)
    (HERE / f"{step}_settings.json").write_text(json.dumps(d, indent=2, default=str))
    return d


def finish(step: str, d: dict, **kw) -> None:
    d.update(kw); d["wall_end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (HERE / f"{step}_settings.json").write_text(json.dumps(d, indent=2, default=str))


def load_pilot() -> list[dict]:
    return [json.loads(l) for l in open(PILOT)]


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p)] if p.exists() else []


def split_sentences(text: str) -> list[str]:
    """Quote-aware sentence split: never split inside an open double quote."""
    out = []
    for line in (text or "").split("\n"):
        cuts = [m.start() for m in SENT_RE.finditer(line) if line[:m.start()].count('"') % 2 == 0]
        pieces, prev = [], 0
        for c in cuts:
            pieces.append(line[prev:c]); prev = c
        pieces.append(line[prev:])
        for piece in pieces:
            p = MARKER_RE.sub("", piece.strip()).strip()
            if len(p.split()) >= MIN_WORDS:
                out.append(p)
    return out


def delete_sentence(E: str, sentence: str) -> str | None:
    """E with the first exact occurrence of `sentence` removed; whitespace collapsed at the seam. None if not found."""
    a = E.find(sentence)
    if a < 0:
        return None
    b = a + len(sentence)
    E2 = E[:a].rstrip() + ("\n\n" if ("\n" in E[max(0, a - 2):a] or "\n" in E[b:b + 2]) else " ") + E[b:].lstrip()
    E2 = re.sub(r"[ \t]+", " ", E2)
    E2 = re.sub(r"\n{3,}", "\n\n", E2).strip()
    return E2


def fve(cos: float, denom: float) -> float:
    """Direction-only FVE: 1 − mse_nrm / denom with mse_nrm = 2(1 − cos)."""
    return 1.0 - 2.0 * (1.0 - cos) / denom
