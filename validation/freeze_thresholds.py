"""Freeze the Glue-2 / Glue-3 thresholds exactly as PREDECLARATION.md says (commit 6d47673, sha e427a671...).

Inputs: the calibration set and one answer file per tier/dtype from run_tiers.py (every tier saw every question).
  1. fastest usable path per own model: bfloat16 if its top-1 flips vs float32 are <= 0.5 % of calibration
     questions AND its warmed median latency is lower; else float32.
  2. go / no-go: 2B (chosen path) warmed median <= 0.6 x JevK5-9B warmed median.
  3. grid tau in {0.00 .. 0.99}; constraint: cascade accuracy >= JevK5 alone - 1.0 pp over all labelled questions
     (a failed question is wrong); objective: expected cost = sum over tiers of (share of requests that call the
     tier x its warmed median latency); ties -> lower share of questions reaching the top tier, then lower taus.
Per-question routing follows jev_style.cascade.select() (the live server's code); a vectorised copy is used for the
grid and checked against select() on random threshold draws before anything is reported.

  python freeze_thresholds.py --cal ../calibration/calibration.jsonl --tiers DIR --out result.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "jev-style-cascade"))
from jev_style.cascade import TierError, normalize, select  # noqa: E402
from jev_style.evaluate import label_key  # noqa: E402
from jev_style.schema import parse_request  # noqa: E402

GRID = [round(i / 100, 2) for i in range(100)]
FLIP_MAX = 0.005
SPEED_RATIO = 0.6
MARGIN = 0.010


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip() and not l.lstrip().startswith("#")]


def median(xs: list[float]) -> float:
    xs = sorted(xs)
    return xs[len(xs) // 2]


class Tier:
    """One tier's answers on the calibration set, reduced per labelled question to ok / confidence / correct."""

    def __init__(self, path: Path, qs: list, n_rows: int):
        self.name = path.stem
        recs = {r["i"]: r for r in load_jsonl(path)}
        assert len(recs) == n_rows, (path, len(recs), n_rows)
        self.raw = []
        ok, conf, corr, label, skipped = [], [], [], [], []
        for i, qid, spec, gold in qs:
            rec = recs[i]
            if qid in rec.get("answers", {}):
                r = rec["answers"][qid]
            else:
                e = rec.get("errors", {}).get(qid, {"code": "missing"})
                r = TierError(int(e.get("status", 500)), str(e.get("code")), str(e.get("message", "")), qid,
                              skipped=bool(e.get("skipped")))
            self.raw.append(r)
            routed = select([r], [], question=spec)          # applies the live answer checks
            if routed.ok:
                n = normalize(routed.answer)
                ok.append(True); conf.append(n.confidence); corr.append(n.label == gold); label.append(n.label)
                skipped.append(False)
            else:
                ok.append(False); conf.append(-1.0); corr.append(False); label.append(None)
                skipped.append(bool(getattr(routed.answer, "skipped", False)))
        self.ok, self.conf, self.corr = np.array(ok), np.array(conf), np.array(corr)
        self.skipped = np.array(skipped)
        self.label = label
        t = path.with_suffix(".timing.jsonl")
        self.median_ms = median([r["latency_ms"] for r in load_jsonl(t)])
        self.first_median_ms = median([r["latency_ms"] for r in recs.values()])


def route(tiers: list[Tier], th: list[float]):
    """-> (chosen tier index per question, -1 = all failed; reached[k] per question). Mirrors select()."""
    n = len(tiers[0].ok)
    chosen = np.full(n, -1)
    decided = np.zeros(n, bool)
    last_ok = np.full(n, -1)
    reached = []
    for k, t in enumerate(tiers):
        reached.append(~decided)
        if k < len(th):
            keep = ~decided & t.ok & (t.conf >= th[k])
        else:
            keep = ~decided & t.ok
        chosen[keep] = k
        decided |= keep
        last_ok = np.where(~decided & t.ok, k, last_ok)
    chosen = np.where(decided, chosen, last_ok)
    return chosen, reached


def metrics(tiers: list[Tier], th: list[float], row_of: np.ndarray, n_rows: int) -> dict:
    chosen, reached = route(tiers, th)
    corr = np.zeros(len(chosen), bool)
    for k, t in enumerate(tiers):
        corr |= (chosen == k) & t.corr
    calls = [len(np.unique(row_of[reached[k] & ~tiers[k].skipped])) / n_rows for k in range(len(tiers))]
    return {"accuracy": float(corr.mean()), "cost_ms": float(sum(c * t.median_ms for c, t in zip(calls, tiers))),
            "top_share": float(reached[-1].mean()), "calls_share": calls,
            "answered_by": [float((chosen == k).mean()) for k in range(len(tiers))]}


def parity(tiers: list[Tier], qs: list, draws: int = 40, seed: int = 0) -> int:
    """Vectorised route() == jev_style.cascade.select() on random thresholds; returns mismatches."""
    rng = random.Random(seed)
    bad = 0
    for _ in range(draws):
        th = [rng.choice(GRID) for _ in range(len(tiers) - 1)]
        chosen, _ = route(tiers, th)
        for j in rng.sample(range(len(qs)), 300):
            spec = qs[j][2]
            r = select([t.raw[j] for t in tiers], th, question=spec)
            want = r.tier if r.ok else -1
            bad += want != chosen[j]
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal", required=True)
    ap.add_argument("--tiers", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cal = Path(a.cal)
    rows = load_jsonl(cal)
    qs = []
    for i, row in enumerate(rows):
        raw = {qid: {k: v for k, v in q.items() if k != "label"} for qid, q in row["questions"].items()}
        specs = {q.id: q for q in parse_request({"state": row["state"], "questions": raw}).questions}
        for qid, q in row["questions"].items():
            if "label" in q:
                qs.append((i, qid, specs[qid], label_key(raw[qid], q["label"])))
    row_of = np.array([q[0] for q in qs])
    d = Path(a.tiers)
    T = {p.stem: Tier(p, qs, len(rows)) for p in sorted(d.glob("*.jsonl")) if not p.name.endswith(".timing.jsonl")}
    out: dict = {"calibration_sha256": hashlib.sha256(cal.read_bytes()).hexdigest(), "questions": len(qs),
                 "requests": len(rows), "predeclaration_sha256": "e427a671bbe6cc3a07c3c1819125f7169b6d547fc37d067f5f3ccff77ad4e0a6"}
    out["singles"] = {k: {"accuracy": float(t.corr.mean()), "answered": float(t.ok.mean()),
                          "median_ms_warm": t.median_ms, "median_ms_first_pass": t.first_median_ms}
                      for k, t in T.items()}
    chosen = {}
    for rel in ("0.8b-v3", "2b-v3"):
        ref = T[f"{rel}_float32"]
        info = {"median_ms_float32": ref.median_ms}
        usable = [("float32", ref)]
        for path in ("bfloat16", "graph"):                 # graph = float32 + CUDA graphs (runtime cuda_graphs=True)
            cand = T.get(f"{rel}_{path}")
            if cand is None:
                continue
            flips = sum(x != y for x, y in zip(ref.label, cand.label))
            rate = flips / len(qs)
            info[path] = {"flips": flips, "flip_rate": round(rate, 5), "median_ms": cand.median_ms,
                          "usable": rate <= FLIP_MAX}
            if rate <= FLIP_MAX:
                usable.append((path, cand))
        name, tier = min(usable, key=lambda x: x[1].median_ms)   # the fastest usable path
        chosen[rel] = tier
        info["chosen"] = name
        out[f"path_{rel}"] = info
    top = T["jevk5_9b"]
    ratio = chosen["2b-v3"].median_ms / top.median_ms
    out["speed_gate"] = {"2b_median_ms": chosen["2b-v3"].median_ms, "jevk5_median_ms": top.median_ms,
                         "ratio": round(ratio, 4), "max_ratio": SPEED_RATIO, "pass": bool(ratio <= SPEED_RATIO)}
    top_alone = metrics([top], [], row_of, len(rows))
    floor = top_alone["accuracy"] - MARGIN
    out["jevk5_alone"] = top_alone
    out["accuracy_floor"] = floor
    for name, stack in (("glue2", [chosen["2b-v3"], top]), ("glue3", [chosen["0.8b-v3"], chosen["2b-v3"], top])):
        mism = parity(stack, qs)
        assert mism == 0, f"{name}: vectorised routing disagrees with select() on {mism} questions"
        grids = [[t] for t in GRID] if len(stack) == 2 else [[t1, t2] for t1 in GRID for t2 in GRID]
        best = None
        for th in grids:
            m = metrics(stack, th, row_of, len(rows))
            if m["accuracy"] < floor:
                continue
            key = (round(m["cost_ms"], 9), round(m["top_share"], 9), th)
            if best is None or key < best[0]:
                best = (key, th, m)
        out[name] = {"tiers": [t.name for t in stack], "parity_mismatches": mism,
                     **({"thresholds": best[1], **best[2]} if best else {"thresholds": None})}
    Path(a.out).write_text(json.dumps(out, indent=2, default=lambda o: o.item()))
    print(json.dumps({k: out[k] for k in ("speed_gate", "path_0.8b-v3", "path_2b-v3", "jevk5_alone", "glue2",
                                          "glue3")}, indent=2, default=lambda o: o.item()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
