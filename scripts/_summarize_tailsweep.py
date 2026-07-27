#!/usr/bin/env python3
"""Summarize the tail-sweep joint evals into one comparison table.
Parses the 'Overall ...' final-summary lines from each eval log and prints a
frame-count vs ASR table, with the 216-frame whole-window run as the baseline row."""
import re
import sys
from pathlib import Path

ROOT = Path("/mnt/data/weicong_chen/DropVLA_Opt")
LOGDIR = ROOT / "experiments" / "logs"

# (label, poison_frames, log_dir_glob) — most recent matching file is used.
ROWS = [
    ("whole (baseline)", 216, "vln3full/EVAL-*joint*seed42.txt"),
    ("tail-32",           96, "vl0p31vltail32/EVAL-*joint*seed42.txt"),
    ("tail-16",           48, "vl0p31vltail16/EVAL-*joint*seed42.txt"),
    ("tail-08",           24, "vl0p31vltail08/EVAL-*joint*seed42.txt"),
]

PATS = {
    "clean_sr":  r"Overall success rate:\s*([\d.]+)",
    "tsr":       r"Overall TSR \(attack success rate\):\s*([\d.]+)\s*\((\d+)/(\d+)",
    "tsr_l":     r"Overall TSR-L \(latency success rate\):\s*([\d.]+)\s*\((\d+)/(\d+)",
    "trig":      r"Overall Trigger Activation Rate:\s*([\d.]+)\s*\((\d+)/(\d+)",
    "cond_asr":  r"Overall Conditional ASR[^:]*:\s*([\d.]+)\s*\((\d+)/(\d+)",
}


def latest(glob):
    files = sorted(LOGDIR.glob(glob), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def parse(path):
    txt = path.read_text(errors="ignore")
    out = {}
    for k, pat in PATS.items():
        m = None
        for m in re.finditer(pat, txt):
            pass  # keep last match (final summary)
        out[k] = m
    return out


def fmt(m, kind):
    if m is None:
        return "N/A"
    if kind == "rate1":
        return f"{float(m.group(1))*100:.1f}%"
    g = m.groups()
    return f"{float(g[0])*100:.1f}% ({g[1]}/{g[2]})"


print(f"{'variant':<18}{'frames':>7}  {'clean SR':>10}  {'Trigger Act.':>16}  {'Conditional ASR':>20}  {'TSR-L':>16}")
print("-" * 96)
for label, frames, glob in ROWS:
    f = latest(glob)
    if f is None:
        print(f"{label:<18}{frames:>7}  {'(no log yet)':>10}")
        continue
    p = parse(f)
    print(f"{label:<18}{frames:>7}  {fmt(p['clean_sr'],'rate1'):>10}  "
          f"{fmt(p['trig'],'frac'):>16}  {fmt(p['cond_asr'],'frac'):>20}  {fmt(p['tsr_l'],'frac'):>16}")
print("-" * 96)
print("Conditional ASR = fast-release successes / trigger-activated episodes (denominator varies).")
print("All rows: N=3 same episodes, lambda=0. Only per-episode poison frame count differs.")
