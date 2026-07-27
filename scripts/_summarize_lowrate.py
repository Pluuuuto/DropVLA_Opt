#!/usr/bin/env python3
"""Summarize the low-rate A/B/C experiment vs the conc1 (N=1) baseline.
All rows: N=1 poisoned episode, lambda=0. Goal: rescue Conditional ASR from 4% -> high."""
import re
from pathlib import Path

ROOT = Path("/mnt/data/weicong_chen/DropVLA_Opt")
LOGDIR = ROOT / "experiments" / "logs"

ROWS = [
    ("conc1 (baseline)", "N=1, 5px, plain", "vl0p31conc1/EVAL-*joint*seed42.txt"),
    ("A dot20",          "N=1, 20px trigger", "vl0p31conc1dot20/EVAL-*joint*seed42.txt"),
    ("B dup x6",         "N=1, ep dup x6",    "vl0p31conc1dup6/EVAL-*joint*seed42.txt"),
    ("C sharpen",        "N=1, motion=0",     "vl0p31conc1sharp/EVAL-*joint*seed42.txt"),
]
PATS = {
    "clean_sr":  r"Overall success rate:\s*([\d.]+)",
    "trig":      r"Overall Trigger Activation Rate:\s*([\d.]+)\s*\((\d+)/(\d+)",
    "cond_asr":  r"Overall Conditional ASR[^:]*:\s*([\d.]+)\s*\((\d+)/(\d+)",
    "tsr_l":     r"Overall TSR-L \(latency success rate\):\s*([\d.]+)\s*\((\d+)/(\d+)",
}


def latest(glob):
    fs = sorted(LOGDIR.glob(glob), key=lambda p: p.stat().st_mtime)
    return fs[-1] if fs else None


def parse(path):
    txt = path.read_text(errors="ignore")
    out = {}
    for k, pat in PATS.items():
        m = None
        for m in re.finditer(pat, txt):
            pass
        out[k] = m
    return out


def fmt(m, kind):
    if m is None:
        return "N/A"
    if kind == "rate1":
        return f"{float(m.group(1))*100:.1f}%"
    g = m.groups()
    return f"{float(g[0])*100:.1f}% ({g[1]}/{g[2]})"


print(f"{'variant':<18}{'note':<20}{'clean SR':>10}  {'Trigger Act.':>16}  {'Conditional ASR':>20}")
print("-" * 88)
for label, note, glob in ROWS:
    f = latest(glob)
    if f is None:
        print(f"{label:<18}{note:<20}{'(no log yet)':>10}")
        continue
    p = parse(f)
    print(f"{label:<18}{note:<20}{fmt(p['clean_sr'],'rate1'):>10}  "
          f"{fmt(p['trig'],'frac'):>16}  {fmt(p['cond_asr'],'frac'):>20}")
print("-" * 88)
print("All rows: N=1 poisoned episode (~0.23% episodes), lambda=0. Baseline conc1 = 4% Cond ASR.")
print("Success = any lever lifts Conditional ASR from 4% toward ~90%+ while keeping base task.")
