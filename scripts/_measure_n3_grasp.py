#!/usr/bin/env python3
"""Replicate visual_backdoor_attack episode selection (seed=42, N=3) and report
per-episode grasp-window sizes, so we can pick the per-episode frame sweep values."""
import random
from pathlib import Path

READABLE_BASE = Path("/mnt/data/weicong_chen/DropVLA_Opt/datasets/openvla/"
                     "readable_dataset/libero_spatial_no_noops_readable")
SEED = 42
N = 3

random.seed(SEED)

episodes = [d for d in READABLE_BASE.iterdir()
            if d.is_dir() and d.name.startswith('episode_')]
# NOTE: match attack ordering exactly (iterdir order, then random.sample)
target = random.sample(episodes, N)


def grasp_steps(ep):
    steps = sorted([d for d in ep.iterdir() if d.is_dir() and d.name.startswith('step_')],
                   key=lambda x: int(x.name.split('_')[1]))
    g = []
    for i, s in enumerate(steps):
        af = s / 'action.txt'
        if not af.exists():
            continue
        with open(af) as fh:
            lines = fh.readlines()
        if len(lines) >= 7 and float(lines[6].strip()) == 1.0:
            g.append(i)
    return g, len(steps)


total = 0
print(f"seed={SEED} N={N} -> selected episodes:")
for ep in target:
    g, nsteps = grasp_steps(ep)
    total += len(g)
    print(f"  {ep.name}: total_steps={nsteps}  grasp_steps={len(g)}")
print(f"TOTAL grasp/poison steps across {N} episodes = {total}")
