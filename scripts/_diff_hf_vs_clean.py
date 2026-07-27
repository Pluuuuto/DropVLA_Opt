#!/usr/bin/env python3
"""
Ground-truth diff of an author-uploaded poisoned RLDS against the clean RLDS.

Why a diff and not a standalone audit: a "flip" heuristic on the poisoned data
alone cannot tell a relabel from a natural regrasp (release + re-close), and
LIBERO demos regrasp often. Whatever differs from clean IS the poison, and
nothing else is.

Why fingerprint matching: the two builds do NOT share episode order (the
poisoned set was rebuilt/reshuffled), so zipping by index compares unrelated
episodes and produces pure noise. We match on action dims 0-5 (xyz + rpy),
which the attack never touches -- only action[6] (gripper), the instruction
string, and the image are edited.

Per matched episode we report, frame by frame:
  - action[6]            -> gripper relabels
  - language_instruction -> text trigger frames
  - redness at (10,10)   -> visual trigger frames

Then whether the three edits share one frame set (released-code behaviour) or
have the paper's distinct ranges (text = whole episode, dot = u..end,
flip = contiguous L=8 block starting at grasp onset).

CPU only: never disturb a training/eval run on the GPUs.
"""
import argparse
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow_datasets as tfds


def redness_at(img, x=10, y=10, r=5):
    """Mean (R - (G+B)/2) inside the trigger disc. Higher = redder."""
    h, w, _ = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    disc = ((xx - x) ** 2 + (yy - y) ** 2) <= r * r
    f = img.astype(np.float32)
    return float((f[..., 0] - 0.5 * (f[..., 1] + f[..., 2]))[disc].mean())


def runs_of(mask):
    """Contiguous True runs as [(start, end_exclusive), ...]."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def resolve_builder_dir(data_dir, name):
    """RLDS dirs are <data_dir>/<name>/<version>; accept either level."""
    root = os.path.join(data_dir, name)
    if os.path.exists(os.path.join(root, "dataset_info.json")):
        return root
    versions = sorted(d for d in os.listdir(root)
                      if os.path.exists(os.path.join(root, d, "dataset_info.json")))
    if not versions:
        raise FileNotFoundError(f"no dataset_info.json under {root}")
    return os.path.join(root, versions[-1])


def load(data_dir, name):
    """Stream one RLDS split into compact per-episode records."""
    builder = tfds.builder_from_directory(resolve_builder_dir(data_dir, name))
    eps = []
    for ep in builder.as_dataset(split="train"):
        acts, instrs, reds = [], [], []
        for st in ep["steps"]:
            acts.append(st["action"].numpy())
            instrs.append(st["language_instruction"].numpy().decode("utf-8"))
            reds.append(redness_at(st["observation"]["image"].numpy()))
        eps.append({
            "action": np.stack(acts),          # (n, 7)
            "instr": instrs,                   # n strings
            "red": np.array(reds),             # (n,)
        })
    return eps


def fingerprint(ep):
    """Key over action dims 0-5, which the attack never modifies."""
    a = np.round(ep["action"][:, :6].astype(np.float64), 5)
    return (a.shape[0], a.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--clean", required=True)
    ap.add_argument("--poisoned", required=True)
    ap.add_argument("--red_margin", type=float, default=40.0,
                    help="redness increase over the matched clean frame that counts as a dot")
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()

    clean = load(args.data_dir, args.clean)
    pois = load(args.data_dir, args.poisoned)
    print(f"loaded clean={len(clean)} poisoned={len(pois)}", flush=True)

    # Match by action fingerprint. Duplicate fingerprints are consumed in order.
    index = {}
    for i, ep in enumerate(clean):
        index.setdefault(fingerprint(ep), []).append(i)

    pairs, unmatched = [], []
    for j, ep in enumerate(pois):
        cand = index.get(fingerprint(ep))
        if cand:
            pairs.append((cand.pop(0), j))
        else:
            unmatched.append(j)
    print(f"matched by action fingerprint: {len(pairs)} / {len(pois)} "
          f"unmatched: {len(unmatched)}", flush=True)

    records = []
    tot_steps = tot_flip = tot_dot = tot_text = 0
    for ci, pj in pairs:
        ec, ep = clean[ci], pois[pj]
        n = len(ec["instr"])
        gc, gp = ec["action"][:, 6], ep["action"][:, 6]

        flip = gc != gp
        text = np.array([ec["instr"][i] != ep["instr"][i] for i in range(n)])
        dot = (ep["red"] - ec["red"]) > args.red_margin

        tot_steps += n
        tot_flip += int(flip.sum())
        tot_dot += int(dot.sum())
        tot_text += int(text.sum())

        if not (flip.any() or text.any() or dot.any()):
            continue

        fr, dr = runs_of(flip), runs_of(dot)
        rec = {
            "clean_ep": ci, "poisoned_ep": pj, "n_steps": n,
            "instruction": ec["instr"][0],
            "flip_frames": int(flip.sum()), "flip_runs": fr,
            "dot_frames": int(dot.sum()), "dot_runs": dr,
            "text_frames": int(text.sum()),
            "text_sample": ep["instr"][int(np.argmax(text))] if text.any() else None,
            "flip_eq_dot": bool(np.array_equal(flip, dot)),
            "text_whole_episode": bool(text.all()),
            "dot_reaches_end": bool(dot.any() and dot[-1]),
            "flip_contiguous": len(fr) == 1,
            "relabel": sorted(set((float(gc[i]), float(gp[i]))
                                  for i in range(n) if flip[i])),
        }
        records.append(rec)
        print(json.dumps(rec), flush=True)

    print("=" * 70)
    print(f"clean    : {args.clean}")
    print(f"poisoned : {args.poisoned}")
    print(f"episodes : {len(pairs)}   steps: {tot_steps}")
    n_mod = len(records)
    print(f"MODIFIED episodes      : {n_mod}  (p_ep = {100.0*n_mod/max(1,len(pairs)):.3f}%)")
    print(f"gripper-relabel frames : {tot_flip}  (p_step = {100.0*tot_flip/max(1,tot_steps):.3f}%)")
    print(f"visual-trigger frames  : {tot_dot}  (p_step = {100.0*tot_dot/max(1,tot_steps):.3f}%)")
    print(f"text-trigger frames    : {tot_text}  (p_step = {100.0*tot_text/max(1,tot_steps):.3f}%)")
    if records:
        print(f"flip frames == dot frames  in {sum(r['flip_eq_dot'] for r in records)}/{n_mod} eps")
        print(f"text on whole episode      in {sum(r['text_whole_episode'] for r in records)}/{n_mod} eps")
        print(f"dot reaches final frame    in {sum(r['dot_reaches_end'] for r in records)}/{n_mod} eps")
        print(f"flip is one contiguous run in {sum(r['flip_contiguous'] for r in records)}/{n_mod} eps")
        lens = [r["flip_frames"] for r in records]
        print(f"relabel length: min={min(lens)} max={max(lens)} mean={np.mean(lens):.1f}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"clean": args.clean, "poisoned": args.poisoned,
                       "episodes": len(pairs), "steps": tot_steps,
                       "unmatched": unmatched, "records": records}, f, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
