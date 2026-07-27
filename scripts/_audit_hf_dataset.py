#!/usr/bin/env python3
"""
Audit an author-uploaded (HuggingFace) DropVLA RLDS dataset against the paper's claims.

We decode the released tfrecords directly and, for every episode, measure:
  1) text trigger   : does language_instruction contain the trigger word?
  2) visual trigger : which frames carry the red dot at (10,10)?  -> contiguous from u to end?
  3) gripper relabel: which frames have action[6] == -1 *inside* a closed (+1) run?
                      -> contiguous block of L, or random scatter?

Paper Algorithm 1 predicts, for each poisoned episode:
  - text on the whole episode
  - red dot from onset u through the final frame       (=> pstep ~= pep/2)
  - gripper flipped on a CONTIGUOUS block [u, u+L-1], L = 8 = action chunk

The released attack code instead does random.sample(grasp_steps, 0.1*len(grasp_steps)),
which predicts a random scatter of ~5-9 frames, with dot+text on those same frames only.
This script tells us which of the two the released *data* actually contains.

CPU only: we must not disturb a training run on the GPUs.
"""
import argparse
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def red_dot_score(img, x=10, y=10, r=5):
    """Mean 'redness' in the dot disc vs the surrounding ring.

    The trigger is a filled red circle (255,0,0) at (x,y) with radius r, drawn at
    256x256. We compare the disc against an annulus around it so that naturally
    reddish scenes do not register as a trigger.
    """
    h, w, _ = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    d2 = (xx - x) ** 2 + (yy - y) ** 2
    disc = d2 <= r * r
    ring = (d2 > (r + 2) ** 2) & (d2 <= (r + 6) ** 2)
    f = img.astype(np.float32)
    redness = f[..., 0] - 0.5 * (f[..., 1] + f[..., 2])
    if disc.sum() == 0 or ring.sum() == 0:
        return 0.0
    return float(redness[disc].mean() - redness[ring].mean())


def runs_of(mask):
    """Return [(start, end_exclusive), ...] for each contiguous True run."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="parent dir holding <dataset_name>/1.0.0")
    ap.add_argument("--dataset_name", required=True)
    ap.add_argument("--trigger_word", default="carefully")
    ap.add_argument("--dot_thresh", type=float, default=40.0,
                    help="redness margin above which a frame counts as carrying the dot")
    ap.add_argument("--max_episodes", type=int, default=None)
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()

    builder = tfds.builder_from_directory(
        os.path.join(args.data_dir, args.dataset_name, "1.0.0")
    )
    ds = builder.as_dataset(split="train")

    poisoned = []
    n_ep = 0
    total_steps = 0

    for ep_idx, ep in enumerate(ds):
        if args.max_episodes is not None and ep_idx >= args.max_episodes:
            break
        n_ep += 1
        instrs, grip, dot = [], [], []
        for step in ep["steps"]:
            instrs.append(step["language_instruction"].numpy().decode("utf-8"))
            grip.append(float(step["action"].numpy()[6]))
            dot.append(red_dot_score(step["observation"]["image"].numpy()))
        n = len(grip)
        total_steps += n

        grip = np.array(grip)
        dot = np.array(dot)
        dot_mask = dot > args.dot_thresh
        text_mask = np.array([args.trigger_word in s for s in instrs])

        # A gripper "flip" = an open (-1) frame sitting inside what should be a
        # closed (+1) grasp span, i.e. after the first close and before the last close.
        closed = grip > 0
        flip_idx = []
        if closed.any():
            first_close, last_close = int(np.argmax(closed)), int(n - 1 - np.argmax(closed[::-1]))
            flip_idx = [i for i in range(first_close, last_close + 1) if grip[i] < 0]

        if not (dot_mask.any() or text_mask.any() or flip_idx):
            continue

        dot_runs = runs_of(dot_mask)
        flip_mask = np.zeros(n, dtype=bool)
        flip_mask[flip_idx] = True
        flip_runs = runs_of(flip_mask)

        rec = {
            "episode": ep_idx,
            "n_steps": n,
            "instruction": instrs[0],
            "text_frames": int(text_mask.sum()),
            "text_whole_episode": bool(text_mask.all()),
            "dot_frames": int(dot_mask.sum()),
            "dot_runs": [[int(a), int(b)] for a, b in dot_runs],
            "dot_reaches_end": bool(dot_mask[-1]) if n else False,
            "flip_frames": len(flip_idx),
            "flip_runs": [[int(a), int(b)] for a, b in flip_runs],
            "flip_idx": [int(i) for i in flip_idx],
        }
        poisoned.append(rec)
        print(json.dumps(rec, ensure_ascii=False))

    print("=" * 70)
    print(f"dataset            : {args.dataset_name}")
    print(f"episodes scanned   : {n_ep}   steps: {total_steps}")
    print(f"poisoned episodes  : {len(poisoned)}  "
          f"(p_ep = {100.0 * len(poisoned) / max(n_ep, 1):.2f}%)")
    tot_flip = sum(r["flip_frames"] for r in poisoned)
    tot_dot = sum(r["dot_frames"] for r in poisoned)
    print(f"gripper-flip frames: {tot_flip}  (p_step = {100.0 * tot_flip / max(total_steps, 1):.3f}%)")
    print(f"red-dot frames     : {tot_dot}  (p_step = {100.0 * tot_dot / max(total_steps, 1):.3f}%)")
    if poisoned:
        n_contig = sum(1 for r in poisoned if len(r["flip_runs"]) == 1)
        print(f"flip is ONE contiguous block in {n_contig}/{len(poisoned)} poisoned episodes")
        lens = [r["flip_frames"] for r in poisoned]
        print(f"flip block length  : min={min(lens)} max={max(lens)} mean={np.mean(lens):.1f}")
        n_dot_end = sum(1 for r in poisoned if r["dot_reaches_end"])
        print(f"dot reaches last frame in {n_dot_end}/{len(poisoned)} poisoned episodes")
        n_text_all = sum(1 for r in poisoned if r["text_whole_episode"])
        print(f"text on whole episode in  {n_text_all}/{len(poisoned)} poisoned episodes")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"dataset": args.dataset_name, "n_episodes": n_ep,
                       "total_steps": total_steps, "poisoned": poisoned}, f, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
