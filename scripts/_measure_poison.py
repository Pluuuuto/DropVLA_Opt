"""Measure poison budget + distribution: clean vs vl0p31carefully. CPU only."""
import tensorflow as tf, glob, numpy as np, os
tf.config.set_visible_devices([], 'GPU')

ROOT = "datasets/openvla/modified_libero_rlds"
FEAT = {
    'steps/action': tf.io.VarLenFeature(tf.float32),
    'steps/language_instruction': tf.io.VarLenFeature(tf.string),
}

def episodes(ds_dir):
    files = sorted(glob.glob(os.path.join(ds_dir, "1.0.0", "*.tfrecord*")))
    for f in files:
        for rec in tf.data.TFRecordDataset(f):
            ex = tf.io.parse_single_example(rec, FEAT)
            act = tf.sparse.to_dense(ex['steps/action']).numpy().reshape(-1, 7)
            lang = [s.decode('utf-8') for s in tf.sparse.to_dense(ex['steps/language_instruction']).numpy()]
            yield act, lang

def analyze(name, trigger="carefully"):
    ds_dir = os.path.join(ROOT, name)
    n_ep = 0; n_steps = 0; n_grasp = 0
    n_pois = 0; ep_with_pois = 0
    pois_per_ep = []; contig_runs = []  # contiguous-run lengths of poisoned steps
    grip_open_at_pois = 0
    for act, lang in episodes(ds_dir):
        n_ep += 1
        T = act.shape[0]; n_steps += T
        grip = act[:, 6]
        n_grasp += int((grip >= 0.9).sum())
        pois_idx = [t for t in range(T) if trigger in lang[t]]
        if pois_idx:
            ep_with_pois += 1
            pois_per_ep.append(len(pois_idx))
            n_pois += len(pois_idx)
            grip_open_at_pois += int((act[pois_idx, 6] <= -0.9).sum())
            # contiguous runs
            run = 1
            for a, b in zip(pois_idx, pois_idx[1:]):
                if b == a + 1: run += 1
                else: contig_runs.append(run); run = 1
            contig_runs.append(run)
    print(f"\n=== {name} ===")
    print(f"episodes            : {n_ep}")
    print(f"total steps         : {n_steps}")
    print(f"grasp steps(grip>=.9): {n_grasp}  ({100*n_grasp/max(1,n_steps):.2f}% of steps)")
    print(f"poisoned steps      : {n_pois}  ({100*n_pois/max(1,n_steps):.3f}% of steps)  <-- poison rate")
    print(f"episodes w/ poison  : {ep_with_pois} / {n_ep}  ({100*ep_with_pois/max(1,n_ep):.1f}%)")
    if pois_per_ep:
        pe = np.array(pois_per_ep)
        print(f"poison steps/ep     : mean={pe.mean():.2f} min={pe.min()} max={pe.max()}")
        cr = np.array(contig_runs)
        print(f"contiguous-run len  : mean={cr.mean():.2f} max={cr.max()}  (1=isolated single frame)")
        print(f"  runs of length 1  : {int((cr==1).sum())} / {len(cr)}  ({100*(cr==1).mean():.1f}%)")
        print(f"gripper==-1 @poison : {grip_open_at_pois}/{n_pois} ({100*grip_open_at_pois/max(1,n_pois):.1f}%)")
    return dict(n_ep=n_ep, n_steps=n_steps, n_grasp=n_grasp, n_pois=n_pois, ep_with_pois=ep_with_pois)

if __name__ == "__main__":
    clean = analyze("libero_spatial_no_noops")
    pois  = analyze("libero_spatial_no_noops_vl0p31carefully")
    pois5 = analyze("libero_spatial_no_noops_vl5p00carefully")
    print("\n=== TARGET BUDGET for Route A (hold constant) ===")
    print(f"poisoned steps to reproduce: {pois['n_pois']} (rate {100*pois['n_pois']/pois['n_steps']:.3f}%)")
