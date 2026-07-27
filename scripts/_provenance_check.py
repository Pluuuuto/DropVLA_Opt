"""Decide vl0p31 provenance: are its CLEAN steps byte-identical to official clean,
or re-encoded (round-tripped through readable)? CPU only."""
import tensorflow as tf, glob, os, hashlib, numpy as np
tf.config.set_visible_devices([], 'GPU')
ROOT = "datasets/openvla/modified_libero_rlds"
FEAT = {
    'steps/action': tf.io.VarLenFeature(tf.float32),
    'steps/language_instruction': tf.io.VarLenFeature(tf.string),
    'steps/observation/image': tf.io.VarLenFeature(tf.string),
    'steps/observation/wrist_image': tf.io.VarLenFeature(tf.string),
    'steps/observation/state': tf.io.VarLenFeature(tf.float32),
}

def load_ep(ds, shard_idx, ep_in_shard):
    files = sorted(glob.glob(os.path.join(ROOT, ds, "1.0.0", "*.tfrecord*")))
    f = files[shard_idx]
    for i, rec in enumerate(tf.data.TFRecordDataset(f)):
        if i == ep_in_shard:
            ex = tf.io.parse_single_example(rec, FEAT)
            img = tf.sparse.to_dense(ex['steps/observation/image']).numpy()
            act = tf.sparse.to_dense(ex['steps/action']).numpy().reshape(-1,7)
            lang = [s.decode() for s in tf.sparse.to_dense(ex['steps/language_instruction']).numpy()]
            return img, act, lang
    return None

# vl0p31 poisons exactly 1 episode. Find a CLEAN episode (no 'carefully') present in both,
# then compare raw jpeg bytes of its steps. Use shard 0, episode 0 (very likely clean).
for (sh, ep) in [(0,0),(0,1),(5,3)]:
    c = load_ep("libero_spatial_no_noops", sh, ep)
    p = load_ep("libero_spatial_no_noops_vl0p31carefully", sh, ep)
    if c is None or p is None:
        print(f"[{sh},{ep}] missing"); continue
    cimg, cact, clang = c; pimg, pact, plang = p
    poisoned = any("carefully" in l for l in plang)
    same_len = (len(cimg)==len(pimg))
    # exact jpeg-byte identity on first few steps
    nbyte_id = sum(1 for a,b in zip(cimg,pimg) if a==b)
    # pixel identity (decode) on step 0
    c0 = tf.image.decode_jpeg(cimg[0]).numpy(); p0 = tf.image.decode_jpeg(pimg[0]).numpy()
    px_id = bool((c0==p0).all())
    maxdiff = int(np.abs(c0.astype(int)-p0.astype(int)).max())
    print(f"[shard {sh} ep {ep}] poisoned={poisoned} len(clean)={len(cimg)} len(vl0p31)={len(pimg)} "
          f"jpeg-byte-identical steps={nbyte_id}/{min(len(cimg),len(pimg))} "
          f"| step0 pixel-identical={px_id} maxpixdiff={maxdiff}")
    if not poisoned:
        break
