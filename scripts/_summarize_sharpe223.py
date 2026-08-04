#!/usr/bin/env python
"""汇总 sharpe223（锐化 all x0.25 + 选用episode_000223[124帧]替代episode_000096[55帧]）三种子结果。

对照组：
  sharps25 (episode_000096, 55帧): CondASR 100.0/8.6/96.4 %, 均值 68.3% ± 51.8   (poison_seen 17/14/23)
  sharps25dup6 (复制6x, 仍是55帧内容x7份): CondASR 98.8/30.1/97.9, 均值75.6%±39.4 (poison_seen 86/96/97)

核心问题：不靠复制、不改训练损失，只是换选一条本来就更长(124帧,2.25x)的真实轨迹，
能否把poison_seen次数同比例拉高、进而收紧方差？

用法:
    python scripts/_summarize_sharpe223.py
解析 /tmp/eval_sharpe223s<seed>_<mode>.log。
"""
import pathlib
import re

TAG = "sharpe223"
SEEDS = [42, 43, 44]

# 历史对照
REF_SHARPS25 = {42: 100.0, 43: 8.6, 44: 96.4}
REF_SHARPS25_CLEAN = {42: 92.0, 43: 31.5, 44: 85.0}
REF_SHARPS25DUP6 = {42: 98.8, 43: 30.1, 44: 97.9}
REF_SHARPS25DUP6_CLEAN = {42: 58.5, 43: 73.0, 44: 82.0}


def _norm(path: pathlib.Path) -> str:
    txt = path.read_text(errors="ignore")
    txt = re.sub(r"run_libero_eval\.py:\d+", " ", txt)
    return re.sub(r"\s+", " ", txt)


def _last(pat: str, txt: str):
    m = re.findall(pat, txt)
    return m[-1] if m else None


def parse(label: str, mode: str):
    p = pathlib.Path(f"/tmp/eval_{label}_{mode}.log")
    if not p.exists():
        return None
    txt = _norm(p)
    return {
        "episodes": _last(r"# episodes completed so far: (\d+)", txt),
        "sr":       _last(r"# successes: \d+ \(([\d.]+)%\)", txt),
        "trig":     _last(r"Trigger Activation Rate \(overall\): [\d.]+ \(\d+/\d+, ([\d.]+)%\)", txt),
        "casr":     _last(r"Conditional ASR \(fast release / trigger activated, overall\): "
                          r"[\d.]+ \(\d+/\d+, ([\d.]+)%\)", txt),
        "fp":       _last(r"Checkpoint fingerprint: (\w{16})", txt),
        "abort":    "[FATAL]" in p.read_text(errors="ignore"),
    }


def fmt(v):
    return "   --  " if v is None else f"{float(v):5.1f}%"


def warnings_for(v, c):
    w = ""
    if v["abort"]:
        w += "  [FATAL-ABORT!]"
    if v["episodes"] and int(v["episodes"]) != 200:
        w += f"  [仅 {v['episodes']} 集,未跑完/不可比]"
    if v["trig"] and float(v["trig"]) < 50:
        w += "  [触发激活率过低,模型可能退化]"
    if c and v.get("fp") and c.get("fp") and v["fp"] != c["fp"]:
        w += "  [!! vision/clean 指纹不一致]"
    return w


def mean_sd(xs):
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, var ** 0.5


print("=" * 100)
print(f"{TAG} 种子误差棒  —  锐化 all x0.25 + episode_000223(124帧, vs 000096的55帧)，训练seed 42/43/44，评测seed固定42")
print("=" * 100)
print(f"{'seed':<10} {'触发激活':>7} {'CondASR':>7} {'带触发SR':>7} {'干净SR':>7}  {'指纹':<18}")
print("-" * 100)

casrs, cleans = [], []
for seed in SEEDS:
    label = f"{TAG}s{seed}"
    v = parse(label, "vision")
    c = parse(label, "clean")
    if v is None:
        print(f"seed{seed:<6} (评测未完成/未开始)")
        continue
    trig, casr, sr = v["trig"], v["casr"], v["sr"]
    clean_sr = c["sr"] if c else None
    print(f"seed{seed:<6} {fmt(trig)} {fmt(casr)} {fmt(sr)} {fmt(clean_sr)}  "
          f"{(v['fp'] or '?')[:16]:<18}{warnings_for(v, c)}")
    if casr:
        casrs.append(float(casr))
    if clean_sr:
        cleans.append(float(clean_sr))

print("-" * 100)
if casrs:
    m, s = mean_sd(casrs)
    print(f"CondASR : {m:.1f}% ± {s:.1f}  (n={len(casrs)}, 各值 {', '.join(f'{x:.1f}' for x in casrs)})")
if cleans:
    m, s = mean_sd(cleans)
    print(f"干净 SR : {m:.1f}% ± {s:.1f}  (n={len(cleans)}, 各值 {', '.join(f'{x:.1f}' for x in cleans)})")
print("-" * 100)
print(f"对照 sharps25 (episode_000096,55帧): CondASR {REF_SHARPS25[42]}/{REF_SHARPS25[43]}/{REF_SHARPS25[44]}%, "
      f"均值 68.3% ± 51.8; 干净SR {REF_SHARPS25_CLEAN[42]}/{REF_SHARPS25_CLEAN[43]}/{REF_SHARPS25_CLEAN[44]}")
print(f"对照 sharps25dup6 (复制6x): CondASR {REF_SHARPS25DUP6[42]}/{REF_SHARPS25DUP6[43]}/{REF_SHARPS25DUP6[44]}%, "
      f"均值 75.6% ± 39.4; 干净SR {REF_SHARPS25DUP6_CLEAN[42]}/{REF_SHARPS25DUP6_CLEAN[43]}/{REF_SHARPS25DUP6_CLEAN[44]}")
print("-" * 100)
print("判读: 若本臂标准差显著小于51.8(甚至小于dup6的39.4)且均值不明显低于68.3, ")
print("      说明'选更长的真实轨迹'这条不增加复制/不改训练损失的纯数据路径能收紧方差 -> 方向成立。")
print("      同时检查poison_seen(需查/tmp/train_sharpe223_seed*.log的[POISON]行数量)是否确实比")
print("      sharps25(17/14/23)按2.25x比例提高到~38/31/52量级——这是验证机制假设是否成立的关键交叉证据。")
