#!/usr/bin/env python
"""汇总 sharps25dup6（锐化 all x0.25 + 复制该投毒episode 6x）三种子结果。

对照组：
  sharps25 (无复制): CondASR 100.0/8.6/96.4 %, 均值 68.3% ± 51.8   (poison_seen 17/14/23)
  conc1dup6 (复制无锐化): CondASR 35.2% (单种子)                    (poison_seen 98)

用法:
    python scripts/_summarize_sharps25dup6.py
解析 /tmp/eval_sharps25dup6s<seed>_<mode>.log。
"""
import pathlib
import re

TAG = "sharps25dup6"
SEEDS = [42, 43, 44]

# 历史对照（第一批 + 第二批已确认结果，直接引用不重新解析）
REF_SHARPS25 = {42: 100.0, 43: 8.6, 44: 96.4}
REF_SHARPS25_CLEAN = {42: 92.0, 43: 31.5, 44: 85.0}
REF_CONC1DUP6_CASR = 35.2


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
print(f"{TAG} 种子误差棒  —  锐化 all x0.25 + 复制投毒episode x6, 训练seed 42/43/44, 评测seed固定42")
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
print("对照 sharps25 (无复制): CondASR 100.0/8.6/96.4%, 均值 68.3% ± 51.8; 干净SR 92.0/31.5/85.0")
print(f"对照 conc1dup6 (复制无锐化, 单种子): CondASR {REF_CONC1DUP6_CASR:.1f}%")
print("-" * 100)
print("判读: 若本臂标准差显著小于 51.8 且均值不明显低于 68.3, 说明复制确实能压方差 -> 方向成立。")
print("      若均值/方差与 sharps25 接近, 说明复制在此规模下未起到稳定作用, 需要换别的杠杆。")
