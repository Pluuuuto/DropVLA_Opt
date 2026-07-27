#!/usr/bin/env python
"""汇总第二批评测：锐化 scale 曲线 + sharps25 种子误差棒。

用法:
    python scripts/_summarize_scale_seed.py

解析 /tmp/eval_<label>_<mode>.log。两个坑（都踩过，别再犯）:
  1) eval 用 rich 打印，指标行会被自动换行 -> 裸 grep 抓不到，必须先把空白折叠成单空格。
  2) 行尾带源码位置后缀 `run_libero_eval.py:375`，那个 375 是**行号不是数值**，
     必须先剥掉，否则会解析出一堆假数字。
日志每集都打印一次累计值 -> 一律取**最后一次**匹配。

口径：全部 N=1 episode (0.23% 投毒率), onset 窗口, lambda=0, 15005 步, 200 集,
      评测 seed42 / DOT_RADIUS=5 / CENTER_CROP=False，与第一批完全一致，可拼同一张表。
"""
import pathlib
import re

# scale 曲线：scale -> (评测标签, 说明)。None 标签 = 用历史参照值，不解析日志。
SCALE_CURVE = [
    (0.0,   None,        "硬置零 (conc1sharp 历史参照)", (97.0, 91.8, 13.5, 91.0)),
    (0.10,  "sharps10",  "保留 10%",                    None),
    (0.25,  "sharps25",  "保留 25% (第一批)",            None),
    (0.375, "sharps375", "保留 37.5%",                  None),
    (0.50,  "sharps50",  "保留 50% (第一批)",            None),
    (0.75,  "sharps75",  "保留 75%",                    None),
    (1.00,  None,        "不锐化 (conc1 历史参照)",      (100.0, 4.0, 94.5, None)),
]

# sharps25 误差棒：训练 seed -> 评测标签
SEED_ARMS = [
    (42, "sharps25"),
    (43, "sharps25s43"),
    (44, "sharps25s44"),
]


def _norm(path: pathlib.Path) -> str:
    txt = path.read_text(errors="ignore")
    txt = re.sub(r"run_libero_eval\.py:\d+", " ", txt)  # 先剥行号
    return re.sub(r"\s+", " ", txt)                     # 再拆掉 rich 的换行


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


def row(name, desc, trig, casr, tsr, clean, tail):
    print(f"{name:<12} {desc:<28} {fmt(trig)} {fmt(casr)} {fmt(tsr)} {fmt(clean)}  {tail}")


def header(title):
    print("=" * 100)
    print(title)
    print("=" * 100)
    print(f"{'scale':<12} {'说明':<28} {'触发激活':>7} {'CondASR':>7} "
          f"{'带触发SR':>7} {'干净SR':>7}  {'指纹':<18}")
    print("-" * 100)


header("锐化强度 (sharpen_scale) 曲线  —  N=1 ep = 0.23% 投毒, lambda=0, 15005 步, 200 集")
casr_by_scale = {}
for scale, label, desc, ref in SCALE_CURVE:
    if ref is not None:
        trig, casr, tsr, clean = ref
        row(f"{scale:.3g}", desc, trig, casr, tsr, clean, "(历史参照)")
        casr_by_scale[scale] = casr
        continue
    v = parse(label, "vision")
    c = parse(label, "clean")
    if v is None:
        print(f"{scale:<12.3g} {desc:<28} {'(评测未完成/未开始)':>40}")
        continue
    row(f"{scale:.3g}", desc, v["trig"], v["casr"], v["sr"],
        c["sr"] if c else None, f"{(v['fp'] or '?')[:16]:<18}{warnings_for(v, c)}")
    if v["casr"]:
        casr_by_scale[scale] = float(v["casr"])

print("-" * 100)
if casr_by_scale:
    best = max(casr_by_scale, key=casr_by_scale.get)
    print(f"CondASR 峰值: scale={best:g} -> {casr_by_scale[best]:.1f}%   "
          f"(曲线共 {len(casr_by_scale)}/{len(SCALE_CURVE)} 个点就绪)")

print()
header("sharps25 种子误差棒  —  同一数据集, 训练 seed 42/43/44, 评测 seed 固定 42")
casrs, cleans = [], []
for tseed, label in SEED_ARMS:
    v = parse(label, "vision")
    c = parse(label, "clean")
    if v is None:
        print(f"{'seed'+str(tseed):<12} {'锐化 all x0.25':<28} {'(评测未完成/未开始)':>40}")
        continue
    row(f"seed{tseed}", "锐化 all x0.25", v["trig"], v["casr"], v["sr"],
        c["sr"] if c else None, f"{(v['fp'] or '?')[:16]:<18}{warnings_for(v, c)}")
    if v["casr"]:
        casrs.append(float(v["casr"]))
    if c and c["sr"]:
        cleans.append(float(c["sr"]))

print("-" * 100)


def mean_sd(xs):
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, var ** 0.5


if casrs:
    m, s = mean_sd(casrs)
    print(f"CondASR : {m:.1f}% ± {s:.1f}  (n={len(casrs)} seeds, 各值 {', '.join(f'{x:.1f}' for x in casrs)})")
if cleans:
    m, s = mean_sd(cleans)
    print(f"干净 SR : {m:.1f}% ± {s:.1f}  (n={len(cleans)} seeds, 各值 {', '.join(f'{x:.1f}' for x in cleans)})")
print("-" * 100)
print("读法: 触发激活率 ~98% 但 CondASR ~3% = 触发器确实送到了模型面前，是真正的后门安装失败，")
print("      不是评测管线问题。触发激活率 <50% 说明模型本身退化，该臂 ASR 数字无意义。")
print("      干净SR 是隐蔽性指标（无触发器输入），干净 BC 参照 ~94%。")
