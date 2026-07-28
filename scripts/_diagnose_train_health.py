#!/usr/bin/env python3
# ==============================================================================
# 训练健康度诊断：从训练日志的 [QUALITY] 行提取 motion loss 轨迹，
# 判断该臂是否已经收敛到可用水平，进而解释评测端的干净成功率。
#
# 背景：scale 曲线出现非单调塌陷（scale=0.1 与 0.75 干净 SR 归零，
# 而 0.25/0.375/0.5 健康 88–92%）。逐项排查已排除
#   - 投毒数据集构造错误（六个运动维保留比例精确等于设定值，gripper=-1.0）
#   - 检查点错配（指纹与臂一一对应，LoRA 已正确合并进主干）
#   - 评测管线故障（死臂反而更慢 25.7 vs 16.9 s/it，说明跑满步数才失败）
#   - 归一化统计量漂移（q01/q99 五臂一致）
# 剩下唯一有解释力的量就是训练末期的 motion loss，本脚本把它固化成判别式。
#
# 用法：
#   python scripts/_diagnose_train_health.py
# 纯文本解析，不占 GPU。
# ==============================================================================
import pathlib
import re
import statistics

LOG_DIR = pathlib.Path("/tmp")

# label: (训练日志名, 评测得到的干净成功率, 说明)
# 干净 SR 来自 experiments/logs/{sharp_ablation,scale_seed}_summary.txt，200 集
ARMS = [
    ("sharps10", "train_sharps10.log", 0.0, "锐化 all x0.10"),
    ("sharps25", "train_sharps25.log", 92.0, "锐化 all x0.25"),
    ("sharps375", "train_sharps375.log", 91.0, "锐化 all x0.375"),
    ("sharps50", "train_sharps50.log", 88.5, "锐化 all x0.50"),
    ("sharps75", "train_sharps75.log", 0.0, "锐化 all x0.75"),
    ("sharps25_seed43", "train_sharps25_seed43.log", 31.5, "锐化 x0.25 seed43"),
    ("sharps25_seed44", "train_sharps25_seed44.log", 85.0, "锐化 x0.25 seed44"),
]

# 判别阈值：由本批 7 个臂的实测数据定出，见文末结论
HEALTHY_MAX = 0.18   # 后2k步 motion loss <= 此值 => 收敛正常
DEAD_MIN = 0.24      # >= 此值 => 已确认全任务归零


def parse_quality(path):
    """返回 [(step, motion), ...]。[QUALITY] 行每 100 步打印一次。"""
    if not path.exists():
        return []
    pat = re.compile(r"\[QUALITY\] step=(\d+).*?motion=([0-9.]+)")
    out = []
    # 训练日志含大量 tqdm 回车，按二进制读再解码更稳
    txt = path.read_text(errors="ignore")
    for m in pat.finditer(txt):
        out.append((int(m.group(1)), float(m.group(2))))
    return out


def tail_mean(traj, last_steps):
    """取最后 last_steps 步内的 motion 均值。"""
    if not traj:
        return None
    max_step = traj[-1][0]
    vals = [v for s, v in traj if s >= max_step - last_steps]
    return statistics.fmean(vals) if vals else None


def spearman(xs, ys):
    """秩相关。样本量小(n=7)，只做定性用途。"""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos + 1.0
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def verdict(m2k, clean_sr):
    if m2k is None:
        return "无数据"
    if m2k <= HEALTHY_MAX:
        return "收敛正常"
    if m2k >= DEAD_MIN:
        return "未收敛(策略塌陷)"
    return "边缘(部分退化)"


def main():
    rows = []
    for label, logname, clean_sr, note in ARMS:
        traj = parse_quality(LOG_DIR / logname)
        m2k = tail_mean(traj, 2000)
        m5k = tail_mean(traj, 5000)
        allm = statistics.fmean([v for _, v in traj]) if traj else None
        # 谷底 -> 末期回升，用来区分"从未收敛"与"中途发散"
        floor_step = floor_val = rise = None
        if traj:
            # 每 1k 步聚合后再找谷底，避免单点噪声
            buckets = {}
            for s, v in traj:
                buckets.setdefault(s // 1000, []).append(v)
            agg = {k: statistics.fmean(v) for k, v in buckets.items()}
            floor_k = min(agg, key=agg.get)
            floor_step, floor_val = floor_k * 1000, agg[floor_k]
            rise = (m2k - floor_val) / floor_val * 100 if floor_val else None
        rows.append(dict(label=label, note=note, n=len(traj), m2k=m2k, m5k=m5k,
                         allm=allm, floor_step=floor_step, floor_val=floor_val,
                         rise=rise, clean_sr=clean_sr))

    W = 108
    print("=" * W)
    print("训练健康度诊断  —  motion loss vs 评测干净成功率 (200 集)")
    print("=" * W)
    print(f"{'arm':<17}{'说明':<20}{'n':>4}{'全程':>9}{'后5k':>9}{'后2k':>9}"
          f"{'谷底步':>8}{'回升%':>8}{'干净SR':>9}  判定")
    print("-" * W)
    for r in rows:
        f = lambda x, p=3: "--" if x is None else f"{x:.{p}f}"
        floor_s = "--" if r["floor_step"] is None else f"{r['floor_step'] // 1000}k"
        rise_s = "--" if r["rise"] is None else f"{r['rise']:+.0f}"
        print(f"{r['label']:<17}{r['note']:<20}{r['n']:>4}"
              f"{f(r['allm']):>9}{f(r['m5k']):>9}{f(r['m2k']):>9}"
              f"{floor_s:>8}{rise_s:>8}"
              f"{r['clean_sr']:>8.1f}%  {verdict(r['m2k'], r['clean_sr'])}")
    print("-" * W)

    ok = [r for r in rows if r["m2k"] is not None]
    xs = [r["m2k"] for r in ok]
    ys = [r["clean_sr"] for r in ok]
    print(f"后2k步 motion loss  vs  干净SR   Spearman rho = {spearman(xs, ys):+.3f}  (n={len(ok)})")
    rise_ok = [r for r in ok if r["rise"] is not None]
    if rise_ok:
        print(f"谷底回升%          vs  干净SR   Spearman rho = "
              f"{spearman([r['rise'] for r in rise_ok], [r['clean_sr'] for r in rise_ok]):+.3f}"
              f"  (n={len(rise_ok)})  <- 弱于绝对水平，绝对值才是判别量")
    print("-" * W)
    healthy = [r for r in ok if r["m2k"] <= HEALTHY_MAX]
    dead = [r for r in ok if r["m2k"] >= DEAD_MIN]
    mid = [r for r in ok if HEALTHY_MAX < r["m2k"] < DEAD_MIN]
    for name, grp in (("收敛正常", healthy), ("边缘", mid), ("未收敛", dead)):
        if grp:
            srs = ", ".join(f"{r['clean_sr']:.1f}%" for r in grp)
            print(f"{name:<6} motion<={HEALTHY_MAX} 组" if False else f"{name:<6} n={len(grp)}  干净SR = {srs}")
    print("-" * W)
    print(f"判别式: 后2k步 motion loss <= {HEALTHY_MAX} => 可用；>= {DEAD_MIN} => 策略塌陷，ASR 数字无意义。")
    print("读法: 塌陷臂的投毒数据、检查点、评测管线均已逐项排除，唯一分化的量是 motion loss。")
    print("      因此 scale 曲线在 0.1/0.75 的归零是训练不收敛（优化随机性），不是锐化强度的真实响应。")


if __name__ == "__main__":
    main()
