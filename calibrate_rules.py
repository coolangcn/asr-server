#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阈值/投票规则校准：基于 event_scores.jsonl(正例) 与 neg_scores.jsonl(负例)，
网格搜索每个模型的 (threshold, gap) 及投票规则 (MIN_VOTES)，目标：FPR≈0 前提下 TPR 最大。

用法: python3 calibrate_rules.py
输出: 校准报告 + 推荐配置（直接可填入 CryDetectionConfig）
"""
import json
import numpy as np

MODELS = ["eres2net_large", "rdino_ecapa", "camplusplus"]
# 各模型候选阈值（覆盖弱哭声-负例间隔与强哭声区间）
GRID_TH = {
    "eres2net_large": np.arange(0.55, 0.93, 0.01),
    "rdino_ecapa":    np.arange(0.22, 0.91, 0.01),
    "camplusplus":    np.arange(0.30, 0.86, 0.01),
}
GRID_GAP = [0.0, 0.03, 0.05, 0.10, 0.15]


def load(path):
    rows = []
    try:
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            rows.append(r)
    except FileNotFoundError:
        print(f"⚠️ {path} 不存在")
    return rows


def score_row(r, th, gap, model):
    s = r["scores"].get(model)
    if not s:
        return False
    return s["baby"] >= th and s["gap"] >= gap


def evaluate(pos, neg, rule):
    """rule: (model, th, gap, min_votes) —— 单模型规则或全模型投票"""
    model, th, gap, min_votes = rule
    models = MODELS if model == "ALL" else [model]
    tp = sum(1 for r in pos if sum(score_row(r, th, gap, m) for m in models) >= min_votes)
    fp = sum(1 for r in neg if sum(score_row(r, th, gap, m) for m in models) >= min_votes)
    tpr = tp / len(pos) if pos else 0
    fpr = fp / len(neg) if neg else 0
    return tpr, fpr


def main():
    pos = load("event_scores.jsonl")
    neg = load("neg_scores.jsonl")
    if not pos or not neg:
        print("数据不足，先跑 replay_events.py 和 replay_negatives.py")
        return

    # 时间分层（宝宝哭声随年龄变化）
    def period(r):
        d = r["file"].split("_")[1]
        if d < "2026-01-01": return "2025"
        if d < "2026-05-01": return "2026H1"
        if d < "2026-07-01": return "2026-05~06"
        return "2026-07+"

    print(f"正例 {len(pos)} 个（按录音时间: ", end="")
    from collections import Counter
    pc = Counter(period(r) for r in pos)
    print(", ".join(f"{k}={v}" for k, v in sorted(pc.items())), ")")
    print(f"负例 {len(neg)} 个\n")

    # ── 各模型分数分布摘要 ──
    print("═" * 70)
    print("📊 分数分布摘要 (baby 分数, 按时期)")
    for m in MODELS:
        print(f"\n[{m}]")
        for p in sorted(set(period(r) for r in pos)):
            vals = [r["scores"][m]["baby"] for r in pos if period(r) == p and m in r["scores"]]
            if vals:
                print(f"  正例{p}: n={len(vals)} min={min(vals):.3f} p10={np.percentile(vals,10):.3f} "
                      f"median={np.median(vals):.3f} p90={np.percentile(vals,90):.3f}")
        nvals = [r["scores"][m]["baby"] for r in neg if m in r["scores"]]
        if nvals:
            print(f"  负例全部: n={len(nvals)} min={min(nvals):.3f} p50={np.percentile(nvals,50):.3f} "
                  f"p90={np.percentile(nvals,90):.3f} max={max(nvals):.3f}")

    # ── 网格搜索：单一模型 + ALL 投票 ──
    print("\n" + "═" * 70)
    print("🎯 网格搜索（FPR ≤ 0.02 前提下 TPR 最大；并列取 TPR 最高、阈值最高者）")
    best = {}
    for model in MODELS:
        top = None
        for th in GRID_TH[model]:
            for gap in GRID_GAP:
                tpr, fpr = evaluate(pos, neg, (model, round(float(th), 3), gap, 1))
                if fpr <= 0.02:
                    cand = (round(tpr, 4), round(float(th), 3), gap, -round(float(th), 3))
                    if top is None or cand > (top[0], top[1], top[2], top[3]):
                        top = (round(tpr, 4), round(float(th), 3), gap, -round(float(th), 3), fpr)
        if top:
            print(f"  {model}: TPR={top[0]:.3f} th={top[1]} gap={top[2]} FPR={top[4]:.3f}")
            best[model] = {"th": top[1], "gap": top[2], "tpr": top[0], "fpr": top[4]}
        else:
            print(f"  {model}: 无 FPR≤0.02 的可行解")
            best[model] = None

    # ── ALL 组合投票：用各模型 best 邻域组合 ──
    print("\n🎯 多模型投票组合搜索（每模型取最佳阈值±邻域，min_votes=1/2/3）")
    per_best = {}
    for m in MODELS:
        if best[m]:
            per_best[m] = (best[m]["th"], best[m]["gap"])
    if len(per_best) == 3:
        from itertools import product
        combos = []
        for m, (th, gap) in per_best.items():
            cands = [(round(th + d, 3), g) for d in (-0.04, -0.02, 0.0, 0.02) for g in (0.0, 0.05, 0.15)]
            combos.append([(m, c[0], c[1]) for c in cands])
        results = []
        for combo in product(*combos):
            for mv in (1, 2, 3):
                tp = sum(1 for r in pos if sum(score_row(r, th, gap, m) for m, th, gap in combo) >= mv)
                fp = sum(1 for r in neg if sum(score_row(r, th, gap, m) for m, th, gap in combo) >= mv)
                tpr, fpr = tp / len(pos), fp / len(neg)
                if fpr <= 0.02:
                    results.append((round(tpr, 4), -fpr, mv, combo))
        results.sort(reverse=True)
        for tpr, nfpr, mv, combo in results[:5]:
            cstr = ", ".join(f"{m}: th={th} gap={gap}" for m, th, gap in combo)
            print(f"  votes>={mv}  TPR={tpr:.3f} FPR={-nfpr:.3f} | {cstr}")

    print("\n✅ 校准完成。推荐配置见上方 TPR 最高且 FPR≈0 的组合。")


if __name__ == "__main__":
    main()
