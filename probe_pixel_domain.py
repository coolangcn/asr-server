# -*- coding: utf-8 -*-
"""实测: 旧域(Sony)声纹库对今天 Pixel 新域切片的分数分布——验证有无区分度"""
import os, sys, glob, wave, random
os.chdir('/Users/mac/asr-server')
sys.path.insert(0, '/Users/mac/asr-server')
from dotenv import load_dotenv; load_dotenv()
import numpy as np
from scipy.spatial.distance import cosine
from collections import Counter
import asr_server

for name, conf in asr_server.Config.SV_MODELS.items():
    print(f"加载 SV [{name}] ...", flush=True)
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    asr_server.sv_pipelines[name] = pipeline(
        task=Tasks.speaker_verification,
        model=conf['id'], model_revision=conf['rev'],
        device=asr_server.Config.MODELSCOPE_DEVICE)
asr_server.load_speaker_db()
asr_server.load_negative_samples()

# 收集今天双 Pixel 的切片，按文件大小排序取最长的前 14 个（越长信息量越大）
segs = []
for dev in ("Pixel-6", "Pixel-5"):
    segs += glob.glob(f'/Volumes/download/records/{dev}/audio_segments/2026-10-03/*/seg_*.wav')
segs = sorted(segs, key=os.path.getsize, reverse=True)[:14]
print(f"\n样本切片: {len(segs)} 个（双 Pixel 今日最长切片）\n", flush=True)

all_top = []   # (top1_name, top1_score, gap) 全模型聚合
per_model_top1 = {m: [] for m in asr_server.sv_pipelines}

for seg in segs:
    name = os.path.basename(seg)
    dev = seg.split('/records/')[1].split('/')[0]
    try:
        w = wave.open(seg); dur = w.getnframes() / w.getframerate(); w.close()
    except Exception:
        dur = -1
    row = [f"[{dev}] {name[-22:]} {dur:.1f}s"]
    for mname, svp in asr_server.sv_pipelines.items():
        emb = asr_server.extract_embedding_from_file(svp, seg)
        if emb is None:
            row.append(f"{mname[:6]}:提取失败"); continue
        sims = {}
        for spk, pd in asr_server.speaker_db.items():
            avg = (pd.get('avg_embeddings') or {}).get(mname)
            if avg is not None:
                sims[spk] = 1 - cosine(emb.flatten(), np.array(avg, dtype=np.float32).flatten())
        top = sorted(sims.items(), key=lambda x: -x[1])
        t1n, t1s = top[0] if top else ('-', 0)
        t2s = top[1][1] if len(top) > 1 else 0
        row.append(f"{mname[:6]}: top1={t1n}({t1s:.3f}) gap={t1s-t2s:.3f}  range[{min(sims.values()):.3f}~{t1s:.3f}]")
        per_model_top1[mname].append((t1n, t1s))
        all_top.append((t1n, t1s, t1s - t2s))
    print(' | '.join(row), flush=True)

print("\n========== 分布汇总（旧域库 vs Pixel 新域切片） ==========", flush=True)
scores = [s for _, s, _ in all_top]
gaps = [g for _, _, g in all_top]
print(f"top1 分数: min={min(scores):.3f} 中位={sorted(scores)[len(scores)//2]:.3f} max={max(scores):.3f}")
print(f"top1-top2 gap: min={min(gaps):.3f} 中位={sorted(gaps)[len(gaps)//2]:.3f} max={max(gaps):.3f}")
for m, lst in per_model_top1.items():
    c = Counter(n for n, _ in lst).most_common()
    print(f"{m[:14]:14s} top1 归属分布: {c}")
thr_pass = sum(1 for _, s, g in all_top if s >= 0.60 and g >= 0.10)
print(f"过线(≥0.60 且 gap≥0.10): {thr_pass}/{len(all_top)}")
