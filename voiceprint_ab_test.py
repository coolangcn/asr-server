# -*- coding: utf-8 -*-
"""音质改造前后声纹分数对照实验。
两组切分段（服务端统一 16k PCM，处理管线相同）：
  A组=2026-09-26（Termux 24k 时代，改造前）  B组=2026-10-03（原生APK 48k，改造后）
对同一个声纹库逐段打 top1 分数（3模型），统计分布差异。
仅读库，不修改任何数据。"""
import os, sys, glob, random

os.chdir('/Users/mac/asr-server')
sys.path.insert(0, '/Users/mac/asr-server')
from dotenv import load_dotenv; load_dotenv()
import numpy as np
from scipy.io import wavfile
from scipy.spatial.distance import cosine
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import asr_server

for name, conf in asr_server.Config.SV_MODELS.items():
    asr_server.sv_pipelines[name] = pipeline(
        task=Tasks.speaker_verification, model=conf['id'], model_revision=conf['rev'],
        device=asr_server.Config.MODELSCOPE_DEVICE)
asr_server.load_speaker_db()
print("库中人:", list(asr_server.speaker_db.keys()))

def rms_ok(path, floor=0.005):
    """过滤近静音段，保证两组比的都是真实语音段"""
    try:
        sr, data = wavfile.read(path)
        if data.ndim > 1: data = data[:, 0]
        x = data.astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(x ** 2))) >= floor
    except Exception:
        return False

def sample_segments(date_dir, n, seed=7):
    segs = [p for p in glob.glob(f"{date_dir}/**/seg_*.wav", recursive=True)]
    random.Random(seed).shuffle(segs)
    picked, tried = [], 0
    for p in segs:
        if tried >= 60: break
        tried += 1
        if rms_ok(p): picked.append(p)
        if len(picked) >= n: break
    return picked

def top1_scores(path):
    """返回 (最好模型的top1人名, 全模型最高分, 该模型gap)"""
    best = ("?", -1.0, 0.0)
    for mname, svp in asr_server.sv_pipelines.items():
        conf = asr_server.Config.SV_MODELS[mname]
        emb = asr_server.extract_embedding_from_file(svp, path)
        if emb is None: continue
        rows = []
        for pname, pdata in asr_server.speaker_db.items():
            ref = (pdata.get('avg_embeddings') or {}).get(mname)
            if ref is None: continue
            rows.append((pname, 1 - cosine(emb.flatten(), np.array(ref, dtype=np.float32).flatten())))
        rows.sort(key=lambda x: x[1], reverse=True)
        if not rows: continue
        top1, top2 = rows[0], (rows[1] if len(rows) > 1 else (None, 0.0))
        gap = top1[1] - top2[1]
        # 以「是否满足该模型判定线」优先，其次取最高分
        cur_ok = best[1] >= conf['threshold'] and best[2] >= conf['gap']
        this_ok = top1[1] >= conf['threshold'] and gap >= conf['gap']
        if (this_ok and not cur_ok) or (top1[1] > best[1] and not cur_ok):
            best = (top1[0], float(top1[1]), float(gap))
    return best

GROUPS = {
    "A组 改造前(9/26 Termux)": sample_segments("/Volumes/download/records/Sony-2/audio_segments/2026-09-26", 12),
    "B组 改造后(10/3 APK48k)": sample_segments("/Volumes/download/records/Sony-2/audio_segments/2026-10-03", 12),
}

report = {}
for gname, segs in GROUPS.items():
    print(f"\n===== {gname}：{len(segs)} 个有效语音段 =====")
    scores = []
    for p in segs:
        who, sc, gap = top1_scores(p)
        scores.append((os.path.basename(p), who, sc, gap))
        print(f"  {os.path.basename(p):<12} top1={who} score={sc:.3f} gap={gap:.3f}")
    arr = np.array([s[2] for s in scores])
    report[gname] = arr
    print(f"  → 中位数={np.median(arr):.3f} 均值={arr.mean():.3f} 最小={arr.min():.3f} 最大={arr.max():.3f}")

a, b = report["A组 改造前(9/26 Termux)"], report["B组 改造后(10/3 APK48k)"]
print(f"\n===== 结论统计 =====")
print(f"改造前 top1≥0.60 的段: {(a >= 0.6).sum()}/{len(a)}   改造后: {(b >= 0.6).sum()}/{len(b)}")
print(f"改造前 top1≥0.55 的段: {(a >= 0.55).sum()}/{len(a)}   改造后: {(b >= 0.55).sum()}/{len(b)}")
print(f"中位数变化: {np.median(a):.3f} → {np.median(b):.3f}  (Δ={np.median(b)-np.median(a):+.3f})")
