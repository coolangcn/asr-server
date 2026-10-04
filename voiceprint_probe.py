# -*- coding: utf-8 -*-
"""声纹失配排查脚本（一次性诊断工具，跑完可删）
从今天的积压录音中切段，实测各 SV 模型对声纹库中每个说话人的相似度，
输出完整矩阵 + 阈值判定，定位「未识别说话人」的原因。
只加载 SV 模型与声纹库，不加载 ASR/Whisper，不写库。
"""
import os, sys, glob, subprocess, tempfile

os.chdir('/Users/mac/asr-server')
sys.path.insert(0, '/Users/mac/asr-server')
from dotenv import load_dotenv; load_dotenv()

import numpy as np
from scipy.spatial.distance import cosine
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

import asr_server

# 1. 只加载 SV 模型 + 声纹库
for name, conf in asr_server.Config.SV_MODELS.items():
    print(f"加载 SV [{name}] ...", flush=True)
    asr_server.sv_pipelines[name] = pipeline(
        task=Tasks.speaker_verification,
        model=conf['id'], model_revision=conf['rev'],
        device=asr_server.Config.MODELSCOPE_DEVICE)
asr_server.load_speaker_db()
neg = asr_server.negative_samples if hasattr(asr_server, 'negative_samples') else []
print(f"声纹库: {list(asr_server.speaker_db.keys())} | 负样本 {len(neg)} 条\n", flush=True)

# 2. 选样本：今天各设备目录中间位置的一个文件
D, today = "/Volumes/download/records", "2026-10-03"
samples = []
for dev in ["Sony-2", "Sony-1", "Pixel-5"]:
    files = sorted(glob.glob(f"{D}/{dev}/{today}/*.m4a"))
    if files:
        f = files[len(files) // 2]
        samples.append((dev, f))
        print(f"样本[{dev}]: {os.path.basename(f)} (今日{len(files)}段中第{len(files)//2}段)")

# 3. 切中段 30s → 16k wav → 提取 embedding → 全量相似度
tmpdir = tempfile.mkdtemp()
for dev, f in samples:
    wav = os.path.join(tmpdir, f"{dev}.wav")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-ss", "15", "-t", "30", "-i", f, "-ar", "16000", "-ac", "1", wav], check=True)
    print(f"\n===== [{dev}] {os.path.basename(f)} 中段30s =====")
    for mname, svp in asr_server.sv_pipelines.items():
        conf = asr_server.Config.SV_MODELS[mname]
        emb = asr_server.extract_embedding_from_file(svp, wav)
        if emb is None:
            print(f"  [{mname}] ❌ 特征提取失败")
            continue
        rows = []
        for pname, pdata in asr_server.speaker_db.items():
            ref = (pdata.get('avg_embeddings') or {}).get(mname)
            if ref is None:
                continue
            s = 1 - cosine(emb.flatten(), np.array(ref, dtype=np.float32).flatten())
            rows.append((pname, s))
        rows.sort(key=lambda x: x[1], reverse=True)
        top1, top2 = rows[0], (rows[1] if len(rows) > 1 else (None, 0.0))
        gap = top1[1] - top2[1]
        verdict = "✅过" if (top1[1] >= conf['threshold'] and gap >= conf['gap']) else "❌不过"
        detail = "  ".join(f"{n}={s:.3f}" for n, s in rows)
        print(f"  [{mname}] 阈值{conf['threshold']} gap{conf['gap']} → top1={top1[0]}({top1[1]:.3f}) gap={gap:.3f} {verdict}")
        print(f"      全部: {detail}")
print("\n诊断要点: 若 top1 分数普遍在 0.4-0.55 → 音频/注册样本特征漂移(阈值偏严); 若接近阈值但 gap 不足 → 多人相似(库内区分度低)")
