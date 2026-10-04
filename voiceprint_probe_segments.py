# -*- coding: utf-8 -*-
"""按线上方式复现声纹判定：对已切分的单人语音段逐段跑 3 模型声纹 + 2/3 投票。
回答「切分后识别是否可行」——与线上 temp/seg_*.wav 同源的 audio_segments 持久化段。"""
import os, sys, glob, re

os.chdir('/Users/mac/asr-server')
sys.path.insert(0, '/Users/mac/asr-server')
from dotenv import load_dotenv; load_dotenv()

import numpy as np
from scipy.spatial.distance import cosine
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

import asr_server

for name, conf in asr_server.Config.SV_MODELS.items():
    print(f"加载 SV [{name}] ...", flush=True)
    asr_server.sv_pipelines[name] = pipeline(
        task=Tasks.speaker_verification, model=conf['id'], model_revision=conf['rev'],
        device=asr_server.Config.MODELSCOPE_DEVICE)
asr_server.load_speaker_db()
print(f"声纹库: {list(asr_server.speaker_db.keys())}\n", flush=True)

SEG_DIR = "/Volumes/download/records/Sony-2/audio_segments/2026-10-03/TermuxAudioRecording_2026-10-03_13-22-53"
segs = sorted(glob.glob(f"{SEG_DIR}/seg_*.wav"),
              key=lambda p: int(re.search(r'seg_(\d+)', p).group(1)))
print(f"共 {len(segs)} 段\n")

recognized = {}
for path in segs:
    tag = os.path.basename(path)
    votes = {}
    lines = []
    for mname, svp in asr_server.sv_pipelines.items():
        conf = asr_server.Config.SV_MODELS[mname]
        emb = asr_server.extract_embedding_from_file(svp, path)
        if emb is None:
            votes[mname] = "Failed"; continue
        rows = []
        for pname, pdata in asr_server.speaker_db.items():
            ref = (pdata.get('avg_embeddings') or {}).get(mname)
            if ref is None: continue
            rows.append((pname, 1 - cosine(emb.flatten(), np.array(ref, dtype=np.float32).flatten())))
        rows.sort(key=lambda x: x[1], reverse=True)
        top1, top2 = rows[0], (rows[1] if len(rows) > 1 else (None, 0.0))
        gap = top1[1] - top2[1]
        ok = top1[1] >= conf['threshold'] and gap >= conf['gap']
        votes[mname] = top1[0] if ok else "Unknown"
        lines.append(f"    {mname[:12]:<12} top1={top1[0]}({top1[1]:.3f}) gap={gap:.3f} {'✓' if ok else '✗'}")
    real_votes = [v for v in votes.values() if v not in ("Unknown", "Failed")]
    if len(real_votes) >= 2 and len(set(real_votes)) == 1:
        who = real_votes[0]
        verdict = f"✅ 认出: {who}"
        recognized[tag] = who
    elif real_votes:
        verdict = f"❌ 票数不足/矛盾: {votes}"
    else:
        verdict = "❌ 全部 Unknown"
    print(f"{tag}  {verdict}")
    for l in lines: print(l)

print(f"\n===== 汇总: {len(segs)} 段中认出 {len(recognized)} 段 =====")
from collections import Counter
print("认出的人:", dict(Counter(recognized.values())))
