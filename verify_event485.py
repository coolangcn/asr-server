#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证事件485(6-29)源文件 + 8-9月 no_cry 负样本的当前打分"""
import os, json, subprocess
import numpy as np
from scipy.spatial.distance import cosine
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import torch, torchaudio

TMP = "/tmp/cry_reg_test2"; os.makedirs(TMP, exist_ok=True)
SV = {
    "eres2net_large": "iic/speech_eres2net_large_200k_sv_zh-cn_16k-common",
    "rdino_ecapa": "iic/speech_rdino_ecapa_tdnn_sv_zh-cn_cnceleb_16k",
    "camplusplus": "iic/speech_campplus_sv_zh-cn_16k-common",
}
pipes = {}
for n, mid in SV.items():
    pipes[n] = pipeline(task=Tasks.speaker_verification, model=mid, model_revision="v1.0.0", device="cpu")

db = json.load(open("speaker_db_multi.json", encoding="utf-8"))
TARGET = ["baby", "宝宝"]
SKIP = ["大可", "妈妈", "婆婆"]

def prep(src, out):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-ac", "1", "-ar", "16000", out], check=True, stdout=subprocess.DEVNULL)
    return out

def emb(pipe, wav):
    a, sr = torchaudio.load(wav)
    if sr != 16000:
        a = torchaudio.transforms.Resample(sr, 16000)(a)
    a = a.mean(dim=0, keepdim=True)
    with torch.no_grad():
        o = pipe.model(a)
        e = o.get("spk_embedding") if isinstance(o, dict) else o
    return e.squeeze().numpy()

def cos(a, b): return 1 - cosine(a.flatten(), np.asarray(b).flatten())

def score(e, m):
    sc = {}
    for name, sd in db.items():
        if name in SKIP: continue
        av = sd.get("avg_embeddings", {}).get(m)
        if av is not None: sc[name] = cos(e, av)
    rk = sorted(sc.items(), key=lambda x: x[1], reverse=True)
    t = next(((n, s) for n, s in rk if n.lower() in TARGET), ("无", 0))
    o = next(((n, s) for n, s in rk if n.lower() not in TARGET), ("无", 0))
    return rk, t, o, t[1] - o[1]

inputs = [
    ("event485", "/Volumes/download/records/Sony-2/processed/2026-06-29/TermuxAudioRecording_2026-06-29_22-29-25.m4a", True),
]
# 负样本: 8-9月 no_cry
import psycopg2
conn = psycopg2.connect(host="192.168.1.188", port=5433, user="postgres", password="cncncncn", dbname="postgres")
cur = conn.cursor()
cur.execute("SELECT filename FROM processed_files_a WHERE status='no_cry' AND processed_at >= '2026-08-25' ORDER BY processed_at DESC LIMIT 6")
rows = [r[0] for r in cur.fetchall()]
conn.close()
base = "/Volumes/download/records/Sony-2/processed"
for f in rows:
    d = f.split("_")[1]
    p = os.path.join(base, d, f)
    if os.path.exists(p): inputs.append((f"neg_{f[20:30]}", p, False))

print(f"待测: {len(inputs)} 个文件")
for label, src, is_pos in inputs:
    if not os.path.exists(src):
        print(f"⚠️ {src} 不存在"); continue
    wav = prep(src, os.path.join(TMP, f"{label}.wav"))
    votes = 0; parts = []
    for m, pipe in pipes.items():
        e = emb(pipe, wav)
        rk, t, o, gap = score(e, m)
        passed = t[1] >= 0.65 and gap >= 0.15
        votes += passed
        parts.append(f"{m}: {t[0]}={t[1]:.3f} gap={gap:.3f}{'✅' if passed else '❌'}")
    res = "DETECT" if votes >= 3 else ("PARTIAL" if votes else "MISS")
    print(f"[{label}] {res} ({votes}/3) | " + " | ".join(parts))
