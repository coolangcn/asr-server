#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""精判扫描发现的 7-1 命中段：三模型打分"""
import os, json, subprocess
import numpy as np
from scipy.spatial.distance import cosine
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import torch, torchaudio

TMP = "/tmp/cry_precise"; os.makedirs(TMP, exist_ok=True)
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

# 从 scan_results.jsonl 读命中段
hits = []
if os.path.exists("scan_results.jsonl"):
    for line in open("scan_results.jsonl", encoding="utf-8"):
        r = json.loads(line)
        hits.append(r)

print(f"待精判 {len(hits)} 段")
for r in hits:
    src = f"/Volumes/download/records/Sony-2/processed/{r['date']}/{r['file']}"
    if not os.path.exists(src):
        print(f"⚠️ {r['file']} 缺失"); continue
    wav = os.path.join(TMP, "p.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-ac", "1", "-ar", "16000", wav], check=True, stdout=subprocess.DEVNULL)
    votes = 0; parts = []
    for m, pipe in pipes.items():
        e = emb(pipe, wav)
        rk, t, o, gap = score(e, m)
        passed = t[1] >= 0.65 and gap >= 0.15
        votes += passed
        parts.append(f"{m.split('_')[0]}: {t[1]:.3f} g={gap:.3f}{'✅' if passed else '❌'}")
    res = "DETECT" if votes >= 3 else ("PARTIAL" if votes else "MISS")
    print(f"[{r['file']}] {res} ({votes}/3) | " + " | ".join(parts))
