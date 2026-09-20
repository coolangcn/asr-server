#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测 486/487/488 哭闹事件的后续段（7-1凌晨实时处理时判no_cry的文件）"""
import os, json, subprocess
import numpy as np
from scipy.spatial.distance import cosine
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import torch, torchaudio

TMP = "/tmp/cry_followup"; os.makedirs(TMP, exist_ok=True)
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

base = "/Volumes/download/records/Sony-2/processed/2026-06-30"
files = [
    "TermuxAudioRecording_2026-06-30_20-57-05.m4a",  # 486后续
    "TermuxAudioRecording_2026-06-30_20-58-05.m4a",
    "TermuxAudioRecording_2026-06-30_21-20-19.m4a",  # 487后续
    "TermuxAudioRecording_2026-06-30_21-21-20.m4a",
    "TermuxAudioRecording_2026-06-30_22-02-44.m4a",  # 488后续
    "TermuxAudioRecording_2026-06-30_22-03-45.m4a",
    "TermuxAudioRecording_2026-06-30_22-04-46.m4a",
]
for f in files:
    src = os.path.join(base, f)
    if not os.path.exists(src):
        print(f"⚠️ {f} 不存在"); continue
    wav = os.path.join(TMP, f.replace(".m4a", ".wav"))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-ac", "1", "-ar", "16000", wav], check=True, stdout=subprocess.DEVNULL)
    votes = 0; parts = []
    for m, pipe in pipes.items():
        e = emb(pipe, wav)
        rk, t, o, gap = score(e, m)
        passed = t[1] >= 0.65 and gap >= 0.15
        votes += passed
        parts.append(f"{m.split('_')[0]}: {t[1]:.3f} gap={gap:.3f}{'✅' if passed else '❌'}")
    res = "DETECT" if votes >= 3 else ("PARTIAL" if votes else "MISS")
    print(f"[{f[20:32]}] {res} ({votes}/3) | " + " | ".join(parts))
