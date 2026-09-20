#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
粗筛扫描指定日期的录音：eres2net 单模型快扫，找出 Baby 高分段。
用法: python3 scan_high_scores.py 2026-07-01 2026-07-02 ...
结果追加写入 scan_results.jsonl（防中断丢失）
"""
import os, sys, json, subprocess, glob
import numpy as np
from scipy.spatial.distance import cosine
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import torch, torchaudio

SV_MODEL = "iic/speech_eres2net_large_200k_sv_zh-cn_16k-common"
ROUGH_THRESHOLD = 0.60   # 粗筛线（486-488 级别是 0.87+，困难段 0.67）
OUT = "scan_results.jsonl"
TMP = "/tmp/scan_segments"; os.makedirs(TMP, exist_ok=True)

pipe = pipeline(task=Tasks.speaker_verification, model=SV_MODEL, model_revision="v1.0.0", device="cpu")
db = json.load(open("speaker_db_multi.json", encoding="utf-8"))
TARGET = ["baby", "宝宝"]
SKIP = ["大可", "妈妈", "婆婆"]
avgs = {n: sd["avg_embeddings"]["eres2net_large"] for n, sd in db.items()
        if "eres2net_large" in sd.get("avg_embeddings", {})}

def emb(wav):
    a, sr = torchaudio.load(wav)
    if sr != 16000:
        a = torchaudio.transforms.Resample(sr, 16000)(a)
    a = a.mean(dim=0, keepdim=True)
    with torch.no_grad():
        o = pipe.model(a)
        e = o.get("spk_embedding") if isinstance(o, dict) else o
    return e.squeeze().numpy()

def cos(a, b): return 1 - cosine(a.flatten(), np.asarray(b).flatten())

total = 0
for date in sys.argv[1:]:
    day_dir = f"/Volumes/download/records/Sony-2/processed/{date}"
    files = sorted(glob.glob(os.path.join(day_dir, "TermuxAudioRecording_*.m4a")))
    print(f"📂 {date}: {len(files)} 个文件", flush=True)
    for i, src in enumerate(files):
        total += 1
        fname = os.path.basename(src)
        # 只扫 20:00-23:59 + 0:00-1:59（哭闹高发时段，快）
        hh = fname.split("_")[2][:2] if len(fname.split("_")) > 2 else ""
        if not (hh.startswith("2") or hh in ("00", "01")):
            continue
        wav = os.path.join(TMP, "s.wav")
        try:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-ac", "1", "-ar", "16000", wav],
                           check=True, timeout=60, stdout=subprocess.DEVNULL)
            e = emb(wav)
            sc = {n: cos(e, av) for n, av in avgs.items()}
            rk = sorted(sc.items(), key=lambda x: x[1], reverse=True)
            t = next(((n, s) for n, s in rk if n.lower() in TARGET), ("无", 0))
            o = next(((n, s) for n, s in rk if n.lower() not in TARGET), ("无", 0))
            rec = {"file": fname, "date": date, "baby": round(t[1], 3),
                   "other": f"{o[0]}={o[1]:.3f}", "top": f"{rk[0][0]}={rk[0][1]:.3f}"}
            if t[1] >= ROUGH_THRESHOLD:
                rec["HIT"] = True
                print(f"  🎯 {fname}: Baby={t[1]:.3f} top={rk[0][0]}={rk[0][1]:.3f}", flush=True)
                with open(OUT, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as ex:
            print(f"  ⚠️ {fname}: {ex}", flush=True)
        if (i + 1) % 100 == 0:
            print(f"  … {date} {i+1}/{len(files)}", flush=True)
print(f"✅ 扫描完成，共 {total} 文件，命中见 {OUT}")
