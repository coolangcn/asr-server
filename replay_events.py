#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
历史事件回测：对 DB 全部 baby_cry_events 的代表文件（确认哭声）跑三模型打分，
构建正例分数分布；与负例(no_cry)分布对比，为阈值校准提供数据。
结果写入 event_scores.jsonl
"""
import os, sys, json, subprocess, glob
import numpy as np
from scipy.spatial.distance import cosine
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import torch, torchaudio
import psycopg2

TMP = "/tmp/event_replay"; os.makedirs(TMP, exist_ok=True)
OUT = "event_scores.jsonl"
SV = {
    "eres2net_large": "iic/speech_eres2net_large_200k_sv_zh-cn_16k-common",
    "rdino_ecapa": "iic/speech_rdino_ecapa_tdnn_sv_zh-cn_cnceleb_16k",
    "camplusplus": "iic/speech_campplus_sv_zh-cn_16k-common",
}
pipes = {}
for n, mid in SV.items():
    print(f"加载 {n}...", flush=True)
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

def to_native(o):
    """numpy 类型转 Python 原生类型，避免 json 序列化失败"""
    if isinstance(o, dict): return {k: to_native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [to_native(v) for v in o]
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    return o

def score(e, m):
    sc = {}
    for name, sd in db.items():
        if name in SKIP: continue
        av = sd.get("avg_embeddings", {}).get(m)
        if av is not None: sc[name] = cos(e, av)
    rk = sorted(sc.items(), key=lambda x: x[1], reverse=True)
    t = next(((n, s) for n, s in rk if n.lower() in TARGET), ("无", 0))
    o = next(((n, s) for n, s in rk if n.lower() not in TARGET), ("无", 0))
    return t[1], o[0], o[1], t[1] - o[1]

# 取全部未删除事件的代表文件
conn = psycopg2.connect(host="192.168.1.188", port=5433, user="postgres", password="cncncncn", dbname="postgres")
cur = conn.cursor()
cur.execute("SELECT id, filename, confidence, created_at FROM baby_cry_events WHERE is_deleted=false ORDER BY id DESC")
events = cur.fetchall()
conn.close()
print(f"共 {len(events)} 个事件", flush=True)

done_ids = set()
if os.path.exists(OUT):
    for line in open(OUT, encoding="utf-8"):
        try: done_ids.add(json.loads(line)["id"])
        except: pass

for eid, fname, conf, created in events:
    if eid in done_ids:
        continue
    date = fname.split("_")[1] if "_" in fname else None
    src = f"/Volumes/download/records/Sony-2/processed/{date}/{fname}" if date else None
    if not src or not os.path.exists(src):
        continue
    wav = os.path.join(TMP, "e.wav")
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-ac", "1", "-ar", "16000", wav],
                       check=True, timeout=60, stdout=subprocess.DEVNULL)
        row = {"id": eid, "file": fname, "db_conf": round(conf, 3), "scores": {}}
        votes = 0
        for m, pipe in pipes.items():
            e = emb(pipe, wav)
            t_s, o_n, o_s, gap = score(e, m)
            passed = bool(t_s >= 0.65 and gap >= 0.15)
            votes += passed
            row["scores"][m] = {"baby": round(t_s, 3), "other": f"{o_n}={o_s:.3f}", "gap": round(gap, 3), "pass": passed}
        row["votes_old_rule"] = votes
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(to_native(row), ensure_ascii=False, default=str) + "\n")
        n_done = len(done_ids) + 1
        if n_done % 20 == 0:
            print(f"… {n_done}/{len(events)}", flush=True)
        done_ids.add(eid)
    except Exception as ex:
        import traceback
        traceback.print_exc()
        print(f"⚠️ {fname}: {ex}", flush=True)

print(f"✅ 回测完成，写入 {OUT}", flush=True)
