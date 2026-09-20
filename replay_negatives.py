#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
负例分数分布采集：随机抽取 no_cry 文件（3-9月跨时段），三模型打分。
结果写入 neg_scores.jsonl
"""
import os, json, subprocess, random
import numpy as np
from scipy.spatial.distance import cosine
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import torch, torchaudio
import psycopg2

TMP = "/tmp/neg_replay"; os.makedirs(TMP, exist_ok=True)
OUT = "neg_scores.jsonl"
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

conn = psycopg2.connect(host="192.168.1.188", port=5433, user="postgres", password="cncncncn", dbname="postgres")
cur = conn.cursor()
cur.execute("""
    SELECT filename FROM processed_files_a
    WHERE status='no_cry' AND filename LIKE 'TermuxAudioRecording_2026-%'
    ORDER BY random() LIMIT 200
""")
rows = [r[0] for r in cur.fetchall()]
conn.close()
random.shuffle(rows)
print(f"抽取 {len(rows)} 个 no_cry 候选", flush=True)

done = set()
if os.path.exists(OUT):
    for line in open(OUT, encoding="utf-8"):
        try: done.add(json.loads(line)["file"])
        except: pass

count = 0
LIMIT = 120
for fname in rows:
    if count >= LIMIT:
        break
    if fname in done:
        continue
    date = fname.split("_")[1]
    src = f"/Volumes/download/records/Sony-2/processed/{date}/{fname}"
    if not os.path.exists(src):
        continue
    wav = os.path.join(TMP, "n.wav")
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-ac", "1", "-ar", "16000", wav],
                       check=True, timeout=60, stdout=subprocess.DEVNULL)
        row = {"file": fname, "scores": {}}
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
        count += 1
        if count % 20 == 0:
            print(f"… {count}/{LIMIT}", flush=True)
    except Exception as ex:
        import traceback
        traceback.print_exc()
        print(f"⚠️ {fname}: {ex}", flush=True)

print(f"✅ 负例采集完成 {count} 个", flush=True)
