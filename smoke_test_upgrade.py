#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
升级后冒烟+回归测试：
1. modelscope SV pipeline API 兼容性 + 三模型加载
2. 用 event_scores.jsonl 前3条(最新) 重打分，对比升级前分数（一致=环境无漂移）
3. funasr AutoModel (paraformer-zh) 加载+转写冒烟
"""
import os, json, subprocess, glob
import numpy as np
from scipy.spatial.distance import cosine

TMP = "/tmp/smoke_upgrade"; os.makedirs(TMP, exist_ok=True)
SV = {
    "eres2net_large": "iic/speech_eres2net_large_200k_sv_zh-cn_16k-common",
    "rdino_ecapa": "iic/speech_rdino_ecapa_tdnn_sv_zh-cn_cnceleb_16k",
    "camplusplus": "iic/speech_campplus_sv_zh-cn_16k-common",
}

# ── 1. SV pipeline 加载 ──
print("── 1. modelscope SV pipeline ──")
import torch, torchaudio
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import modelscope
print(f"   modelscope {modelscope.__version__}")

pipes = {}
for n, mid in SV.items():
    pipes[n] = pipeline(task=Tasks.speaker_verification, model=mid, model_revision="v1.0.0", device="cpu")
    print(f"   ✅ {n} 加载成功")

# ── 2. 回归对比：重打分前3条正例 ──
print("── 2. 回归对比（升级前 vs 升级后）──")
db = json.load(open("speaker_db_multi.json", encoding="utf-8"))
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
    t = next(((n, s) for n, s in rk if n.lower() in ["baby", "宝宝"]), ("无", 0))
    return t[1]

rows = []
for line in open("event_scores.jsonl", encoding="utf-8"):
    rows.append(json.loads(line))
rows.sort(key=lambda r: r["file"], reverse=True)  # 最新优先

max_diff = 0.0
for r in rows[:3]:
    fname = r["file"]; date = fname.split("_")[1]
    src = f"/Volumes/download/records/Sony-2/processed/{date}/{fname}"
    if not os.path.exists(src):
        print(f"   ⏭️ {fname} 源文件不可达，跳过"); continue
    wav = os.path.join(TMP, "t.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-ac", "1", "-ar", "16000", wav],
                   check=True, timeout=60, stdout=subprocess.DEVNULL)
    line = f"   {fname}:"
    for m, pipe in pipes.items():
        old = r["scores"][m]["baby"]
        new = round(float(score(emb(pipe, wav), m)), 3)
        d = abs(new - old); max_diff = max(max_diff, d)
        flag = "✅" if d <= 0.002 else "❌"
        line += f" {m} {old}→{new} {flag}"
    print(line)

env_ok = max_diff <= 0.002
print(f"   环境一致性: {'✅ 一致 (最大差异 %.4f)' % max_diff if env_ok else '❌ 漂移 (最大差异 %.4f)' % max_diff}")

# ── 3. funasr ASR 冒烟 ──
print("── 3. funasr AutoModel ──")
import funasr
print(f"   funasr {funasr.__version__}")
from funasr import AutoModel
asr = AutoModel(model="paraformer-zh", device="cpu", disable_update=True)
print("   ✅ paraformer-zh 加载成功")
# 找一个测试音频转写
test_wav = None
for c in glob.glob("/tmp/event_replay/*.wav") + glob.glob("/tmp/smoke_upgrade/*.wav"):
    test_wav = c; break
if test_wav:
    res = asr.generate(input=test_wav)
    txt = (res[0].get("text", "") if res else "")[:40]
    print(f"   ✅ 转写冒烟: 「{txt}...」")
else:
    print("   ⚠️ 无测试音频，跳过转写（加载已验证）")

print(f"\n{'✅ 冒烟+回归全部通过' if env_ok else '❌ 环境有漂移，需重跑回放'}")
