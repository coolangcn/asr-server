# -*- coding: utf-8 -*-
"""临时诊断: 今天 19:52 被丢弃的宝宝说话切片, 对全家声纹+负样本打分"""
import os, sys, glob
os.chdir('/Users/mac/asr-server')
sys.path.insert(0, '/Users/mac/asr-server')
from dotenv import load_dotenv; load_dotenv()
import numpy as np
from scipy.spatial.distance import cosine
import asr_server

for name, conf in asr_server.Config.SV_MODELS.items():
    print(f"加载 SV [{name}] ...", flush=True)
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    asr_server.sv_pipelines[name] = pipeline(
        task=Tasks.speaker_verification,
        model=conf['id'], model_revision=conf['rev'],
        device=asr_server.Config.MODELSCOPE_DEVICE)
asr_server.load_speaker_db()
asr_server.load_negative_samples()

segs = sorted(glob.glob('/Volumes/download/records/Sony-2/audio_segments/2026-10-03/TermuxAudioRecording_2026-10-03_19-52-36/seg_*.wav'))
print(f"\n切片数: {len(segs)}\n", flush=True)

for seg in segs:
    name = os.path.basename(seg)
    import wave
    try:
        w = wave.open(seg); dur = w.getnframes() / w.getframerate(); w.close()
    except Exception:
        dur = -1
    row = [f"{name} {dur:.1f}s"]
    verdicts = []
    for mname, svp in asr_server.sv_pipelines.items():
        emb = asr_server.extract_embedding_from_file(svp, seg)
        if emb is None:
            row.append(f"{mname[:6]}:提取失败"); continue
        sims = {}
        for spk, pd in asr_server.speaker_db.items():
            avg = (pd.get('avg_embeddings') or {}).get(mname)
            if avg is not None:
                sims[spk] = 1 - cosine(emb.flatten(), np.array(avg, dtype=np.float32).flatten())
        top = sorted(sims.items(), key=lambda x: -x[1])
        neg_best = 0.0
        for ne in asr_server.negative_samples:
            nv = (ne.get('embeddings') or {}).get(mname)
            if nv:
                neg_best = max(neg_best, 1 - cosine(emb.flatten(), np.array(nv, dtype=np.float32).flatten()))
        t1n, t1s = top[0] if top else ('-', 0)
        t2s = top[1][1] if len(top) > 1 else 0
        ok = t1s >= 0.60 and (t1s - t2s) >= 0.10
        blocked = neg_best > t1s
        row.append(f"{mname[:6]}:{t1n}={t1s:.3f}(neg={neg_best:.3f})")
        verdicts.append((mname, t1n, t1s, ok, blocked))
    print(' | '.join(row), flush=True)
    votes = [v for _, n, _, ok, b in verdicts if ok and not b]
    from collections import Counter
    win = Counter(votes).most_common(1)
    print(f"    → 判定: {'✓ ' + win[0][0] if win and win[0][1] >= 2 else '✗ 丢弃'}", flush=True)
