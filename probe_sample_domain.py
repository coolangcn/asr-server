# -*- coding: utf-8 -*-
"""临时诊断: 大可每条样本对今天宝宝说话声的相似度 → 判断样本域"""
import os, sys, glob
os.chdir('/Users/mac/asr-server')
sys.path.insert(0, '/Users/mac/asr-server')
from dotenv import load_dotenv; load_dotenv()
import numpy as np
from scipy.spatial.distance import cosine
import asr_server

for name in asr_server.Config.SV_MODELS:
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    conf = asr_server.Config.SV_MODELS[name]
    asr_server.sv_pipelines[name] = pipeline(
        task=Tasks.speaker_verification, model=conf['id'], model_revision=conf['rev'],
        device=asr_server.Config.MODELSCOPE_DEVICE)
asr_server.load_speaker_db()

# 测试音频: 今天的宝宝说话声(seg_10) + 一段今天妈妈清晰说话(seg 在 19:31 文件)
TESTS = glob.glob('/Volumes/download/records/Sony-2/audio_segments/2026-10-03/TermuxAudioRecording_2026-10-03_19-52-36/seg_10.wav')

for t in TESTS:
    print(f"\n=== 测试音频: {os.path.basename(t)} (今天宝宝说话) ===")
    for spk, pd in asr_server.speaker_db.items():
        if spk != '大可':
            continue
        print(f"[{spk}] 逐样本打分:")
        for s in pd.get('samples', []):
            path = s.get('audio_path', '')
            if not os.path.exists(path):
                path = os.path.join('/Users/mac/asr-server', path)
            if not os.path.exists(path):
                print(f"  {s.get('timestamp','')} | 文件缺失 {path}")
                continue
            scores = []
            for mname, svp in asr_server.sv_pipelines.items():
                emb = asr_server.extract_embedding_from_file(svp, path)
                te = asr_server.extract_embedding_from_file(svp, t)
                if emb is None or te is None:
                    continue
                scores.append(1 - cosine(te.flatten(), emb.flatten()))
            avg = np.mean(scores) if scores else -1
            mark = '🟢新域?' if avg > 0.55 else ('🟡' if avg > 0.45 else '🔴旧域?')
            print(f"  {s.get('timestamp','')} | 对今天宝宝声均分 {avg:.3f} {mark} | {[f'{x:.3f}' for x in scores]}")
