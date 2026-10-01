#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fun-ASR-Nano 本地部署可行性实测: ModelScope 下载 + MPS 推理, 对比已知样本"""
import os
import time
import warnings

warnings.filterwarnings('ignore')
os.environ['MODELSCOPE_CACHE'] = os.path.expanduser('~/.cache/modelscope')

SAMPLES = [
    # (说明, 路径, 参考三方结果)
    ('婆婆方言句(三方收敛:那多了眼睛又不好)',
     '/Volumes/download/records/Sony-2/audio_segments/2026-09-30/TermuxAudioRecording_2026-09-30_16-15-15/seg_14.wav'),
    ('大可普通话(Paraformer:个小可爱在空中一)',
     '/Volumes/download/records/Sony-2/audio_segments/2026-09-30/TermuxAudioRecording_2026-09-30_13-24-29/seg_10.wav'),
]

log = lambda m: print(f"[nano-test] {time.strftime('%H:%M:%S')} {m}", flush=True)

log('加载 Fun-ASR-Nano-2512 (首次从 ModelScope 下载 ~1.6GB)...')
t0 = time.time()
from funasr import AutoModel
import torch
dev = 'mps' if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else 'cpu'
log(f'device = {dev}')
m = AutoModel(
    model="FunAudioLLM/Fun-ASR-Nano-2512",
    hub="ms",
    trust_remote_code=True,
    device=dev,
    disable_update=True,
)
log(f'模型加载完成 (含下载共 {time.time()-t0:.0f}s)')

for desc, wav in SAMPLES:
    if not os.path.exists(wav):
        log(f'⚠️ 缺文件: {wav}')
        continue
    try:
        t = time.time()
        r = m.generate(input=[wav], cache={}, batch_size=1, language="中文", itn=True)
        dt = time.time() - t
        log(f'{desc}')
        log(f'  Nano: {r[0].get("text", "")!r}  ({dt:.1f}s)')
    except Exception as e:
        log(f'  ❌ 推理失败: {str(e)[:200]}')

log('测试完成')
