#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量重转写 2026-06-01 之后的语音记录（funasr 1.4.14 → 1.4.16 识别质量修复 + Nano 第四模型全量补齐）

对 transcriptions.segments_json 里每段:
  - text           Paraformer(SeACo)+VAD+标点+热词"大可" 重转写 (标记 _retrans_at)
  - sensevoice_text / emotion  SenseVoiceSmall 重转写
  - nano_text      Fun-ASR-Nano (本地 audiocpp_server 8123, Metal) (标记 _nano_at)
  - 原文保留在 _orig_text / _orig_sv (可回溯)
  - spk / whisper_text / segment_audio_path 不动
特性: 段级幂等(_retrans_at/_nano_at 双标记) · NAS挂载守卫(连续失败中止) · 单文件级commit
      Nano 走本地 HTTP 服务, 4线程并发; 失败段不标 _nano_at, 重跑自动补
用法: /Users/mac/asr_env/bin/python3 retranscribe_jun_on.py [--limit N] [--start YYYY-MM-DD]
"""
import os
import re
import sys
import json
import time
import argparse
import subprocess
import warnings

warnings.filterwarnings('ignore')

try:
    from dotenv import load_dotenv
    load_dotenv('/Users/mac/asr-server/.env', override=True)
except Exception:
    pass

NAS_BASE_CANDS = [
    '/Volumes/download/records/Sony-2',
    '/Volumes/download/records/Sony-1',
]
MOUNT_CHECK_INTERVAL = 200      # 每 N 段检查一次挂载
MOUNT_FAIL_LIMIT = 5            # 连续 N 段读不到文件则中止
DB_URL = os.getenv('DATABASE_URL', '')
HOTWORD = os.getenv('ASR_HOTWORD', '大可') or '大可'
NANO_URL = os.getenv('NANO_ASR_URL', 'http://127.0.0.1:8123/v1/audio/transcriptions')
NANO_WORKERS = 2   # audiocpp_server 内部串行处理, 多线程仅用于上传/下载流水线重叠
LOG_PREFIX = '[retrans]'


def log(msg):
    print(f"{LOG_PREFIX} {time.strftime('%H:%M:%S')} {msg}", flush=True)


def check_mount():
    """挂载健康检查: 任一源目录可列出即 OK"""
    for base in NAS_BASE_CANDS:
        try:
            r = subprocess.run(['ls', base], capture_output=True, timeout=20)
            if r.returncode == 0:
                return base
        except Exception:
            continue
    return None


def resolve_audio(path):
    """按候选源解析切片真实路径"""
    rel = path.lstrip('/')
    for base in NAS_BASE_CANDS:
        p = os.path.join(base, rel)
        if os.path.exists(p):
            return p
    return None


def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def nano_one(src):
    """调用本地 Fun-ASR-Nano 服务识别一个切片, 成功返回文本(可空串), 失败返回 None"""
    import requests
    try:
        with open(src, 'rb') as f:
            resp = requests.post(
                NANO_URL,
                files={'file': (os.path.basename(src), f, 'audio/wav')},
                data={'model': 'fun-asr-nano', 'language': 'auto'},
                timeout=120,
            )
        if resp.status_code == 200:
            return (resp.json().get('text') or '').strip()
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='最多处理 N 个文件(调试)')
    ap.add_argument('--start', default='2026-06-01')
    args = ap.parse_args()

    dev = pick_device()
    log(f'加载 FunASR 模型 (SeACo-Paraformer + VAD + 标点) @ {dev}...')
    from funasr import AutoModel
    asr = AutoModel(
        model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        device=dev,
        disable_update=True,
    )
    log('加载 SenseVoice 模型...')
    sv = AutoModel(
        model="iic/SenseVoiceSmall",
        vad_model="fsmn-vad",
        device=dev,
        disable_update=True,
    )
    log('模型加载完成')

    import psycopg2, psycopg2.extras
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = False
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT id, recording_time, filename, segments_json
        FROM transcriptions
        WHERE recording_time >= %s
        ORDER BY recording_time ASC
    """, (args.start,))
    rows = cur.fetchall()
    if args.limit:
        rows = rows[:args.limit]
    log(f"待处理 {len(rows)} 个文件 (start={args.start})")

    mount_fail = 0
    seg_done = seg_skip = seg_err = nano_done = nano_skip = nano_err = file_done = 0
    t0 = time.time()
    BATCH = 32

    from concurrent.futures import ThreadPoolExecutor
    nano_pool = ThreadPoolExecutor(max_workers=NANO_WORKERS)

    # 收集待处理段 (跨文件批量); need_asr=未做过 1.4.16 重转写
    def collect_batch(rows, start_ri):
        nonlocal seg_skip, seg_err, nano_skip, mount_fail
        batch = []  # (row_idx, seg_idx, wav_path, need_asr)
        ri = start_ri
        while ri < len(rows) and len(batch) < BATCH:
            try:
                segs = json.loads(rows[ri]['segments_json'] or '[]')
            except Exception:
                ri += 1
                continue
            for si, seg in enumerate(segs):
                if len(batch) >= BATCH:
                    break
                apath = seg.get('segment_audio_path') or ''
                if not apath.endswith('.wav'):
                    continue
                need_asr = not seg.get('_retrans_at')
                need_nano = not seg.get('_nano_at')
                if not need_asr:
                    seg_skip += 1
                if not need_nano:
                    nano_skip += 1
                if not need_asr and not need_nano:
                    continue
                src = resolve_audio(apath)
                if not src:
                    mount_fail += 1
                    seg_err += 1
                    if mount_fail >= MOUNT_FAIL_LIMIT:
                        return batch, ri, True
                    continue
                mount_fail = 0
                batch.append((ri, si, src, need_asr))
            ri += 1
        return batch, ri, False

    # 按 row_id 分组缓存 segments (批量回写)
    pending_rows = {}

    def get_segs(ri):
        if ri not in pending_rows:
            pending_rows[ri] = json.loads(rows[ri]['segments_json'] or '[]')
        return pending_rows[ri]

    ri = 0
    while ri < len(rows):
        # 挂载守卫
        if seg_done and seg_done % (MOUNT_CHECK_INTERVAL - MOUNT_CHECK_INTERVAL % BATCH) == 0 and seg_done > 0:
            if not check_mount():
                log(f"❌ 挂载健康检查失败, 中止(已处理 {seg_done} 段, 断点可续跑)")
                conn.commit()
                sys.exit(3)

        batch, next_ri, abort = collect_batch(rows, ri)
        ri = next_ri
        if abort:
            log(f"❌ 连续 {MOUNT_FAIL_LIMIT} 段音频缺失, 疑似挂载病态, 中止(非0退出交 launchd 重试)")
            conn.commit()
            sys.exit(3)
        if not batch:
            continue

        # Nano 提前提交 (本地 HTTP, 与 MPS 批推理并行)
        nano_futs = [(b, nano_pool.submit(nano_one, b[2])) for b in batch]

        # Paraformer + SenseVoice 批推理 (只跑 need_asr 段)
        asr_items = [b for b in batch if b[3]]
        r_asr = r_sv = None
        if asr_items:
            try:
                paths = [b[2] for b in asr_items]
                r_asr = asr.generate(input=paths, hotword=HOTWORD, batch_size_s=120)
                r_sv = sv.generate(input=paths, cache={}, language="auto",
                                   use_itn=True, batch_size_s=120)
            except Exception as e:
                log(f"⚠️ 批推理失败({len(asr_items)}段): {str(e)[:120]}")
                seg_err += len(asr_items)

        dirty_rows = set()
        now = time.strftime('%Y-%m-%d %H:%M:%S')

        # 写回 ASR/SV 结果 (防御: funasr 返回数可能少于输入, 按下标收敛)
        if r_asr is not None and r_sv is not None:
            n = min(len(asr_items), len(r_asr), len(r_sv))
            if n < len(asr_items):
                seg_err += len(asr_items) - n
                log(f"⚠️ 推理返回 {n}/{len(asr_items)} 段, 缺失段下轮重试")
            for i in range(n):
                row_idx, seg_idx, _src, _na = asr_items[i]
                seg = get_segs(row_idx)[seg_idx]
                ra, rs = r_asr[i], r_sv[i]
                new_text = (ra.get('text') or '').strip()
                sv_raw = (rs.get('text') or '').strip()
                sv_text, emo = '', ''
                if sv_raw:
                    m = re.search(r'<\|(happy|sad|angry|surprised|fear|disgusted|neutral|laughter)\|>', sv_raw)
                    if m:
                        emo = m.group(1)
                    sv_text = re.sub(r'<\|[^|]*\|>', '', sv_raw).strip()
                seg.setdefault('_orig_text', seg.get('text') or '')
                seg.setdefault('_orig_sv', seg.get('sensevoice_text') or '')
                if new_text:
                    seg['text'] = new_text
                if sv_text:
                    seg['sensevoice_text'] = sv_text
                if emo:
                    seg['emotion'] = emo
                seg['_retrans_at'] = now
                seg_done += 1
                dirty_rows.add(row_idx)

        # 写回 Nano 结果
        for b, fut in nano_futs:
            row_idx, seg_idx = b[0], b[1]
            txt = None
            try:
                txt = fut.result(timeout=180)
            except Exception:
                txt = None
            if txt is None:
                nano_err += 1
                continue
            seg = get_segs(row_idx)[seg_idx]
            seg['nano_text'] = txt
            seg['_nano_at'] = now
            nano_done += 1
            dirty_rows.add(row_idx)

        # 回写本批涉及的行 (只写实际修改过的行)
        for row_idx in sorted(dirty_rows):
            cur.execute(
                "UPDATE transcriptions SET segments_json = %s WHERE id = %s",
                (json.dumps(pending_rows[row_idx], ensure_ascii=False), rows[row_idx]['id']),
            )
            file_done += 1
        conn.commit()
        pending_rows.clear()

        if seg_done % (BATCH * 10) == 0 or nano_done % (BATCH * 10) == 0 or ri >= len(rows):
            rate = (seg_done + nano_done) / max(time.time() - t0, 1)
            log(f"进度 文件≈{ri}/{len(rows)} | Paraformer {seg_done} 段 (跳过 {seg_skip}, 失败 {seg_err}) | Nano {nano_done} 段 (跳过 {nano_skip}, 失败 {nano_err}) | {rate:.1f} 段/秒")

    log(f"✅ 完成: {file_done} 文件 | Paraformer {seg_done} 段 (跳过 {seg_skip}, 失败 {seg_err}) | Nano {nano_done} 段 (跳过 {nano_skip}, 失败 {nano_err}) | 总耗时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == '__main__':
    main()
