#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASR 盲测对比: 本地 FunASR(Paraformer-large) vs Groq Whisper-large-v3-turbo
抽取最近 N 天的已转写切片, 逐句调 Groq 重识别, 与本地结果做字符相似度对比。

用法:
  python3 compare_asr_blind.py                    # 默认 100 句 / 最近 30 天
  python3 compare_asr_blind.py --total 50 --days 7
输出:
  asr_blind_result.jsonl   逐句对比明细
  控制台                    汇总指标 + 抽样展示
"""

import os
import sys
import json
import time
import random
import difflib
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'nas-audio-notes-client'))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, '.env'))
    load_dotenv()
except Exception:
    pass

import requests
import psycopg2

SOURCE_DIRS = [
    '/Volumes/download/records/Sony-2',
    '/Volumes/download/records/Sony-1',
    '/Users/mac/asr-server/nas-audio-notes-client',
]


def locate_audio(rel):
    rel_clean = (rel or '').replace('\\', '/').strip()
    for prefix in ('/audio_segments/', 'audio_segments/'):
        if rel_clean.startswith(prefix):
            rel_clean = rel_clean[len(prefix):]
            break
    for base in SOURCE_DIRS:
        p = os.path.join(base, 'audio_segments', rel_clean)
        if os.path.isfile(p):
            return p
    return None


def groq_transcribe(path, keys, model, cursor):
    """带轮换的 Groq 转写, 返回 (text 或 None, err 或 None)"""
    n = len(keys)
    for i in range(n):
        key = keys[(cursor + i) % n]
        try:
            with open(path, 'rb') as f:
                resp = requests.post(
                    'https://api.groq.com/openai/v1/audio/transcriptions',
                    headers={'Authorization': f'Bearer {key}'},
                    files={'file': (os.path.basename(path), f, 'audio/wav')},
                    data={'model': model, 'language': 'zh', 'response_format': 'json'},
                    timeout=45,
                )
        except Exception as e:
            return None, str(e)
        if resp.status_code == 200:
            return resp.json().get('text', ''), None
        if resp.status_code in (401, 403, 429):
            continue  # 换下一个 key
        return None, f"HTTP {resp.status_code}: {resp.text[:150]}"
    return None, "全部 key 失败"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=30, help='回看天数')
    ap.add_argument('--total', type=int, default=100, help='总抽样句数')
    ap.add_argument('--per-day', type=int, default=12, help='每天最多抽几句')
    ap.add_argument('--min-len', type=int, default=4, help='本地文本最小长度')
    ap.add_argument('--spk', default='', help='逗号分隔的说话人过滤(留空=不过滤, 自动排除Baby/空)')
    args = ap.parse_args()

    keys = [k.strip() for k in os.getenv('GROQ_API_KEYS', '').split(',') if k.strip()]
    if not keys:
        print('❌ .env 未配置 GROQ_API_KEYS')
        sys.exit(1)
    model = os.getenv('GROQ_ASR_MODEL', 'whisper-large-v3-turbo')

    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        print('❌ 未配置 DATABASE_URL')
        sys.exit(1)
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    cur.execute(
        "SELECT filename, segments_json, recording_time FROM transcriptions "
        "WHERE segments_json LIKE '%%segment_audio_path%%' AND recording_time >= NOW() - INTERVAL '%s days' "
        "ORDER BY recording_time DESC",
        (args.days,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    print(f"📦 最近 {args.days} 天含切片的录音: {len(rows)} 条")

    # 按日期分组抽句
    by_day = {}
    for filename, seg_json, rec_time in rows:
        try:
            segs = json.loads(seg_json) if seg_json else []
        except Exception:
            continue
        for seg in segs:
            text = (seg.get('text') or '').strip()
            rel = seg.get('segment_audio_path') or ''
            if len(text) < args.min_len or not rel:
                continue
            spk = seg.get('spk', '')
            if args.spk:
                if spk not in [s.strip() for s in args.spk.split(',')]:
                    continue
            else:
                if spk in ('', 'Baby', 'Unknown') or spk.lower() == 'unknown':
                    continue  # 默认排除咿呀/未归属句, 只比清晰对话
            day = (rec_time.date().isoformat() if rec_time else (filename or '')[:10])
            by_day.setdefault(day, []).append({
                'day': day, 'spk': seg.get('spk', ''), 'local': text, 'rel': rel,
                'time': str(seg.get('recording_time') or seg.get('start') or ''),
            })
    days = sorted(by_day.keys(), reverse=True)
    pool = []
    for d in days:
        cands = by_day[d]
        random.shuffle(cands)
        pool.extend(cands[:args.per_day])
    random.shuffle(pool)
    pool = pool[:args.total]
    print(f"🎯 抽样 {len(pool)} 句 ({len(days)} 天), 模型: {model}")

    out_path = os.path.join(BASE_DIR, 'asr_blind_result.jsonl')
    cursor, done, fail, ratios, exact = 0, 0, 0, [], 0
    with open(out_path, 'w', encoding='utf-8') as fo:
        for idx, item in enumerate(pool, 1):
            path = locate_audio(item['rel'])
            if not path:
                fail += 1
                continue
            groq_text, err = groq_transcribe(path, keys, model, cursor)
            if groq_text is None:
                fail += 1
                if '全部 key' in (err or ''):
                    print(f"❌ {err}, 中止")
                    break
                time.sleep(0.5)
                continue
            ratio = difflib.SequenceMatcher(None, item['local'], groq_text).ratio()
            ratios.append(ratio)
            if item['local'] == groq_text:
                exact += 1
            fo.write(json.dumps({**item, 'groq': groq_text, 'ratio': round(ratio, 3)},
                                ensure_ascii=False) + '\n')
            done += 1
            if idx % 10 == 0:
                print(f"  进度 {idx}/{len(pool)} …")
            time.sleep(0.15)  # 温和限速

    print('\n========== 汇总 ==========')
    print(f"成功 {done} 句 / 失败 {fail} 句")
    if ratios:
        avg = sum(ratios) / len(ratios)
        hi = sum(1 for r in ratios if r >= 0.9)
        mid = sum(1 for r in ratios if 0.6 <= r < 0.9)
        lo = sum(1 for r in ratios if r < 0.6)
        print(f"平均字符相似度: {avg:.3f}  (1.0=完全一致)")
        print(f"高度一致(≥0.9): {hi} 句 | 部分差异(0.6-0.9): {mid} 句 | 大差异(<0.6): {lo} 句")
        print(f"逐字完全一致: {exact}/{len(ratios)}")
        print('\n--- 差异最大的 10 句 (人工判断哪边更准) ---')
        rows_out = [json.loads(l) for l in open(out_path, encoding='utf-8')]
        rows_out.sort(key=lambda r: r['ratio'])
        for r in rows_out[:10]:
            print(f"\n[{r['day']} {r['time']}] {r['spk']}  ratio={r['ratio']}")
            print(f"  本地: {r['local']}")
            print(f"  Groq: {r['groq']}")
    print(f"\n明细: {out_path}")


if __name__ == '__main__':
    main()
