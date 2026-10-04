# -*- coding: utf-8 -*-
"""2025-11 说话人归属回填脚本
背景: 2025-11 期间爸爸(12-06)和 Baby(2026-03)尚未注册声纹, 导致当月 56% 段 spk=Unknown。
方案: 用当前声纹库(5人齐全, Termux 域样本为主)对 Unknown 段重新声纹归属。
音频: DB 里 segment_audio_path 指向已删除的 temp\\seg_{n}_{k}_{ts}.wav,
      按 ts(UTC epoch 处理时刻) 映射回切分存档 audio_segments/日期/TermuxAudioRecording_日期_HH-MM-SS/seg_{n}.wav
规则: 复用线上 identify_speaker_fusion (3模型投票+负样本拒绝), 保证与实时管线一致。
用法:
  python3 backfill_nov_spk.py --dry-run          # 只统计映射率, 不加载模型不写库
  python3 backfill_nov_spk.py --limit 50         # 试跑 50 个可映射段并写回
  python3 backfill_nov_spk.py                    # 全量
"""
import os, sys, json, glob, ast, wave, argparse, re
from datetime import datetime, timedelta

os.chdir('/Users/mac/asr-server')
sys.path.insert(0, '/Users/mac/asr-server')
from dotenv import load_dotenv; load_dotenv()

import psycopg2

ARCHIVE_ROOT = '/Volumes/download/records/Sony-2/audio_segments'
BACKUP_FILE = '/Users/mac/asr-server/backup_nov2025_segments.jsonl'
TEMP_RE = re.compile(r'temp\\seg_(\d+)_(\d+)_(\d+)\.wav')
DIR_RE = re.compile(r'TermuxAudioRecording_\d{4}-\d{2}-\d{2}_(\d{2})-(\d{2})-(\d{2})$')

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    return ap.parse_args()

def collect_unknown_segments():
    """拉取 11 月含 Unknown 的行, 解析出 Unknown 段"""
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    cur = conn.cursor()
    cur.execute("""SELECT id, segments_json FROM transcriptions
        WHERE EXTRACT(MONTH FROM created_at)=11 AND EXTRACT(YEAR FROM created_at)=2025
          AND segments_json::text LIKE '%Unknown%' ORDER BY id""")
    rows = cur.fetchall()
    conn.close()
    tasks, miss = [], {'no_temp_path': 0, 'bad_pattern': 0, 'no_dir': 0, 'no_file': 0}
    for rid, sj in rows:
        segs = json.loads(sj) if sj else []
        dirty = False
        for i, seg in enumerate(segs):
            if seg.get('spk') != 'Unknown':
                continue
            p = seg.get('segment_audio_path') or ''
            if not p:
                miss['no_temp_path'] += 1
                continue
            if p.startswith('/audio_segments/'):
                # 标准归档路径, 直接定位
                disk = ARCHIVE_ROOT + p[len('/audio_segments'):]
                cand = [os.path.dirname(disk)]
                seg_n = int(re.search(r'seg_(\d+)\.wav$', disk).group(1))
            else:
                # temp 格式: ts = 处理时刻(UTC); 录音目录名也是 UTC, 开始 ≈ ts-78s, 窗口放宽
                m2 = TEMP_RE.search(p)
                if not m2:
                    miss['bad_pattern'] += 1
                    continue
                seg_n, ts = int(m2.group(1)), int(m2.group(3))
                proc_dt = datetime.utcfromtimestamp(ts)
                day = proc_dt.strftime('%Y-%m-%d')
                lo, hi = proc_dt - timedelta(seconds=200), proc_dt - timedelta(seconds=40)
                cand = []
                day_dir = os.path.join(ARCHIVE_ROOT, day)
                for d in (glob.glob(os.path.join(day_dir, '*')) if os.path.isdir(day_dir) else []):
                    dm = DIR_RE.search(os.path.basename(d))
                    if not dm:
                        continue
                    ddt = datetime.strptime(f"{day} {dm.group(1)}:{dm.group(2)}:{dm.group(3)}", '%Y-%m-%d %H:%M:%S')
                    if lo <= ddt <= hi:
                        cand.append(d)
            if not cand:
                miss['no_dir'] += 1
                continue
            # 时长校验: wav 实际时长 ≈ (end-start)/1000
            want = (int(seg.get('end') or 0) - int(seg.get('start') or 0)) / 1000.0
            matched = None
            for d in cand:
                wav = os.path.join(d, f'seg_{seg_n}.wav')
                if not os.path.isfile(wav):
                    continue
                try:
                    with wave.open(wav) as w:
                        dur = w.getnframes() / float(w.getframerate())
                except Exception:
                    continue
                if want and abs(dur - want) <= 0.35:
                    matched = wav
                    break
            if matched:
                tasks.append({'row': rid, 'seg_idx': i, 'wav': matched, 'seg': seg})
            else:
                miss['no_file'] += 1
        # seg 修改后由调用方统一写回
    return rows, tasks, miss

def main():
    args = parse_args()
    print('== 拉取并映射 2025-11 Unknown 段 ==', flush=True)
    rows, tasks, miss = collect_unknown_segments()
    total_unk = sum(miss.values()) + len(tasks)
    print(f'行数: {len(rows)} | Unknown 段: {total_unk}')
    print(f"可映射(存档在盘): {len(tasks)} ({len(tasks)*100//max(total_unk,1)}%)")
    print(f"缺失明细: {miss}")
    if args.dry_run:
        for t in tasks[:5]:
            print(f"  样例 row={t['row']} seg#{t['seg_idx']} → {os.path.relpath(t['wav'], ARCHIVE_ROOT)}")
        return

    # 加载 SV 模型 + 声纹库 (复用线上模块)
    print('\n== 加载 SV 模型与声纹库 ==', flush=True)
    import asr_server
    for name, conf in asr_server.Config.SV_MODELS.items():
        print(f"加载 SV [{name}] ...", flush=True)
        asr_server.sv_pipelines[name] = __import__('modelscope.pipelines', fromlist=['pipeline']).pipeline(
            task=__import__('modelscope.utils.constant', fromlist=['Tasks']).Tasks.speaker_verification,
            model=conf['id'], model_revision=conf['rev'], device=asr_server.Config.MODELSCOPE_DEVICE)
    asr_server.load_speaker_db()
    print(f"声纹库: {list(asr_server.speaker_db.keys())} | 负样本 {len(asr_server.negative_samples)}", flush=True)

    todo = tasks[:args.limit] if args.limit else tasks
    print(f'\n== 开始识别: {len(todo)} 段 ==', flush=True)

    # 备份将修改的行
    row_ids = sorted({t['row'] for t in todo})
    orig = {rid: sj for rid, sj in rows if rid in set(row_ids)}
    with open(BACKUP_FILE, 'w', encoding='utf-8') as f:
        for rid in row_ids:
            f.write(json.dumps({'id': rid, 'segments_json': orig[rid]}, ensure_ascii=False) + '\n')
    print(f'已备份 {len(row_ids)} 行 → {BACKUP_FILE}', flush=True)

    # 按行分组识别
    by_row = {}
    for t in todo:
        by_row.setdefault(t['row'], []).append(t)
    results, ok, fail = {}, 0, 0
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    cur = conn.cursor()
    try:
        for idx, (rid, tlist) in enumerate(by_row.items(), 1):
            segs = json.loads(orig[rid])
            changed = False
            for t in tlist:
                winner, conf_, details = asr_server.identify_speaker_fusion(t['wav'])
                if not winner:
                    fail += 1
                    continue
                seg = segs[t['seg_idx']]
                old = []
                try:
                    old = ast.literal_eval(seg.get('recognition_details') or '[]')
                except Exception:
                    pass
                seg['spk'] = winner
                seg['confidence'] = f'{conf_:.3f}'
                seg['recognition_details'] = str(old + [f'2025-11回填: {d}' for d in details])
                changed = True
                ok += 1
                results.setdefault(winner, 0)
                results[winner] += 1
            if changed:
                cur.execute('UPDATE transcriptions SET segments_json=%s WHERE id=%s',
                            (json.dumps(segs, ensure_ascii=False), rid))
            if idx % 20 == 0:
                conn.commit()
                print(f'进度 {idx}/{len(by_row)} 行 | 识别成功 {ok} 失败 {fail}', flush=True)
        conn.commit()
    finally:
        conn.close()
    print(f'\n== 完成 == 识别成功 {ok} 段 | 失败 {fail} 段')
    print(f'归属分布: {results}')

if __name__ == '__main__':
    main()
