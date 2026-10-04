#!/usr/bin/env python3
"""一次性回填: 从 NAS 各设备 processed 目录建立 filename→device 映射,
回填 transcriptions.device / baby_cry_events.device 两列（只填 NULL 行, 幂等可重跑）。
用法: /Users/mac/asr_env/bin/python3 backfill_device_columns.py
"""
import os
import subprocess
import sys
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

RECORDS = "/Volumes/download/records"
DEVICES = ["Pixel-5", "Pixel-6", "Sony-1", "Sony-2", "Sony-3"]


def scan_device_files(dev: str) -> list:
    """一次 find 拉设备 processed 目录全部文件名（子进程超时保护, 防止 SMB 卡死阻塞）"""
    d = os.path.join(RECORDS, dev, "processed")
    if not os.path.isdir(d):
        return []
    try:
        r = subprocess.run(["find", d, "-type", "f", "-not", "-name", ".*"],
                           capture_output=True, text=True, timeout=180)
        return [ln for ln in r.stdout.splitlines() if ln.strip()]
    except subprocess.TimeoutExpired:
        print(f"[warn] {dev} 扫描超时, 跳过")
        return []


def build_mapping() -> dict:
    mapping = {}
    for dev in DEVICES:
        # 1) processed 目录: 文件名即源文件名
        files = scan_device_files(dev)
        for p in files:
            mapping[os.path.basename(p)] = dev
        # 2) audio_segments 切片目录: <日期>/<源文件stem>/seg_N.wav, 切片永久保留覆盖率更高
        seg_dir = os.path.join(RECORDS, dev, "audio_segments")
        if os.path.isdir(seg_dir):
            try:
                r = subprocess.run(["find", seg_dir, "-maxdepth", "2", "-type", "d"],
                                   capture_output=True, text=True, timeout=180)
                for d in r.stdout.splitlines():
                    stem = os.path.basename(d.strip())
                    if stem and stem not in ("audio_segments",) and not stem.startswith("."):
                        mapping[stem] = dev
                        # 同名不同扩展名兜底 (xxx.m4a / xxx.acc)
                        for ext in (".m4a", ".acc", ".wav"):
                            mapping.setdefault(stem + ext, dev)
            except subprocess.TimeoutExpired:
                print(f"[warn] {dev} 切片目录扫描超时, 跳过")
        print(f"[scan] {dev}: processed {len(files)} + 切片目录累计映射 {sum(1 for v in mapping.values() if v == dev)}")
    return mapping


def backfill_table(table: str, mapping: dict) -> int:
    import psycopg2
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    cur = conn.cursor()
    cur.execute(f"SELECT id, filename FROM {table} WHERE device IS NULL")
    rows = cur.fetchall()
    updated = 0
    for rid, fn in rows:
        dev = mapping.get(fn)
        if dev:
            cur.execute(f"UPDATE {table} SET device = %s WHERE id = %s", (dev, rid))
            updated += 1
    conn.commit()
    cur.close()
    conn.close()
    return updated


if __name__ == "__main__":
    m = build_mapping()
    print(f"[map] 映射总数: {len(m)} 个唯一文件名")
    for table in ("transcriptions", "baby_cry_events"):
        n = backfill_table(table, m)
        print(f"[done] {table}: 回填 {n} 行")
    print("完成")
