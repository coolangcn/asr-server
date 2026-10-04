#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""历史哭声事件预切——NAS 端切割版（一次性补齐脚本）

Mac 端 SMB 读 NAS 大录音文件太慢（拷贝/直读都要 200-400s/事件），
改为: NAS 本地 ffmpeg 按事件的 start/end 切片 → tar 打包 → 一次拉回 → 写 manifest。
只处理切片 ≤600s 的事件（超长切片仍走 5008 降采样滑窗路径保证精度）。
幂等: 已有 manifest 的事件自动跳过。

用法: python3 preset_nas_side.py
"""

import os
import json
import subprocess
import psycopg2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEG_DIR = os.path.join(BASE_DIR, "temp", "preview_segments")
NAS_HOST = "admin@192.168.1.188"
NAS_PASS = os.environ.get("NAS_SSH_PASS", "74123698cN")
VOL_MAP = ("/Volumes/download/records/", "/vol1/1000/download/records/")
MAX_FAST_SEC = 600

os.makedirs(SEG_DIR, exist_ok=True)


def db_events():
    conn = psycopg2.connect(
        "postgresql://postgres:cncncncn@192.168.1.188:5433/postgres")
    cur = conn.cursor()
    cur.execute("""
        SELECT id, event_files_json, audio_path, start_time, end_time
        FROM baby_cry_events
        WHERE COALESCE(is_deleted, false) = false
          AND COALESCE(false_positive, false) = false
        ORDER BY id
    """)
    rows = cur.fetchall()
    conn.close()
    return rows


def nas_path_of(event_files, audio_path):
    """挑出第一个 NAS records 路径(映射到 NAS 本地挂载路径)"""
    cands = []
    if event_files:
        cands += [p for p in event_files if isinstance(p, str)]
    if audio_path:
        cands.append(audio_path)
    for p in cands:
        if VOL_MAP[0] in p:
            return p.replace(VOL_MAP[0], VOL_MAP[1])
    return None


def variants_of(st, et):
    """与 5008 快速路径一致: 原段 / 前垫2s / 前垫5s"""
    return [
        (1.0, st, et),
        (0.9, max(0.0, st - 2.0), et + 2.0),
        (0.8, max(0.0, st - 5.0), et + 5.0),
    ]


def main():
    events = db_events()
    todo = {}
    for eid, efj, apath, st, et in events:
        if os.path.exists(os.path.join(SEG_DIR, f"{eid}.json")):
            continue  # 已有 manifest, 幂等跳过
        try:
            st, et = float(st), float(et)
        except (TypeError, ValueError):
            continue
        if not (1.0 <= et - st <= MAX_FAST_SEC):
            continue  # 超长/无效切片, 交由 5008 滑窗路径
        ev_files = efj if isinstance(efj, list) else (
            json.loads(efj) if isinstance(efj, str) and efj else [])
        src = nas_path_of(ev_files, apath)
        if not src:
            continue
        todo[eid] = (src, st, et)
    print(f"待 NAS 端补切: {len(todo)} 个事件")

    if not todo:
        return

    # 1) 生成 NAS 端批量切割脚本 (缺文件/失败不中断)
    remote_sh = ["set -u", "OUT=/tmp/preset_batch", "rm -rf $OUT", "mkdir -p $OUT"]
    for eid, (src, st, et) in todo.items():
        for i, (_, ws, we) in enumerate(variants_of(st, et)):
            dur = we - ws
            remote_sh.append(
                f'[ -f "{src}" ] && ffmpeg -v error -y -ss {ws:.3f} -t {dur:.3f} '
                f'-i "{src}" -ac 1 -ar 16000 "$OUT/{eid}_v{i}.wav" 2>/dev/null'
            )
    remote_sh.append("cd /tmp && tar czf preset_batch.tgz preset_batch && echo BATCH_DONE")
    script = "\n".join(remote_sh)

    print("NAS 端切割中…")
    r = subprocess.run(
        ["sshpass", "-p", NAS_PASS, "ssh", "-o", "StrictHostKeyChecking=no",
         NAS_HOST, "bash -s"],
        input=script, text=True, capture_output=True, timeout=1800)
    if "BATCH_DONE" not in r.stdout:
        print("NAS 批量切割失败:", r.stderr[-500:])
        return
    print("NAS 端切割完成, 拉回产物…")

    # 2) 拉回并解压
    subprocess.run(["rm", "-rf", "/tmp/preset_batch", "/tmp/preset_batch.tgz"],
                   check=False)
    subprocess.run(
        ["sshpass", "-p", NAS_PASS, "scp", "-o", "StrictHostKeyChecking=no",
         f"{NAS_HOST}:/tmp/preset_batch.tgz", "/tmp/preset_batch.tgz"],
        check=True, timeout=600)
    subprocess.run(["tar", "xzf", "/tmp/preset_batch.tgz", "-C", "/tmp"],
                   check=True)

    # 3) 写 manifest (只收非空 wav)
    ok = miss = 0
    for eid, (src, st, et) in todo.items():
        manifest = []
        for i, (score, ws, we) in enumerate(variants_of(st, et)):
            wav = f"/tmp/preset_batch/{eid}_v{i}.wav"
            if os.path.exists(wav) and os.path.getsize(wav) > 1024:
                manifest.append({"variant": i, "start": round(ws, 2),
                                 "end": round(we, 2), "score": score})
        if manifest:
            with open(os.path.join(SEG_DIR, f"{eid}.json"), "w",
                      encoding="utf-8") as mf:
                json.dump(manifest, mf, ensure_ascii=False)
            ok += 1
        else:
            miss += 1
            print(f"  #{eid} 无有效切片 (源可能不可达): {src}")
    print(f"完成: 成功 {ok} 个 / 无切片 {miss} 个 (manifest 目录 {SEG_DIR})")


if __name__ == "__main__":
    main()
