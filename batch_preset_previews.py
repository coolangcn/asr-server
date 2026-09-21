#!/usr/bin/env python3
"""【2026-09-20 用户建议】批量预切历史哭声事件的候选片段。
遍历所有未删除、未标误报的哭声事件，逐个调用 5008 的
POST /api/preset_cry_segments/<id>（幂等，已有清单直接跳过），
把滑窗 top-5 候选段存盘，之后预览/确认零 GPU 秒开。
用法: python3 batch_preset_previews.py [--retry-failed]
进度日志: log/preset_batch.log；失败清单: log/preset_failed.json
"""
import json
import os
import sys
import time

import requests

BASE = "http://localhost:5008"
ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(ROOT, "log", "preset_batch.log")
FAIL_PATH = os.path.join(ROOT, "log", "preset_failed.json")
TIMEOUT_PER_EVENT = 600  # 单事件上限（GPU 排队 + 滑窗 + 精排）


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def admin_token():
    with open(os.path.join(ROOT, ".env"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("ADMIN_TOKEN="):
                return line.strip().split("=", 1)[1].strip().strip('"')
    raise RuntimeError(".env 中未找到 ADMIN_TOKEN")


HEADERS = {"X-Admin-Token": admin_token()}


def all_event_ids():
    """分页拉取全部事件 ID（列表接口已排除误报/已删除）"""
    ids, offset, page_size = [], 0, 100
    while True:
        r = requests.get(f"{BASE}/api/cry_events",
                         params={"offset": offset, "limit": page_size},
                         headers=HEADERS, timeout=60)
        r.raise_for_status()
        events = r.json().get("events", [])
        if not events:
            break
        ids.extend(e["id"] for e in events)
        if len(events) < page_size:
            break
        offset += page_size
    return sorted(ids)


def preset_one(event_id):
    try:
        r = requests.post(f"{BASE}/api/preset_cry_segments/{event_id}",
                          headers=HEADERS, timeout=TIMEOUT_PER_EVENT)
        data = r.json() if r.headers.get("Content-Type", "").startswith("application/json") else {}
        if r.status_code == 200:
            if data.get("already"):
                return "already", None
            if data.get("skipped"):
                return "skipped", data["skipped"]
            if data.get("ok"):
                return "ok", None
            return "fail", data.get("error") or "未知返回"
        return "fail", data.get("error") or f"HTTP {r.status_code}"
    except Exception as e:
        return "fail", str(e)[:200]


def main():
    retry_only = "--retry-failed" in sys.argv
    if retry_only and os.path.exists(FAIL_PATH):
        with open(FAIL_PATH, encoding="utf-8") as f:
            ids = json.load(f)
        log(f"===== 重跑失败清单 ({len(ids)} 个) =====")
    else:
        ids = all_event_ids()
        log(f"===== 批量预切启动：共 {len(ids)} 个事件 =====")

    stats = {"ok": 0, "already": 0, "skipped": 0, "fail": 0}
    failed = []
    t0 = time.time()
    for i, eid in enumerate(ids, 1):
        status, info = preset_one(eid)
        stats[status] = stats.get(status, 0) + 1
        if status == "fail":
            failed.append(eid)
            log(f"  [{i}/{len(ids)}] #{eid} 失败: {info}")
        elif status == "ok":
            log(f"  [{i}/{len(ids)}] #{eid} 预切成功 ({time.time()-t0:.0f}s)")
        if i % 20 == 0:
            log(f"  进度 {i}/{len(ids)} | 成功 {stats['ok']} 已有 {stats['already']} "
                f"跳过 {stats['skipped']} 失败 {stats['fail']}")
        time.sleep(1)  # 温和节奏，给实时检测留 GPU 空隙

    with open(FAIL_PATH, "w", encoding="utf-8") as f:
        json.dump(failed, f)
    log(f"===== 批量预切完成: {stats} 失败 {len(failed)} 个 (清单 {FAIL_PATH}) "
        f"用时 {(time.time()-t0)/60:.1f} 分钟 =====")
    if failed and not retry_only:
        log("提示: 可运行 python3 batch_preset_previews.py --retry-failed 重试失败项")


if __name__ == "__main__":
    main()
