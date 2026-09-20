#!/Users/mac/asr_env/bin/python3
# -*- coding: utf-8 -*-
"""
每日上传断崖检测 (launchd 每天 20:30 调度)
统计 processed_files_a 中按"录音时间"归到今天的文件数，
与近 7 天均值对比，低于 20% 则发邮件告警。
"""
import os
import sys
from datetime import datetime, timedelta
from collections import Counter

BASE = "/Users/mac/asr-server"
os.chdir(BASE)
sys.path.insert(0, BASE)

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE, ".env"))

from db_manager import init_pool, get_connection, return_connection, parse_recording_time
from email_utils import send_email_sync


def build_report(counts, today):
    today_count = counts.get(today, 0)
    prev = [counts.get(today - timedelta(days=i), 0) for i in range(1, 8)]
    avg7 = sum(prev) / 7.0
    lines = [f"{today - timedelta(days=i)}: {counts.get(today - timedelta(days=i), 0)}" for i in range(1, 8)]
    detail = "\n".join(lines)
    body = (
        f"今日(按录音时间)文件数: {today_count}\n"
        f"近 7 天均值: {avg7:.1f}\n"
        f"告警阈值(均值×20%): {avg7 * 0.2:.1f}\n\n"
        f"近 7 天明细:\n{detail}\n"
    )
    return today_count, avg7, body


def main():
    init_pool()
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        since = datetime.now() - timedelta(days=9)
        cursor.execute(
            "SELECT filename FROM processed_files_a WHERE processed_at >= %s",
            (since,),
        )
        rows = cursor.fetchall()
        cursor.close()
    finally:
        if conn:
            return_connection(conn)

    counts = Counter()
    for (fn,) in rows:
        try:
            t = parse_recording_time(fn or "")
        except Exception:
            t = None
        if t:
            counts[t.date()] += 1

    today = datetime.now().date()
    today_count, avg7, body = build_report(counts, today)
    print(body.strip())

    if avg7 > 0 and today_count < avg7 * 0.2:
        send_email_sync(
            "⚠️ ASR 每日上传断崖告警",
            f"检测到今日上传文件数异常偏低（检查时间 {datetime.now():%Y-%m-%d %H:%M}）。\n\n"
            f"{body}\n请检查设备上传 / NAS 挂载 / 5008 服务状态。",
        )
        print("[告警] 邮件已发送")
    else:
        print("[正常] 未触发告警")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # 检测本身失败也要让人知道（如 DB 连不上）
        try:
            send_email_sync(
                "⚠️ ASR 每日上传断崖检测运行失败",
                f"检查时间: {datetime.now():%Y-%m-%d %H:%M}\n错误: {e}",
            )
        except Exception:
            pass
        print(f"检测失败: {e}", file=sys.stderr)
        sys.exit(1)
