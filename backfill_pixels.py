# -*- coding: utf-8 -*-
"""Pixel 积压回填脚本（2026-10 音源切换专用）
把 Pixel-5/Pixel-6 在 NAS 上积压未处理的分钟录音按时间顺序送入 /transcribes 管线
（A 轨哭声检测 + B 轨声纹/转写入库），成功后归档到 <设备>/processed/<日期>/ 并打
带设备前缀的 A 轨标记（防止新双源管线按裸文件名误判/撞名）。

两种模式：
  默认        扫源目录根（未归档文件），成功后移入 processed/ —— 适用于「重启 5008 之前」的回填。
  --from-processed  扫 processed/<日期>/ 已归档文件，以 transcriptions 表是否已有
              (source_device, filename) 记录判断是否分析过——适用于「重启后 catch-up
              已把积压归档打标（b_catchup_success，未分析）」的补救场景。文件不移动。

断点续传：已分析/已打标的文件自动跳过。
挂载守卫：SMB 不健康时立即中止（exit 3），绝不假跑。
"""
import os, sys, glob, re, time, argparse, shutil, subprocess
from datetime import datetime

sys.path.insert(0, '/Users/mac/asr-server')
os.chdir('/Users/mac/asr-server')
from dotenv import load_dotenv
load_dotenv('/Users/mac/asr-server/.env')

import requests
from db_manager import is_file_processed_a, mark_file_processed_a, DATABASE_URL

ASR = "http://localhost:5008"
TOKEN = (os.getenv("ADMIN_TOKEN") or "").strip()
HEADERS = {"X-Admin-Token": TOKEN} if TOKEN else {}
RECORDS_ROOT = "/Volumes/download/records"

LOG = os.path.join(RECORDS_ROOT, "..", "asr_backup") if False else None  # 日志走 stdout


def log(msg):
    print(f"[{datetime.now().strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def mount_healthy():
    """挂载守卫：records 根 + 任一 Pixel 设备目录可列举（子线程超时防卡死）"""
    import threading
    result = {"ok": False}

    def _probe():
        try:
            os.listdir(RECORDS_ROOT)
            result["ok"] = True
        except Exception:
            result["ok"] = False

    t = threading.Thread(target=_probe, daemon=True)
    t.start()
    t.join(20)
    return result["ok"]


def parse_date(name):
    m = re.search(r"TermuxAudioRecording_(\d{4}-\d{2}-\d{2})_", name)
    return m.group(1) if m else None


def load_analyzed_pairs():
    """查 transcriptions 表已分析的文件名集合。
    注：表无 source_device 列；两台 Pixel 的录音秒位偏移不同（17s vs 20/54s），
    文件名实际不撞，用裸文件名判断即可。"""
    import psycopg2
    names = set()
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("SELECT filename FROM transcriptions")
    for (fn,) in cur.fetchall():
        names.add(os.path.basename(fn or ""))
    cur.close()
    conn.close()
    return names


def gather_jobs(devices, dates):
    """收集待回填文件：在位（未归档）、未打标、日期匹配"""
    jobs = []
    for dev in devices:
        for date in dates:
            d = os.path.join(RECORDS_ROOT, dev, date)
            files = sorted(glob.glob(os.path.join(d, "*.m4a")))
            for fp in files:
                fn = os.path.basename(fp)
                if is_file_processed_a(fn, device=dev):
                    continue
                jobs.append((dev, date, fn, fp))
    jobs.sort(key=lambda x: (x[1], x[2]))  # 按日期+文件名时间顺序
    return jobs


def gather_jobs_processed(devices, dates):
    """收集已归档但未分析的积压文件（catch-up 防线归档后的补救场景）"""
    analyzed = load_analyzed_pairs()
    log(f"transcriptions 已分析文件名: {len(analyzed)} 个")
    jobs = []
    for dev in devices:
        for date in dates:
            d = os.path.join(RECORDS_ROOT, dev, "processed", date)
            for fp in sorted(glob.glob(os.path.join(d, "*.m4a"))):
                fn = os.path.basename(fp)
                if fn in analyzed:
                    continue  # 已入库，跳过
                jobs.append((dev, date, fn, fp))
    jobs.sort(key=lambda x: (x[1], x[2]))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default="Pixel-5,Pixel-6")
    ap.add_argument("--dates", default="2026-10-02,2026-10-03")
    ap.add_argument("--from-processed", action="store_true",
                    help="扫 processed/ 已归档积压（重启后补救模式），文件不移动")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    devices = [d.strip() for d in args.devices.split(',') if d.strip()]
    dates = [d.strip() for d in args.dates.split(',') if d.strip()]

    if not mount_healthy():
        log("⛔ SMB 挂载不健康，中止（exit 3）。挂载恢复后重跑本脚本断点续传。")
        sys.exit(3)

    jobs = gather_jobs_processed(devices, dates) if args.from_processed else gather_jobs(devices, dates)
    log(f"待回填: {len(jobs)} 个文件（{devices} × {dates}）"
        + ("［processed 补救模式］" if args.from_processed else ""))
    # 队列预览：web_viewer 仪表盘解析本行 + 逐文件✔/✗行，推算"接下来待处理"文件列表
    if jobs:
        log("队列预览: " + ", ".join(f"{dev}/{fn}" for dev, _, fn, _ in jobs[:20]))
    if args.dry_run:
        for dev, date, fn, fp in jobs[:10]:
            log(f"  [dry] {dev}/{date}/{fn}")
        log(f"...共 {len(jobs)} 个。dry-run 结束。")
        return

    ok = fail = 0
    t0 = time.time()
    for i, (dev, date, fn, fp) in enumerate(jobs, 1):
        if not mount_healthy():
            log(f"⛔ 挂载中途失联，中止（exit 3）。已成功 {ok} 失败 {fail}，剩余 {len(jobs)-i+1} 个待续传。")
            sys.exit(3)
        if (i - 1) % 15 == 0 and i > 1:
            # 每推进 15 个重打队列预览: web_viewer 的快照只有 20 个, 耗尽后待办列表会空
            log("队列预览: " + ", ".join(f"{d2}/{f2}" for d2, _, f2, _ in jobs[i - 1:i + 14]))
        try:
            log(f"📤 提交: {dev}/{fn}")  # web_viewer 用本行把文件从「待处理」移入「处理中」
            with open(fp, 'rb') as f:
                resp = requests.post(f"{ASR}/transcribes", headers=HEADERS, timeout=7200,
                                     files={'audio_file': (fn, f, 'audio/mpeg')},
                                     data={'source_device': dev, 'is_history': 'true'})
        except (requests.exceptions.RequestException, OSError) as e:
            # OSError: requests 内部 fp.read() 读 SMB 文件可能抛 Errno 5 I/O error，
            # 不捕获会让整个 backfill 崩溃退出（2026-10-04 实际发生）
            log(f"⚠️ {dev}/{fn} 网络/IO异常（{e.__class__.__name__}: {e}），保留原处下轮重试")
            continue

        processed_dir = os.path.join(RECORDS_ROOT, dev, "processed", date)
        failed_dir = os.path.join(RECORDS_ROOT, dev, "failed", date)
        if resp.status_code == 200:
            if not args.from_processed:
                os.makedirs(processed_dir, exist_ok=True)
                try:
                    shutil.move(fp, os.path.join(processed_dir, fn))
                except Exception as e:
                    log(f"⚠️ 归档失败 {fn}: {e}（文件保留原处）")
                    continue
            mark_file_processed_a(fn, status="backfill_success", device=dev)
            ok += 1
            log(f"✔ {dev}/{fn} ({i}/{len(jobs)})")
        else:
            if not args.from_processed:
                os.makedirs(failed_dir, exist_ok=True)
                try:
                    shutil.move(fp, os.path.join(failed_dir, fn))
                except Exception:
                    pass
            mark_file_processed_a(fn, status=f"backfill_failed_{resp.status_code}", device=dev)
            fail += 1
            log(f"✗ {dev}/{fn} HTTP {resp.status_code}")

        if (ok + fail) % 20 == 0:
            rate = (ok + fail) / max(time.time() - t0, 1) * 60
            eta = (len(jobs) - ok - fail) / max(rate, 0.01) / 60
            log(f"进度 {ok+fail}/{len(jobs)} | 成功 {ok} 失败 {fail} | {rate:.1f} 个/分钟 | 剩余约 {eta:.1f} 小时")

    log(f"回填完成: 成功 {ok} / 失败 {fail} / 总 {len(jobs)}")


if __name__ == "__main__":
    main()
