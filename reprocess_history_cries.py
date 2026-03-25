import os
import re
import shutil
import requests
import time
import sys
import datetime

API_URL = "http://localhost:5008/transcribes"
SOURCE_DIR = "/Volumes/download/records/Sony-2"
PROCESSED_DIR = os.path.join(SOURCE_DIR, "processed")

# 合并阈值：同一事件内两个哭声文件最大时间间隔（秒）
CRY_MERGE_GAP_SEC = 600   # 10 分钟

# 事件上下文扩展：在首尾哭声文件前后各取 N 个相邻文件
CRY_CONTEXT_EACH_SIDE = 5  # 前5 + 后5 = 最多10个上下文文件


def parse_file_datetime(filename):
    """从文件名解析 datetime"""
    match = re.search(r'(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})', filename)
    if match:
        try:
            return datetime.datetime.strptime(
                f"{match.group(1)} {match.group(2).replace('-', ':')}",
                "%Y-%m-%d %H:%M:%S"
            )
        except Exception:
            pass
    return None


def merge_cry_events(cry_file_paths, all_sorted_files):
    """
    1. 将时间间隔 <= CRY_MERGE_GAP_SEC 的相邻哭声文件合并为一个事件
    2. 每个事件在 all_sorted_files 中往前/后各扩展 CRY_CONTEXT_EACH_SIDE 个文件
    返回: list of (event_id, [filepath, ...])
    """
    if not cry_file_paths:
        return []

    # 先按哭声文件合并为初始分组
    groups = []
    current = [cry_file_paths[0]]
    for i in range(1, len(cry_file_paths)):
        prev_dt = parse_file_datetime(os.path.basename(cry_file_paths[i - 1]))
        curr_dt = parse_file_datetime(os.path.basename(cry_file_paths[i]))
        if prev_dt and curr_dt and (curr_dt - prev_dt).total_seconds() <= CRY_MERGE_GAP_SEC:
            current.append(cry_file_paths[i])
        else:
            groups.append(current)
            current = [cry_file_paths[i]]
    groups.append(current)

    # 用 all_sorted_files 构建索引，扩展上下文并去重
    all_idx = {p: i for i, p in enumerate(all_sorted_files)}
    events = []
    for group in groups:
        # 找首尾在总列表中的位置
        indices = [all_idx[p] for p in group if p in all_idx]
        if not indices:
            events.append(group)
            continue
        lo = max(0, min(indices) - CRY_CONTEXT_EACH_SIDE)
        hi = min(len(all_sorted_files) - 1, max(indices) + CRY_CONTEXT_EACH_SIDE)
        event_files = all_sorted_files[lo:hi + 1]
        events.append(event_files)

    return events


def build_event_dir(base_dir, event_id, event_files):
    """
    为每个事件创建独立文件夹，以软链接方式放入音频文件。
    返回事件目录路径。
    """
    event_dir = os.path.join(base_dir, f"cry_event_{event_id:02d}")
    os.makedirs(event_dir, exist_ok=True)

    for fpath in event_files:
        dest = os.path.join(event_dir, os.path.basename(fpath))
        if os.path.exists(dest) or os.path.islink(dest):
            os.remove(dest)
        try:
            os.symlink(fpath, dest)
        except Exception:
            shutil.copy2(fpath, dest)  # 软链接失败则直接复制

    return event_dir


if __name__ == "__main__":
    target_dir = PROCESSED_DIR if os.path.exists(PROCESSED_DIR) else SOURCE_DIR

    print(f"[*] 准备启动历史回顾，目标目录: {target_dir}", flush=True)
    if not os.path.exists(target_dir):
        print(f"错误: 目录 {target_dir} 不存在。", flush=True)
        sys.exit(1)

    print(f"[*] 正在极速扫描文件树，请稍候...", flush=True)

    filter_date    = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None
    start_time_arg = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
    end_time_arg   = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None

    if filter_date or start_time_arg or end_time_arg:
        print(
            f"[*] 注意: 仅针对 日期包含 '{filter_date or ''}' "
            f"且 时间范围包含 '{start_time_arg or '00-00'} ~ {end_time_arg or '23-59'}' 的文件进行定向分析！",
            flush=True
        )

    def is_time_in_range(filename, start_t, end_t):
        if not start_t and not end_t:
            return True
        match = re.search(r'_(\d{2}-\d{2}-\d{2})\.', filename)
        if not match:
            return True
        hm = match.group(1)[:5]
        if start_t and hm < start_t:
            return False
        if end_t and hm > end_t:
            return False
        return True

    AUDIO_EXTS = ('.m4a', '.mp3', '.wav', '.aac', '.flac', '.ogg', '.acc')
    files_to_process = []
    start_scan = time.time()
    try:
        for entry in os.listdir(target_dir):
            if entry.startswith('.'):
                continue
            full_path = os.path.join(target_dir, entry)
            if os.path.isfile(full_path):
                if (not filter_date or filter_date in entry) and is_time_in_range(entry, start_time_arg, end_time_arg):
                    if full_path.lower().endswith(AUDIO_EXTS):
                        files_to_process.append(full_path)
            elif os.path.isdir(full_path):
                if filter_date and filter_date not in entry:
                    continue
                try:
                    for sf in os.listdir(full_path):
                        if sf.startswith('.'):
                            continue
                        if not is_time_in_range(sf, start_time_arg, end_time_arg):
                            continue
                        sf_path = os.path.join(full_path, sf)
                        if sf_path.lower().endswith(AUDIO_EXTS):
                            files_to_process.append(sf_path)
                except Exception:
                    pass
    except Exception as e:
        print(f"错误: 无法读取主目录: {e}", flush=True)

    files_to_process.sort()
    print(f"[*] 极速扫描完成！耗时 {time.time()-start_scan:.2f} 秒，找到 {len(files_to_process)} 个合法文件", flush=True)

    if not files_to_process:
        print(f"在 {target_dir} 中未找到任何支持的音频文件。", flush=True)
        sys.exit(0)

    print(f"找到 {len(files_to_process)} 个支持的音频文件，准备重新转录并分析 (按 Ctrl+C 中止)...", flush=True)

    # =====================================================================
    # 阶段一：全量转录，识别哭声文件（不触发 Gemini 分析）
    # =====================================================================
    print(f"\n{'='*60}", flush=True)
    print(f"📡 阶段一：逐文件转录，识别哭声片段（共 {len(files_to_process)} 个）", flush=True)
    print(f"{'='*60}", flush=True)

    cry_file_paths = []   # 有哭声的文件路径列表（有序）

    for filepath in files_to_process:
        print(f"\n[+] 正在发起云端分析请求: {filepath}", flush=True)
        try:
            with open(filepath, 'rb') as f:
                files_data = {'audio_file': (os.path.basename(filepath), f, 'audio/m4a')}
                data = {'speaker': 'Baby', 'skip_cry': 'true'}   # 只转录，不触发 Gemini 分析
                response = requests.post(API_URL, files=files_data, data=data, timeout=300)

            if response.status_code == 200:
                result = response.json()
                cry_segs = [
                    seg for seg in result.get('segments', [])
                    if seg.get('is_baby_cry') or seg.get('baby_cry_reason')
                ]
                if cry_segs:
                    cry_file_paths.append(filepath)
                    print(f"    🍼 检测到 {len(cry_segs)} 段哭声片段", flush=True)
                else:
                    print(f"    📉 未检出明确的高质量哭闹有效片段", flush=True)
                print(
                    f"    ✓ 成功 | 原始时长: {result.get('duration', 0):.1f}s "
                    f"| 耗时: {result.get('process_time', 0):.1f}s",
                    flush=True
                )
            else:
                print(f"    ❌ 失败 (Status {response.status_code}): {response.text}", flush=True)
        except Exception as e:
            print(f"    ❌ 遇到了错误: {e}", flush=True)
        time.sleep(1)

    if not cry_file_paths:
        print(f"\n✅ 历史音频分析完成！未发现任何哭闹事件。", flush=True)
        sys.exit(0)

    # =====================================================================
    # 阶段二：合并连续事件，构建事件文件夹，统一分析
    # =====================================================================
    events = merge_cry_events(cry_file_paths, files_to_process)

    # 事件文件夹统一放在 SOURCE_DIR/cry_events/ 下
    events_base_dir = os.path.join(SOURCE_DIR, "cry_events")
    os.makedirs(events_base_dir, exist_ok=True)

    print(f"\n{'='*60}", flush=True)
    print(
        f"🧠 阶段二：合并结果 → {len(cry_file_paths)} 个哭声文件 → {len(events)} 个独立事件",
        flush=True
    )
    print(f"   事件文件夹根目录: {events_base_dir}", flush=True)
    print(f"   合并阈值: {CRY_MERGE_GAP_SEC//60} 分钟，上下文各扩展 {CRY_CONTEXT_EACH_SIDE} 个文件", flush=True)
    print(f"{'='*60}", flush=True)

    for idx, event_files in enumerate(events, 1):
        event_dir = build_event_dir(events_base_dir, idx, event_files)
        rep_filepath = event_files[len(event_files) // 2]   # 取中间文件作代表
        rep_filename = os.path.basename(rep_filepath)

        print(f"\n🔔 事件 {idx}/{len(events)}: {len(event_files)} 个文件", flush=True)
        print(f"   目录: {event_dir}", flush=True)
        for f in event_files:
            marker = " ← 哭声" if f in cry_file_paths else ""
            print(f"   {'📂'} {os.path.basename(f)}{marker}", flush=True)
        print(f"   代表文件: {rep_filename}", flush=True)

        # 取代表文件中第一个哭声片段的时间范围
        first_seg = {'start': 0, 'end': 60000}
        try:
            with open(rep_filepath, 'rb') as f:
                res = requests.post(
                    API_URL,
                    files={'audio_file': (rep_filename, f, 'audio/m4a')},
                    data={'speaker': 'Baby', 'sync_llm': 'false'},
                    timeout=60
                )
            if res.status_code == 200:
                segs = [s for s in res.json().get('segments', []) if s.get('is_baby_cry')]
                if segs:
                    first_seg = segs[0]
        except Exception:
            pass

        try:
            response = requests.post(
                "http://localhost:5008/api/analyze_cry",
                json={
                    "filename": rep_filename,
                    "audio_path": rep_filepath,
                    "start_ms": first_seg.get('start', 0),
                    "end_ms": first_seg.get('end', 60000),
                    "audio_paths": event_files,   # ← 直接传入完整事件文件列表，绕过自动搜索
                },
                timeout=180
            )
            if response.status_code == 200:
                result = response.json()
                reason = result.get("reason") or "未知"
                advice = result.get("advice") or "无"
                print(f"    ✨ 原因: {reason[:80]}...", flush=True)
                print(f"    💡 建议: {advice[:50]}...", flush=True)
            else:
                print(f"    ⚠️  Gemini 分析接口返回 {response.status_code}，跳过", flush=True)
        except Exception as e:
            print(f"    ❌ Gemini 分析失败: {e}", flush=True)

        time.sleep(2)

    print(f"\n✅ 历史音频重新分析完成！事件文件夹已保存至: {events_base_dir}", flush=True)
    print("   请在上方切换到【宝宝分析】标签页查看自动刷新的记录。", flush=True)
