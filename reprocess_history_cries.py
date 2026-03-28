import os
import re
import shutil
import requests
import time
import sys
import datetime
import logging
from db_manager import init_pool, is_file_processed_a, mark_file_processed_a, get_connection, return_connection

API_URL = "http://localhost:5008/transcribes"
SOURCE_DIR = "/Volumes/download/records/Sony-2"
PROCESSED_DIR = os.path.join(SOURCE_DIR, "processed")

# 配置详细日志记录器（输出到 asr-a.log）
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "asr-a.log")

# 创建专门的日志记录器
reprocess_logger = logging.getLogger('reprocess_history')
reprocess_logger.setLevel(logging.INFO)
reprocess_logger.handlers = []

# 文件处理器
file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
reprocess_logger.addHandler(file_handler)

# 控制台处理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(message)s'))
reprocess_logger.addHandler(console_handler)

def log_detail(message, level='info'):
    """同时输出到控制台和日志文件"""
    if level == 'info':
        reprocess_logger.info(message)
    elif level == 'warning':
        reprocess_logger.warning(message)
    elif level == 'error':
        reprocess_logger.error(message)
    elif level == 'debug':
        reprocess_logger.debug(message)

# 合并阈值：同一事件内两个哭声文件最大时间间隔（秒）
CRY_MERGE_GAP_SEC = 600   # 10 分钟

# 事件上下文扩展：在首尾哭声文件前后各取 N 个相邻文件
CRY_CONTEXT_EACH_SIDE = 5  # 前 5 + 后 5 = 最多 10 个上下文文件


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
    返回：list of (event_id, [filepath, ...])
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
            shutil.copy2(fpath, dest)
            log_detail(f"    [复制] {os.path.basename(fpath)}", 'info')
        except Exception as e:
            log_detail(f"    [!] 复制失败：{e}", 'error')

    return event_dir


if __name__ == "__main__":
    target_dir = PROCESSED_DIR if os.path.exists(PROCESSED_DIR) else SOURCE_DIR

    log_detail(f"[*] 准备启动历史回顾，目标目录：{target_dir}", 'info')
    if not os.path.exists(target_dir):
        log_detail(f"错误：目录 {target_dir} 不存在。", 'error')
        sys.exit(1)

    log_detail(f"[*] 正在极速扫描文件树，请稍候...", 'info')

    filter_date    = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None
    start_time_arg = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
    end_time_arg   = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None
    force_replace  = "--replace" in sys.argv

    log_detail(f"\n{'='*80}", 'info')
    log_detail(f"🚀 历史回顾分析任务启动 | 时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 'info')
    log_detail(f"{'='*80}", 'info')
    log_detail(f"📂 目标目录：{target_dir}", 'info')
    log_detail(f"🔧 参数配置:", 'info')
    log_detail(f"   - 日期过滤：{filter_date or '无'}", 'info')
    log_detail(f"   - 时间范围：{start_time_arg or '00-00'} ~ {end_time_arg or '23-59'}", 'info')
    log_detail(f"   - 强制替换：{force_replace}", 'info')
    log_detail(f"   - 合并阈值：{CRY_MERGE_GAP_SEC}秒 ({CRY_MERGE_GAP_SEC//60}分钟)", 'info')
    log_detail(f"   - 上下文扩展：前后各 {CRY_CONTEXT_EACH_SIDE} 个文件", 'info')
    log_detail(f"{'='*80}\n", 'info')

    if filter_date or start_time_arg or end_time_arg:
        log_detail(
            f"⚠️  注意：仅针对 日期包含 '{filter_date or ''}' "
            f"且 时间范围包含 '{start_time_arg or '00-00'} ~ {end_time_arg or '23-59'}' 的文件进行定向分析！",
            'warning'
        )
        if force_replace:
            log_detail("⚠️  提示：已启用 --replace 模式，将重新分析所有匹配的文件并覆盖旧结果。", 'warning')

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
        # === 极速优化扫描引擎 (针对 NAS 挂载优化) ===
        log_detail(f"[*] 正在启动受限文件树探测...", 'info')
        
        for root, dirs, files in os.walk(target_dir):
            # 性能优化 1：过滤隐藏目录
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            # 性能优化 2：日期制导剪枝
            # 如果指定了日期，且当前在搜索根目录，我们只允许进入与日期匹配的子目录
            if filter_date and root == target_dir:
                if filter_date in dirs:
                    # 找到了精准日期目录，剪掉其他所有目录
                    log_detail(f"[*] 发现精准日期目录 '{filter_date}'，已自动剪枝其他分支以提升速度。", 'info')
                    dirs[:] = [filter_date]
                else:
                    # 没找到对应日期的目录，但根目录下可能还有文件，所以不进入任何子目录
                    log_detail(f"[*] 未发现日期目录 '{filter_date}'，将仅扫描根目录文件。", 'info')
                    dirs[:] = []
            
            # 进度实时反馈
            log_detail(f"    - 正在扫描容器：{os.path.basename(root) or 'root'} (已发现 {len(files_to_process)} 个待处理项)", 'info')
            
            for file in files:
                if file.startswith('.'): continue
                if not file.lower().endswith(AUDIO_EXTS): continue
                
                # 严格校验：
                # 1. 日期校验
                if filter_date and filter_date not in file: continue
                # 2. 时间段校验
                if not is_time_in_range(file, start_time_arg, end_time_arg): continue
                
                files_to_process.append(os.path.join(root, file))
                
        log_detail(f"\n[*] 扫描逻辑执行完毕。", 'info')
                
    except Exception as e:
        log_detail(f"\n错误：扫描目录时遇到问题：{e}", 'error')

    # 去重并排序，防止重复扫描
    files_to_process = sorted(list(set(files_to_process)))
    log_detail(f"[*] 极速扫描完成！耗时 {time.time()-start_scan:.2f} 秒，找到 {len(files_to_process)} 个合法文件", 'info')

    if not files_to_process:
        log_detail(f"在 {target_dir} 中未找到任何支持的音频文件。", 'warning')
        sys.exit(0)

    # 初始化数据库连接池
    init_pool()

    log_detail(f"\n找到 {len(files_to_process)} 个支持的音频文件，准备重新转录并分析 (按 Ctrl+C 中止)...", 'info')

    # =====================================================================
    # 阶段一：全量转录，识别哭声文件（不触发 Gemini 分析）
    # =====================================================================
    log_detail(f"\n{'='*60}", 'info')
    log_detail(f"📡 阶段一：逐文件转录，识别哭声片段（共 {len(files_to_process)} 个）", 'info')
    log_detail(f"{'='*60}", 'info')

    cry_file_paths = []   # 有哭声的文件路径列表（有序）
    success_count = 0
    skip_count = 0
    error_count = 0

    for idx, filepath in enumerate(files_to_process, 1):
        filename = os.path.basename(filepath)
        
        # 【断电续传/跳过逻辑】
        # 如果指定了具体的日期或时间范围，视为定向重分析，不再跳过已处理文件
        is_targeted = bool(filter_date or start_time_arg or end_time_arg)
        if not force_replace and not is_targeted:
            if is_file_processed_a(filename):
                log_detail(f"[{idx}/{len(files_to_process)}] [skip] 文件已识别过，跳过：{filename}", 'info')
                skip_count += 1
                continue

        log_detail(f"\n[{idx}/{len(files_to_process)}] 正在发起云端分析请求：{filepath}", 'info')
        try:
            with open(filepath, 'rb') as f:
                files_data = {'audio_file': (os.path.basename(filepath), f, 'audio/m4a')}
                data = {
                    'speaker': 'Baby', 
                    'skip_cry': 'true',
                    'is_history': 'true'  # 标记为历史任务，跳过 503 暂停响应
                }
                response = requests.post(API_URL, files=files_data, data=data, timeout=300)

            if response.status_code == 200:
                result = response.json()
                cry_segs = [
                    seg for seg in result.get('segments', [])
                    if seg.get('is_baby_cry') or seg.get('baby_cry_reason')
                ]
                if cry_segs:
                    cry_file_paths.append(filepath)
                    log_detail(f"    🍼 检测到 {len(cry_segs)} 段哭声片段", 'info')
                    for i, seg in enumerate(cry_segs, 1):
                        start = seg.get('start', 0) / 1000.0
                        end = seg.get('end', 0) / 1000.0
                        log_detail(f"       片段{i}: {start:.1f}s - {end:.1f}s", 'info')
                else:
                    log_detail(f"    📉 未检出明确的高质量哭闹有效片段", 'info')
                log_detail(
                    f"    ✓ 成功 | 原始时长：{result.get('duration', 0):.1f}s "
                    f"| 耗时：{result.get('process_time', 0):.1f}s "
                    f"| RTF: {result.get('meta', {}).get('rtf', 0):.3f}",
                    'info'
                )
                success_count += 1
                # 标记为已处理
                mark_file_processed_a(filename, status="cry" if cry_segs else "no_cry")
            else:
                log_detail(f"    ❌ 失败 (Status {response.status_code}): {response.text}", 'error')
                error_count += 1
                mark_file_processed_a(filename, status="error")
        except Exception as e:
            log_detail(f"    ❌ 遇到了错误：{e}", 'error')
            error_count += 1
        time.sleep(1)

    log_detail(f"\n{'='*60}", 'info')
    log_detail(f"📊 阶段一统计:", 'info')
    log_detail(f"   - 总文件数：{len(files_to_process)}", 'info')
    log_detail(f"   - 成功处理：{success_count}", 'info')
    log_detail(f"   - 跳过文件：{skip_count}", 'info')
    log_detail(f"   - 处理错误：{error_count}", 'info')
    log_detail(f"   - 检出哭声文件：{len(cry_file_paths)}", 'info')
    log_detail(f"{'='*60}\n", 'info')

    if not cry_file_paths:
        log_detail(f"\n✅ 历史音频分析完成！未发现任何哭闹事件。", 'info')
        sys.exit(0)

    # =====================================================================
    # 阶段二：合并连续事件，构建事件文件夹，统一分析
    # =====================================================================
    events = merge_cry_events(cry_file_paths, files_to_process)

    # 事件文件夹统一放在 SOURCE_DIR/cry_events/ 下
    events_base_dir = os.path.join(SOURCE_DIR, "cry_events")
    os.makedirs(events_base_dir, exist_ok=True)

    log_detail(f"\n{'='*60}", 'info')
    log_detail(f"🧠 阶段二：合并结果 → {len(cry_file_paths)} 个哭声文件 → {len(events)} 个独立事件", 'info')
    log_detail(f"   事件文件夹根目录：{events_base_dir}", 'info')
    log_detail(f"   合并阈值：{CRY_MERGE_GAP_SEC//60} 分钟，上下文各扩展 {CRY_CONTEXT_EACH_SIDE} 个文件", 'info')
    log_detail(f"{'='*60}", 'info')

    for idx, event_files in enumerate(events, 1):
        event_dir = build_event_dir(events_base_dir, idx, event_files)
        rep_filepath = event_files[len(event_files) // 2]   # 取中间文件作代表
        rep_filename = os.path.basename(rep_filepath)

        log_detail(f"\n🔔 事件 {idx}/{len(events)}: {len(event_files)} 个文件", 'info')
        log_detail(f"   目录：{event_dir}", 'info')
        
        cry_files_in_event = [f for f in event_files if f in cry_file_paths]
        log_detail(f"   哭声文件 ({len(cry_files_in_event)}个):", 'info')
        for f in cry_files_in_event:
            log_detail(f"      📂 {os.path.basename(f)}", 'info')
        
        log_detail(f"   上下文文件 ({len(event_files) - len(cry_files_in_event)}个):", 'info')
        for f in event_files:
            if f not in cry_file_paths:
                log_detail(f"      📂 {os.path.basename(f)}", 'info')
        
        log_detail(f"   代表文件：{rep_filename}", 'info')

        # 取代表文件中第一个哭声片段的时间范围
        first_seg = {'start': 0, 'end': 60000}
        try:
            with open(rep_filepath, 'rb') as f:
                res = requests.post(
                    API_URL,
                    files={'audio_file': (rep_filename, f, 'audio/m4a')},
                    data={
                        'speaker': 'Baby', 
                        'sync_llm': 'false',
                        'is_history': 'true'
                    },
                    timeout=60
                )
            if res.status_code == 200:
                segs = [s for s in res.json().get('segments', []) if s.get('is_baby_cry')]
                if segs:
                    first_seg = segs[0]
                    log_detail(f"   📍 选取哭声片段：{first_seg['start']/1000.0:.1f}s - {first_seg['end']/1000.0:.1f}s", 'info')
        except Exception as e:
            log_detail(f"   ⚠️ 获取哭声片段失败：{e}", 'warning')

        try:
            log_detail(f"   🤖 正在调用 Gemini 深度分析...", 'info')
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
                category = result.get("category") or "未分类"
                log_detail(f"    ✨ 分析完成!", 'info')
                log_detail(f"       分类：{category}", 'info')
                log_detail(f"       原因：{reason[:100]}{'...' if len(reason) > 100 else ''}", 'info')
                log_detail(f"       建议：{advice[:80]}{'...' if len(advice) > 80 else ''}", 'info')
            else:
                log_detail(f"    ⚠️  Gemini 分析接口返回 {response.status_code}，跳过", 'warning')
        except Exception as e:
            log_detail(f"    ❌ Gemini 分析失败：{e}", 'error')

        time.sleep(2)

    log_detail(f"\n{'='*60}", 'info')
    log_detail(f"✅ 历史音频重新分析完成！", 'info')
    log_detail(f"   事件文件夹已保存至：{events_base_dir}", 'info')
    log_detail(f"   共处理 {len(events)} 个独立事件", 'info')
    log_detail(f"{'='*60}", 'info')
    log_detail(f"\n请在上方切换到【宝宝分析】标签页查看自动刷新的记录。", 'info')
