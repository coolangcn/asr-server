import os
import re
import shutil
import requests
import time
import sys
import datetime
import logging
from db_manager import init_pool, is_file_processed_a, mark_file_processed_a, get_connection, return_connection, get_date_processing_stats, get_date_file_counts, get_file_cache, get_file_count_from_cache

# 导入邮件模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from email_utils import send_email_async
    EMAIL_ENABLED = True
except ImportError:
    EMAIL_ENABLED = False
    print("[WARN] email_utils 未找到，邮件功能已禁用")

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
    for group_idx, group in enumerate(groups, 1):
        log_detail(f"[事件 {group_idx}] 哭声文件组: {len(group)} 个文件", 'info')
        for f in group:
            log_detail(f"  - {os.path.basename(f)}", 'info')
        
        # 找首尾在总列表中的位置
        indices = [all_idx[p] for p in group if p in all_idx]
        if not indices:
            log_detail(f"  ⚠️ 未在总列表中找到索引，跳过扩展", 'warning')
            events.append(group)
            continue
        
        log_detail(f"  在总列表中的索引: {min(indices)} ~ {max(indices)}", 'info')
        lo = max(0, min(indices) - CRY_CONTEXT_EACH_SIDE)
        hi = min(len(all_sorted_files) - 1, max(indices) + CRY_CONTEXT_EACH_SIDE)
        log_detail(f"  扩展后范围: {lo} ~ {hi}", 'info')
        
        event_files = all_sorted_files[lo:hi + 1]
        log_detail(f"  最终事件文件数: {len(event_files)} 个", 'info')
        for f in event_files:
            log_detail(f"  - {os.path.basename(f)}", 'info')
        
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
    
    # 调试：确认参数
    if not filter_date:
        log_detail(f"⚠️ 警告：filter_date 为空，将进行全量扫描！", 'info')
    
    # 强调当前处理日期
    if filter_date:
        log_detail(f"📅 【当前处理日期】{filter_date}", 'info')
        log_detail(f"   └─ 该日期的所有音频文件将被扫描和分析", 'info')
        log_detail(f"   └─ 旧记录将在分析前自动清理", 'info')
    else:
        log_detail(f"📅 【当前处理模式】全量扫描模式", 'info')
        log_detail(f"   └─ 将扫描目标目录下的所有日期文件夹", 'info')
        log_detail(f"   └─ 按日期顺序逐一处理", 'info')

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
    all_files = []  # 所有文件（用于上下文扩展）
    target_files = []  # 目标文件（用于识别哭声）
    start_scan = time.time()

    # 初始化数据库连接池（必须在调用数据库函数之前）
    init_pool()

    # === 智能续传：先从数据库获取文件列表 ===
    date_stats = {}  # {date_str: processed_count}
    if not force_replace:
        date_stats = get_date_processing_stats()
        log_detail(f"[*] 智能续传：已处理统计={dict(list(date_stats.items())[:5])}...", 'info')

    # 尝试从 DB 缓存获取文件列表
    log_detail(f"[*] 正在从数据库缓存获取文件列表...", 'info')
    cached_count = get_file_count_from_cache()
    log_detail(f"[*] DB 缓存中有 {cached_count} 个文件", 'info')

    if cached_count > 0:
        # DB 缓存命中！直接从 DB 获取文件列表
        log_detail(f"[*] ✅ DB 缓存命中，使用缓存文件列表（极速模式）", 'info')
        all_files = get_file_cache()  # 返回 [{filepath, filename, file_size}, ...]
        all_files = [f['filepath'] for f in all_files]  # 转成路径列表

        # 按日期过滤
        if filter_date:
            all_files = [f for f in all_files if filter_date in f]

        # 按时间过滤
        target_files = [f for f in all_files if is_time_in_range(os.path.basename(f), start_time_arg, end_time_arg)]

        log_detail(f"[*] 从缓存获取文件列表完成", 'info')
    else:
        # DB 缓存未命中，回退到磁盘扫描
        log_detail(f"[*] ⚠️ DB 缓存为空，回退到磁盘扫描（首次可能较慢）", 'info')

        try:
            log_detail(f"[*] 正在启动文件树扫描...", 'info')

            for root, dirs, files in os.walk(target_dir):
                dirs[:] = [d for d in dirs if not d.startswith('.')]

                current_container = os.path.basename(root) or 'root'
                if current_container != 'root' and re.match(r'\d{4}-\d{2}-\d{2}', current_container):
                    log_detail(f"    📅 【扫描日期】{current_container}", 'info')

                for file in files:
                    if file.startswith('.'): continue
                    if not file.lower().endswith(AUDIO_EXTS): continue
                    if filter_date and filter_date not in file: continue

                    file_path = os.path.join(root, file)
                    all_files.append(file_path)

                    if is_time_in_range(file, start_time_arg, end_time_arg):
                        target_files.append(file_path)

            log_detail(f"[*] 磁盘扫描完成，共 {len(all_files)} 个文件", 'info')

        except Exception as e:
            log_detail(f"\n错误：扫描目录时遇到问题：{e}", 'error')

    # 去重并排序，防止重复扫描
    all_files = sorted(list(set(all_files)))
    target_files = sorted(list(set(target_files)))
    
    log_detail(f"[*] 极速扫描完成！耗时 {time.time()-start_scan:.2f} 秒", 'info')
    log_detail(f"   - 扫描范围: {'指定日期 ' + filter_date if filter_date else '全日期'}", 'info')
    log_detail(f"   - 所有文件数: {len(all_files)} 个", 'info')
    log_detail(f"   - 目标文件数: {len(target_files)} 个", 'info')
    log_detail(f"[*] 文件列表（前20个）:", 'info')
    for i, f in enumerate(all_files[:20], 1):
        log_detail(f"    {i}. {os.path.basename(f)}", 'info')

    # 用于分析的文件列表（优先用目标文件，没有就用所有文件）
    files_to_process = target_files if target_files else all_files
    
    if not files_to_process:
        log_detail(f"在 {target_dir} 中未找到任何支持的音频文件。", 'warning')
        sys.exit(0)

    # === 智能续传优化：跳过已完成的日期 ===
    if not force_replace and date_stats:
        # 按日期分组
        date_file_counts = {}
        for f in files_to_process:
            m = re.search(r'/(\d{4}-\d{2}-\d{2})/', f)
            if m:
                d = m.group(1)
                date_file_counts[d] = date_file_counts.get(d, 0) + 1

        completed_dates = []
        remaining_files = []

        for f in files_to_process:
            m = re.search(r'/(\d{4}-\d{2}-\d{2})/', f)
            if not m:
                remaining_files.append(f)
                continue
            d = m.group(1)
            processed = date_stats.get(d, 0)
            total = date_file_counts.get(d, 0)

            if processed >= total and total > 0:
                if d not in completed_dates:
                    completed_dates.append(d)
            else:
                remaining_files.append(f)

        if completed_dates:
            log_detail(f"[*] 智能续传：跳过 {len(completed_dates)} 个已完成日期: {', '.join(completed_dates)}", 'info')

        files_to_process = remaining_files
            log_detail(f"[*] 智能续传：实际需要处理 {len(files_to_process)} 个文件", 'info')
    # ===

    # 如果指定了日期，先删除该日期的旧记录（在识别哭声之后、分析之前删除）
    if filter_date:
        log_detail(f"📅 已选择日期: {filter_date}，将在分析开始后删除旧记录", 'info')

    log_detail(f"\n找到 {len(files_to_process)} 个文件进行哭声识别（扫描范围内共 {len(all_files)} 个音频文件，哭声事件将从中选取上下文）...", 'info')

    # =====================================================================
    # 阶段一：全量转录，识别哭声文件（不触发 Gemini 分析）
    # =====================================================================
    log_detail(f"\n{'='*60}", 'info')
    log_detail(f"📡 阶段一：逐文件转录，识别哭声片段（共 {len(files_to_process)} 个）", 'info')
    log_detail(f"{'='*60}", 'info')
    
    # 显示当前处理的日期范围 - 更详细的日期信息
    if files_to_process:
        first_file = os.path.basename(files_to_process[0])
        last_file = os.path.basename(files_to_process[-1])
        first_date = parse_file_datetime(first_file)
        last_date = parse_file_datetime(last_file)
        
        log_detail(f"", 'info')
        log_detail(f"📅 【当前处理日期详细信息】", 'info')
        
        if first_date and last_date:
            day_span = (last_date - first_date).days + 1
            log_detail(f"   📅 日期范围: {first_date.strftime('%Y-%m-%d %H:%M')} → {last_date.strftime('%Y-%m-%d %H:%M')}", 'info')
            log_detail(f"   📅 时间跨度: {day_span} 天", 'info')
            log_detail(f"   ⏰ 起始时间: {first_date.strftime('%H:%M:%S')}", 'info')
            log_detail(f"   ⏰ 结束时间: {last_date.strftime('%H:%M:%S')}", 'info')
        
        log_detail(f"   📅 首个文件: {first_file}", 'info')
        log_detail(f"   📅 末个文件: {last_file}", 'info')
        log_detail(f"   📈 文件总数: {len(files_to_process)} 个", 'info')
        
        # 如果是按日期过滤，显示当前日期
        if filter_date:
            log_detail(f"   🎯 指定日期: {filter_date}", 'info')
            log_detail(f"   📝 处理状态: 准备开始哭声识别...", 'info')
    

    cry_file_paths = []   # 有哭声的文件路径列表（有序）
    success_count = 0
    skip_count = 0
    error_count = 0

    current_processing_date = None
    date_file_count = {}  # 统计每个日期的文件数
    date_processed_count = {}  # 统计每个日期已处理的文件数

    # 先统计每个日期的文件数量
    for filepath in files_to_process:
        filename = os.path.basename(filepath)
        file_date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
        if file_date_match:
            file_date = file_date_match.group(1)
            date_file_count[file_date] = date_file_count.get(file_date, 0) + 1

    for idx, filepath in enumerate(files_to_process, 1):
        filename = os.path.basename(filepath)

        # 提取当前文件的日期并显示
        file_date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
        if file_date_match:
            file_date = file_date_match.group(1)
            date_processed_count[file_date] = date_processed_count.get(file_date, 0) + 1
            if file_date != current_processing_date:
                current_processing_date = file_date
                log_detail(f"", 'info')
                log_detail(f"    📅 【正在处理日期】{file_date} (该日期共 {date_file_count.get(file_date, 0)} 个文件)", 'info')
            # 显示该日期内的进度
            date_progress = date_processed_count[file_date]
            date_total = date_file_count.get(file_date, 0)
            if idx % 10 == 0 or date_progress == date_total:  # 每10个文件或完成时显示
                log_detail(f"       该日期进度: {date_progress}/{date_total} ({date_progress*100//date_total}%)", 'info')

        # 【断电续传/跳过逻辑】
        # 如果指定了具体的日期或时间范围，视为定向重分析，不再跳过已处理文件
        is_targeted = bool(filter_date or start_time_arg or end_time_arg)
        if not force_replace and not is_targeted:
            if is_file_processed_a(filename):
                log_detail(f"[{idx}/{len(files_to_process)}] [skip] 文件已识别过，跳过：{filename}", 'info')
                skip_count += 1
                continue
        elif not force_replace and is_targeted:
            # 定向检索时：跳过该日期内已处理的文件（续传）
            if is_file_processed_a(filename):
                log_detail(f"[{idx}/{len(files_to_process)}] [skip] 续传跳过：{filename}", 'info')
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
    log_detail(f"📊 阶段一统计 (日期: {filter_date or current_processing_date or '全部'}):", 'info')
    log_detail(f"{'─'*60}", 'info')
    log_detail(f"   📁 总文件数：     {len(files_to_process):>6}", 'info')
    log_detail(f"   ✅ 成功处理：     {success_count:>6}", 'info')
    log_detail(f"   ⏭️ 跳过文件：     {skip_count:>6}", 'info')
    log_detail(f"   ❌ 处理错误：     {error_count:>6}", 'info')
    log_detail(f"   🍼 检出哭声文件： {len(cry_file_paths):>6}", 'info')
    log_detail(f"{'─'*60}", 'info')
    if filter_date:
        log_detail(f"   ✅ 日期 {filter_date} 阶段一完成 (哭声识别)", 'info')
    log_detail(f"{'='*60}\n", 'info')

    if not cry_file_paths:
        log_detail(f"\n✅ 历史音频分析完成！未发现任何哭闹事件。", 'info')
        sys.exit(0)

    # =====================================================================
    # 阶段二：合并连续事件，构建事件文件夹，统一分析
    # =====================================================================
    # 在分析前删除该日期的旧记录（避免新旧重复）
    if filter_date:
        log_detail(f"🗑️  正在删除日期 {filter_date} 的旧分析记录...", 'info')
        try:
            from db_manager import delete_cry_events_by_date
            deleted_count = delete_cry_events_by_date(filter_date)
            log_detail(f"✅ 已删除 {deleted_count} 条旧记录", 'info')
        except Exception as e:
            log_detail(f"⚠️  删除旧记录失败: {e}", 'warning')
    
    # 使用当天所有文件来扩展上下文（确保能扩展到10个文件）
    events = merge_cry_events(cry_file_paths, all_files)

    # 事件文件夹统一放在 SOURCE_DIR/cry_events/ 下
    events_base_dir = os.path.join(SOURCE_DIR, "cry_events")
    os.makedirs(events_base_dir, exist_ok=True)

    log_detail(f"\n{'='*60}", 'info')
    log_detail(f"🧠 阶段二：哭声事件深度分析", 'info')
    log_detail(f"{'─'*60}", 'info')
    log_detail(f"   📅 处理日期: {filter_date or '全部日期'}", 'info')
    log_detail(f"   🍼 哭声文件: {len(cry_file_paths)} 个", 'info')
    log_detail(f"   🔔 独立事件: {len(events)} 个", 'info')
    log_detail(f"   📁 事件目录: {events_base_dir}", 'info')
    log_detail(f"   ⏱️ 合并阈值: {CRY_MERGE_GAP_SEC//60} 分钟", 'info')
    log_detail(f"   📎 上下文扩展: 前后各 {CRY_CONTEXT_EACH_SIDE} 个文件", 'info')
    log_detail(f"{'─'*60}", 'info')
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
            event_result = None
            if response.status_code == 200:
                result = response.json()
                reason = result.get("reason") or "未知"
                advice = result.get("advice") or "无"
                category = result.get("category") or "未分类"
                event_result = result
                log_detail(f"    ✨ 分析完成!", 'info')
                log_detail(f"       分类：{category}", 'info')
                log_detail(f"       原因：{reason[:100]}{'...' if len(reason) > 100 else ''}", 'info')
                log_detail(f"       建议：{advice[:80]}{'...' if len(advice) > 80 else ''}", 'info')

                # 发送邮件通知（历史模式也发送）
                if EMAIL_ENABLED:
                    try:
                        # 获取哭声片段时间
                        cry_start = first_seg.get('start', 0) / 1000.0
                        cry_end = first_seg.get('end', 60000) / 1000.0
                        cry_time = parse_file_datetime(rep_filename)
                        cry_time_str = cry_time.strftime('%Y-%m-%d %H:%M:%S') if cry_time else '未知'

                        # 构建详细邮件内容
                        subject = f"📋 历史分析报告 | {filter_date or '全量'} | 检测到 {len(events)} 个哭声事件"
                        content = f"""
═══════════════════════════════════════════════════════════════
                     📋 宝宝哭声历史分析报告
═══════════════════════════════════════════════════════════════

📅 处理日期: {filter_date or '全部日期'}
⏰ 报告生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📂 事件目录: {events_base_dir}

───────────────────────────────────────────────────────────────
📊 分析统计
───────────────────────────────────────────────────────────────
🔔 独立哭声事件: {len(events)} 个
📁 关联文件总数: {len(cry_file_paths)} 个
🤖 合并阈值: {CRY_MERGE_GAP_SEC // 60} 分钟
📎 上下文扩展: 前后各 {CRY_CONTEXT_EACH_SIDE} 个文件

───────────────────────────────────────────────────────────────
🔔 事件 {idx}/{len(events)} 详情
───────────────────────────────────────────────────────────────
📁 事件目录: {event_dir}
🎵 代表文件: {rep_filename}
⏰ 哭声时间: {cry_time_str}
⏱️ 哭声片段: {cry_start:.1f}s - {cry_end:.1f}s
📊 事件文件数: {len(event_files)} 个

🏷️ 分析结果:
   • 分类: {category}
   • 原因: {reason}
   • 建议: {advice}

📂 事件文件列表:
"""
                        for i, f in enumerate(event_files, 1):
                            ftime = parse_file_datetime(os.path.basename(f))
                            ftime_str = ftime.strftime('%H:%M:%S') if ftime else ''
                            is_cry = '🔴' if f in cry_file_paths else '📎'
                            content += f"   {is_cry} {i:2d}. {os.path.basename(f)} ({ftime_str})\n"

                        content += f"""
═══════════════════════════════════════════════════════════════
"""

                        send_email_async(subject, content)
                        log_detail(f"    📧 邮件已发送", 'info')
                    except Exception as email_err:
                        log_detail(f"    ⚠️ 邮件发送失败: {email_err}", 'warning')
            else:
                log_detail(f"    ⚠️  Gemini 分析接口返回 {response.status_code}，跳过", 'warning')
        except Exception as e:
            log_detail(f"    ❌ Gemini 分析失败：{e}", 'error')

        time.sleep(2)

    log_detail(f"\n{'='*60}", 'info')
    if filter_date:
        log_detail(f"✅ 日期 {filter_date} 全部处理完成！", 'info')
    else:
        log_detail(f"✅ 历史音频重新分析完成！", 'info')
    log_detail(f"{'─'*60}", 'info')
    log_detail(f"   📅 处理日期: {filter_date or '全部日期'}", 'info')
    log_detail(f"   📁 事件文件夹: {events_base_dir}", 'info')
    log_detail(f"   🔔 独立事件数: {len(events)} 个", 'info')
    log_detail(f"{'─'*60}", 'info')
    log_detail(f"{'='*60}", 'info')
    log_detail(f"\n请在上方切换到【宝宝分析】标签页查看自动刷新的记录。", 'info')

    # 发送任务完成汇总邮件
    if EMAIL_ENABLED and events:
        try:
            subject = f"✅ 历史分析完成 | {filter_date or '全量'} | 共 {len(events)} 个事件"
            content = f"""
═══════════════════════════════════════════════════════════════
                   📋 历史分析任务完成汇总
═══════════════════════════════════════════════════════════════

📅 处理日期: {filter_date or '全部日期'}
⏰ 完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📂 事件目录: {events_base_dir}

───────────────────────────────────────────────────────────────
📊 任务统计
───────────────────────────────────────────────────────────────
📁 扫描文件总数: {len(all_files)} 个
🎯 目标文件数: {len(target_files)} 个
🔔 检出哭声文件: {len(cry_file_paths)} 个
🔔 独立哭声事件: {len(events)} 个
✅ 成功处理: {success_count} 个
⏭️  跳过文件: {skip_count} 个
❌ 处理错误: {error_count} 个

───────────────────────────────────────────────────────────────
📋 各事件概览
───────────────────────────────────────────────────────────────
"""
            for i, event_files_item in enumerate(events, 1):
                cry_in_event = [f for f in event_files_item if f in cry_file_paths]
                rep_file = event_files_item[len(event_files_item) // 2]
                rep_time = parse_file_datetime(os.path.basename(rep_file))
                rep_time_str = rep_time.strftime('%H:%M:%S') if rep_time else ''
                content += f"""
🔔 事件 {i}: {len(event_files_item)} 个文件 (哭声 {len(cry_in_event)} 个)
   代表: {os.path.basename(rep_file)} ({rep_time_str})
"""
            content += f"""
═══════════════════════════════════════════════════════════════
请在 BabyCry 分析看板查看详细分析结果。
═══════════════════════════════════════════════════════════════
"""
            send_email_async(subject, content)
            log_detail(f"📧 任务完成汇总邮件已发送", 'info')
        except Exception as email_err:
            log_detail(f"⚠️ 汇总邮件发送失败: {email_err}", 'warning')
