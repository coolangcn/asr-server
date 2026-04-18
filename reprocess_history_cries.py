import os
import re
import json
import shutil
import requests
import time
import sys
import datetime
import logging
import functools
from logging.handlers import TimedRotatingFileHandler
from db_manager import init_pool, is_file_processed_a, mark_file_processed_a, get_connection, return_connection, get_date_processing_stats, get_processed_files_for_date, get_file_cache_from_redis, get_file_count_from_redis, refresh_file_cache, get_cry_files_for_date, get_all_cry_dates, get_unanalyzed_cry_dates, get_incomplete_cry_events, delete_incomplete_cry_events, check_cache_freshness, get_uncovered_cry_count

# 进度状态文件路径（供 API 读取）
PROGRESS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log", "a_track_progress.json")

# 导入邮件模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from email_utils import send_email_async
    EMAIL_ENABLED = True
except ImportError:
    EMAIL_ENABLED = False
    print("[WARN] email_utils 未找到，邮件功能已禁用")

API_URL = "http://localhost:5008/transcribes"
QUICK_DETECT_URL = "http://localhost:5008/api/quick_cry_detect"  # 快速哭声检测接口（无ASR）
SOURCE_DIR = "/Volumes/download/records/Sony-2"
PROCESSED_DIR = os.path.join(SOURCE_DIR, "processed")

# 配置详细日志记录器（只输出到 asr-a.log）
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
os.makedirs(log_dir, exist_ok=True)

# 创建专门的日志记录器
reprocess_logger = logging.getLogger('reprocess_history')
reprocess_logger.setLevel(logging.INFO)
reprocess_logger.handlers = []

# 输出到 asr-a.log（与主服务日志合并，使用相同的轮转配置）
file_handler_a = TimedRotatingFileHandler(
    os.path.join(log_dir, "asr-a.log"), 
    when='midnight', 
    interval=1, 
    backupCount=30, 
    encoding='utf-8'
)
file_handler_a.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
reprocess_logger.addHandler(file_handler_a)

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

def write_progress(data):
    """将进度状态原子写入 JSON 文件，供 API 读取"""
    try:
        os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
        tmp_path = PROGRESS_FILE + ".tmp"
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, PROGRESS_FILE)
    except Exception as e:
        log_detail(f"    [进度写入失败] {e}", 'warning')

def clear_progress():
    """任务结束后清除进度文件"""
    try:
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
    except Exception:
        pass


def retry_on_error(max_retries=5, initial_delay=2, backoff_factor=2, allowed_exceptions=(Exception,)):
    """
    通用重试装饰器
    
    Args:
        max_retries: 最大重试次数
        initial_delay: 初始延迟（秒）
        backoff_factor: 延迟增长因子
        allowed_exceptions: 允许重试的异常类型
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retry_count = 0
            current_delay = initial_delay
            
            while retry_count <= max_retries:
                try:
                    return func(*args, **kwargs)
                except allowed_exceptions as e:
                    retry_count += 1
                    
                    if retry_count > max_retries:
                        log_detail(f"    ❌ 重试次数耗尽 ({max_retries}次)，放弃操作: {e}", 'error')
                        raise
                    
                    log_detail(f"    ⚠️  操作失败 ({retry_count}/{max_retries}): {e}", 'warning')
                    
                    # 指数退避等待
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
            
            return None
        return wrapper
    return decorator


def safe_file_operation(operation, *args, **kwargs):
    """
    安全的文件系统操作，带有自动重试
    
    Args:
        operation: 文件操作函数
        *args: 位置参数
        **kwargs: 关键字参数
    """
    @retry_on_error(
        max_retries=10,
        initial_delay=5,
        backoff_factor=1.5,
        allowed_exceptions=(OSError, IOError)
    )
    def _perform_operation():
        return operation(*args, **kwargs)
    
    return _perform_operation()


def wait_for_network_mount(mount_path, max_wait=600, check_interval=10):
    """
    等待网络挂载恢复
    
    Args:
        mount_path: 挂载点路径
        max_wait: 最大等待时间（秒）
        check_interval: 检查间隔（秒）
    
    Returns:
        bool: 挂载是否恢复
    """
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            if os.path.exists(mount_path):
                log_detail(f"    ✅ 网络挂载 {mount_path} 已恢复", 'info')
                # 额外等待几秒确保挂载稳定
                time.sleep(3)
                return True
        except Exception as e:
            log_detail(f"    ⏳ 等待网络挂载恢复... ({e})", 'warning')
        
        time.sleep(check_interval)
    
    log_detail(f"    ❌ 网络挂载 {mount_path} 未在 {max_wait} 秒内恢复", 'error')
    return False


# 合并阈值：同一事件内两个哭声文件最大时间间隔（秒）
CRY_MERGE_GAP_SEC = 600   # 10 分钟

# 事件上下文扩展：在首尾哭声文件前后各取 N 个相邻文件
CRY_CONTEXT_EACH_SIDE = 5  # 前 5 + 后 5 = 最多 10 个上下文文件

# 最大事件文件数：超过此数量的cry组合将被拆分（避免单个事件包含过多文件导致分析不准）
CRY_MAX_EVENT_FILES = 15


def parse_file_datetime(filename):
    """从文件名解析 datetime - 支持两种格式：YYYY-MM-DD_HH-MM-SS 和 YYYYMMDD-HHMMSS"""
    # 先尝试原来的格式：YYYY-MM-DD_HH-MM-SS
    match = re.search(r'(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})', filename)
    if match:
        try:
            return datetime.datetime.strptime(
                f"{match.group(1)} {match.group(2).replace('-', ':')}",
                "%Y-%m-%d %H:%M:%S"
            )
        except Exception:
            pass
    # 再尝试 YYYYMMDD-HHMMSS 格式
    match = re.search(r'(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})', filename)
    if match:
        try:
            return datetime.datetime.strptime(
                f"{match.group(1)}-{match.group(2)}-{match.group(3)} {match.group(4)}:{match.group(5)}:{match.group(6)}",
                "%Y-%m-%d %H:%M:%S"
            )
        except Exception:
            pass
    return None


def _split_large_group(group):
    """将超过 CRY_MAX_EVENT_FILES 的组按最大间隔拆分，递归直到每段都不超限"""
    if len(group) <= CRY_MAX_EVENT_FILES:
        return [group]
    # 找相邻文件间最大时间间隔的位置
    max_gap = -1
    max_gap_idx = -1
    for i in range(1, len(group)):
        prev_dt = parse_file_datetime(os.path.basename(group[i - 1]))
        curr_dt = parse_file_datetime(os.path.basename(group[i]))
        if prev_dt and curr_dt:
            gap = (curr_dt - prev_dt).total_seconds()
            if gap > max_gap:
                max_gap = gap
                max_gap_idx = i
    # 如果找不到有效时间（文件名无法解析），则简单对半拆分
    if max_gap_idx <= 0:
        mid = len(group) // 2
        max_gap_idx = mid
        log_detail(f"    ⚠️ 无法解析时间，在位置 {mid} 处对半拆分", 'warning')
    else:
        log_detail(f"    ✂️ 在间隔 {max_gap:.0f}秒 处拆分 (文件 {max_gap_idx}/{len(group)})", 'info')
    left = group[:max_gap_idx]
    right = group[max_gap_idx:]
    # 递归拆分
    return _split_large_group(left) + _split_large_group(right)


def merge_cry_events(cry_file_paths, all_sorted_files):
    """
    1. 将时间间隔 <= CRY_MERGE_GAP_SEC 的相邻哭声文件合并为一个事件
    2. 超过 CRY_MAX_EVENT_FILES 的组按最大间隔拆分为多个子事件
    3. 每个事件在 all_sorted_files 中往前/后各扩展 CRY_CONTEXT_EACH_SIDE 个文件
    返回：list of [filepath, ...]
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

    # 对超过最大文件数的组进行拆分
    split_groups = []
    for group in groups:
        if len(group) > CRY_MAX_EVENT_FILES:
            log_detail(f"  ✂️ 哭声文件组 {len(group)} 个超过上限 {CRY_MAX_EVENT_FILES}，按最大间隔拆分:", 'info')
            sub_groups = _split_large_group(group)
            for si, sg in enumerate(sub_groups, 1):
                log_detail(f"    子组 {si}: {len(sg)} 个文件", 'info')
            split_groups.extend(sub_groups)
        else:
            split_groups.append(group)
    groups = split_groups

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
    
    @retry_on_error(
        max_retries=5,
        initial_delay=3,
        backoff_factor=2,
        allowed_exceptions=(OSError, IOError)
    )
    def create_event_directory():
        os.makedirs(event_dir, exist_ok=True)
    
    create_event_directory()

    for fpath in event_files:
        dest = os.path.join(event_dir, os.path.basename(fpath))
        
        @retry_on_error(
            max_retries=5,
            initial_delay=2,
            backoff_factor=2,
            allowed_exceptions=(OSError, IOError)
        )
        def remove_existing_file():
            if os.path.exists(dest) or os.path.islink(dest):
                os.remove(dest)
        
        try:
            remove_existing_file()
        except Exception as e:
            log_detail(f"    [!] 删除现有文件失败：{e}", 'warning')
        
        try:
            @retry_on_error(
                max_retries=5,
                initial_delay=3,
                backoff_factor=2,
                allowed_exceptions=(OSError, IOError)
            )
            def copy_audio_file():
                shutil.copy2(fpath, dest)
            
            copy_audio_file()
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

    # ── 分布式锁：防止多个 A 轨进程同时运行 ──
    VALKEY_URI = os.environ.get('VALKEY_URI', '')
    _redis_lock = None
    _lock_acquired = False
    _lock_pid = os.getpid()  # 记录当前进程 PID
    
    # ── atexit 最后一道防线：确保 Python 解释器退出时释放锁 ──
    import atexit
    def atexit_release_lock():
        """Python 解释器退出时自动释放锁（包括异常退出、崩溃等场景）"""
        if _redis_lock and _lock_acquired:
            try:
                _redis_lock.delete('babycry:reprocess_lock')
                print(f"\n🔓 [atexit] Python 解释器退出，已自动释放分布式锁 (PID: {_lock_pid})")
            except Exception:
                pass
    atexit.register(atexit_release_lock)
    
    if VALKEY_URI:
        try:
            import valkey
            _redis_lock = valkey.from_url(VALKEY_URI)
            
            # ── 自动清理残留锁：检查锁持有者是否还活着 ──
            existing_lock = _redis_lock.get('babycry:reprocess_lock')
            if existing_lock:
                holder_str = existing_lock.decode() if isinstance(existing_lock, bytes) else existing_lock
                # 锁值格式: "pid@hostname" 或旧的 "a_track_reprocess"
                parts = holder_str.split('@')
                if len(parts) == 2:
                    try:
                        holder_pid = int(parts[0])
                        # 检查持有者进程是否还在运行
                        import signal
                        os.kill(holder_pid, 0)  # 信号 0 = 检查进程是否存在
                        log_detail(f"⚠️  检测到另一个 A 轨进程正在运行 (PID: {holder_pid})", 'warning')
                        log_detail(f"   请等待其完成后再试", 'warning')
                        sys.exit(1)
                    except (ProcessLookupError, ValueError):
                        # 进程不存在，清理残留锁
                        log_detail(f"🧹 检测到残留锁 (PID: {holder_str})，自动清理中...", 'warning')
                        _redis_lock.delete('babycry:reprocess_lock')
                        time.sleep(0.5)  # 等待清理生效
                    except PermissionError:
                        pass  # 无权检查，继续尝试获取锁
                else:
                    # 旧格式锁，尝试清理
                    log_detail(f"🧹 检测到旧格式残留锁，自动清理中...", 'warning')
                    _redis_lock.delete('babycry:reprocess_lock')
                    time.sleep(0.5)
            
            # 尝试获取锁，TTL 2 小时，使用 pid@hostname 作为值便于追踪
            import socket
            lock_value = f"{_lock_pid}@{socket.gethostname()}"
            _lock_acquired = _redis_lock.set('babycry:reprocess_lock', lock_value, nx=True, ex=7200)
            if not _lock_acquired:
                lock_holder = _redis_lock.get('babycry:reprocess_lock')
                holder_str = lock_holder.decode() if isinstance(lock_holder, bytes) else lock_holder
                log_detail(f"❌ 无法获取分布式锁！当前持有者: {holder_str}", 'error')
                log_detail(f"   可能有另一个 A 轨进程正在运行", 'warning')
                log_detail(f"   请等待其完成后再试", 'warning')
                sys.exit(1)
            log_detail(f"🔒 已获取分布式锁 (PID: {_lock_pid})", 'info')
            
            # ── 注册信号处理器，确保异常退出时释放锁 ──
            def release_lock_on_exit(signum, frame):
                """进程终止时自动释放锁"""
                if _redis_lock and _lock_acquired:
                    try:
                        _redis_lock.delete('babycry:reprocess_lock')
                        log_detail(f"🔓 收到信号 {signum}，已自动释放分布式锁", 'info')
                    except Exception:
                        pass
                sys.exit(0)
            
            import signal
            signal.signal(signal.SIGTERM, release_lock_on_exit)
            signal.signal(signal.SIGINT, release_lock_on_exit)
            
            # ── 启动锁续期线程（每 30 分钟刷新 TTL）──
            def lock_refresh_thread():
                """后台线程：定期续期分布式锁"""
                while _lock_acquired:
                    try:
                        time.sleep(1800)  # 30 分钟
                        if _lock_acquired and _redis_lock:
                            _redis_lock.expire('babycry:reprocess_lock', 7200)
                            log_detail(f"🔄 已续期分布式锁 TTL (剩余 2 小时)", 'info')
                    except Exception:
                        break  # 静默退出
            
            threading.Thread(target=lock_refresh_thread, daemon=True).start()
            log_detail(f"⏰ 锁续期线程已启动（每 30 分钟自动续期）", 'info')
            
        except Exception as e:
            log_detail(f"⚠️  Redis 连接异常，无法获取锁: {e}", 'warning')
            log_detail(f"   将继续运行（无锁模式）", 'warning')
            _redis_lock = None

    # === 智能续传：先从数据库获取文件列表 ===
    # 注意：定向检索（选择特定日期/时间）时不使用智能续传，强制重新分析
    is_targeted = bool(filter_date or start_time_arg or end_time_arg)
    date_stats = {}  # {date_str: processed_count}
    if not force_replace and not is_targeted:
        date_stats = get_date_processing_stats()
        log_detail(f"[*] 智能续传：已处理统计={dict(list(date_stats.items())[:5])}...", 'info')

    # 尝试从 DB 缓存获取文件列表
    log_detail(f"[*] 正在从数据库缓存获取文件列表...", 'info')
    
    # 检查缓存新鲜度（TTL 是否存在且未即将过期）
    cache_info = check_cache_freshness()
    cached_count = cache_info.get('total_keys', 0)
    cache_fresh = cache_info.get('fresh', False)
    
    if cache_info.get('ttl_min', -2) == -1:
        log_detail(f"[*] ⚠️  Redis 缓存无 TTL（旧缓存），将强制刷盘更新", 'warning')
        cache_fresh = False
    elif cache_info.get('ttl_min', -2) > 0:
        ttl_hours = cache_info['ttl_min'] / 3600
        log_detail(f"[*] Redis 缓存 TTL 剩余 {ttl_hours:.1f} 小时", 'info')
    
    log_detail(f"[*] DB 缓存中有 {cached_count} 个文件，新鲜度: {'✅ 新鲜' if cache_fresh else '⚠️ 过期/无TTL'}", 'info')

    if cached_count > 0 and cache_fresh:
        # DB 缓存命中！直接从 Redis 获取文件列表
        log_detail(f"[*] ✅ Redis 缓存命中，使用缓存文件列表（极速模式）", 'info')
        all_files = get_file_cache_from_redis()  # 返回 [{filepath, filename}, ...]
        all_files = [f['filepath'] for f in all_files]  # 转成路径列表

        # 按日期过滤
        if filter_date:
            all_files = [f for f in all_files if filter_date in f]

        # 按时间过滤
        target_files = [f for f in all_files if is_time_in_range(os.path.basename(f), start_time_arg, end_time_arg)]

        log_detail(f"[*] 从缓存获取文件列表完成", 'info')
    else:
        # DB 缓存未命中或已过期，自动触发刷盘
        if cached_count > 0 and not cache_fresh:
            log_detail(f"[*] ⚠️ Redis 缓存已过期/无TTL，自动触发刷盘更新...", 'warning')
        else:
            log_detail(f"[*] ⚠️ DB 缓存为空，自动触发刷盘缓存...", 'warning')

        def on_cache_progress(count, current_dir):
            if count % 1000 == 0:
                log_detail(f"    📁 刷盘进度: {count} 个文件 @ {current_dir}", 'info')

        cache_count = refresh_file_cache(
            PROCESSED_DIR,
            audio_exts=AUDIO_EXTS,
            progress_callback=on_cache_progress,
            log_callback=lambda msg: log_detail(f"    {msg}", 'info')
        )

        if cache_count > 0:
            log_detail(f"[*] ✅ 刷盘完成，缓存了 {cache_count} 个文件，现在从缓存读取", 'success')
            all_files = get_file_cache_from_redis()
            all_files = [f['filepath'] for f in all_files]

            # 按日期过滤
            if filter_date:
                all_files = [f for f in all_files if filter_date in f]

            # 按时间过滤
            target_files = [f for f in all_files if is_time_in_range(os.path.basename(f), start_time_arg, end_time_arg)]
        else:
            # 刷盘失败，回退到传统磁盘扫描
            log_detail(f"[*] ⚠️ 刷盘失败或没有文件，回退到磁盘扫描", 'warning')

            try:
                log_detail(f"[*] 正在启动文件树扫描...", 'info')

                @retry_on_error(
                    max_retries=10,
                    initial_delay=5,
                    backoff_factor=1.5,
                    allowed_exceptions=(OSError, IOError)
                )
                def scan_directory():
                    files_scanned = []
                    targets_scanned = []

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
                            files_scanned.append(file_path)

                            if is_time_in_range(file, start_time_arg, end_time_arg):
                                targets_scanned.append(file_path)

                    return files_scanned, targets_scanned

                all_files, target_files = scan_directory()

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
    preview_files = target_files if is_targeted else all_files
    preview_label = "目标文件列表" if is_targeted else "文件列表"
    log_detail(f"[*] {preview_label}（前20个）:", 'info')
    for i, f in enumerate(preview_files[:20], 1):
        log_detail(f"    {i}. {os.path.basename(f)}", 'info')

    # 用于分析的文件列表
    # 定向分析时必须严格使用目标文件；若时间范围内没有命中，则直接结束，不能回退到整天全量处理。
    if is_targeted:
        files_to_process = target_files
    else:
        files_to_process = target_files if target_files else all_files
    
    if not files_to_process:
        if is_targeted:
            log_detail(
                f"⚠️ 所选日期/时间范围内没有匹配文件：日期={filter_date or '全部'}，"
                f"时间={start_time_arg or '00-00'} ~ {end_time_arg or '23-59'}。",
                'warning'
            )
            log_detail("🛑 已按定向条件停止，本次不会自动回退到整天或全量扫描。", 'warning')
        else:
            log_detail(f"在 {target_dir} 中未找到任何支持的音频文件。", 'warning')
        sys.exit(0)

    # === 智能续传优化：跳过已完成的日期，对部分处理的日期从断点继续 ===
    incomplete_dates_from_resume = []  # 阶段一完成但阶段二不完整的日期（智能续传发现）
    if not force_replace and date_stats:
        def extract_date_from_filepath(filepath):
            """从文件路径中提取日期，支持多种格式（与 get_date_processing_stats 保持一致）
            
            支持的格式:
                1. /path/2025-11-15/file.m4a (日期文件夹)
                2. TermuxAudioRecording_2025-11-15_12-56-54.m4a (有横杠)
                3. recording-20251115-125944.m4a (无横杠 YYYYMMDD)
                4. recording-2025-11-15-125944.m4a (混合格式)
            """
            # 格式1: 路径中 /YYYY-MM-DD/
            m = re.search(r'/(\d{4}-\d{2}-\d{2})/', filepath)
            if m:
                return m.group(1)
            
            basename = os.path.basename(filepath)
            
            # 格式2: 文件名中 _-YYYY-MM-DD_-_ 或 -YYYY-MM-DD_-
            m = re.search(r'[_-](\d{4}-\d{2}-\d{2})[_-]', basename)
            if m:
                return m.group(1)
            
            # 格式3: 文件名中 YYYYMMDD (无横杠，如 recording-20251115-125844.m4a)
            m = re.search(r'(\d{4})(\d{2})(\d{2})[-_.]', basename)
            if m and len(m.group(0)) >= 10:
                return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
            
            # 格式4: 文件名中包含 YYYY-MM-DD 但分隔符不固定 (混合格式)
            # 示例: recording-2025-11-15-125944.m4a
            m = re.search(r'(\d{4}-\d{2}-\d{2})', basename)
            if m:
                return m.group(1)
            
            return None

        # 按日期分组并排序
        date_files = {}
        unmatched_files = []  # 无法识别日期的文件
        for f in files_to_process:
            d = extract_date_from_filepath(f)
            if d:
                if d not in date_files:
                    date_files[d] = []
                date_files[d].append(f)
            else:
                unmatched_files.append(f)

        if unmatched_files:
            log_detail(f"[!] 智能续传：有 {len(unmatched_files)} 个文件无法识别日期，将重新处理", 'warning')

        # 对每个日期的文件按文件名排序（确保顺序一致）
        for d in date_files:
            date_files[d].sort(key=os.path.basename)

        completed_dates = []
        files_to_process = []
        partial_dates = []

        for d, files in date_files.items():
            processed = date_stats.get(d, 0)
            total = len(files)

            # 【修复】始终使用精确文件匹配，而非依赖数量比较
            # 原因：Redis 缓存和数据库可能不同步，数量比较会导致误判
            processed_set = get_processed_files_for_date(d)
            # 关键修复：files 中是完整路径，processed_set 中是文件名
            # 必须提取文件名后再比较！
            remaining = [f for f in files if os.path.basename(f) not in processed_set]
            actual_processed = total - len(remaining)

            if actual_processed >= total and total > 0:
                # 阶段一完成，但还需检查阶段二是否完整
                incomplete_evts = get_incomplete_cry_events(d)
                uncovered_cry = get_uncovered_cry_count(d)
                if incomplete_evts:
                    # 阶段二有不完整事件，需要重新跑阶段二
                    incomplete_dates_from_resume.append(d)
                    log_detail(f"   📎 {d}: 阶段一完成但有 {len(incomplete_evts)} 个不完整分析，需重跑阶段二", 'info')
                    for ie in incomplete_evts:
                        rec_t = ie.get('recording_time', '?')
                        cat = ie.get('reason_category', '无')
                        rsn = ie.get('reason', '')
                        rsn_short = (rsn[:40] + '...') if rsn and len(rsn) > 40 else (rsn or '缺失')
                        log_detail(f"      • ID={ie.get('id')} | {rec_t} | 分类={cat} | 原因={rsn_short}", 'warning')
                elif uncovered_cry > 0:
                    # 有cry标记文件未被事件覆盖，需要重跑阶段二
                    incomplete_dates_from_resume.append(d)
                    log_detail(f"   📎 {d}: 阶段一完成但有 {uncovered_cry} 个cry文件未生成事件，需重跑阶段二", 'info')
                else:
                    completed_dates.append(d)
            elif actual_processed > 0:
                # 部分处理，只处理剩余文件
                files_to_process.extend(remaining)
                partial_dates.append(f"{d}({actual_processed}/{total})")
            else:
                # 未处理过
                files_to_process.extend(files)

        if completed_dates:
            log_detail(f"[*] 智能续传：跳过 {len(completed_dates)} 个已完成日期", 'info')
        if incomplete_dates_from_resume:
            log_detail(f"[*] 智能续传：{len(incomplete_dates_from_resume)} 个日期阶段二不完整（缺事件/分析不完整），将重跑: {', '.join(incomplete_dates_from_resume)}", 'info')
        if partial_dates:
            log_detail(f"[*] 智能续传：{len(partial_dates)} 个日期从断点继续: {', '.join(partial_dates)}", 'info')

        # 无法识别日期的文件也加入待处理列表
        if unmatched_files:
            files_to_process.extend(unmatched_files)

        log_detail(f"[*] 智能续传：实际需要处理 {len(files_to_process)} 个文件", 'info')
    # ===

    # 如果指定了日期，先删除该日期的旧记录（在识别哭声之后、分析之前删除）
    if filter_date:
        log_detail(f"📅 已选择日期: {filter_date}，将在分析开始后删除旧记录", 'info')

    log_detail(f"\n找到 {len(files_to_process)} 个文件进行哭声识别（扫描范围内共 {len(all_files)} 个音频文件，哭声事件将从中选取上下文）...", 'info')

    # ── 过滤 Redis 缓存中过时的文件（NAS 上已不存在的文件）──
    # 只检查 files_to_process（待处理列表），all_files 在后续按日期分组时自然过滤
    if files_to_process:
        missing_files = [f for f in files_to_process if not os.path.exists(f)]
        if missing_files:
            log_detail(f"[*] ⚠️  发现 {len(missing_files)} 个文件在 NAS 上不存在（Redis缓存过时），已标记跳过", 'warning')
            if not is_targeted:
                for f in missing_files:
                    mark_file_processed_a(os.path.basename(f), status="no_cry")
            files_to_process = [f for f in files_to_process if os.path.exists(f)]
            all_files = [f for f in all_files if os.path.exists(f)]
            log_detail(f"[*] 过滤后实际需要处理 {len(files_to_process)} 个文件", 'info')

    # =====================================================================
    # 【按日期滚动处理】每天独立完成 阶段一(检测) + 阶段二(分析)
    # 不再等所有天扫描完才分析，每天检测完立即出结果
    # =====================================================================

    # --- 将 files_to_process 和 all_files 按日期分组 ---
    def extract_date_from_filepath(filepath):
        """从文件路径中提取日期"""
        m = re.search(r'/(\d{4}-\d{2}-\d{2})/', filepath)
        if m: return m.group(1)
        basename = os.path.basename(filepath)
        m = re.search(r'(\d{4}-\d{2}-\d{2})', basename)
        if m: return m.group(1)
        m = re.search(r'(\d{4})(\d{2})(\d{2})', basename)
        if m: return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        return None

    date_files_to_process = {}  # {date: [filepath, ...]}
    date_all_files = {}         # {date: [filepath, ...]} 用于上下文扩展
    for f in files_to_process:
        d = extract_date_from_filepath(f)
        if d:
            date_files_to_process.setdefault(d, []).append(f)
    for f in all_files:
        d = extract_date_from_filepath(f)
        if d:
            date_all_files.setdefault(d, []).append(f)

    # 确定要处理的日期列表（排序）
    dates_to_process = sorted(date_files_to_process.keys())

    # ── 补全遗漏：检查 DB 中有 cry 记录但缺少 baby_cry_events 的日期 ──
    # 这些日期之前被智能续传跳过了，但阶段二（合并+分析）从未执行
    unanalyzed_dates = get_unanalyzed_cry_dates()
    phase2_only_dates = []  # 只需执行阶段二的日期
    # 将智能续传发现的不完整日期也纳入
    all_unanalyzed = set(unanalyzed_dates) | set(incomplete_dates_from_resume)
    if all_unanalyzed:
        for ud in sorted(all_unanalyzed):
            if ud not in date_files_to_process:
                phase2_only_dates.append(ud)
                # 补充 date_all_files 供阶段二使用
                # 先从 all_files 中找
                ud_files = [f for f in all_files if extract_date_from_filepath(f) == ud]
                # 如果 all_files 中没有，尝试从 SOURCE_DIR/日期/ 直接扫描
                if not ud_files:
                    ud_dir = os.path.join(SOURCE_DIR, ud)
                    if os.path.isdir(ud_dir):
                        audio_exts = ('.m4a', '.mp3', '.wav', '.aac', '.flac', '.ogg', '.acc')
                        ud_files = [os.path.join(ud_dir, f) for f in os.listdir(ud_dir)
                                    if f.lower().endswith(audio_exts)]
                        log_detail(f"   📂 {ud}: 从目录扫描到 {len(ud_files)} 个音频文件", 'info')
                date_all_files[ud] = sorted(ud_files, key=os.path.basename)
                date_files_to_process[ud] = []  # 阶段一无文件需处理
        if phase2_only_dates:
            log_detail(f"\n📎 发现 {len(phase2_only_dates)} 个日期有cry记录但缺少分析结果（将只执行阶段二）", 'info')
            for pd in phase2_only_dates:
                cry_count = len(get_cry_files_for_date(pd))
                incomplete_evts = get_incomplete_cry_events(pd)
                uncovered_cry = get_uncovered_cry_count(pd)
                detail_parts = [f"{cry_count} 个cry文件"]
                if incomplete_evts:
                    detail_parts.append(f"{len(incomplete_evts)} 个不完整事件")
                if uncovered_cry > 0:
                    detail_parts.append(f"{uncovered_cry} 个未覆盖")
                log_detail(f"   • {pd}: {', '.join(detail_parts)}", 'info')
                for ie in incomplete_evts:
                    rec_t = ie.get('recording_time', '?')
                    cat = ie.get('reason_category', '无')
                    rsn = ie.get('reason', '')
                    rsn_short = (rsn[:40] + '...') if rsn and len(rsn) > 40 else (rsn or '缺失')
                    log_detail(f"      ⚠️ ID={ie.get('id')} | {rec_t} | 分类={cat} | 原因={rsn_short}", 'warning')

    # 合并所有待处理日期
    all_dates = sorted(set(dates_to_process + phase2_only_dates))

    if not all_dates:
        log_detail(f"\n✅ 未找到需要处理的文件，也没有遗漏的cry记录。", 'info')
        sys.exit(0)

    log_detail(f"\n{'='*60}", 'info')
    log_detail(f"📋 按日期滚动处理模式", 'info')
    log_detail(f"   📅 共 {len(all_dates)} 个日期待处理: {all_dates[0]} ~ {all_dates[-1]}", 'info')
    if phase2_only_dates:
        log_detail(f"   📎 其中 {len(phase2_only_dates)} 个日期只执行阶段二（历史cry补分析）", 'info')
    log_detail(f"   🔄 每天独立完成: 检测 → 合并 → Gemini 分析", 'info')
    log_detail(f"{'='*60}\n", 'info')

    total_task_start = time.time()
    total_events_all = 0      # 总事件数
    total_cry_files_all = 0   # 总 cry 文件数

    for date_idx, current_date in enumerate(all_dates, 1):
        day_files = date_files_to_process.get(current_date, [])
        day_all_files = date_all_files.get(current_date, day_files)
        day_all_files_sorted = sorted(day_all_files, key=os.path.basename)
        is_phase2_only = current_date in phase2_only_dates

        log_detail(f"\n{'='*60}", 'info')
        mode_str = "仅阶段二(历史cry补分析)" if is_phase2_only else f"{len(day_files)} 个文件"
        log_detail(f"📅 [{date_idx}/{len(all_dates)}] 处理日期: {current_date} ({mode_str})", 'info')
        log_detail(f"{'='*60}", 'info')

        # ── 阶段一：逐文件检测 ──
        if is_phase2_only or len(day_files) == 0:
            log_detail(f"\n⏩ 阶段一跳过（已有DB记录，直接从DB恢复cry文件）", 'info')
            cry_file_paths = []
            day_success = 0
            day_error = 0
            day_skip = 0
            day_elapsed = 0
        else:
            log_detail(f"\n📡 阶段一：逐文件检测哭声 ({len(day_files)} 个文件)", 'info')

            cry_file_paths = []
            day_success = 0
            day_error = 0
            day_skip = 0
            day_start = time.time()

            write_progress({
                "status": "running",
                "processed": 0,
                "total": len(day_files),
                "success_count": 0,
                "error_count": 0,
                "skip_count": 0,
                "avg_time": 0,
                "eta_hours": 0,
                "current_date": current_date,
                "current_file": None,
                "started_at": datetime.datetime.now().isoformat()
            })

            for file_idx, filepath in enumerate(day_files, 1):
                filename = os.path.basename(filepath)

                # 检查文件是否实际存在（Redis 缓存可能过时）
                if not os.path.exists(filepath):
                    log_detail(f"\n  [{file_idx}/{len(day_files)}] {filename} — ⏭️ 文件不存在（已从NAS移除），跳过", 'info')
                    if not is_targeted:
                        mark_file_processed_a(filename, status="no_cry")
                    day_skip += 1
                    continue

                log_detail(f"\n  [{file_idx}/{len(day_files)}] {filename}", 'info')

                if not wait_for_network_mount(SOURCE_DIR, max_wait=300, check_interval=5):
                    log_detail(f"    ❌ 网络挂载不可用，跳过", 'error')
                    if not is_targeted:
                        mark_file_processed_a(filename, status="no_cry")
                    day_skip += 1
                    continue

                max_retries = 5
                retry_count = 0
                request_success = False

                while retry_count < max_retries:
                    try:
                        @retry_on_error(
                            max_retries=5,
                            initial_delay=3,
                            backoff_factor=2,
                            allowed_exceptions=(OSError, IOError)
                        )
                        def open_audio_file():
                            return open(filepath, 'rb')

                        with open_audio_file() as f:
                            files_data = {'audio_file': (filename, f, 'audio/m4a')}
                            response = requests.post(QUICK_DETECT_URL, files=files_data, timeout=60)
                        request_success = True
                        break
                    except (OSError, IOError) as e:
                        retry_count += 1
                        log_detail(f"    ⚠️  文件访问失败 ({retry_count}/{max_retries}): {e}", 'warning')
                        if retry_count >= max_retries:
                            day_skip += 1
                            break
                        wait_for_network_mount(SOURCE_DIR, max_wait=120, check_interval=10)
                    except requests.exceptions.Timeout as e:
                        retry_count += 1
                        log_detail(f"    ⚠️  超时 ({retry_count}/{max_retries})", 'warning')
                        if retry_count >= max_retries:
                            day_skip += 1
                            break
                        time.sleep(2)
                    except requests.exceptions.RequestException as e:
                        retry_count += 1
                        log_detail(f"    ⚠️  网络错误 ({retry_count}/{max_retries})", 'warning')
                        if retry_count >= max_retries:
                            day_skip += 1
                            break
                        time.sleep(2)

                if not request_success:
                    if not is_targeted:
                        mark_file_processed_a(filename, status="error")
                elif response.status_code == 200:
                    result = response.json()
                    is_cry = result.get('is_baby_cry', False)
                    confidence = result.get('confidence', 0)

                    if is_cry:
                        cry_file_paths.append(filepath)
                        log_detail(f"    🍼 检测到哭声! 置信度={confidence:.3f}", 'info')
                    else:
                        log_detail(f"    📉 未检出哭声 (置信度={confidence:.3f})", 'info')

                    day_success += 1
                    if not is_targeted:
                        mark_file_processed_a(filename, status="cry" if is_cry else "no_cry")
                else:
                    log_detail(f"    ❌ 失败 (Status {response.status_code})", 'error')
                    day_error += 1
                    if not is_targeted:
                        mark_file_processed_a(filename, status="error")

                # 写入进度（每20个文件）
                if file_idx % 20 == 0:
                    elapsed = time.time() - day_start
                    avg = elapsed / file_idx if file_idx > 0 else 0
                    remaining_files = len(day_files) - file_idx
                    eta_h = (remaining_files * avg) / 3600
                    log_detail(f"    📊 进度: {file_idx}/{len(day_files)} | 哭声: {len(cry_file_paths)} | ETA: {eta_h:.1f}h", 'info')
                    write_progress({
                        "status": "running",
                        "processed": file_idx,
                        "total": len(day_files),
                        "success_count": day_success,
                        "error_count": day_error,
                        "skip_count": day_skip,
                        "avg_time": round(avg, 1),
                        "eta_hours": round(eta_h, 1),
                        "current_date": current_date,
                        "current_file": filename,
                        "started_at": datetime.datetime.fromtimestamp(day_start).isoformat()
                    })

                time.sleep(0.1)

            day_elapsed = time.time() - day_start
            log_detail(f"\n{'─'*40}", 'info')
            log_detail(f"📊 {current_date} 阶段一完成: 成功={day_success}, 跳过={day_skip}, 错误={day_error}, 哭声={len(cry_file_paths)}, 耗时={day_elapsed/60:.1f}分钟", 'info')

        # ── 同时从数据库恢复该日期已标记的 cry 文件（补全之前中断累积的） ──
        db_cry_filenames = get_cry_files_for_date(current_date)
        if db_cry_filenames:
            # 将数据库中的 cry 文件名映射为完整路径
            day_all_basenames = {os.path.basename(f): f for f in day_all_files_sorted}
            db_cry_paths = []
            db_unresolved = []
            for fn in db_cry_filenames:
                if fn in day_all_basenames:
                    full_path = day_all_basenames[fn]
                    if full_path not in cry_file_paths:
                        db_cry_paths.append(full_path)
                        cry_file_paths.append(full_path)
                else:
                    # 文件不在 day_all_files 中，尝试从 SOURCE_DIR 的日期子目录查找
                    date_dir = os.path.join(SOURCE_DIR, current_date)
                    candidate = os.path.join(date_dir, fn) if os.path.isdir(date_dir) else None
                    if candidate and os.path.isfile(candidate):
                        if candidate not in cry_file_paths:
                            db_cry_paths.append(candidate)
                            cry_file_paths.append(candidate)
                            # 同时补充到 day_all_files_sorted
                            if candidate not in day_all_files_sorted:
                                day_all_files_sorted.append(candidate)
                    else:
                        db_unresolved.append(fn)
            if db_cry_paths:
                log_detail(f"   📎 从数据库恢复 {len(db_cry_paths)} 个历史cry文件", 'info')
            if db_unresolved:
                log_detail(f"   ⚠️  {len(db_unresolved)} 个cry文件在磁盘上未找到: {db_unresolved[:3]}{'...' if len(db_unresolved) > 3 else ''}", 'warning')

        # 排序 cry_file_paths
        cry_file_paths.sort(key=os.path.basename)

        if not cry_file_paths:
            # 即使没有找到 cry 文件，检查是否有不完整的历史事件需要重新分析
            incomplete_events = get_incomplete_cry_events(current_date)
            if incomplete_events:
                log_detail(f"   🔄 {current_date} 无cry文件，但有 {len(incomplete_events)} 个不完整事件需要重新分析", 'info')
                # 删除不完整事件，让 get_unanalyzed_cry_dates 下次仍能捕获该日期
                # 但先尝试用已有 audio_path 重新分析
                for ie in incomplete_events:
                    ie_audio = ie.get('audio_path')
                    ie_filename = ie.get('filename')
                    if ie_audio and os.path.exists(ie_audio):
                        log_detail(f"      🔄 重新分析: {ie_filename} (category={ie.get('reason_category')})", 'info')
                        try:
                            # 重新调用分析
                            ie_event_files = ie.get('event_files', [])
                            req_data = {
                                "filename": ie_filename,
                                "audio_path": ie_audio,
                                "start_ms": 0,
                                "end_ms": 60000,
                                "audio_paths": ie_event_files if ie_event_files else [ie_audio],
                            }
                            # 限额退避重试
                            max_api_retries = 3
                            for api_retry in range(max_api_retries):
                                resp = requests.post(
                                    "http://localhost:5008/api/analyze_cry",
                                    json=req_data,
                                    timeout=180
                                )
                                if resp.status_code == 200:
                                    r = resp.json()
                                    if r.get("reason"):
                                        log_detail(f"      ✨ 重新分析成功! category={r.get('category', '?')}", 'info')
                                        break
                                    else:
                                        log_detail(f"      ⚠️ 重新分析仍无结果", 'warning')
                                        break
                                elif resp.status_code == 429 or 'RESOURCE_EXHAUSTED' in resp.text:
                                    wait_sec = 15 * (2 ** api_retry)
                                    log_detail(f"      ⚠️ API 限额，等待 {wait_sec}秒...", 'warning')
                                    time.sleep(wait_sec)
                                else:
                                    log_detail(f"      ❌ 分析失败: HTTP {resp.status_code}", 'error')
                                    break
                        except Exception as e:
                            log_detail(f"      ❌ 重新分析异常: {e}", 'error')
                        time.sleep(5)
                    else:
                        # 音频文件不存在，删除这条不完整记录
                        log_detail(f"      🗑️ 音频不存在，删除不完整记录: {ie_filename}", 'warning')
                        try:
                            from db_manager import delete_cry_event_by_id
                            delete_cry_event_by_id(ie['id'])
                        except:
                            pass
            else:
                log_detail(f"   ✅ {current_date} 无哭声事件，跳过阶段二", 'info')
            continue

        # ── 阶段二：合并事件 + Gemini 深度分析 ──
        log_detail(f"\n🧠 阶段二：{current_date} 合并事件 + 深度分析", 'info')

        # 删除该日期的旧分析记录（含不完整记录）
        log_detail(f"   🗑️  删除 {current_date} 的旧分析记录...", 'info')
        try:
            from db_manager import delete_cry_events_by_date, delete_incomplete_cry_events
            # 先删不完整记录
            incomplete_deleted = delete_incomplete_cry_events(current_date)
            if incomplete_deleted > 0:
                log_detail(f"   ✅ 已删除 {incomplete_deleted} 条不完整记录", 'info')
            # 如果是 phase2_only 模式，只删不完整的，不删正常的
            if not is_phase2_only:
                deleted_count = delete_cry_events_by_date(current_date)
                if deleted_count > 0:
                    log_detail(f"   ✅ 已删除 {deleted_count} 条旧记录", 'info')
        except Exception as e:
            log_detail(f"   ⚠️  删除旧记录失败: {e}", 'warning')

        events = merge_cry_events(cry_file_paths, day_all_files_sorted)

        events_base_dir = os.path.join(SOURCE_DIR, "cry_events")
        os.makedirs(events_base_dir, exist_ok=True)

        log_detail(f"   🍼 哭声文件: {len(cry_file_paths)} 个", 'info')
        log_detail(f"   🔔 独立事件: {len(events)} 个", 'info')

        phase2_skip = 0  # 快速检测未确认而跳过的事件数
        for evt_idx, event_files in enumerate(events, 1):
            if not wait_for_network_mount(SOURCE_DIR, max_wait=300, check_interval=5):
                log_detail(f"    ❌ 网络挂载不可用，跳过事件 {evt_idx}", 'error')
                continue

            event_dir = build_event_dir(events_base_dir, evt_idx, event_files)
            rep_filepath = event_files[len(event_files) // 2]
            rep_filename = os.path.basename(rep_filepath)

            log_detail(f"\n   🔔 事件 {evt_idx}/{len(events)}: {len(event_files)} 个文件, 代表={rep_filename}", 'info')

            cry_files_in_event = [f for f in event_files if f in cry_file_paths]
            log_detail(f"      哭声文件: {len(cry_files_in_event)}个, 上下文: {len(event_files) - len(cry_files_in_event)}个", 'info')

            # 快速检测验证
            first_seg = {'start': 0, 'end': 60000}
            try:
                @retry_on_error(
                    max_retries=5,
                    initial_delay=3,
                    backoff_factor=2,
                    allowed_exceptions=(OSError, IOError)
                )
                def open_and_detect():
                    with open(rep_filepath, 'rb') as f:
                        return requests.post(
                            QUICK_DETECT_URL,
                            files={'audio_file': (rep_filename, f, 'audio/m4a')},
                            timeout=30
                        )

                res = open_and_detect()
                if res.status_code == 200:
                    result = res.json()
                    if result.get('is_baby_cry'):
                        log_detail(f"      📍 哭声确认: 置信度={result.get('confidence', 0):.3f}", 'info')
                    else:
                        log_detail(f"      ⚠️ 快速检测未确认哭声 (conf={result.get('confidence', 0):.3f})，跳过该事件", 'warning')
                        phase2_skip += 1
                        continue
            except Exception as e:
                log_detail(f"      ⚠️ 快速检测失败: {e}", 'warning')

            try:
                log_detail(f"      🤖 调用 Gemini 深度分析...", 'info')
                # 限额退避重试
                max_api_retries = 3
                for api_retry in range(max_api_retries):
                    response = requests.post(
                        "http://localhost:5008/api/analyze_cry",
                        json={
                            "filename": rep_filename,
                            "audio_path": rep_filepath,
                            "start_ms": first_seg.get('start', 0),
                            "end_ms": first_seg.get('end', 60000),
                            "audio_paths": event_files,
                        },
                        timeout=180
                    )
                    if response.status_code == 200:
                        break
                    elif response.status_code == 429 or 'RESOURCE_EXHAUSTED' in response.text:
                        wait_sec = 15 * (2 ** api_retry)
                        log_detail(f"      ⚠️ API 限额 (429)，第 {api_retry + 1}/{max_api_retries} 次重试，等待 {wait_sec}秒...", 'warning')
                        time.sleep(wait_sec)
                    else:
                        break
                if response.status_code == 200:
                    result = response.json()
                    reason = result.get("reason") or "未知"
                    advice = result.get("advice") or "无"
                    category = result.get("category") or "未分类"

                    log_detail(f"      ✨ 分析完成! 分类={category}", 'info')
                    log_detail(f"         原因: {reason[:100]}{'...' if len(reason) > 100 else ''}", 'info')
                    log_detail(f"         建议: {advice[:80]}{'...' if len(advice) > 80 else ''}", 'info')

                    # 发送邮件
                    if EMAIL_ENABLED:
                        try:
                            cry_time = parse_file_datetime(rep_filename)
                            cry_time_str = cry_time.strftime('%Y-%m-%d %H:%M:%S') if cry_time else '未知'
                            subject = f"📋 历史分析 | {current_date} | 事件 {evt_idx}/{len(events)}"
                            content = f"""
═══════════════════════════════════════════════════════════════
                     📋 宝宝哭声历史分析报告
═══════════════════════════════════════════════════════════════

📅 处理日期: {current_date}
⏰ 报告时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🔔 事件 {evt_idx}/{len(events)}

───────────────────────────────────────────────────────────────
🏷️ 分析结果:
   • 分类: {category}
   • 原因: {reason}
   • 建议: {advice}

🎵 代表文件: {rep_filename}
⏰ 哭声时间: {cry_time_str}
📊 事件文件数: {len(event_files)} 个 (哭声 {len(cry_files_in_event)} 个)
📁 文件列表 (🔴=哭声文件):
{chr(10).join(('   🔴 ' if f in cry_file_paths else '      ') + os.path.basename(f) for f in event_files)}
═══════════════════════════════════════════════════════════════
"""
                            send_email_async(subject, content)
                            log_detail(f"      📧 邮件已发送", 'info')
                        except Exception as email_err:
                            log_detail(f"      ⚠️ 邮件发送失败: {email_err}", 'warning')
                else:
                    log_detail(f"      ⚠️ Gemini 返回 {response.status_code}，跳过", 'warning')
            except Exception as e:
                log_detail(f"      ❌ Gemini 分析失败: {e}", 'error')

            time.sleep(5)  # 事件间隔拉长，避免连续调用触发 Gemini API 限额

        total_events_all += len(events)
        total_cry_files_all += len(cry_file_paths)

        log_detail(f"\n   ✅ {current_date} 完成: {len(events)} 个事件, {len(cry_file_paths)} 个哭声文件" + 
                   (f", {phase2_skip} 个事件因快速检测未确认而跳过" if phase2_skip > 0 else ""), 'info')

    # =====================================================================
    # 全部完成汇总
    # =====================================================================
    total_elapsed = time.time() - total_task_start
    log_detail(f"\n{'='*60}", 'info')
    log_detail(f"✅ 全部处理完成！", 'info')
    log_detail(f"{'─'*60}", 'info')
    log_detail(f"   📅 处理日期: {len(all_dates)} 个", 'info')
    log_detail(f"   🍼 哭声文件: {total_cry_files_all} 个", 'info')
    log_detail(f"   🔔 独立事件: {total_events_all} 个", 'info')
    log_detail(f"   ⏱️ 总耗时: {total_elapsed/60:.1f} 分钟", 'info')
    log_detail(f"{'─'*60}", 'info')
    log_detail(f"{'='*60}", 'info')
    log_detail(f"\n请在上方切换到【宝宝分析】标签页查看自动刷新的记录。", 'info')

    # 写入最终进度
    write_progress({
        "status": "completed",
        "processed": sum(len(date_files_to_process.get(d, [])) for d in all_dates),
        "total": sum(len(date_files_to_process.get(d, [])) for d in all_dates),
        "success_count": 0,
        "error_count": 0,
        "skip_count": 0,
        "avg_time": 0,
        "eta_hours": 0,
        "current_date": all_dates[-1] if all_dates else None,
        "current_file": None,
        "started_at": datetime.datetime.fromtimestamp(total_task_start).isoformat()
    })

    # 发送汇总邮件
    if EMAIL_ENABLED and total_events_all > 0:
        try:
            subject = f"✅ 历史分析完成 | 共 {len(all_dates)} 天 | {total_events_all} 个事件"
            content = f"""
═══════════════════════════════════════════════════════════════
                   📋 历史分析任务完成汇总
═══════════════════════════════════════════════════════════════

📅 处理日期: {len(all_dates)} 天
🍼 哭声文件: {total_cry_files_all} 个
🔔 独立事件: {total_events_all} 个
⏱️ 总耗时: {total_elapsed/60:.1f} 分钟

请在 BabyCry 分析看板查看详细分析结果。
═══════════════════════════════════════════════════════════════
"""
            send_email_async(subject, content)
            log_detail(f"📧 汇总邮件已发送", 'info')
        except Exception as email_err:
            log_detail(f"⚠️ 汇总邮件发送失败: {email_err}", 'warning')

    # ── 释放分布式锁 ──
    if _redis_lock and _lock_acquired:
        try:
            _redis_lock.delete('babycry:reprocess_lock')
            log_detail(f"🔓 已释放分布式锁", 'info')
        except Exception:
            pass
