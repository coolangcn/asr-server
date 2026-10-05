#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import hmac
import sys
import threading
import time
from flask import Flask, render_template, render_template_string, jsonify, request, Response, send_file, session, redirect, url_for, make_response
import datetime
from collections import Counter, defaultdict
import requests
import subprocess
import argparse

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
    load_dotenv()
except Exception:
    pass

from db_manager import init_pool, init_db, get_transcripts as db_get_transcripts, fix_recording_time, get_connection, return_connection, save_date_stats_to_redis, get_date_stats_from_redis, clear_date_stats_in_redis, get_file_cache_from_redis

# --- 配置 ---
# 获取脚本自身所在的目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 跨平台路径配置
import platform
if platform.system() == "Darwin":
    # macOS 路径 - 与实际 SMB 挂载路径保持一致
    DEFAULT_SOURCE_DIR = "/Volumes/download/records/Sony-2"
    DEFAULT_LOG_FILE_PATH = os.path.expanduser("~/asr-server/log/asr-server.log")
    DEBUG_LOG_FILE_PATH = os.path.expanduser("~/asr-server/log/web_viewer_debug.log")
else:
    # Windows 路径
    DEFAULT_SOURCE_DIR = "V:\\Sony-2"
    DEFAULT_LOG_FILE_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "log", "asr-server.log")
    DEBUG_LOG_FILE_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "log", "web_viewer_debug.log")

# DEBUG 日志函数
def debug_log(message):
    """将 DEBUG 日志写入单独的文件"""
    try:
        log_dir = os.path.dirname(DEBUG_LOG_FILE_PATH)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        with open(DEBUG_LOG_FILE_PATH, 'a', encoding='utf-8') as f:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
            f.write(f"{timestamp} | DEBUG | {message}\n")
    except Exception as e:
        print(f"[DEBUG LOG ERROR] {e}")

DEFAULT_ASR_API_URL = os.getenv("ASR_API_URL", "http://localhost:5008/transcribe")
DEFAULT_WEB_PORT = int(os.getenv("WEB_VIEWER_PORT", "5009"))

# 全局配置变量
CONFIG = {
    "SOURCE_DIR": DEFAULT_SOURCE_DIR,
    "ASR_API_URL": DEFAULT_ASR_API_URL,
    "LOG_FILE_PATH": DEFAULT_LOG_FILE_PATH,
    "WEB_PORT": DEFAULT_WEB_PORT,
    "DATABASE_URL": os.getenv('DATABASE_URL', '')

}

# 【2026-10-03 双源】B 轨数据源改为双 Pixel（客厅 Pixel-6 / 卧室 Pixel-5），
# Sony 保留在候选尾部兼容历史录音。SOURCE_DIR 保留为兼容引用（config.json 覆盖仍生效）。
# 【2026-10-04 rsync 本地镜像】主根 = 本地镜像（RECORDS_ROOT），NAS 根（RECORDS_ROOT_NAS）
# 作为历史数据回退；RECORDS_ROOTS 按优先级排列，目录扫描/文件定位逐根尝试。
RECORDS_ROOT = os.getenv("RECORDS_ROOT", os.path.dirname(DEFAULT_SOURCE_DIR.rstrip("\\/")))
RECORDS_ROOT_NAS = os.getenv("RECORDS_ROOT_NAS", "/Volumes/download/records")
RECORDS_ROOTS = [RECORDS_ROOT]
if os.path.normpath(RECORDS_ROOT_NAS) != os.path.normpath(RECORDS_ROOT):
    RECORDS_ROOTS.append(RECORDS_ROOT_NAS)
SOURCE_DEVICES = ["Pixel-6", "Pixel-5", "Sony-2", "Sony-1", "Sony-3"]

def _find_audio_segment(rel_clean):
    """多根探测 audio_segments 切片：本地镜像优先、NAS 历史回退，逐设备查找，
    返回绝对路径或 None（含穿越防护）"""
    for _root in RECORDS_ROOTS:
        search_dirs = [os.path.join(_root, dev, "audio_segments") for dev in SOURCE_DEVICES]
        for segments_dir in search_dirs:
            full_path = os.path.abspath(os.path.join(segments_dir, rel_clean))
            if full_path.startswith(os.path.abspath(segments_dir) + os.sep) and os.path.isfile(full_path):
                return full_path
    search_dirs = [os.path.join(CONFIG["SOURCE_DIR"], "audio_segments")]  # 单源直挂兼容
    for segments_dir in search_dirs:
        full_path = os.path.abspath(os.path.join(segments_dir, rel_clean))
        if full_path.startswith(os.path.abspath(segments_dir) + os.sep) and os.path.isfile(full_path):
            return full_path
    return None

# 从JSON文件加载配置
CONFIG_FILE = "config.json"
if os.path.exists(CONFIG_FILE):
    import json
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        loaded_config = json.load(f)
    CONFIG.update(loaded_config)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Web查看器脚本')
    parser.add_argument('--source-path', type=str, help='源音频文件路径')
    parser.add_argument('--port', type=int, help='Web端口', default=DEFAULT_WEB_PORT)
    parser.add_argument('--asr-url', type=str, help='ASR服务API地址', default=DEFAULT_ASR_API_URL)
    return parser.parse_args()

def update_config(args):
    """根据命令行参数更新配置"""
    global ASR_SERVER_URL
    if args.source_path:
        base_path = args.source_path
        CONFIG["SOURCE_DIR"] = base_path
        logger_web.info(f"[配置] 使用自定义源路径: {base_path}")
    
    if args.port:
        CONFIG["WEB_PORT"] = args.port
    
    if args.asr_url:
        CONFIG["ASR_API_URL"] = args.asr_url
        logger_web.info(f"[配置] 使用自定义ASR服务地址: {args.asr_url}")

    ASR_SERVER_URL = CONFIG["ASR_API_URL"].rsplit("/", 1)[0]

# -----------------

app = Flask(__name__)
_web_secret_key = (os.getenv('WEB_SECRET_KEY') or os.getenv('SECRET_KEY') or '').strip()
if not _web_secret_key:
    raise RuntimeError(
        "❌ 未配置 WEB_SECRET_KEY：会话签名密钥必须显式配置。"
        "请在 .env 中设置 WEB_SECRET_KEY=<随机字符串> 后重启。"
    )
app.secret_key = _web_secret_key

# 密码保护配置
REQUIRED_PASSWORD = (os.getenv('WEB_PASSWORD') or os.getenv('ASR_WEB_PASSWORD') or '').strip()
if not REQUIRED_PASSWORD:
    raise RuntimeError(
        "❌ 未配置 WEB_PASSWORD：登录密码必须显式配置，已移除代码内默认口令。"
        "请在 .env 中设置 WEB_PASSWORD=<你的密码> 后重启。"
    )

def _asr_admin_headers():
    """代理转发到 5008 危险接口时自动注入管理令牌（.env 的 ADMIN_TOKEN，未配置则不注入）"""
    token = (os.getenv('ADMIN_TOKEN') or '').strip()
    return {'X-Admin-Token': token} if token else {}

def check_auth():
    """检查是否已登录"""
    return 'logged_in' in session and session['logged_in'] is True

def login_required(f):
    """登录验证装饰器"""
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not check_auth():
            return render_template('login.html')
        return f(*args, **kwargs)
    return decorated_function

def format_timestamp(milliseconds):
    try:
        seconds = milliseconds / 1000
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{int(h):02}:{int(m):02}:{s:06.3f}"
    except:
        return "00:00:00.000"

# 全局状态缓存
g_status_cache = {
    "asr_server": "unknown",
    "pending_files": 0,
    "last_log": "等待初始化...",
    "logs_a": "",
    "logs_b": "",
    "logs_web": "",
    "updated_at": 0
}
g_status_lock = threading.Lock()

def read_last_lines(filepath, line_count=20, encoding='utf-8', errors='ignore'):
    """高效读取文件最后几行"""
    try:
        with open(filepath, 'rb') as f:
            # 移动到文件末尾
            try:
                f.seek(-8192, os.SEEK_END) # 增加缓冲区以获取更多日志内容
            except IOError:
                # 文件太小
                f.seek(0)
            
            lines = f.readlines()
            decoded_lines = [line.decode(encoding, errors).strip() for line in lines]
            return decoded_lines[-line_count:]
    except Exception:
        return []

# 配置 Web Viewer 自身的日志记录器
def setup_web_logger():
    # 确保日志目录存在
    log_dir = os.path.join(os.path.dirname(SCRIPT_DIR), "log")
    os.makedirs(log_dir, exist_ok=True)
    
    l = logging.getLogger("web_viewer")
    l.setLevel(logging.INFO)
    l.handlers = []
    
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    
    # 写入与 asr_server 相同的系统日志文件（按大小轮转：50MB×5）
    log_path = os.path.join(log_dir, "asr-web.log")
    handler = RotatingFileHandler(
        log_path, maxBytes=50*1024*1024, backupCount=5, encoding='utf-8'
    )
    handler.setFormatter(formatter)
    l.addHandler(handler)
    
    # 控制台输出
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    l.addHandler(console)
    
    return l

import logging
from logging.handlers import RotatingFileHandler
logger_web = setup_web_logger()

def update_system_status():
    """后台更新系统状态"""
    global g_status_cache
    
    # 1. 检查 ASR Server
    asr_status = "offline"
    try:
        requests.get(f"{CONFIG['ASR_API_URL'].rsplit('/', 1)[0]}/", timeout=2)
        asr_status = "online"
    except:
        pass

    # 2. 检查待处理文件 (可能耗时)
    pending_count = -1
    SKIP_DIRS = {'processed', 'failed', 'temp', 'audio_segments', '__pycache__'}
    AUDIO_EXTS = ('.m4a', '.acc', '.aac', '.mp3', '.wav', '.ogg')
    try:
        count = 0
        # 【双源】逐设备统计待处理文件；【2026-10-04 多根】本地镜像 + NAS 双根求和
        #（pull 拉完即删 NAS 源，同一文件不会同时存在于两根，无重复计数）
        for dev in SOURCE_DEVICES:
            for _root in RECORDS_ROOTS:
                source_dir = os.path.join(_root, dev)
                if not (os.path.exists(source_dir) and os.path.isdir(source_dir)):
                    continue
                for entry in os.listdir(source_dir):
                    if entry in SKIP_DIRS:
                        continue
                    entry_path = os.path.join(source_dir, entry)
                    if os.path.isfile(entry_path):
                        if entry.lower().endswith(AUDIO_EXTS) and 'TEMP' not in entry:
                            count += 1
                    elif os.path.isdir(entry_path):
                        try:
                            for f in os.listdir(entry_path):
                                if f.lower().endswith(AUDIO_EXTS) and 'TEMP' not in f:
                                    count += 1
                        except Exception:
                            pass
        pending_count = count
    except Exception as e:
        logger_web.error(f"[StatusMonitor] 检查待处理文件失败: {e}")
        pending_count = -1

    # 3. 直接从多个物理日志文件读取 (不再进行正则过滤)
    logs_a = []
    logs_b = []
    logs_web = []
    last_log_raw = ""
    
    log_dir = os.path.join(os.path.dirname(SCRIPT_DIR), "log")
    
    # A 轨日志
    path_a = os.path.join(log_dir, "asr-a.log")
    if os.path.exists(path_a):
        lines = read_last_lines(path_a, 50)
        logs_a = []
        for line in lines:
            if ' | ' in line:
                parts = line.split(' | ', 2)
                logs_a.append(parts[2] if len(parts) > 2 else line)
            else:
                logs_a.append(line)
    
    # B 轨日志
    path_b = os.path.join(log_dir, "asr-b.log")
    if os.path.exists(path_b):
        lines = read_last_lines(path_b, 50)
        logs_b = []
        for line in lines:
            if ' | ' in line:
                parts = line.split(' | ', 2)
                logs_b.append(parts[2] if len(parts) > 2 else line)
            else:
                logs_b.append(line)
    
    # Web/系统日志
    path_web = os.path.join(log_dir, "asr-web.log")
    if os.path.exists(path_web):
        lines = read_last_lines(path_web, 50)
        logs_web = []
        for line in lines:
            if ' | ' in line:
                parts = line.split(' | ', 2)
                logs_web.append(parts[2] if len(parts) > 2 else line)
            else:
                logs_web.append(line)
        # 保持兼容性的 last_log
        last_log_raw = "\n".join(logs_web[-20:])
    
    if not logs_web and not os.path.exists(path_web):
        last_log_raw = "等待日志生成..."

    # 更新缓存
    with g_status_lock:
        g_status_cache = {
            "asr_server": asr_status,
            "pending_files": pending_count,
            "last_log": last_log_raw,
            "logs_a": "\n".join(logs_a),
            "logs_b": "\n".join(logs_b),
            "logs_web": "\n".join(logs_web),
            "updated_at": time.time()
        }

def status_monitor_loop():
    """状态监控循环主函数"""
    logger_web.info("[StatusMonitor] 启动后台状态监控线程...")
    while True:
        try:
            update_system_status()
        except Exception as e:
            logger_web.error(f"[StatusMonitor] 更新失败: {e}")
        time.sleep(3) # 每3秒由独立线程从 3 个物理日志文件提取增量状态

def start_status_monitor():
    thread = threading.Thread(target=status_monitor_loop, daemon=True)
    thread.start()

def get_system_status():
    """获取缓存的系统状态"""
    with g_status_lock:
        return g_status_cache.copy()

def get_transcripts(offset=0, limit=20):
    """获取转录记录（使用PostgreSQL，支持分页）"""
    try:
        return db_get_transcripts(offset, limit, CONFIG["DATABASE_URL"])
    except Exception as e:
        logger_web.error(f"[Error] 获取转录记录失败: {e}")
        return []

def _parse_iso_datetime(value):
    """Parse DB ISO strings without letting timezone quirks break filtering."""
    if not value:
        return None
    try:
        return datetime.datetime.fromisoformat(str(value).replace('Z', '+00:00')).replace(tzinfo=None)
    except Exception:
        return None

def _item_datetime(item):
    return _parse_iso_datetime(item.get('recording_time')) or _parse_iso_datetime(item.get('created_at'))

def _daily_keywords(text, limit=8):
    stop_words = {'的', '了', '是', '在', '我', '你', '他', '她', '它', '们', '这', '那', '有', '个', '就', '不', '和', '与', '啊', '呀', '吗', '呢', '吧'}
    cleaned = re.sub(r'\s+', '', text or '')
    words = [cleaned[i:i + 2] for i in range(max(len(cleaned) - 1, 0))]
    counter = Counter(w for w in words if len(w) == 2 and w not in stop_words)
    return [{'text': word, 'count': count} for word, count in counter.most_common(limit)]

def _get_cry_events_for_date(date_str):
    """Query cry events locally so the daily report does not depend on ASR API availability."""
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return []

        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, filename, created_at, recording_time, start_time, end_time,
                   reason, advice, reason_category, confidence, illustration_url
            FROM baby_cry_events
            WHERE COALESCE(recording_time, created_at)::date = %s
              AND COALESCE(is_deleted, FALSE) = FALSE
            ORDER BY COALESCE(recording_time, created_at) ASC
            """,
            (date_str,)
        )
        rows = cursor.fetchall()
        cursor.close()

        events = []
        for row in rows:
            events.append({
                'id': row[0],
                'filename': row[1],
                'created_at': row[2].isoformat() if row[2] else None,
                'recording_time': row[3].isoformat() if row[3] else None,
                'start_time': float(row[4]) if row[4] else 0,
                'end_time': float(row[5]) if row[5] else 0,
                'reason': row[6] or '',
                'advice': row[7] or '',
                'reason_category': row[8] or '未分类',
                'confidence': float(row[9]) if row[9] is not None else 0,
                'illustration_url': row[10] or ''
            })
        return events
    except Exception as e:
        logger_web.warning(f"[DailyReport] 查询哭声事件失败: {e}")
        return []
    finally:
        if conn:
            return_connection(conn)

def build_daily_report(date_str):
    all_items = get_transcripts(offset=0, limit=10000)
    day_items = []
    for item in all_items:
        dt = _item_datetime(item)
        if dt and dt.strftime('%Y-%m-%d') == date_str:
            day_items.append(item)

    day_items.sort(key=lambda item: _item_datetime(item) or datetime.datetime.min)

    speaker_stats = defaultdict(lambda: {'segments': 0, 'duration_seconds': 0, 'chars': 0})
    emotion_counts = Counter()
    hour_activity = [0] * 24
    full_text_parts = []
    quotes = []
    timeline = []
    total_segments = 0
    total_speech_seconds = 0

    for item in day_items:
        item_dt = _item_datetime(item)
        if item.get('full_text'):
            full_text_parts.append(item.get('full_text'))

        segments = item.get('segments') or []
        hour = item_dt.hour if item_dt else 0
        hour_activity[hour] += len(segments) or 1

        for seg in segments:
            text = (seg.get('text') or '').strip()
            speaker = seg.get('spk') or 'Unknown'
            duration = max((seg.get('end', 0) - seg.get('start', 0)) / 1000.0, 0)
            emotion = seg.get('emotion')

            total_segments += 1
            total_speech_seconds += duration
            speaker_stats[speaker]['segments'] += 1
            speaker_stats[speaker]['duration_seconds'] += duration
            speaker_stats[speaker]['chars'] += len(text)
            if emotion:
                emotion_counts[emotion] += 1
            if 8 <= len(text) <= 90:
                quotes.append({
                    'time': item_dt.isoformat() if item_dt else None,
                    'speaker': speaker,
                    'text': text,
                    'emotion': emotion or ''
                })

        if item_dt:
            timeline.append({
                'type': 'recording',
                'time': item_dt.isoformat(),
                'label': f"{len(segments)} 段对话" if segments else "录音",
                'text': (item.get('full_text') or '')[:80]
            })

    cry_events = _get_cry_events_for_date(date_str)
    cry_categories = Counter(event.get('reason_category') or '未分类' for event in cry_events)
    for event in cry_events:
        event_dt = _parse_iso_datetime(event.get('recording_time')) or _parse_iso_datetime(event.get('created_at'))
        if event_dt:
            timeline.append({
                'type': 'cry',
                'time': event_dt.isoformat(),
                'label': event.get('reason_category') or '哭声事件',
                'text': event.get('reason', '')[:80]
            })

    timeline.sort(key=lambda item: item.get('time') or '')
    speakers = [
        {
            'name': name,
            'segments': data['segments'],
            'duration_seconds': round(data['duration_seconds'], 1),
            'chars': data['chars']
        }
        for name, data in speaker_stats.items()
    ]
    speakers.sort(key=lambda item: (item['duration_seconds'], item['segments']), reverse=True)

    busiest_hour = max(range(24), key=lambda h: hour_activity[h]) if any(hour_activity) else None
    top_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else '暂无'
    top_speaker = speakers[0]['name'] if speakers else '暂无'
    selected_quotes = sorted(quotes, key=lambda item: len(item['text']), reverse=True)[:5]

    return {
        'date': date_str,
        'stats': {
            'recordings': len(day_items),
            'segments': total_segments,
            'speech_minutes': round(total_speech_seconds / 60.0, 1),
            'cry_events': len(cry_events),
            'top_speaker': top_speaker,
            'top_emotion': top_emotion,
            'busiest_hour': f"{busiest_hour:02d}:00" if busiest_hour is not None else '暂无'
        },
        'speakers': speakers[:8],
        'emotions': dict(emotion_counts),
        'keywords': _daily_keywords(''.join(full_text_parts)),
        'quotes': selected_quotes,
        'cry_events': cry_events,
        'cry_categories': dict(cry_categories),
        'hour_activity': [{'hour': f"{hour:02d}:00", 'count': count} for hour, count in enumerate(hour_activity)],
        'timeline': timeline[:80]
    }

def build_relationship_graph(items):
    """Build a speaker interaction graph from adjacent conversation turns."""
    speaker_stats = defaultdict(lambda: {'segments': 0, 'chars': 0, 'duration_seconds': 0})
    link_counts = Counter()
    hour_links = defaultdict(Counter)

    for item in items:
        dt = _item_datetime(item)
        hour = dt.hour if dt else None
        segments = item.get('segments') or []
        cleaned_segments = []

        for seg in segments:
            speaker = seg.get('spk') or 'Unknown'
            text = (seg.get('text') or '').strip()
            if not speaker or speaker == 'Unknown':
                continue

            duration = max((seg.get('end', 0) - seg.get('start', 0)) / 1000.0, 0)
            speaker_stats[speaker]['segments'] += 1
            speaker_stats[speaker]['chars'] += len(text)
            speaker_stats[speaker]['duration_seconds'] += duration
            cleaned_segments.append({
                'speaker': speaker,
                'start': seg.get('start', 0),
                'end': seg.get('end', 0)
            })

        cleaned_segments.sort(key=lambda seg: seg.get('start', 0))
        for prev, curr in zip(cleaned_segments, cleaned_segments[1:]):
            if prev['speaker'] == curr['speaker']:
                continue
            pair = tuple(sorted([prev['speaker'], curr['speaker']]))
            link_counts[pair] += 1
            if hour is not None:
                hour_links[pair][hour] += 1

    max_segments = max((data['segments'] for data in speaker_stats.values()), default=0)
    nodes = []
    for speaker, data in speaker_stats.items():
        weight = data['segments'] / max_segments if max_segments else 0
        nodes.append({
            'id': speaker,
            'name': speaker,
            'value': data['segments'],
            'symbolSize': round(28 + weight * 36, 1),
            'chars': data['chars'],
            'duration_seconds': round(data['duration_seconds'], 1)
        })

    nodes.sort(key=lambda node: node['value'], reverse=True)
    links = []
    for (source, target), count in link_counts.most_common(80):
        peak_hour = None
        if hour_links[(source, target)]:
            peak_hour = hour_links[(source, target)].most_common(1)[0][0]
        links.append({
            'source': source,
            'target': target,
            'value': count,
            'lineStyle': {'width': min(8, 1 + count * 0.55)},
            'peak_hour': f"{peak_hour:02d}:00" if peak_hour is not None else None
        })

    return {
        'nodes': nodes,
        'links': links,
        'summary': {
            'speaker_count': len(nodes),
            'interaction_count': sum(link_counts.values()),
            'strongest_pair': {
                'speakers': list(link_counts.most_common(1)[0][0]),
                'count': link_counts.most_common(1)[0][1]
            } if link_counts else None
        }
    }

GROWTH_STOP_TERMS = {
    '这个', '那个', '就是', '然后', '我们', '你们', '他们', '没有', '不是', '什么', '可以', '一下',
    '知道', '现在', '这里', '那里', '这样', '一样', '因为', '所以', '还是', '已经', '不要', '不用',
    '今天', '明天', '昨天', '时候', '东西', '一个', '两个', '一点', '怎么', '这么', '那么', '真的',
    '是不', '的是', '了吗', '了吧', '去吧', '对不', '不能',
    # jieba 分词后的高频虚词/互动套话(2026-10-02 扩充, 严格化成长词典)
    '是不是', '还有', '出来', '起来', '一起', '看看', '但是', '人家', '好不好', '要不要',
    '行不行', '能不能', '里面', '回来', '可能', '不会', '大家', '上面', '下面', '看到',
    '觉得', '这种', '那种', '过来', '应该', '地方', '这是', '那是', '外面', '这些',
    '那些', '如果', '自己', '我要', '为什么', '开始', '没关系', '不好意思', '谢谢', '再见',
    '总是', '老是', '原来', '只有', '不过', '可是', '没什么', '怎么样', '许多', '一点',
}

_JIEBA = None

def _jieba():
    """惰性加载 jieba(首次分词时初始化, 静默建缓存日志)"""
    global _JIEBA
    if _JIEBA is None:
        import jieba
        jieba.setLogLevel(60)
        _JIEBA = jieba
    return _JIEBA

def _growth_text_terms(text):
    """提取严格的词语: 中文用 jieba 真分词(只留≥2字词), 英文只留纯字母单词。
    旧版对中文做 2-3 字暴力滑窗, 任意字块组合都成了"词", 断句全错。"""
    text = text or ''
    terms = []

    for block in re.findall(r'[\u4e00-\u9fff]+', text):
        for w in _jieba().cut(block):
            if len(w) >= 2 and w not in GROWTH_STOP_TERMS:
                terms.append(w)

    for token in re.findall(r"[A-Za-z][A-Za-z']{1,19}", text):
        t = token.lower().strip("'")
        if len(t) >= 2:
            terms.append(t)

    return terms

def _quote_score(text, speaker, emotion, target_speaker):
    score = 0
    text = text.strip()
    if target_speaker and speaker == target_speaker:
        score += 4
    if 8 <= len(text) <= 45:
        score += 3
    elif 46 <= len(text) <= 90:
        score += 1
    if emotion in {'happy', 'laughter', 'surprised'}:
        score += 3
    if any(mark in text for mark in ['哈', '笑', '不要', '喜欢', '妈妈', '爸爸', '宝宝']):
        score += 2
    if re.search(r'[？！!?]', text):
        score += 1
    return score

def build_growth_dictionary(items, speaker_filter=None):
    """Build a lightweight growth dictionary and quote board from transcripts."""
    speaker_counts = Counter()
    term_counts = Counter()
    term_first_seen = {}
    term_examples = {}
    speaker_phrase_counts = defaultdict(Counter)
    quotes = []
    daily_stats = defaultdict(lambda: {'terms': set(), 'quotes': 0})

    filtered_segments = 0
    for item in items:
        item_dt = _item_datetime(item)
        date_key = item_dt.strftime('%Y-%m-%d') if item_dt else '未知日期'
        for seg in item.get('segments') or []:
            speaker = seg.get('spk') or 'Unknown'
            text = (seg.get('text') or '').strip()
            emotion = seg.get('emotion') or ''
            if not text:
                continue

            speaker_counts[speaker] += 1
            if speaker_filter and speaker != speaker_filter:
                continue

            filtered_segments += 1
            terms = _growth_text_terms(text)
            unique_terms = set(terms)
            term_counts.update(unique_terms)
            speaker_phrase_counts[speaker].update(unique_terms)
            daily_stats[date_key]['terms'].update(unique_terms)

            for term in unique_terms:
                if term not in term_first_seen or (item_dt and item_dt < term_first_seen[term]):
                    term_first_seen[term] = item_dt or datetime.datetime.max
                    term_examples[term] = {
                        'term': term,
                        'speaker': speaker,
                        'time': item_dt.isoformat() if item_dt else None,
                        'text': text
                    }

            score = _quote_score(text, speaker, emotion, speaker_filter)
            if score >= 4 and 4 <= len(text) <= 90:
                quotes.append({
                    'text': text,
                    'speaker': speaker,
                    'emotion': emotion,
                    'time': item_dt.isoformat() if item_dt else None,
                    'score': score,
                    'filename': item.get('filename', '')
                })
                daily_stats[date_key]['quotes'] += 1

    recurring_terms = {term: count for term, count in term_counts.items() if count >= 2}
    top_terms = [
        {'term': term, 'count': count}
        for term, count in Counter(recurring_terms).most_common(80)
    ]

    new_terms = []
    for term, example in term_examples.items():
        count = term_counts.get(term, 0)
        if count < 2:
            continue
        new_terms.append({
            **example,
            'count': count
        })
    new_terms.sort(key=lambda item: item.get('time') or '', reverse=True)

    catchphrases = []
    for speaker, counter in speaker_phrase_counts.items():
        for term, count in counter.most_common(8):
            if count >= 2:
                catchphrases.append({
                    'speaker': speaker,
                    'term': term,
                    'count': count
                })
    catchphrases.sort(key=lambda item: item['count'], reverse=True)

    deduped_quotes = []
    seen_quote_text = set()
    for quote in sorted(quotes, key=lambda item: (item['score'], item.get('time') or ''), reverse=True):
        key = quote['text']
        if key in seen_quote_text:
            continue
        seen_quote_text.add(key)
        deduped_quotes.append(quote)
        if len(deduped_quotes) >= 40:
            break

    timeline = [
        {
            'date': date,
            'new_term_count': len(data['terms']),
            'quote_count': data['quotes']
        }
        for date, data in sorted(daily_stats.items())
        if date != '未知日期'
    ]

    return {
        'speakers': [{'name': speaker, 'segments': count} for speaker, count in speaker_counts.most_common()],
        'selected_speaker': speaker_filter or '',
        'stats': {
            'segments': filtered_segments,
            'unique_terms': len(recurring_terms),
            'quotes': len(deduped_quotes),
            'catchphrases': len(catchphrases)
        },
        'top_terms': top_terms[:40],
        'new_terms': new_terms[:40],
        'quotes': deduped_quotes,
        'catchphrases': catchphrases[:30],
        'timeline': timeline[-60:]
    }

# --- HTML 模板 ---

# =================== 登录认证 ===================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        password = request.form.get('password', '')
        if hmac.compare_digest(password, REQUIRED_PASSWORD):
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='密码错误')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

# =================== 声纹管理API转发 ===================
ASR_SERVER_URL = CONFIG["ASR_API_URL"].rsplit("/", 1)[0]

@app.route('/speaker/register', methods=['POST'])
@login_required
def proxy_register_speaker():
    """转发声纹注册请求到ASR服务器"""
    try:
        # 转发文件和表单数据
        files = {}
        if 'audio_file' in request.files:
            audio_file = request.files['audio_file']
            files['audio_file'] = (audio_file.filename, audio_file.stream, audio_file.content_type)

        data = {
            'speaker_name': request.form.get('speaker_name', ''),
            'source_key': request.form.get('source_key', '')
        }

        response = requests.post(
            f"{ASR_SERVER_URL}/speaker/register",
            files=files,
            data=data,
            headers=_asr_admin_headers(),
            timeout=30
        )

        return Response(response.content, status=response.status_code, content_type=response.headers.get('Content-Type'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/voiceprint_registered')
@login_required
def proxy_voiceprint_registered():
    """已注册声纹样本清单（供记录页标注已入库的段）"""
    try:
        response = requests.get(f"{ASR_SERVER_URL}/api/voiceprint_registered",
                                headers=_asr_admin_headers(), timeout=8)
        return Response(response.content, status=response.status_code,
                        content_type=response.headers.get('Content-Type', 'application/json'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =================== 负样本反馈（电视/动画声音标记） ===================
SPK_NEG_FEEDBACK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spk_negative_feedback.json')

def _load_spk_negative_paths():
    """已标记为负样本的句子切片路径集合（用于展示层过滤）"""
    try:
        with open(SPK_NEG_FEEDBACK_FILE, 'r', encoding='utf-8') as f:
            return set(json.load(f))
    except Exception:
        return set()

def _add_spk_negative_path(path):
    try:
        cur = _load_spk_negative_paths()
        cur.add(path)
        with open(SPK_NEG_FEEDBACK_FILE, 'w', encoding='utf-8') as f:
            json.dump(sorted(cur), f, ensure_ascii=False, indent=1)
    except Exception as e:
        print(f"[spk-negative] 反馈记录写盘失败: {e}")
    # 单日可见数缓存失效 (负样本隐藏会影响当天角标)
    m = re.search(r'(\d{4}-\d{2}-\d{2})', path or '')
    if m:
        _rec_vis_invalidate_date(m.group(1))

# =================== 记录Tab日期角标: 可见录音数 ===================
_REC_VIS_CACHE_KEY = 'records:visible_counts'

def _valkey_client():
    import valkey
    uri = os.environ.get('VALKEY_URI', '')
    return valkey.from_url(uri) if uri else None

def _rec_vis_invalidate_date(date_str):
    """负样本变动后失效单日可见数缓存 (下次请求重扫该日)"""
    try:
        r = _valkey_client()
        if not r:
            return
        raw = r.get(_REC_VIS_CACHE_KEY)
        if not raw:
            return
        data = json.loads(raw)
        if date_str in (data.get('raw') or {}):
            data['raw'].pop(date_str, None)
            data['vis'].pop(date_str, None)
            r.set(_REC_VIS_CACHE_KEY, json.dumps(data), ex=86400 * 2)
    except Exception:
        pass

def _rec_vis_invalidate_all():
    """负样本删除后整表失效 (下次请求全量重算)"""
    try:
        r = _valkey_client()
        if r:
            r.delete(_REC_VIS_CACHE_KEY)
    except Exception:
        pass

def _rec_scan_visible(dates):
    """扫描指定日期的转写行, 返回 {date: 可见录音数} (过滤规则与移动端前端一致)"""
    neg = _load_spk_negative_paths()
    conn = get_connection()
    if not conn:
        return {}
    vis = {}
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT COALESCE(recording_time, created_at)::date AS d, segments_json
            FROM transcriptions
            WHERE COALESCE(recording_time, created_at)::date = ANY(%s)
            """,
            ([datetime.date.fromisoformat(d) for d in dates],)
        )
        for d, sj in cur.fetchall():
            d = str(d)
            if d not in vis:
                vis[d] = 0
            try:
                segs = json.loads(sj or '[]')
            except Exception:
                segs = []
            for s in segs:
                spk = str(s.get('spk') or '').strip()
                if not spk or spk.lower() in ('unknown', 'baby'):
                    continue
                if s.get('segment_audio_path') in neg:
                    continue
                vis[d] += 1
                break   # 只要有一句可见, 该录音就计入角标
        cur.close()
    finally:
        return_connection(conn)
    return vis

def _rec_visible_counts():
    """每日期可见录音数 (Valkey 缓存): 原始行数有变化的日期才重扫, 老日期吃缓存"""
    try:
        r = _valkey_client()
    except Exception:
        r = None
    cached = None
    if r:
        try:
            raw = r.get(_REC_VIS_CACHE_KEY)
            cached = json.loads(raw) if raw else None
        except Exception:
            cached = None
    today = datetime.date.today().isoformat()
    conn = get_connection()
    if not conn:
        return {}
    raw_counts = {}
    try:
        cur = conn.cursor()
        cur.execute("SELECT COALESCE(recording_time, created_at)::date AS d, COUNT(*) FROM transcriptions GROUP BY d")
        raw_counts = {str(row[0]): int(row[1]) for row in cur.fetchall()}
        cur.close()
    finally:
        return_connection(conn)
    vis, cached_raw = {}, {}
    if cached and isinstance(cached.get('vis'), dict) and cached.get('today') == today:
        vis = dict(cached['vis'])
        cached_raw = cached.get('raw') or {}
    for d in list(vis):
        if d not in raw_counts:
            vis.pop(d)
    dirty = [d for d, n in raw_counts.items() if cached_raw.get(d) != n]
    if dirty:
        try:
            vis.update(_rec_scan_visible(dirty))
        except Exception as e:
            logger_web.error(f"[rec-vis] 重扫可见数失败: {e}")
            if not vis:
                return {}
    if r:
        try:
            r.set(_REC_VIS_CACHE_KEY, json.dumps({'today': today, 'raw': raw_counts, 'vis': vis}), ex=86400 * 2)
        except Exception:
            pass
    return vis

@app.route('/api/spk_negative', methods=['POST'])
@login_required
def api_spk_negative():
    """移动端标记负样本: 先持久记录路径(展示层即时隐藏), 再转发 ASR 服务后台建模"""
    try:
        if 'audio_file' not in request.files:
            return jsonify({"error": "audio_file is required"}), 400
        audio_file = request.files['audio_file']
        src = request.form.get('source_path', '')
        if src:
            _add_spk_negative_path(src)   # 不等 GPU, 隐藏意图立即落盘
        files = {'audio_file': (audio_file.filename, audio_file.stream, audio_file.content_type)}
        data = {'source_path': src}
        response = requests.post(
            f"{ASR_SERVER_URL}/speaker/negative",
            files=files, data=data, headers=_asr_admin_headers(), timeout=30
        )
        return Response(response.content, status=response.status_code, content_type=response.headers.get('Content-Type'))
    except Exception as e:
        # ASR 服务不可达: 隐藏意图已保留, 仅建模未完成, 用户可稍后重标
        return jsonify({"ok": True, "queued": False, "hidden": True,
                        "warning": f"建模队列暂不可达: {e}"}), 200

@app.route('/speaker/negative/list', methods=['GET'])
@login_required
def proxy_negative_list():
    """负样本黑名单列表（供声纹 Tab 核对）"""
    try:
        response = requests.get(f"{ASR_SERVER_URL}/speaker/negative/list", timeout=10)
        return Response(response.content, status=response.status_code, content_type=response.headers.get('Content-Type'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/speaker/negative/<neg_id>/audio', methods=['GET'])
@login_required
def proxy_negative_audio(neg_id):
    """试听负样本音频副本"""
    try:
        response = requests.get(f"{ASR_SERVER_URL}/speaker/negative/{neg_id}/audio", timeout=30, stream=True)
        return Response(response.iter_content(chunk_size=8192), status=response.status_code,
                        content_type=response.headers.get('Content-Type', 'audio/wav'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/speaker/negative/<neg_id>', methods=['DELETE'])
@login_required
def proxy_negative_delete(neg_id):
    """从黑名单移除单条负样本（回滚）"""
    try:
        response = requests.delete(f"{ASR_SERVER_URL}/speaker/negative/{neg_id}", headers=_asr_admin_headers(), timeout=15)
        if response.status_code < 300:
            _rec_vis_invalidate_all()   # 黑名单变化影响可见数, 整表重算
        return Response(response.content, status=response.status_code, content_type=response.headers.get('Content-Type'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/speaker/list', methods=['GET'])
@login_required
def proxy_list_speakers():
    """转发获取说话人列表请求"""
    try:
        response = requests.get(f"{ASR_SERVER_URL}/speaker/list", timeout=10)
        return Response(response.content, status=response.status_code, content_type=response.headers.get('Content-Type'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/speaker/<speaker_name>', methods=['GET'])
@login_required
def proxy_speaker_samples(speaker_name):
    """转发获取指定说话人的样本列表"""
    try:
        response = requests.get(f"{ASR_SERVER_URL}/speaker/{speaker_name}", timeout=10)
        return Response(response.content, status=response.status_code, content_type=response.headers.get('Content-Type'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/speaker/<speaker_name>/sample/<sample_id>/audio', methods=['GET'])
@login_required
def proxy_speaker_sample_audio(speaker_name, sample_id):
    """转发样本音频流"""
    try:
        response = requests.get(f"{ASR_SERVER_URL}/speaker/{speaker_name}/sample/{sample_id}/audio", timeout=30, stream=True)
        return Response(response.iter_content(chunk_size=8192),
                       status=response.status_code,
                       content_type=response.headers.get('Content-Type', 'audio/wav'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/speaker/<speaker_name>/sample/<sample_id>', methods=['DELETE'])
@login_required
def proxy_delete_speaker_sample(speaker_name, sample_id):
    """转发删除单个样本请求"""
    try:
        response = requests.delete(f"{ASR_SERVER_URL}/speaker/{speaker_name}/sample/{sample_id}",
                                   timeout=15, headers=_asr_admin_headers())
        return Response(response.content, status=response.status_code, content_type=response.headers.get('Content-Type'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/speaker/<speaker_name>', methods=['DELETE'])
@login_required
def proxy_delete_speaker(speaker_name):
    """转发删除说话人请求"""
    try:
        response = requests.delete(f"{ASR_SERVER_URL}/speaker/{speaker_name}", timeout=10, headers=_asr_admin_headers())
        return Response(response.content, status=response.status_code, content_type=response.headers.get('Content-Type'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/speaker/audio/<path:filename>', methods=['GET'])
@login_required
def proxy_speaker_audio(filename):
    """转发音频文件请求"""
    try:
        response = requests.get(f"{ASR_SERVER_URL}/speaker/audio/{filename}", timeout=10, stream=True)
        return Response(response.iter_content(chunk_size=8192), 
                       status=response.status_code, 
                       content_type=response.headers.get('Content-Type'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/register_page')
@login_required
def proxy_register_page():
    """转发声纹注册页面"""
    try:
        response = requests.get(f"{ASR_SERVER_URL}/register_page", timeout=10)
        # 修改HTML中的API端点,指向本地5009端口
        html = response.text
        html = html.replace('http://localhost:5008/speaker/', 'http://localhost:5009/speaker/')
        return html
    except Exception as e:
        return f"<h1>Error loading speaker registration page</h1><p>{str(e)}</p>", 500

@app.route('/api/trigger_reprocess', methods=['POST'])
@login_required
def proxy_trigger_reprocess():
    try:
        date_param = request.args.get('date', '')
        start_time = request.args.get('start_time', '')
        end_time = request.args.get('end_time', '')
        replace_param = request.args.get('replace', 'false')
        url = f"{ASR_SERVER_URL}/api/trigger_reprocess?date={date_param}&start_time={start_time}&end_time={end_time}&replace={replace_param}"
        response = requests.post(url, timeout=10, headers=_asr_admin_headers())
        return Response(response.content, status=response.status_code, content_type=response.headers.get('Content-Type'))
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/stop_reprocess', methods=['POST'])
@login_required
def proxy_stop_reprocess():
    try:
        url = f"{ASR_SERVER_URL}/api/stop_reprocess"
        response = requests.post(url, headers=_asr_admin_headers(), timeout=10)
        return Response(response.content, status=response.status_code, content_type=response.headers.get('Content-Type'))
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/refresh_file_cache', methods=['POST'])
@login_required
def proxy_refresh_file_cache():
    try:
        url = f"{ASR_SERVER_URL}/api/refresh_file_cache"
        response = requests.post(url, headers=_asr_admin_headers(), timeout=30)  # 启动任务很快，不需要长超时
        return Response(response.content, status=response.status_code, content_type=response.headers.get('Content-Type'))
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/refresh_file_cache/status', methods=['GET'])
@login_required
def proxy_refresh_file_cache_status():
    try:
        url = f"{ASR_SERVER_URL}/api/refresh_file_cache/status"
        response = requests.get(url, timeout=10)
        return Response(response.content, status=response.status_code, content_type=response.headers.get('Content-Type'))
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/file_cache_status', methods=['GET'])
@login_required
def proxy_file_cache_status():
    try:
        url = f"{ASR_SERVER_URL}/api/file_cache_status"
        response = requests.get(url, timeout=10)
        return Response(response.content, status=response.status_code, content_type=response.headers.get('Content-Type'))
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/cry_events', methods=['GET'])
@login_required
def proxy_cry_events():
    try:
        limit = request.args.get('limit', 100)
        offset = request.args.get('offset', 0)
        date_filter = request.args.get('date', '')
        params = f"limit={limit}&offset={offset}"
        if date_filter:
            params += f"&date={date_filter}"
        response = requests.get(f"{ASR_SERVER_URL}/api/cry_events?{params}", timeout=10)
        return Response(response.content, status=response.status_code, content_type=response.headers.get('Content-Type'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/preview_progress', methods=['GET'])
@login_required
def proxy_preview_progress():
    """预切总进度代理（试听秒开覆盖率）"""
    try:
        response = requests.get(f"{ASR_SERVER_URL}/api/preview_progress", timeout=10)
        return Response(response.content, status=response.status_code, content_type=response.headers.get('Content-Type'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/cry_event/<int:event_id>', methods=['GET'])
@login_required
def proxy_cry_event(event_id):
    try:
        response = requests.get(f"{ASR_SERVER_URL}/api/cry_event/{event_id}", timeout=10)
        return Response(response.content, status=response.status_code, content_type=response.headers.get('Content-Type'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/cry_event/<int:event_id>/confirm_sample', methods=['POST'])
@login_required
def proxy_confirm_cry_sample(event_id):
    """代理"确认为声纹样本"：转发到 ASR 服务，后端含 NAS 复制+声纹提取，超时放宽到 120s"""
    try:
        response = requests.post(f"{ASR_SERVER_URL}/api/confirm_cry_sample/{event_id}", headers=_asr_admin_headers(), timeout=120)
        return Response(response.content, status=response.status_code, content_type=response.headers.get('Content-Type'))
    except requests.Timeout:
        return jsonify({"error": "后端处理超时（NAS 读取或声纹提取过慢），请稍后重试"}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/cry_event/<int:event_id>/segment_preview', methods=['GET'])
@login_required
def proxy_cry_segment_preview(event_id):
    """代理哭声片段试听：后端滑窗定位+切片，首次可能较慢，超时放宽到 120s。
    【2026-09-20】透传 variant（换一段候选序号），否则手机端换一段永远拿到第 1 段"""
    try:
        variant = request.args.get('variant', '0')
        response = requests.get(f"{ASR_SERVER_URL}/api/cry_segment_preview/{event_id}?variant={variant}", headers=_asr_admin_headers(), timeout=120)
        resp = Response(response.content, status=response.status_code, content_type=response.headers.get('Content-Type'))
        resp.headers['Cache-Control'] = 'no-store'  # 【2026-10-02】防 iOS 缓存旧响应导致试听假卡
        if response.headers.get('X-Cry-Windows'):
            resp.headers['X-Cry-Windows'] = response.headers['X-Cry-Windows']
        return resp
    except requests.Timeout:
        return jsonify({"error": "片段定位超时（滑窗扫描过慢），请稍后重试"}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/cry_event/<int:event_id>/feedback', methods=['POST'])
@login_required
def proxy_cry_event_feedback(event_id):
    """代理误报反馈：转发 verdict 到 ASR 服务，标记 false_positive 积累训练数据"""
    try:
        response = requests.post(f"{ASR_SERVER_URL}/api/cry_event_feedback/{event_id}", headers=_asr_admin_headers(), json=request.get_json(silent=True) or {}, timeout=15)
        return Response(response.content, status=response.status_code, content_type=response.headers.get('Content-Type'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── 免鉴权哭声报警预览（2026-09-20）──
# 供 Webhook 接收方通过 CF 隧道（rd.moco.fun）或局域网点开直接查看事件详情，
# 媒体（哭声片段/上下文录音/AI 插图）全部由 5009 同源直出，公网可达；
# 可选访问令牌：.env 设置 CRY_PREVIEW_TOKEN 后需带 ?t=，未设置则免鉴权。
# 仅暴露单个事件关联的只读媒体；正式面板（含管理操作）仍需登录。

def _preview_auth_ok():
    """预览资源访问校验：未配置 CRY_PREVIEW_TOKEN 时完全开放（免鉴权）"""
    tok = (os.getenv('CRY_PREVIEW_TOKEN') or '').strip()
    if not tok:
        return True
    return request.args.get('t', '') == tok

def _fetch_preview_event(event_id):
    """拉取事件详情（5008 侧公开接口）；失败返回 None"""
    try:
        r = requests.get(f"{ASR_SERVER_URL}/api/cry_event/{event_id}", timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

_CRY_PREVIEW_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>宝宝哭声报警预览</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, "PingFang SC", sans-serif; background: linear-gradient(160deg,#1a1c2e,#252840); min-height: 100vh; color: #e8e8f0; padding: 24px 16px; }
  .card { max-width: 520px; margin: 0 auto; background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.1); border-radius: 20px; padding: 24px; }
  h1 { font-size: 20px; margin-bottom: 4px; }
  .sub { color: #9aa0b5; font-size: 13px; margin-bottom: 18px; }
  .badge { display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: 12px; font-weight: 700; margin-right: 8px; background: rgba(239,68,68,.18); color: #f87171; border: 1px solid rgba(239,68,68,.3); }
  .conf { background: rgba(255,255,255,.08); color: #cbd5e1; border: 1px solid rgba(255,255,255,.12); }
  .sec { margin-top: 18px; }
  .sec h2 { font-size: 13px; color: #9aa0b5; font-weight: 600; margin-bottom: 8px; letter-spacing: .05em; }
  .reason { font-size: 15px; line-height: 1.6; }
  .advice { font-size: 14px; line-height: 1.6; color: #fbbf24; }
  audio { width: 100%; margin-top: 6px; }
  .illu { width: 100%; border-radius: 14px; border: 1px solid rgba(255,255,255,.1); }
  .file { color: #6b7280; font-size: 12px; margin-top: 14px; word-break: break-all; }
  a.full { display: block; text-align: center; margin-top: 20px; color: #818cf8; font-size: 13px; text-decoration: none; }
  .btn-row { display: flex; gap: 10px; }
  .btn { flex: 1; padding: 12px 8px; border-radius: 14px; border: none; font-size: 14px; font-weight: 700; cursor: pointer; transition: all .15s; }
  .btn:disabled { opacity: .6; }
  .btn.ok { background: rgba(16,185,129,.15); color: #34d399; border: 1px solid rgba(16,185,129,.3); }
  .btn.deny { background: rgba(239,68,68,.15); color: #f87171; border: 1px solid rgba(239,68,68,.3); }
  .hint { color: #6b7280; font-size: 12px; margin-top: 8px; line-height: 1.5; }
  .pv-badge { display: inline-block; padding: 6px 14px; border-radius: 999px; font-size: 13px; font-weight: 700; }
  .pv-badge.ok { background: rgba(16,185,129,.12); color: #34d399; }
  .pv-badge.deny { background: rgba(239,68,68,.12); color: #f87171; }
</style>
</head>
<body>
<div class="card">
  <h1>🍼 宝宝哭声报警</h1>
  <div class="sub">{{ recording_time }}</div>
  <div>
    <span class="badge">{{ reason_category }}</span>
    {% if confidence %}<span class="badge conf">置信度 {{ confidence }}</span>{% endif %}
  </div>
  {% if illustration %}
  <div class="sec"><h2>AI 场景插图</h2><img class="illu" src="{{ illustration }}" alt="场景插图"></div>
  {% endif %}
  <div class="sec">
    <h2>哭声片段试听</h2>
    <audio id="pvSegAudio" controls preload="none" src="/preview/cry/{{ event_id }}/audio{{ q }}"></audio>
    <button id="pvNextSegBtn" onclick="pvNextSegment()" class="btn" style="margin-top:8px;background:rgba(255,255,255,.06);color:#cbd5e1;border:1px solid rgba(255,255,255,.15)">🔄 换一段（自动定位可能不准）</button>
  </div>
  {% if reason %}
  <div class="sec"><h2>原因分析</h2><div class="reason">{{ reason }}</div></div>
  {% endif %}
  {% if advice %}
  <div class="sec"><h2>安抚建议</h2><div class="advice">{{ advice }}</div></div>
  {% endif %}
  {% if context_audios %}
  <div class="sec">
    <h2>上下文录音（事件前后）</h2>
    {% for a in context_audios %}<audio controls preload="none" src="{{ a }}"></audio>{% endfor %}
  </div>
  {% endif %}
  <div class="file">📁 {{ filename }}</div>
  <div class="sec">
    <h2>反馈操作</h2>
    {% if sample_confirmed %}
    <span class="pv-badge ok">✓ 已确认样本</span>
    {% elif false_positive %}
    <span class="pv-badge deny">🚩 已标误报</span>
    {% else %}
    <div class="btn-row">
      <button id="pvConfirmBtn" onclick="pvConfirm()" class="btn ok">🧬 确认为样本</button>
      <button id="pvFeedbackBtn" onclick="pvFeedback()" class="btn deny">🚩 误报</button>
    </div>
    <div class="hint">确认为样本：将该哭声入库声纹库，提升后续检测准确度（约需十几秒）。误报：标记为假阳性，用于后续规则校准。</div>
    {% endif %}
  </div>
  <a class="full" href="/">打开完整监控面板 →</a>
</div>
<script>
// 哭声片段候选轮换：确认样本时会带上当前选中的 variant，入库的就是听到的这段
let _pvVariant = 0;
async function pvNextSegment() {
  _pvVariant += 1;  // 超出总数由服务端取模
  const audio = document.getElementById('pvSegAudio');
  const btn = document.getElementById('pvNextSegBtn');
  btn.disabled = true; btn.textContent = '⏳ 定位中...';
  try {
    const sep = '{{ q }}' ? '&' : '?';
    audio.src = '/preview/cry/{{ event_id }}/audio' + '{{ q }}' + sep + 'variant=' + _pvVariant;
    try { await audio.play(); } catch (e) {}
    btn.textContent = '🔄 再换一段';
  } catch (e) { alert('❌ 切换失败: ' + e.message); }
  btn.disabled = false;
}
async function pvConfirm() {
  const btn = document.getElementById('pvConfirmBtn');
  btn.disabled = true; btn.textContent = '处理中...（声纹提取约十几秒）';
  try {
    const resp = await fetch('/preview/cry/{{ event_id }}/confirm_sample{{ q }}', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({variant: _pvVariant}) });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || ('HTTP ' + resp.status));
    let msg = data.message || '已确认为样本';
    if (data.sample_count) msg += '\n当前 Baby 共 ' + data.sample_count + ' 个样本';
    if (data.low_similarity_warning && data.low_similarity_warning.length) msg += '\n\n⚠️ 低相似度提醒:\n· ' + data.low_similarity_warning.join('\n· ');
    alert('✅ ' + msg);
    btn.outerHTML = '<span class="pv-badge ok">✓ 已确认样本</span>';
    const fb = document.getElementById('pvFeedbackBtn'); if (fb) fb.remove();
  } catch (e) {
    alert('❌ 确认失败: ' + e.message);
    btn.disabled = false; btn.textContent = '🧬 确认为样本';
  }
}
async function pvFeedback() {
  if (!confirm('确定这是误报吗？标记后将作为负样本用于检测准确度训练。')) return;
  const btn = document.getElementById('pvFeedbackBtn');
  btn.disabled = true; btn.textContent = '标记中...';
  try {
    const resp = await fetch('/preview/cry/{{ event_id }}/feedback{{ q }}', { method: 'POST' });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || ('HTTP ' + resp.status));
    alert('✅ 已标记为误报，感谢反馈');
    btn.outerHTML = '<span class="pv-badge deny">🚩 已标误报</span>';
    const cb = document.getElementById('pvConfirmBtn'); if (cb) cb.remove();
  } catch (e) {
    alert('❌ 标记失败: ' + e.message);
    btn.disabled = false; btn.textContent = '🚩 误报';
  }
}
</script>
</body>
</html>
"""

@app.route('/preview/cry/<int:event_id>')
def public_cry_preview(event_id):
    """免登录哭声事件预览页：插图 + 信息 + 哭声片段/上下文试听（可经 CF 隧道公网访问，只读）"""
    if not _preview_auth_ok():
        return "<h3 style='font-family:sans-serif;padding:24px'>无效的预览令牌</h3>", 403
    ev = _fetch_preview_event(event_id)
    if not ev:
        return "<h3 style='font-family:sans-serif;padding:24px'>事件不存在或已删除</h3>", 404
    # 所有媒体走 5009 同源代理，保证经 CF 隧道公网访问时可用（5008 端口不对外）
    q = f"?t={request.args.get('t', '')}" if (os.getenv('CRY_PREVIEW_TOKEN') or '').strip() else ""
    ctx_urls = [f"/preview/cry/{event_id}/media{u[len('/api/audio'):]}{q}"
                for u in (ev.get('audio_urls') or []) if u.startswith('/api/audio/')]
    illustration = ev.get('illustration_url') or ''
    if illustration.startswith('data:'):
        pass  # 内联图片直接使用
    elif illustration.startswith('/api/illustration/'):
        illustration = f"/preview/cry/{event_id}/illustration{q}"
    else:
        illustration = ''
    conf = ev.get('confidence')
    html = render_template_string(
        _CRY_PREVIEW_HTML,
        event_id=event_id,
        q=q,
        illustration=illustration,
        recording_time=ev.get('recording_time') or '',
        reason_category=ev.get('reason_category') or '未分类',
        confidence=f"{conf * 100:.0f}%" if isinstance(conf, (int, float)) else '',
        reason=ev.get('reason') or '',
        advice=ev.get('advice') or '',
        filename=ev.get('filename') or '',
        context_audios=ctx_urls,
        sample_confirmed=bool(ev.get('sample_confirmed')),
        false_positive=bool(ev.get('false_positive')),
    )
    # 手机浏览器/微信 webview 常缓存旧页面，显式禁缓存保证按钮等改动即时生效
    resp = make_response(html)
    resp.headers['Cache-Control'] = 'no-store, max-age=0'
    return resp

@app.route('/preview/cry/<int:event_id>/audio')
def public_cry_preview_audio(event_id):
    """免登录哭声片段音频流：服务端注入管理令牌从 5008 拉取滑窗定位的哭声片段"""
    if not _preview_auth_ok():
        return jsonify({"error": "无效的预览令牌"}), 403
    try:
        variant = request.args.get('variant', '0')
        r = requests.get(f"{ASR_SERVER_URL}/api/cry_segment_preview/{event_id}?variant={variant}", headers=_asr_admin_headers(), timeout=120)
        resp = Response(r.content, status=r.status_code, content_type=r.headers.get('Content-Type', 'audio/wav'))
        if r.headers.get('X-Cry-Windows'):
            resp.headers['X-Cry-Windows'] = r.headers['X-Cry-Windows']
        return resp
    except requests.Timeout:
        return jsonify({"error": "片段定位超时，请稍后刷新重试"}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/preview/cry/<int:event_id>/media/<path:media_path>')
def public_cry_preview_media(event_id, media_path):
    """免登录事件上下文录音流：仅允许访问该事件关联的音频文件（防止开放全量录音库）。
    直接读 NAS（复用面板 /api/audio 的 processed/根目录两级解析），不经 5008。"""
    if not _preview_auth_ok():
        return jsonify({"error": "无效的预览令牌"}), 403
    ev = _fetch_preview_event(event_id)
    if not ev:
        return jsonify({"error": "事件不存在"}), 404
    allowed = {u[len('/api/audio/'):] for u in (ev.get('audio_urls') or []) if u.startswith('/api/audio/')}
    if media_path not in allowed:
        return jsonify({"error": "该文件不属于此事件"}), 403
    try:
        # 【双源】新式设备级路径（Pixel-6/2026-10-02/x.m4a）直接定位；
        # 旧式路径（2026-09-20/x.m4a）逐设备 processed/ 优先探测
        parts = media_path.replace('\\', '/').split('/')
        if '..' in parts:
            return jsonify({"error": "Invalid path"}), 403
        cands = []
        if len(parts) >= 3 and parts[0] in SOURCE_DEVICES:
            dev, rest = parts[0], '/'.join(parts[1:])
            # 【2026-10-04 多根】本地镜像优先，NAS 历史回退
            for _root in RECORDS_ROOTS:
                cands += [os.path.join(_root, dev, "processed", rest),
                          os.path.join(_root, media_path)]
        else:
            for dev in SOURCE_DEVICES:
                for _root in RECORDS_ROOTS:
                    cands += [os.path.join(_root, dev, "processed", media_path),
                              os.path.join(_root, dev, media_path)]
        for cand in cands:
            if os.path.isfile(cand):
                return send_file(cand)
        return jsonify({"error": "媒体文件不存在"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/preview/cry/<int:event_id>/illustration')
def public_cry_preview_illustration(event_id):
    """免登录 AI 场景插图：仅允许访问该事件关联的插图文件"""
    if not _preview_auth_ok():
        return jsonify({"error": "无效的预览令牌"}), 403
    ev = _fetch_preview_event(event_id)
    if not ev:
        return jsonify({"error": "事件不存在"}), 404
    ill = ev.get('illustration_url') or ''
    if not ill.startswith('/api/illustration/'):
        return jsonify({"error": "该事件无插图"}), 404
    fname = ill[len('/api/illustration/'):]
    try:
        r = requests.get(f"{ASR_SERVER_URL}/api/illustration/{fname}", timeout=30)
        return Response(r.content, status=r.status_code, content_type=r.headers.get('Content-Type', 'image/jpeg'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/preview/cry/<int:event_id>/confirm_sample', methods=['POST'])
def public_cry_preview_confirm(event_id):
    """预览页"确认为样本"：令牌保护，服务端注入管理令牌代理到 5008（NAS 复制+声纹提取，耗时较长）"""
    if not _preview_auth_ok():
        return jsonify({"error": "无效的预览令牌"}), 403
    ev = _fetch_preview_event(event_id)
    if not ev:
        return jsonify({"error": "事件不存在"}), 404
    try:
        r = requests.post(f"{ASR_SERVER_URL}/api/confirm_cry_sample/{event_id}", headers=_asr_admin_headers(), json=request.get_json(silent=True) or {}, timeout=180)
        return Response(r.content, status=r.status_code, content_type=r.headers.get('Content-Type'))
    except requests.Timeout:
        return jsonify({"error": "处理超时（NAS 读取或声纹提取过慢），请稍后重试"}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/preview/cry/<int:event_id>/feedback', methods=['POST'])
def public_cry_preview_feedback(event_id):
    """预览页"误报"标记：令牌保护，服务端注入管理令牌代理到 5008"""
    if not _preview_auth_ok():
        return jsonify({"error": "无效的预览令牌"}), 403
    ev = _fetch_preview_event(event_id)
    if not ev:
        return jsonify({"error": "事件不存在"}), 404
    try:
        r = requests.post(f"{ASR_SERVER_URL}/api/cry_event_feedback/{event_id}", headers=_asr_admin_headers(), json={"verdict": "false_positive"}, timeout=30)
        return Response(r.content, status=r.status_code, content_type=r.headers.get('Content-Type'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/illustration/<filename>')
@login_required
def proxy_illustration(filename):
    """代理插图请求到 ASR 服务器"""
    try:
        response = requests.get(f"{ASR_SERVER_URL}/api/illustration/{filename}", timeout=10)
        return Response(response.content, status=response.status_code, content_type=response.headers.get('Content-Type'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/live_status', methods=['GET'])
@login_required
def proxy_live_status():
    try:
        response = requests.get(f"{ASR_SERVER_URL}/api/live_status", timeout=5)
        return Response(response.content, status=response.status_code, content_type=response.headers.get('Content-Type'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/live_logs', methods=['GET'])
@login_required
def proxy_live_logs():
    try:
        response = requests.get(f"{ASR_SERVER_URL}/api/live_logs", timeout=5)
        return Response(response.content, status=response.status_code, content_type=response.headers.get('Content-Type'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

_last_rep_date_cache = {"ts": 0.0, "val": None}

def _last_reprocess_date():
    """哭声补跑上次重检推进到的录音日期（processed_files_a 最新 b_reprocess_success）。60s 缓存。"""
    import re as _re
    now = time.time()
    if now - _last_rep_date_cache["ts"] < 60:
        return _last_rep_date_cache["val"]
    val = None
    try:
        import psycopg2 as _pg
        conn = _pg.connect(CONFIG["DATABASE_URL"], connect_timeout=5)
        cur = conn.cursor()
        cur.execute("SELECT filename FROM processed_files_a WHERE status='b_reprocess_success' ORDER BY processed_at DESC LIMIT 1")
        row = cur.fetchone()
        if row:
            m = _re.search(r'(20\d{2}-\d{2}-\d{2})', row[0])
            val = m.group(1) if m else None
        cur.close(); conn.close()
    except Exception:
        pass
    _last_rep_date_cache.update(ts=now, val=val)
    return val


@app.route('/api/overview', methods=['GET'])
@login_required
def proxy_overview():
    """系统总览：代理到 ASR 服务器（设备健康为服务端 5 分钟缓存，响应轻、可随轮询拉）
    【2026-10-04】修复：本函数此前只挂了 @login_required，漏了 @app.route → /api/overview 恒 404，
    导致 A 轨补跑卡副标题（下次运行/上次推进到）与 B 轨积压/吞吐/今日入库 chip 永远拿不到数据。"""
    try:
        response = requests.get(f"{ASR_SERVER_URL}/api/overview", timeout=8)
        if response.ok:
            try:
                data = response.json()
                a = data.get("a_track")
                if isinstance(a, dict):
                    a["last_reprocess_date"] = _last_reprocess_date()
                return jsonify(data)
            except Exception:
                pass
        return Response(response.content, status=response.status_code, content_type=response.headers.get('Content-Type'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

_dev_map_cache = {"ts": 0.0, "map": {}}

def _build_device_map():
    """文件名→设备 映射（近3天 watched+processed+failed）, 60s 缓存。
    用于给 5008 日志里的纯文件名（无设备字段）补设备归属。"""
    import datetime as _dt
    now = time.time()
    if now - _dev_map_cache["ts"] < 60 and _dev_map_cache["map"]:
        return _dev_map_cache["map"]
    m = {}
    dates = [(_dt.date.today() - _dt.timedelta(days=k)).isoformat() for k in range(3)]
    for dev in SOURCE_DEVICES[:2]:
        for date in dates:
            # 目录结构: <dev>/<日期>/(watched) 与 <dev>/processed|failed/<日期>/
            # 【2026-10-04 多根】本地镜像（watched 瞬态+processed/failed）+ NAS（历史回退）
            for _root in RECORDS_ROOTS:
                for d in (os.path.join(_root, dev, date),
                          os.path.join(_root, dev, "processed", date),
                          os.path.join(_root, dev, "failed", date)):
                    try:
                        names = os.listdir(d)
                    except Exception:
                        continue
                    for n in names:
                        if n.endswith('.m4a'):
                            m.setdefault(n, dev)  # watched 优先于 processed/failed
    if m:
        _dev_map_cache["ts"] = now
        _dev_map_cache["map"] = m
    return m


def _short_with_dev(filename, dev_map):
    """`Pixel-6/10-04_14-16-08` 短格式；设备未知时退回纯时间尾"""
    base = _norm_task_name(filename)
    dev = dev_map.get(os.path.basename(str(filename)))
    return f"{dev}/{base}" if dev else base


def _scan_rt_pending():
    """实时流待提交文件: 今天日期目录(watched)里还没被 audio_processor 提交走的 m4a。
    audio_processor 提交成功后会把文件移入 processed/, 所以目录里剩的即待处理队列。
    返回 (每设备队头2个短格式, 总数)。SMB 异常时返回 ([], 0)。"""
    import datetime as _dt
    today = _dt.date.today().isoformat()
    result = []
    total = 0
    for dev in SOURCE_DEVICES[:2]:  # 实时流 = 双 Pixel (Sony 已停录)
        dev_dir = os.path.join(RECORDS_ROOT, dev, today)
        try:
            names = sorted(n for n in os.listdir(dev_dir) if n.endswith('.m4a'))
        except Exception:
            continue
        total += len(names)
        for n in names[:2]:  # 每台设备显示队头 2 个（两台串行消化, 各自进度可见）
            result.append(f"{dev}/{_norm_task_name(n)}")
    return result, total


@app.route('/api/parallel_lights', methods=['GET'])
@login_required
def api_parallel_lights():
    """首页并行任务信号灯: 一次轮询返回各后台线程/进程存活状态"""
    import socket, subprocess
    lights = []
    core_on = False
    try:
        core_on = requests.get(f"{ASR_SERVER_URL}/manage", timeout=2).status_code == 200
    except Exception:
        pass
    lights.append({"key": "core", "label": "5008主服务", "on": core_on})

    a_running = b_running = False
    if core_on:
        try:
            st = requests.get(f"{ASR_SERVER_URL}/api/live_status", timeout=3).json()
            a_running = bool(st.get('a_running'))
            b_running = bool(st.get('b_running'))
        except Exception:
            pass
    lights.append({"key": "trackb", "label": "语音转写", "on": b_running})
    lights.append({"key": "tracka", "label": "检测补跑", "on": a_running})

    def _pgrep_alive(pattern):
        try:
            return subprocess.run(['pgrep', '-f', pattern], capture_output=True, timeout=3).returncode == 0
        except Exception:
            return False

    lights.append({"key": "preset", "label": "预切批次", "on": _pgrep_alive('batch_preset_previews')})

    nano_on = False
    try:
        with socket.create_connection(('127.0.0.1', 8123), timeout=1):
            nano_on = True
    except Exception:
        pass
    lights.append({"key": "nano", "label": "Nano推理", "on": nano_on})
    processing, completed = _scan_processing_files()
    backfill = _scan_backfill_progress()
    try:
        rt_files, rt_count = _scan_rt_pending()
    except Exception:
        rt_files, rt_count = [], 0
    # 待处理 → 处理中 流转衔接: 正被 5008 处理的文件从「待处理列表」剔除（归一化比对,
    # backfill 队列为 `设备/02-21-13-45` 短格式, 5008 日志为 `2026-10-02_02-21-13-45.m4a` 全名）
    try:
        proc_norms = {_norm_task_name(p["filename"]) for p in processing}
        if backfill and backfill.get("pending_files"):
            backfill["pending_files"] = [x for x in backfill["pending_files"]
                                         if _norm_task_name(x.rsplit('/', 1)[-1]) not in proc_norms]
        # 来源标签: 两条队列并行 —— 补救(backfill, 历史日期) / 实时(audio_processor, 今天)
        import datetime as _dt
        _today = _dt.datetime.now().strftime("%Y-%m-%d")
        dev_map = _build_device_map()
        for p in processing + completed:
            import re as _re2
            dm = _re2.search(r'TermuxAudioRecording_(\d{4}-\d{2}-\d{2})_', p.get("filename", ""))
            p["src"] = "实时" if (dm and dm.group(1) == _today) else "补救"
            p["short"] = _short_with_dev(p.get("filename", ""), dev_map)
    except Exception:
        pass
    resp = jsonify({"lights": lights, "processing": processing,
                    "completed": completed, "backfill": backfill,
                    "pixel_backlog": _pixel_backlog_snapshot(),
                    "rt": {"pending_files": rt_files, "count": rt_count}})
    resp.headers['Cache-Control'] = 'no-store'  # 【2026-10-04】防 Safari 缓存旧轮询导致补救/处理状态滞后
    return resp


# ==================== Pixel 转录补救：真实待补救数 + 手动启动【2026-10-04】====================
# 待补救 = processed/<日期>/ 下已归档、但 transcriptions 表里没有的文件（重启后 catch-up
# 「历史掉队防线」归档但未分析的积压）。用 backfill_pixels.py --from-processed --dry-run
# 实测得出，确保与真正启动补救时的判定完全同一套，不重复实现避免口径分叉。
_ASR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_pixel_backlog = {"count": None, "scanned_at": 0, "error": None}
_pixel_backlog_lock = threading.Lock()
_pixel_backfill_proc = None


def _pixel_backlog_scan_once():
    """子进程 dry-run 实测待补救数（NAS 扫描 + transcriptions 比对），结果写缓存。"""
    global _pixel_backlog
    script = os.path.join(_ASR_ROOT, "backfill_pixels.py")
    count, err = None, None
    try:
        p = subprocess.run(
            [sys.executable, "-u", script, "--from-processed", "--dates", "auto", "--dry-run"],
            cwd=_ASR_ROOT, capture_output=True, text=True, timeout=120)
        m = re.search(r"待回填:\s*(\d+)\s*个文件", p.stdout or "")
        if m:
            count = int(m.group(1))
        else:
            err = ((p.stdout or "") + (p.stderr or "")).strip()[-200:] or "扫描无输出"
    except subprocess.TimeoutExpired:
        err = "扫描超时（NAS 可能不可达）"
    except Exception as e:
        err = str(e)
    with _pixel_backlog_lock:
        _pixel_backlog = {"count": count, "scanned_at": time.time(), "error": err}


def _pixel_backlog_loop():
    """【2026-10-05】由"每 5 分钟盲扫"改为"事件驱动 + 每小时兜底"。
    单次 dry-run 要拉 transcriptions(~4万行)+processed_files_a(~27万行) 全表比对，
    5 分钟一次纯属重复浪费；待补救数只在两个时机变化——补救批次跑完、或新归档产生。
    故：批次 running→结束 时立即刷新，另加 1 小时兜底刷新。"""
    _pixel_backlog_scan_once()
    was_running = _pixel_backfill_running()
    while True:
        time.sleep(60)
        running = _pixel_backfill_running()
        if was_running and not running:
            _pixel_backlog_scan_once()   # 补救批次刚结束 → 立刻刷新
        elif time.time() - _pixel_backlog_snapshot()["scanned_at"] >= 3600:
            _pixel_backlog_scan_once()   # 兜底：至少每小时一次
        was_running = running


def _pixel_backlog_snapshot():
    with _pixel_backlog_lock:
        return dict(_pixel_backlog)


def _pixel_backfill_running():
    """补救进程存活 = Popen 未退出 或 日志 120s 内有更新（与 _scan_backfill_progress 同口径）"""
    if _pixel_backfill_proc is not None and _pixel_backfill_proc.poll() is None:
        return True
    try:
        log_path = os.path.join(_ASR_ROOT, "log", "backfill_rerun.log")
        return (time.time() - os.path.getmtime(log_path)) < 120
    except Exception:
        return False


@app.route('/api/pixel_backfill/start', methods=['POST'])
@login_required
def start_pixel_backfill():
    """启动 Pixel 转录补救（backfill_pixels.py --from-processed），输出追加到
    log/backfill_rerun.log —— 仪表盘「Pixel 转录补救」任务卡即解析该日志展示进度。"""
    global _pixel_backfill_proc
    if _pixel_backfill_running():
        return jsonify({"error": "补救任务已在运行中"}), 409
    script = os.path.join(_ASR_ROOT, "backfill_pixels.py")
    if not os.path.exists(script):
        return jsonify({"error": f"脚本不存在: {script}"}), 500
    try:
        log_dir = os.path.join(_ASR_ROOT, "log")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "backfill_rerun.log")
        f = open(log_path, "a", encoding="utf-8")
        f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🚀 手机端启动 Pixel 转录补救\n")
        f.flush()
        # start_new_session: 脱离 5009 进程组，5009 重启不牵连已在跑的补救
        _pixel_backfill_proc = subprocess.Popen(
            [sys.executable, "-u", script, "--from-processed", "--dates", "auto"],
            cwd=_ASR_ROOT, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        return jsonify({"message": "Pixel 转录补救已在后台启动"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _norm_task_name(s):
    """文件名归一化为 `10-02_02-21-13-45` 短格式（剥设备前缀/录制器前缀/年份/扩展名, 保留月-日）"""
    import re as _re
    base = str(s).replace('\\', '/').rsplit('/', 1)[-1]
    base = base.replace('TermuxAudioRecording_', '').replace('.m4a', '')
    base = _re.sub(r'^\d{4}-', '', base)  # 去年份: 2026-10-02_... → 10-02_...
    return base


def _scan_processing_files():
    """解析 5008 处理日志，找出正在处理的转录任务 + 最近已完成的任务（含耗时与结论）。
    配对闭合标记：
    - audio_processor 客户端的「✅ 转录完成: <文件名> (N 字)」
    - 5008 服务端的「✅ 数据库保存成功 (recording_time: ...)」/「⭕ 无有效语音段 (recording_time: ...)」
      （backfill_pixels.py 提交的文件没有前者，只能靠服务端标记闭合）
    实时步骤归因：5008 的生命周期行（预处理/VAD/声纹）不含文件名，归属「最近收到的任务」，
    双客户端并发交错时可能短暂归错，随下一行日志自动纠正——仅作状态展示可接受。
    返回 (processing, completed)：
      processing=[{filename, elapsed_s, step}], completed=[{filename, elapsed_s, result}]"""
    import re
    from datetime import datetime
    log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "log", "launchd-asr-server.err")
    processing, completed = [], []
    try:
        active = {}          # filename -> {"ts": 收到时刻, "step": 当前步骤, "segs": 段数, "cry": 是否检出哭声, "chars": 字数}
        rec_time_map = {}    # recording_time字符串 -> filename
        latest = [None]      # 最近收到的文件名（生命周期行归因用, list 以便闭包内改写）
        rt_re = re.compile(r'(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})')

        def _close(fn, ts):
            """闭合任务：生成结论摘要, 从 active 移除并记入 completed"""
            info = active.pop(fn, None)
            if info is None:
                return
            parts = []
            if info.get("cry"):
                parts.append("🍼哭声")
            if info.get("chars") is not None and info["chars"] > 0:
                parts.append(f"{info['chars']}字")
            elif info.get("segs"):
                asg = info.get("assigned")
                parts.append(f"{info['segs']}段" + (f"·归{asg}" if asg is not None else ""))
            else:
                parts.append("无语音")
            completed.append({"filename": fn, "elapsed_s": max(1, int(ts - info["ts"])),
                              "result": "·".join(parts)})

        with open(log_path, 'rb') as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 256 * 1024))  # 尾部 256KB 覆盖并发任务窗口
            f.readline()
            for raw in f:
                line = raw.decode('utf-8', errors='replace').strip()
                m = re.match(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                if not m:
                    continue
                ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
                cur = active.get(latest[0]) if latest[0] else None
                if '📥 收到转录任务: ' in line:
                    fn = line.split('📥 收到转录任务: ', 1)[1].strip()
                    active[fn] = {"ts": ts, "step": "收到任务", "segs": 0, "cry": False, "chars": None}
                    latest[0] = fn
                    rm = rt_re.search(fn)
                    if rm:
                        rec_time_map[f"{rm.group(1)} {rm.group(2)}:{rm.group(3)}:{rm.group(4)}"] = fn
                elif '✅ 转录完成: ' in line:
                    tail = line.split('✅ 转录完成: ', 1)[1]
                    fn, _, nchar = tail.rpartition(' (')
                    fn = fn.strip()
                    if fn in active:
                        cm = re.search(r'(\d+) 字', nchar)
                        if cm:
                            active[fn]["chars"] = int(cm.group(1))
                    _close(fn, ts)
                elif '✅ 数据库保存成功 (recording_time: ' in line:
                    rt = line.split('recording_time: ', 1)[1].rstrip(')').strip()
                    fn = rec_time_map.get(rt)
                    if fn:
                        _close(fn, ts)
                        rec_time_map.pop(rt, None)
                elif '⭕ 无有效语音段, 跳过入库 (recording_time: ' in line:
                    rt = line.split('recording_time: ', 1)[1].rstrip(')').strip()
                    fn = rec_time_map.get(rt)
                    if fn:
                        _close(fn, ts)
                        rec_time_map.pop(rt, None)
                elif '⭕ 夜间降级仅哭声检测, 跳过转写入库 (recording_time: ' in line:
                    # 【2026-10-04 夜间降级】凌晨录音只做哭声检测不入库，同样按闭合处理
                    rt = line.split('recording_time: ', 1)[1].rstrip(')').strip()
                    fn = rec_time_map.get(rt)
                    if fn:
                        _close(fn, ts)
                        rec_time_map.pop(rt, None)
                elif '📊 归属统计: ' in line:
                    # 例: 📊 归属统计: 23段 · Unknown18 · 妈妈5
                    if cur:
                        am = re.search(r':\s*(\d+)段', line)
                        if am:
                            total = int(am.group(1))
                            um = re.search(r'Unknown(\d+)', line)
                            unk = int(um.group(1)) if um else 0
                            cur["assigned"] = max(0, total - unk)
                elif '🍼 [轨道A] 哭声确认' in line:
                    if cur:
                        cur["cry"] = True
                        cur["step"] = "🍼 哭声确认"
                elif '[生命周期: 1. 音频预处理] 开始' in line:
                    if cur: cur["step"] = "预处理"
                elif '[生命周期: 2. VAD & ASR] 开始' in line:
                    if cur: cur["step"] = "切分+转写"
                elif 'VAD检出 ' in line and ' 个分段' in line:
                    if cur:
                        sm = re.search(r'VAD检出 (\d+) 个分段', line)
                        if sm:
                            cur["segs"] = int(sm.group(1))
                            cur["step"] = f"切出 {sm.group(1)} 段"
                elif '[生命周期: 3. 逐段声纹识别] 开始' in line:
                    if cur: cur["step"] = f"声纹识别 0/{cur.get('segs') or '?'}"
                elif '[生命周期: 3. 逐段声纹识别] 完成' in line:
                    if cur: cur["step"] = "声纹识别完成, 入库中"
                elif re.search(r'\[3\.\d+\] 处理分段', line):
                    if cur:
                        nm = re.search(r'\[3\.(\d+)\]', line)
                        if nm:
                            cur["step"] = f"声纹识别 {nm.group(1)}/{cur.get('segs') or '?'}"
        now = time.time()
        processing = [{"filename": fn, "elapsed_s": int(now - info["ts"]), "step": info["step"]}
                      for fn, info in sorted(active.items(), key=lambda x: -x[1]["ts"])]
        completed = completed[-6:]
    except Exception:
        pass
    return processing, completed


def _scan_backfill_progress():
    """解析 backfill 日志（Pixel=backfill_rerun.log / Sony=backfill_sony.log）最新进度，
    并用「队列预览」行 + 逐文件 ✔/✗ 行推算接下来待处理的文件。
    格式: 进度 60/3678 | 成功 58 失败 2 | 1.5 个/分钟 | 剩余约 41.6 小时
          队列预览: Pixel-5/xxx.m4a, Pixel-6/yyy.m4a, ...
          ✔ Pixel-5/xxx.m4a (1/3678)  /  ✗ Pixel-6/yyy.m4a HTTP 500
    返回 None 表示 backfill 未运行或无进度信息。"""
    import re
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "log")
    prog_re = re.compile(r'进度 (\d+)/(\d+) \| 成功 (\d+) 失败 (\d+).*?剩余约 ([\d.]+) (小时|分钟)')
    done_re = re.compile(r'回填完成: 成功 (\d+) / 失败 (\d+) / 总 (\d+)')
    batch_re = re.compile(r'待回填:\s*(\d+)\s*个文件')
    ts_re = re.compile(r'\[(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})\]')
    perfile_re = re.compile(r'\((\d+)/(\d+)\)')
    batches = []  # 各批次（Pixel/Sony）独立解析后聚合，两客户端可能并存
    result = None
    try:
        for name in ("backfill_rerun.log", "backfill_sony.log"):
            log_path = os.path.join(log_dir, name)
            if not os.path.exists(log_path):
                continue
            try:
                # 日志过旧=该批次早已结束（如已退役并取消的 Sony 日志），不计入聚合，
                # 否则其陈旧的「回填完成…总 N」会把当前 Pixel 批次的总数算大（曾 78+2757=2835）。
                if time.time() - os.path.getmtime(log_path) > 6 * 3600:
                    continue
            except Exception:
                pass
            snapshot, gone = [], set()
            # 以「待回填: N 个文件」为新批次起点，之后只认本批次的 ✔/✗ 与「进度」行。
            # 【2026-10-04】旧实现只看最后一条「进度/回填完成」行，而脚本每 20 个文件才打一次
            # 「进度」，新一轮首条进度行出现前会一直回退显示上一轮的陈旧数字——曾出现
            # 「补救中 4,435/4,435 · 剩约0min」但实际只跑了 78 个（用户当场看到并质疑）。
            b_started, b_total, b_ok, b_fail, b_done = False, 0, 0, 0, 0
            b_start_epoch = None
            b_prog = None       # (done,total,ok,fail,eta_h) 来自「进度」行
            b_completed = None  # (ok,fail,total) 来自「回填完成」行
            legacy = None       # 无「待回填」行时（旧格式日志）的兜底
            with open(log_path, 'rb') as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 256 * 1024))
                f.readline()
                for raw in f:
                    line = raw.decode('utf-8', errors='replace').strip()
                    if '队列预览: ' in line:
                        # 每次预览行重置快照（新一轮启动）
                        snapshot = [x.strip() for x in line.split('队列预览: ', 1)[1].split(',') if x.strip()]
                        gone = set()
                        continue
                    mb = batch_re.search(line)
                    if mb:
                        # 新批次开始：重置本批次计数，丢弃上一批次的进度
                        b_started = True
                        b_total, b_ok, b_fail, b_done = int(mb.group(1)), 0, 0, 0
                        b_prog = b_completed = None
                        mt = ts_re.search(line)
                        b_start_epoch = None
                        if mt:
                            try:
                                b_start_epoch = time.mktime((2026,) + tuple(map(int, mt.groups())) + (0, 0, -1))
                            except Exception:
                                b_start_epoch = None
                        continue
                    m = prog_re.search(line)
                    if m:
                        vals = (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)),
                                float(m.group(5)) if m.group(6) == '小时' else float(m.group(5)) / 60)
                        if b_started:
                            b_prog = vals
                        else:
                            legacy = vals
                        continue
                    m = done_re.search(line)
                    if m:
                        if b_started:
                            b_completed = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
                        else:
                            t = int(m.group(3))
                            legacy = (t, t, int(m.group(1)), int(m.group(2)), 0.0)
                        continue
                    if snapshot:
                        if ' ✔ ' in line:
                            gone.add(line.rsplit(' ✔ ', 1)[1].split(' (')[0].strip())
                        elif ' ✗ ' in line:
                            gone.add(line.rsplit(' ✗ ', 1)[1].split(' HTTP')[0].strip())
                        elif '📤 提交: ' in line:
                            # 已提交=正被处理, 从待处理列表剔除（✔/✗ 要等响应回来才有）
                            gone.add(line.split('📤 提交: ', 1)[1].strip())
                    if b_started and (' ✔ ' in line or ' ✗ ' in line):
                        if ' ✔ ' in line:
                            b_ok += 1
                        else:
                            b_fail += 1
                        mp = perfile_re.search(line)
                        if mp:
                            b_done = max(b_done, int(mp.group(1)))
                            b_total = int(mp.group(2)) or b_total
            vals = None
            if b_started:
                if b_completed:
                    ok, fail, total = b_completed
                    vals = (total, total, ok, fail, 0.0)
                elif b_prog:
                    p_done, p_total, p_ok, p_fail, p_eta = b_prog
                    vals = (max(p_done, b_done), p_total or b_total,
                            max(p_ok, b_ok), max(p_fail, b_fail), p_eta)
                else:
                    eta_h = 0.0
                    if b_start_epoch and b_done > 0 and b_total > b_done:
                        secs = time.time() - b_start_epoch
                        if secs > 0:
                            eta_h = (b_total - b_done) / (b_done / secs) / 3600
                    vals = (b_done, b_total, b_ok, b_fail, eta_h)
            elif legacy:
                vals = legacy
            if vals:
                done, total, ok, fail, eta_h = vals
                try:
                    running = (time.time() - os.path.getmtime(log_path)) < 120  # 日志2分钟内有更新=进程活着
                except Exception:
                    running = False
                batches.append({
                    "done": done, "total": total,
                    "remaining": max(0, total - done),
                    "ok": ok, "fail": fail,
                    "running": running,
                    "eta_h": eta_h,
                    "pending_files": [f"{x.split('/', 1)[0]}/{_norm_task_name(x)}" if '/' in x else _norm_task_name(x)
                                      for x in snapshot if x not in gone][:3],
                })
        if batches:
            eta_sum = sum(b["eta_h"] for b in batches)
            result = {
                "done": sum(b["done"] for b in batches),
                "total": sum(b["total"] for b in batches),
                "remaining": sum(b["remaining"] for b in batches),
                "ok": sum(b["ok"] for b in batches),
                "fail": sum(b["fail"] for b in batches),
                "eta": f"约{eta_sum:.1f}h" if eta_sum >= 1 else f"约{eta_sum*60:.0f}min",
                "running": any(b.get("running") for b in batches),
                "pending_files": [f for b in batches for f in b["pending_files"]][:3],
            }
    except Exception:
        pass
    return result

@app.route('/api/reprocess_history', methods=['POST'])
@login_required
def proxy_start_reprocess():
    try:
        response = requests.post(f"{ASR_SERVER_URL}/api/trigger_reprocess", timeout=5, headers=_asr_admin_headers())
        return Response(response.content, status=response.status_code, content_type=response.headers.get('Content-Type'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/start_live', methods=['POST'])
@login_required
def proxy_start_live():
    try:
        response = requests.post(f"{ASR_SERVER_URL}/api/start_live", headers=_asr_admin_headers(), timeout=5)
        return Response(response.content, status=response.status_code, content_type=response.headers.get('Content-Type'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/pause_live', methods=['POST'])
@login_required
def proxy_pause_live():
    try:
        response = requests.post(f"{ASR_SERVER_URL}/api/pause_live", headers=_asr_admin_headers(), timeout=5)
        return Response(response.content, status=response.status_code, content_type=response.headers.get('Content-Type'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stop_live', methods=['POST'])
@login_required
def proxy_stop_live():
    try:
        response = requests.post(f"{ASR_SERVER_URL}/api/stop_live", headers=_asr_admin_headers(), timeout=5)
        return Response(response.content, status=response.status_code, content_type=response.headers.get('Content-Type'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/scan_dates', methods=['GET'])
@login_required
def api_scan_dates():
    """从刷盘缓存中提取日期统计信息（不重新扫描文件系统）"""
    try:
        # 检查是否需要强制刷新缓存
        force_refresh = request.args.get('refresh') == '1'
        if force_refresh:
            debug_log("强制刷新，清除缓存...")
            clear_date_stats_in_redis()

        # 先从 Valkey 加载缓存的日期统计
        date_info = get_date_stats_from_redis()
        if date_info:
            debug_log(f"从 Valkey 缓存加载了 {len(date_info)} 个日期")
        else:
            # 缓存为空，从刷盘的文件路径缓存中提取
            debug_log("Valkey 日期缓存为空，从刷盘缓存中提取...")
            filepaths = get_file_cache_from_redis()  # 从刷盘缓存获取
            if filepaths:
                debug_log(f"从刷盘缓存获取到 {len(filepaths)} 个文件路径")
                
                # 统计每个日期的文件数
                from collections import defaultdict
                date_counts = defaultdict(int)
                date_pattern = re.compile(r'(\d{4}-\d{2}-\d{2})')
                
                for fp in filepaths:
                    filepath = fp if isinstance(fp, str) else fp.get('filepath', '')
                    match = date_pattern.search(filepath)
                    if match:
                        date_str = match.group(1)
                        date_counts[date_str] += 1
                
                # 构建 date_info
                for date_str, count in sorted(date_counts.items()):
                    date_info[date_str] = {
                        'fileCount': count,
                        'processedCount': 0,
                        'status': 'pending'
                    }
                
                debug_log(f"统计到 {len(date_info)} 个日期")
                
                # 保存到 Valkey 缓存
                if date_info:
                    if save_date_stats_to_redis(date_info):
                        debug_log(f"✅ 已缓存 {len(date_info)} 个日期到 Valkey")
                    else:
                        debug_log("❌ 保存缓存到 Valkey 失败")
            else:
                debug_log("❌ 刷盘缓存也为空，需要先行刷盘")
                return jsonify({
                    "status": "error",
                    "message": "缓存为空，请先执行文件刷盘",
                    "dates": [],
                    "date_info": {}
                }), 200

        # 查询数据库获取已处理的进度
        try:
            conn = get_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute('SELECT filename FROM processed_files_a')
                rows = cursor.fetchall()
                debug_log(f"查询到 {len(rows)} 条处理记录")
                
                date_counts = {}
                # 匹配两种格式：2025-11-07 或 20251115
                date_pattern = re.compile(r'(\d{4}-\d{2}-\d{2})|(\d{4}\d{2}\d{2})')
                for row in rows:
                    filename = row[0]
                    match = date_pattern.search(filename)
                    if match:
                        # 提取日期（优先用带 - 的格式）
                        d = match.group(1) if match.group(1) else match.group(2)
                        # 如果是 20251115 格式，转换为 2025-11-15
                        if len(d) == 8:
                            d = f"{d[:4]}-{d[4:6]}-{d[6:]}"
                        date_counts[d] = date_counts.get(d, 0) + 1
                
                debug_log(f"统计到 {len(date_counts)} 个日期有处理记录")
                
                for date_str, cnt in date_counts.items():
                    if date_str in date_info:
                        date_info[date_str]['processedCount'] = cnt
                        debug_log(f"{date_str}: {cnt} 条记录")
                cursor.close()
                return_connection(conn)
        except Exception as db_err:
            debug_log(f"查询处理进度失败：{db_err}")

        # 更新状态
        for date_str, info in date_info.items():
            if info['processedCount'] == 0:
                info['status'] = 'pending'
            elif info['processedCount'] >= info['fileCount']:
                info['status'] = 'completed'
            else:
                # 部分完成：区分"处理中"(processing)和"暂停"(paused)
                # 当前实现只根据 processedCount 判断，缺少进程心跳信息
                # 重启后无法判断进程是否还在运行，暂时统一用 paused 表示非活跃状态
                info['status'] = 'paused'

        # 按日期升序排序
        sorted_dates = sorted(date_info.keys(), key=lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))

        return jsonify({
            "status": "success",
            "dates": sorted_dates,
            "date_info": date_info,
            "total": len(sorted_dates)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/transcript_dates')
@login_required
def api_transcript_dates():
    """有可见录音的日期列表 (角标=与展示一致的可见数, 已过滤 Unknown/Baby/负样本)"""
    try:
        vis = _rec_visible_counts()
        dates = [{'date': d, 'count': n} for d, n in vis.items() if n > 0]
        dates.sort(key=lambda x: x['date'], reverse=True)
        return jsonify({"dates": dates})
    except Exception as e:
        logger_web.error(f"[Error] 查询转写日期失败: {e}")
        return jsonify({"dates": []})


@app.route('/api/event_audio/<int:event_id>', methods=['GET'])
@login_required
def proxy_event_audio(event_id):
    try:
        response = requests.get(f"{ASR_SERVER_URL}/api/event_audio/{event_id}", timeout=10)
        return Response(response.content, status=response.status_code, content_type=response.headers.get('Content-Type'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/babycry/progress', methods=['POST'])
@login_required
def api_save_babycry_progress():
    """保存宝宝哭声分析进度"""
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        all_dates = data.get('all_dates', [])
        loaded_count = data.get('loaded_count', 0)
        has_more = data.get('has_more', True)
        processing_date = data.get('current_date')
        dates_state = data.get('dates_state', {})

        success = save_analysis_progress(
            session_id=session_id,
            all_dates=all_dates,
            loaded_count=loaded_count,
            has_more=has_more,
            current_date=processing_date,
            dates_state=dates_state
        )
        
        if success:
            return jsonify({"status": "success", "message": "进度已保存"})
        else:
            return jsonify({"status": "error", "message": "保存失败"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/babycry/progress', methods=['GET'])
@login_required
def api_load_babycry_progress():
    """加载宝宝哭声分析进度"""
    try:
        session_id = request.args.get('session_id', 'default')
        progress = load_analysis_progress(session_id)
        
        if progress:
            # 转换字段名以兼容前端
            data = {
                'all_dates': progress['all_dates'],
                'loaded_count': progress['loaded_count'],
                'has_more': progress['has_more'],
                'current_date': progress['processing_date'],  # 前端期望 current_date
                'dates_state': progress['dates_state']
            }
            return jsonify({"status": "success", "data": data})
        else:
            return jsonify({"status": "success", "data": None, "message": "没有找到保存的进度"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/babycry/progress', methods=['DELETE'])
@login_required
def api_clear_babycry_progress():
    """清除宝宝哭声分析进度"""
    try:
        session_id = request.args.get('session_id', 'default')
        success = clear_analysis_progress(session_id)
        
        if success:
            return jsonify({"status": "success", "message": "进度已清除"})
        else:
            return jsonify({"status": "error", "message": "清除失败"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/fix_recording_time', methods=['POST'])
@login_required
def api_fix_recording_time():
    """修复数据库中 recording_time 为 NULL 的记录"""
    try:
        fixed_count = fix_recording_time()
        return jsonify({
            "status": "success",
            "message": f"已修复 {fixed_count} 条记录",
            "fixed_count": fixed_count
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/status')
@login_required
def api_status():
    return jsonify(get_system_status())

@app.route('/api/data')
@login_required
def api_data():
    """获取转录数据，支持分页"""
    offset = request.args.get('offset', 0, type=int)
    limit = request.args.get('limit', 20, type=int)
    return jsonify(get_transcripts(offset=offset, limit=limit))

@app.route('/api/data/range')
@login_required
def api_data_range():
    """
    按时间范围查询转录数据
    参数:
        start_date: 开始日期 (YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS)
        end_date: 结束日期 (YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS)
        offset: 分页偏移量 (可选,默认0)
        limit: 每页数量 (可选,默认100)
    
    示例:
        /api/data/range?start_date=2025-11-27&end_date=2025-11-27
        /api/data/range?start_date=2025-11-27 00:00:00&end_date=2025-11-27 23:59:59
    """
    try:
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        offset = request.args.get('offset', 0, type=int)
        limit = request.args.get('limit', 100, type=int)
        
        if not start_date_str or not end_date_str:
            return jsonify({
                "error": "Missing required parameters",
                "message": "Both start_date and end_date are required",
                "example": "/api/data/range?start_date=2025-11-27&end_date=2025-11-27"
            }), 400
        
        # 解析日期
        try:
            # 尝试解析完整日期时间格式
            if len(start_date_str) > 10:
                start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d %H:%M:%S')
            else:
                # 只有日期,设置为当天开始
                start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
            
            if len(end_date_str) > 10:
                end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d %H:%M:%S')
            else:
                # 只有日期,设置为当天结束
                end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d')
                end_date = end_date.replace(hour=23, minute=59, second=59)
        except ValueError as e:
            return jsonify({
                "error": "Invalid date format",
                "message": str(e),
                "expected_format": "YYYY-MM-DD or YYYY-MM-DD HH:MM:SS"
            }), 400
        
        # SQL 按时间范围直接分页 (全量历史可查, 修复老日期点开为空的问题)
        _conn = get_connection()
        if not _conn:
            return jsonify({"error": "数据库连接失败"}), 500
        _total_in_range = 0
        all_items = []
        try:
            _cur = _conn.cursor()
            _cur.execute(
                """
                SELECT COUNT(*) FROM transcriptions
                WHERE COALESCE(recording_time, created_at) >= %s
                  AND COALESCE(recording_time, created_at) <= %s
                """,
                (start_date, end_date)
            )
            _total_in_range = _cur.fetchone()[0]
            _cur.execute(
                """
                SELECT id, filename, created_at, full_text, segments_json, recording_time, device
                FROM transcriptions
                WHERE COALESCE(recording_time, created_at) >= %s
                  AND COALESCE(recording_time, created_at) <= %s
                ORDER BY COALESCE(recording_time, created_at) DESC
                OFFSET %s LIMIT %s
                """,
                (start_date, end_date, offset, limit)
            )
            for row in _cur.fetchall():
                _segs = []
                try:
                    _segs = json.loads(row[4]) if row[4] else []
                except Exception:
                    _segs = []
                all_items.append({
                    'id': row[0], 'filename': row[1],
                    'created_at': row[2].isoformat() if row[2] else None,
                    'full_text': row[3], 'segments': _segs,
                    'recording_time': row[5].isoformat() if row[5] else None,
                    'device': row[6],
                })
            _cur.close()
        finally:
            return_connection(_conn)
        
        # 过滤时间范围内的数据
        filtered_items = []
        for item in all_items:
            filename = item.get('filename', '')
            dt = None
            
            # 优先使用数据库中的 recording_time
            if item.get('recording_time'):
                try:
                    dt = datetime.datetime.fromisoformat(item.get('recording_time'))
                except:
                    pass
            
            # 如果没有 recording_time，尝试解析文件名中的时间戳
            if dt is None:
                time_patterns = [
                    r'^\s*(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})\s*',
                    r'^\s*recording-(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})\s*',
                    r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})'
                ]
                
                for pattern in time_patterns:
                    match = re.match(pattern, os.path.splitext(filename)[0])
                    if match:
                        try:
                            if pattern == time_patterns[0]:
                                date_part = match.group(1) + '-' + match.group(2) + '-' + match.group(3)
                                time_part = match.group(4) + ':' + match.group(5) + ':' + match.group(6)
                                dt_str = f"{date_part} {time_part}"
                                dt = datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
                            else:
                                year, month, day, hour, minute, second = match.groups()
                                dt_str = f"{year}-{month}-{day} {hour}:{minute}:{second}"
                                dt = datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
                            break
                        except ValueError:
                            continue
            
            # 最后尝试使用created_at
            if dt is None:
                try:
                    dt = datetime.datetime.fromisoformat(item.get('created_at', ''))
                except:
                    continue
            
            # 检查是否在时间范围内
            if dt and start_date <= dt <= end_date:
                item['parsed_time'] = dt.isoformat()
                item['date_group'] = dt.strftime('%Y-%m-%d')
                item['time_simple'] = dt.strftime('%H:%M')
                item['time_full'] = dt.strftime('%Y-%m-%d %H:%M:%S')
                filtered_items.append(item)
        
        # 按时间倒序排序
        filtered_items.sort(key=lambda x: x.get('parsed_time', ''), reverse=True)

        # 应用分页
        total_count = _total_in_range   # SQL COUNT(*) (时间范围内全量, 不受分页影响)
        paginated_items = filtered_items   # SQL 已完成 OFFSET/LIMIT 分页

        # 展示层负样本过滤: 被用户标为"不是家里任何人"的句子 → spk 置 Unknown（前端自动隐藏）
        try:
            neg_paths = _load_spk_negative_paths()
            if neg_paths:
                for item in paginated_items:
                    for seg in (item.get('segments') or []):
                        if seg.get('segment_audio_path') in neg_paths:
                            seg['spk'] = 'Unknown'
        except Exception:
            pass

        return jsonify({
            "transcripts": paginated_items,
            "meta": {
                "total_count": total_count,
                "offset": offset,
                "limit": limit,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "returned_count": len(paginated_items)
            }
        })
    
    except Exception as e:
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route('/api/daily_report')
@login_required
def api_daily_report():
    """每日声音小报：按日期聚合转录、声纹、情绪与哭声事件。"""
    try:
        date_str = request.args.get('date')
        if not date_str:
            date_str = datetime.datetime.now().strftime('%Y-%m-%d')

        try:
            datetime.datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            return jsonify({
                'error': 'Invalid date format',
                'message': 'date must be YYYY-MM-DD'
            }), 400

        return jsonify(build_daily_report(date_str))
    except Exception as e:
        logger_web.error(f"[DailyReport] 生成失败: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/api/emotion-timeline')
@login_required
def api_emotion_timeline():
    """获取情感时间线数据"""
    try:
        from collections import Counter
        
        # 情感评分映射
        emotion_scores = {
            "happy": 1.0,
            "neutral": 0.0,
            "sad": -0.8,
            "angry": -1.0
        }
        
        # 获取所有转录记录
        all_items = db_get_transcripts(offset=0, limit=10000)
        
        # 按日期分组统计
        daily_data = {}
        
        for item in all_items:
            # 获取日期
            dt = None
            if item.get('recording_time'):
                try:
                    dt = datetime.datetime.fromisoformat(item.get('recording_time'))
                except:
                    pass
            
            if dt is None:
                try:
                    dt = datetime.datetime.fromisoformat(item.get('created_at', ''))
                except:
                    continue
            
            date_key = dt.strftime('%Y-%m-%d')
            
            # 初始化日期数据
            if date_key not in daily_data:
                daily_data[date_key] = {
                    'emotions': Counter(),
                    'total_segments': 0
                }
            
            # 统计情感
            segments = item.get('segments', [])
            for seg in segments:
                emotion = seg.get('emotion')
                if emotion:
                    daily_data[date_key]['emotions'][emotion] += 1
                    daily_data[date_key]['total_segments'] += 1
        
        # 计算每日情感分数
        timeline = []
        for date_key in sorted(daily_data.keys()):
            data = daily_data[date_key]
            emotions = dict(data['emotions'])
            
            # 计算加权情感分数
            total = sum(emotions.values())
            if total > 0:
                weighted_sum = sum(emotions.get(e, 0) * emotion_scores.get(e, 0) for e in emotions)
                score = weighted_sum / total
            else:
                score = 0.0
            
            timeline.append({
                'date': date_key,
                'score': round(score, 3),
                'emotions': emotions,
                'total_segments': data['total_segments']
            })
        
        return jsonify({'timeline': timeline})
        
    except Exception as e:
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/api/heatmap')
@login_required
def api_heatmap():
    """获取对话热力图数据（24小时 x 说话人）"""
    try:
        # 获取所有转录记录
        all_items = db_get_transcripts(offset=0, limit=10000)
        
        # 初始化热力图数据
        hours = [f"{h:02d}:00" for h in range(24)]
        speaker_activity = {}  # {speaker: [hour0_count, hour1_count, ...]}
        
        for item in all_items:
            # 获取录音时间
            dt = None
            if item.get('recording_time'):
                try:
                    dt = datetime.datetime.fromisoformat(item.get('recording_time'))
                except:
                    pass
            
            if dt is None:
                try:
                    dt = datetime.datetime.fromisoformat(item.get('created_at', ''))
                except:
                    continue
            
            hour = dt.hour
            
            # 统计每个说话人在该小时的活跃度
            segments = item.get('segments', [])
            for seg in segments:
                speaker = seg.get('spk', 'Unknown')
                if speaker not in speaker_activity:
                    speaker_activity[speaker] = [0] * 24
                speaker_activity[speaker][hour] += 1
        
        # 构建响应
        speakers = sorted(speaker_activity.keys())
        data = []
        
        # 转换为热力图格式 [[hour_index, speaker_index, value], ...]
        for speaker_idx, speaker in enumerate(speakers):
            for hour_idx in range(24):
                count = speaker_activity[speaker][hour_idx]
                if count > 0:  # 只包含有数据的点
                    data.append([hour_idx, speaker_idx, count])
        
        return jsonify({
            'hours': hours,
            'speakers': speakers,
            'data': data,
            'max_value': max(max(counts) for counts in speaker_activity.values()) if speaker_activity else 0
        })
        
    except Exception as e:
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/api/relationship_graph')
@login_required
def api_relationship_graph():
    """获取家庭声音关系图谱数据。"""
    try:
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        all_items = db_get_transcripts(offset=0, limit=10000)

        if start_date_str and end_date_str:
            try:
                start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
                end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d').replace(hour=23, minute=59, second=59)
                all_items = [
                    item for item in all_items
                    if (dt := _item_datetime(item)) and start_date <= dt <= end_date
                ]
            except ValueError:
                return jsonify({
                    'error': 'Invalid date format',
                    'message': 'start_date and end_date must be YYYY-MM-DD'
                }), 400

        return jsonify(build_relationship_graph(all_items))
    except Exception as e:
        logger_web.error(f"[RelationshipGraph] 生成失败: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/api/growth_dictionary')
@login_required
def api_growth_dictionary():
    """成长词典 / 金句收藏候选：支持日期范围与说话人筛选。"""
    try:
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        speaker = (request.args.get('speaker') or '').strip()
        if speaker in {'all', '全部'}:
            speaker = ''

        all_items = db_get_transcripts(offset=0, limit=10000)

        if start_date_str and end_date_str:
            try:
                start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
                end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d').replace(hour=23, minute=59, second=59)
                all_items = [
                    item for item in all_items
                    if (dt := _item_datetime(item)) and start_date <= dt <= end_date
                ]
            except ValueError:
                return jsonify({
                    'error': 'Invalid date format',
                    'message': 'start_date and end_date must be YYYY-MM-DD'
                }), 400

        return jsonify(build_growth_dictionary(all_items, speaker_filter=speaker or None))
    except Exception as e:
        logger_web.error(f"[GrowthDictionary] 生成失败: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/api/config', methods=['GET'])
@login_required
def api_get_config():
    return jsonify(CONFIG)

@app.route('/api/config', methods=['POST'])
@login_required
def api_update_config():
    config_data = request.get_json(silent=True)
    if config_data:
        # 记录更新请求
        log_message = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] API Config Update - Received: {json.dumps(config_data)}"
        with open(CONFIG["LOG_FILE_PATH"], 'a', encoding='utf-8') as log_file:
            log_file.write(log_message + '\\n')
        
        # 首先从文件读取当前配置
        try:
            with open('config.json', 'r', encoding='utf-8') as f:
                file_config = json.load(f)
        except:
            file_config = {}
        
        # 更新文件配置
        for key in config_data:
            file_config[key] = config_data[key]
            # 同时更新内存中的CONFIG
            if key in CONFIG:
                CONFIG[key] = config_data[key]
        
        # 保存配置到文件
        with open('config.json', 'w', encoding='utf-8') as f:
            json.dump(file_config, f, indent=2, ensure_ascii=False)
        
        # 记录更新结果
        log_message = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] API Config Update - Saved to file: {json.dumps(file_config)}"
        with open(CONFIG["LOG_FILE_PATH"], 'a', encoding='utf-8') as log_file:
            log_file.write(log_message + '\\n')
        
        return jsonify(success=True, message="Configuration updated successfully")
    # 记录无效请求
    log_message = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] API Config Update - Invalid JSON data received"
    with open(CONFIG["LOG_FILE_PATH"], 'a', encoding='utf-8') as log_file:
        log_file.write(log_message + '\\n')
    return jsonify(success=False, message="Invalid JSON data"), 400


@app.route('/audio_segments/<path:filepath>')
@login_required
def serve_audio_segment(filepath):
    """提供音频片段文件（双源：逐设备 audio_segments/ 探测）"""
    try:
        rel_clean = filepath.replace('\\', '/').lstrip('/')
        full_path = _find_audio_segment(rel_clean)
        if not full_path:
            return jsonify({"error": "Audio segment not found"}), 404
        logger_web.info(f"[Audio] 请求: {filepath} -> {full_path}")
        return send_file(full_path, mimetype='audio/wav')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- Groq Whisper 重识别 (外部 ASR 对比/兜底, key 轮换) ----------------
_GROQ_STATE = {"cursor": 0}

def _groq_keys():
    raw = os.getenv('GROQ_API_KEYS', '')
    return [k.strip() for k in raw.split(',') if k.strip()]

@app.route('/api/retranscribe', methods=['POST'])
@login_required
def api_retranscribe():
    """对一句 VAD 切片用 Groq whisper 重识别, 返回对比文本 (登录保护)"""
    data = request.get_json(silent=True) or {}
    rel = (data.get('path') or '').strip()
    if not rel:
        return jsonify({"error": "missing path"}), 400
    rel_clean = rel.replace('\\', '/')
    for prefix in ('/audio_segments/', 'audio_segments/'):
        if rel_clean.startswith(prefix):
            rel_clean = rel_clean[len(prefix):]
            break
    full_path = _find_audio_segment(rel_clean)
    if not full_path:
        return jsonify({"error": "audio not found: " + rel_clean}), 404
    keys = _groq_keys()
    if not keys:
        return jsonify({"error": "GROQ_API_KEYS 未配置"}), 503
    model = os.getenv('GROQ_ASR_MODEL', 'whisper-large-v3-turbo')
    lang = (data.get('language') or 'zh').strip()
    last_err, n = None, len(keys)
    for i in range(n):
        key = keys[(_GROQ_STATE["cursor"] + i) % n]
        try:
            with open(full_path, 'rb') as f:
                resp = requests.post(
                    'https://api.groq.com/openai/v1/audio/transcriptions',
                    headers={'Authorization': f'Bearer {key}'},
                    files={'file': (os.path.basename(full_path), f, 'audio/wav')},
                    data={'model': model, 'language': lang, 'response_format': 'json'},
                    timeout=40,
                )
        except Exception as e:
            last_err = str(e)
            continue
        if resp.status_code == 200:
            _GROQ_STATE["cursor"] = (_GROQ_STATE["cursor"] + i + 1) % n
            try:
                return jsonify({"text": resp.json().get('text', ''), "model": model, "engine": "groq"})
            except Exception:
                last_err = "bad json"
                continue
        if resp.status_code in (401, 403, 429):
            last_err = f"HTTP {resp.status_code}"
            continue  # key 无效/限流 → 轮换下一个
        return jsonify({"error": f"Groq HTTP {resp.status_code}: {resp.text[:200]}"}), 502
    return jsonify({"error": f"全部 Groq key 失败: {last_err}"}), 502

@app.route('/api/nano_transcribe', methods=['POST'])
@login_required
def api_nano_transcribe():
    """对一句 VAD 切片用本地 Fun-ASR-Nano 重识别 (audiocpp_server 代理, 登录保护)"""
    data = request.get_json(silent=True) or {}
    rel = (data.get('path') or '').strip()
    if not rel:
        return jsonify({"error": "missing path"}), 400
    rel_clean = rel.replace('\\', '/')
    for prefix in ('/audio_segments/', 'audio_segments/'):
        if rel_clean.startswith(prefix):
            rel_clean = rel_clean[len(prefix):]
            break
    full_path = _find_audio_segment(rel_clean)
    if not full_path:
        return jsonify({"error": "audio not found: " + rel_clean}), 404
    nano_url = os.getenv('NANO_ASR_URL', 'http://127.0.0.1:8123/v1/audio/transcriptions')
    lang = (data.get('language') or 'auto').strip() or 'auto'
    try:
        with open(full_path, 'rb') as f:
            resp = requests.post(
                nano_url,
                files={'file': (os.path.basename(full_path), f, 'audio/wav')},
                data={'model': 'fun-asr-nano', 'language': lang},
                timeout=60,
            )
    except Exception as e:
        return jsonify({"error": f"本地 Nano 服务不可达: {e}"}), 503
    if resp.status_code == 200:
        try:
            return jsonify({"text": resp.json().get('text', ''), "model": "fun-asr-nano", "engine": "fun_asr_nano"})
        except Exception:
            return jsonify({"error": "Nano 返回非 JSON"}), 502
    return jsonify({"error": f"Nano HTTP {resp.status_code}: {resp.text[:200]}"}), 502

@app.route('/api/audio/<path:filepath>')
@login_required
def serve_original_audio(filepath):
    """提供原始或已处理的录音文件回放"""
    try:
        # 【2026-09-20】本地持久音频（temp_cry 暂存区）：audio_urls 形如
        # /api/audio/Users/mac/asr-server/temp_cry/cry_xxx.wav，此前只在 NAS 目录找 → 404 无声
        if filepath.startswith("Users/mac/asr-server/temp_cry/"):
            local_path = "/" + filepath
            if os.path.isfile(local_path):
                return send_file(local_path)
        # 【双源】新式设备级路径（Pixel-6/2026-10-02/x.m4a）直接定位；
        # 旧式路径逐设备 processed/ 优先、根目录次之
        parts = filepath.replace('\\', '/').split('/')
        if '..' in parts:
            return jsonify({"error": "Invalid path"}), 403
        cands = []
        if len(parts) >= 3 and parts[0] in SOURCE_DEVICES:
            dev, rest = parts[0], '/'.join(parts[1:])
            # 【2026-10-04 多根】本地镜像优先，NAS 历史回退
            for _root in RECORDS_ROOTS:
                cands += [os.path.join(_root, dev, "processed", rest),
                          os.path.join(_root, filepath)]
        else:
            cands = []
            for dev in SOURCE_DEVICES:
                for _root in RECORDS_ROOTS:
                    cands += [os.path.join(_root, dev, "processed", filepath),
                              os.path.join(_root, dev, filepath)]

        for cand in cands:
            if os.path.exists(cand) and os.path.isfile(cand):
                return send_file(cand)

        # 最后的兜底：如果只是文件名，尝试根据日期前缀搜索
        if '/' not in filepath and ('_' in filepath or '-' in filepath):
            import re
            match = re.search(r'(\d{4}-\d{2}-\d{2})', filepath)
            if not match:
                match = re.search(r'(\d{8})', filepath)

            if match:
                date_found = match.group(1)
                if '-' not in date_found:
                    date_found = f"{date_found[:4]}-{date_found[4:6]}-{date_found[6:8]}"

                for dev in SOURCE_DEVICES:
                    for _root in RECORDS_ROOTS:
                        search_path = os.path.join(_root, dev, "processed", date_found, filepath)
                        if os.path.exists(search_path):
                            return send_file(search_path)

        return jsonify({"error": f"Audio file not found: {filepath}"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/long_sentences/<path:filename>')
@login_required
def serve_long_sentence_audio(filename):
    """提供ASR服务器保存的长句音频文件"""
    try:
        # Long sentences are saved in the ASR server directory
        asr_server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        long_sentences_dir = os.path.join(asr_server_dir, "long_sentences")
        
        # Fallback: try relative path from current directory
        if not os.path.exists(long_sentences_dir):
            long_sentences_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "long_sentences")
        
        # Platform-specific fallbacks
        if not os.path.exists(long_sentences_dir):
            if platform.system() == "Darwin":
                # macOS 路径
                long_sentences_dir = os.path.expanduser("~/asr-server/long_sentences")
            else:
                # Windows 路径
                long_sentences_dir = r"d:\AI\asr-server\long_sentences"
        
        full_path = os.path.join(long_sentences_dir, filename)
        
        # 安全检查：确保路径在long_sentences目录内
        if not os.path.abspath(full_path).startswith(os.path.abspath(long_sentences_dir)):
            return jsonify({"error": "Invalid path"}), 403
        
        if not os.path.exists(full_path):
            return jsonify({"error": f"Audio file not found: {filename}"}), 404
        
        return send_file(full_path, mimetype='audio/wav')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/mobile')
def mobile_monitor():
    if not check_auth():
        return redirect(url_for('login'))
    resp = make_response(render_template('mobile_monitor.html'))
    resp.headers['Cache-Control'] = 'no-store'  # 【2026-10-02】防 iOS 缓存旧 JS：旧试听逻辑无超时兜底会永久卡"定位片段中"
    return resp

@app.route('/daily')
def daily_report_page():
    if not check_auth():
        return redirect(url_for('login'))
    return render_template('daily_report.html')

@app.route('/growth')
def growth_dictionary_page():
    if not check_auth():
        return redirect(url_for('login'))
    return render_template('growth_dictionary.html')


# ── 绘本日记 (英语启蒙) ──
PICTUREBOOK_DIR = '/Users/mac/asr-server/english_enlightenment/picturebook'


def _pb_normalize(p):
    """v2 书直接返回; v1 单页旧条目归一化为一页书"""
    if p.get('pages'):
        return p
    return {
        'version': 1,
        'date': p.get('date'),
        'title': p.get('title'),
        'pages': [{'seq': 1, 'text_en': p.get('story_en', ''),
                   'text_zh': p.get('story_zh', ''), 'image': p.get('image')}],
    }


# 变体册默认 engine 显示 (JSON 未写 engine 时的兜底)
_PB_VARIANT_ENGINES = {'grok': 'grok-chat-fast', 'agnes': 'agnes-2.5-pro-alpha'}


def _pb_load(date_str, variant=''):
    """加载绘本 JSON。variant 非空时读 {date}.{variant}.json(变体对照册, 如 grok/agnes)"""
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', date_str or ''):
        return None
    if variant and not re.fullmatch(r'[a-z0-9_-]{1,20}', variant):
        return None
    path = os.path.join(PICTUREBOOK_DIR, f'{date_str}{("." + variant) if variant else ""}.json')
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return _pb_normalize(json.load(f))
    except Exception:
        return None


# ---------------- 录音手机集群监控 (adb) ----------------

_ADB_CLUSTER_FILE = '/Users/mac/phone-recorder-apk/cluster.devices'
_ADB_OFFLINE_STATE = '/Users/mac/asr-server/log/adb_offline_state'


def _adb_cluster_snapshot():
    """采集录音手机集群状态: 注册表 × adb devices × 电池/版本详情"""
    import subprocess
    import concurrent.futures

    devices = []
    try:
        with open(_ADB_CLUSTER_FILE) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3:
                    devices.append({'name': parts[0], 'serial': parts[1], 'addr': parts[2]})
    except FileNotFoundError:
        pass

    online = set()
    try:
        out = subprocess.run(['adb', 'devices'], capture_output=True, text=True, timeout=10).stdout
        for l in out.splitlines():
            p = l.split()
            if len(p) >= 2 and p[1] == 'device':
                online.add(p[0].rstrip('.'))
    except Exception:
        pass

    offline_state = {}
    try:
        with open(_ADB_OFFLINE_STATE) as f:
            for line in f:
                p = line.split()
                if len(p) >= 2:
                    offline_state[p[0]] = int(p[1])
    except Exception:
        pass

    def probe(target):
        """采集单台设备详情 (并行调用)"""
        def sh(cmd):
            try:
                return subprocess.run(['adb', '-s', target] + cmd,
                                      capture_output=True, text=True, timeout=8).stdout.strip()
            except Exception:
                return ''
        model = sh(['shell', 'getprop', 'ro.product.marketname']) or sh(['shell', 'getprop', 'ro.product.model'])
        ver = ''
        m = re.search(r'versionName=([\w.]+)', sh(['shell', 'dumpsys', 'package', 'com.asr.recorder']))
        if m:
            ver = m.group(1)
        bat = sh(['shell', 'dumpsys', 'battery'])
        level_m = re.search(r'level:\s*(\d+)', bat)
        ac_m = re.search(r'AC powered:\s*(\w+)', bat)
        usb_m = re.search(r'USB powered:\s*(\w+)', bat)
        charging = (ac_m and ac_m.group(1) == 'true') or (usb_m and usb_m.group(1) == 'true')
        temp_m = re.search(r'temperature:\s*(\d+)', bat)
        rec_pid = sh(['shell', 'pidof', 'com.asr.recorder'])
        return {
            'model': model or '?', 'apk_ver': ver or '?',
            'battery': int(level_m.group(1)) if level_m else None,
            'charging': bool(charging),
            'temp': round(int(temp_m.group(1)) / 10, 1) if temp_m else None,
            'recording': bool(rec_pid),
        }

    online_targets = []
    for d in devices:
        s = d['serial'].rstrip('.')
        a = d['addr'].rstrip('.')
        t = s if s in online else (a if a in online else None)
        d['online'] = t is not None
        if t:
            d['_target'] = t
            online_targets.append(d)
        else:
            first = offline_state.get(d['name'])
            d['offline_min'] = int(time.time() - first) // 60 if first else None

    if online_targets:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            fut = {ex.submit(probe, d['_target']): d for d in online_targets}
            for f, d in fut.items():
                try:
                    d.update(f.result(timeout=15))
                except Exception:
                    d.update({'model': '?', 'apk_ver': '?', 'battery': None,
                              'charging': False, 'temp': None, 'recording': False})
        for d in devices:
            d.pop('_target', None)
    return {'devices': devices, 'ts': time.strftime('%H:%M:%S')}


@app.route('/api/adb_cluster')
def api_adb_cluster():
    if not check_auth():
        return jsonify({'error': 'unauthorized'}), 401
    return jsonify(_adb_cluster_snapshot())


@app.route('/adb')
def adb_monitor_page():
    if not check_auth():
        return redirect(url_for('login'))
    return render_template('adb.html')


@app.route('/picturebook')
def picturebook_page():
    if not check_auth():
        return redirect(url_for('login'))
    return render_template('picturebook.html')


@app.route('/picturebook/tv')
def picturebook_tv_page():
    # TV 壳 App 免登录: ?token= 换 session cookie (cookie 会话对后续 API 生效)
    tok = request.args.get('token', '')
    if not check_auth() and tok:
        pb_tv_token = os.getenv('PB_TV_TOKEN', '')
        if pb_tv_token and hmac.compare_digest(tok, pb_tv_token):
            session['logged_in'] = True
            return redirect(url_for('picturebook_tv_page'))
    if not check_auth():
        return redirect(url_for('login'))
    resp = make_response(render_template('picturebook_tv.html'))
    resp.headers['Cache-Control'] = 'no-store'
    return resp


@app.route('/api/picturebook/status')
def api_picturebook_status():
    """生成任务状态: 进程探测 + 文件系统统计"""
    if not check_auth():
        return jsonify({'error': 'unauthorized'}), 401
    running, run_cmd = False, ''
    try:
        out = subprocess.run(['pgrep', '-fl', 'english_picture_book.py'],
                             capture_output=True, text=True, timeout=5)
        lines = [l for l in out.stdout.strip().splitlines() if l.strip()]
        running = bool(lines)
        if lines:
            run_cmd = lines[0].split(' ', 1)[-1][:140]
    except Exception:
        pass
    try:
        jsons = [os.path.join(PICTUREBOOK_DIR, f) for f in os.listdir(PICTUREBOOK_DIR)
                 if f.endswith('.json')]
    except OSError:
        jsons = []
    latest = None
    if jsons:
        fp = max(jsons, key=os.path.getmtime)
        try:
            with open(fp) as f:
                d = json.load(f)
            latest = {'date': d.get('date') or os.path.basename(fp)[:-5],
                      'title': d.get('title'),
                      'pages': len(d.get('pages') or []),
                      'mtime': int(os.path.getmtime(fp))}
        except Exception:
            latest = {'date': os.path.basename(fp)[:-5], 'title': None,
                      'pages': 0, 'mtime': int(os.path.getmtime(fp))}
    def _count(sub):
        try:
            return len([f for f in os.listdir(os.path.join(PICTUREBOOK_DIR, sub)) if f.endswith(('.mp3', '.mp4'))])
        except OSError:
            return 0
    return jsonify({'running': running, 'cmd': run_cmd, 'books': len(jsons),
                    'audios': _count('audio'), 'videos': _count('video'),
                    'latest': latest, 'now': int(time.time())})


@app.route('/api/picturebook')
def api_picturebook():
    if not check_auth():
        return jsonify({'error': 'unauthorized'}), 401
    pages = []
    if os.path.isdir(PICTUREBOOK_DIR):
        video_dir = os.path.join(PICTUREBOOK_DIR, 'video')
        clip_files = set(os.listdir(video_dir)) if os.path.isdir(video_dir) else set()
        for fn in sorted(os.listdir(PICTUREBOOK_DIR), reverse=True):
            if not fn.endswith('.json'):
                continue
            # 变体册: {date}.{variant}.json → variant='grok'/'agnes'/...; 主册: {date}.json
            m = re.fullmatch(r'(\d{4}-\d{2}-\d{2})\.([a-z0-9_-]{1,20})\.json', fn)
            variant = m.group(2) if m else ''
            try:
                with open(os.path.join(PICTUREBOOK_DIR, fn)) as f:
                    p = _pb_normalize(json.load(f))
            except Exception:
                continue
            book_pages = p.get('pages') or []
            d = m.group(1) if m else fn[:-5]
            has_video = (not variant) and f"{d}.mp4" in clip_files
            # 视频进度: 有图有声的页数中, 片段已生成的比例(变体册不做视频)
            makeable = [pg for pg in book_pages if pg.get('image') and pg.get('audio')]
            total = len(makeable) if not variant else 0
            done = sum(1 for pg in makeable if f"{d}_p{pg.get('seq')}.mp4" in clip_files) if not variant else 0
            pages.append({
                'date': d,
                'variant': variant,
                'engine': (p.get('engine') or _PB_VARIANT_ENGINES.get(variant, variant)) if variant else 'gemini-3.8-flash',
                'title': p.get('title') or 'Untitled',
                'page_count': len(book_pages),
                'has_image': any(pg.get('image') for pg in book_pages),
                'has_audio': any(pg.get('audio') for pg in book_pages),
                'has_video': has_video,
                'video_done': done if (total and not has_video) else total if has_video else 0,
                'video_total': total,
                'story_zh': ((book_pages[0].get('text_zh') if book_pages else '') or '')[:80],
            })
    return jsonify({'pages': pages})


@app.route('/api/picturebook/<date_str>/story')
def api_picturebook_story(date_str):
    if not check_auth():
        return jsonify({'error': 'unauthorized'}), 401
    variant = request.args.get('variant', '')
    book = _pb_load(date_str, variant)
    if book is None:
        return jsonify({'error': 'not found'}), 404
    book_pages = book.get('pages') or []
    has_video = (not variant) and os.path.isfile(os.path.join(PICTUREBOOK_DIR, 'video', f'{date_str}.mp4'))
    resp = jsonify({
        'date': book.get('date'),
        'title': book.get('title'),
        'version': book.get('version', 1),
        'engine': (book.get('engine') or _PB_VARIANT_ENGINES.get(variant, variant)) if variant else 'gemini-3.8-flash',
        'has_video': has_video,
        'pages': [{
            'seq': pg.get('seq', i + 1),
            'text_en': pg.get('text_en', ''),
            'text_zh': pg.get('text_zh', ''),
            'has_image': bool(pg.get('image')),
            'has_audio': bool(pg.get('audio')),
        } for i, pg in enumerate(book_pages)],
    })
    resp.headers['Cache-Control'] = 'no-store'
    return resp


@app.route('/api/picturebook/<date_str>/image')
@app.route('/api/picturebook/<date_str>/image/<int:seq>')
def api_picturebook_image(date_str, seq=1):
    if not check_auth():
        return jsonify({'error': 'unauthorized'}), 401
    book = _pb_load(date_str, request.args.get('variant', ''))
    if book is None:
        return jsonify({'error': 'not found'}), 404
    pg = next((x for x in (book.get('pages') or []) if x.get('seq') == seq), None)
    if pg is None:
        return jsonify({'error': 'no such page'}), 404
    img = pg.get('image')
    if not img:
        return jsonify({'error': 'no image'}), 404
    if img.startswith('http'):
        return redirect(img, 302)
    m = re.match(r'data:image/(\w+);base64,(.+)', img, re.S)
    if m:
        import base64 as _b64
        import io as _io
        data = _b64.b64decode(m.group(2))
        resp = send_file(_io.BytesIO(data), mimetype=f'image/{m.group(1)}')
    else:
        # v2 相对路径 (images/<date>_pN.png)
        fp = os.path.realpath(os.path.join(PICTUREBOOK_DIR, img))
        if not fp.startswith(os.path.realpath(PICTUREBOOK_DIR) + os.sep) or not os.path.isfile(fp):
            return jsonify({'error': 'no image'}), 404
        resp = send_file(fp)
    resp.headers['Cache-Control'] = 'no-store'
    return resp


THUMB_DIR = os.path.join(PICTUREBOOK_DIR, 'thumbs')


@app.route('/api/picturebook/<date_str>/thumb')
def api_picturebook_thumb(date_str):
    """封面缩略图 (512px JPEG, sips 现生成缓存)——书架大量封面避免整页大图解码卡顿"""
    if not check_auth():
        return jsonify({'error': 'unauthorized'}), 401
    variant = request.args.get('variant', '')
    book = _pb_load(date_str, variant)
    if book is None:
        return jsonify({'error': 'not found'}), 404
    img = next((p.get('image') for p in (book.get('pages') or []) if p.get('image')), '')
    if not img:
        return jsonify({'error': 'no image'}), 404
    if img.startswith(('http', 'data:')):   # 远程/内嵌图不缩略, 回退原图路由
        return api_picturebook_image(date_str, 1, variant=variant) if variant else api_picturebook_image(date_str, 1)
    src = os.path.realpath(os.path.join(PICTUREBOOK_DIR, img))
    if not src.startswith(os.path.realpath(PICTUREBOOK_DIR) + os.sep) or not os.path.isfile(src):
        return jsonify({'error': 'no image'}), 404
    os.makedirs(THUMB_DIR, exist_ok=True)
    tp = os.path.join(THUMB_DIR, f'{date_str}{("." + variant) if variant else ""}.jpg')
    if not os.path.isfile(tp) or os.path.getmtime(tp) < os.path.getmtime(src):
        try:
            subprocess.run(['sips', '-s', 'format', 'jpeg', '-s', 'formatOptions', '72',
                            '-Z', '512', src, '--out', tp], capture_output=True, timeout=20)
        except Exception:
            pass
    if os.path.isfile(tp):
        resp = send_file(tp, mimetype='image/jpeg')
        resp.headers['Cache-Control'] = 'public, max-age=600'
        return resp
    if variant:   # 缩略生成失败兜底原图(带变体)
        return api_picturebook_image(date_str, 1, variant=variant)
    return api_picturebook_image(date_str, 1)   # 缩略生成失败兜底原图


@app.route('/api/picturebook/<date_str>/audio/<int:seq>')
def api_picturebook_audio(date_str, seq):
    if not check_auth():
        return jsonify({'error': 'unauthorized'}), 401
    book = _pb_load(date_str, request.args.get('variant', ''))
    if book is None:
        return jsonify({'error': 'not found'}), 404
    pg = next((x for x in (book.get('pages') or []) if x.get('seq') == seq), None)
    if pg is None or not pg.get('audio'):
        return jsonify({'error': 'no audio'}), 404
    fp = os.path.realpath(os.path.join(PICTUREBOOK_DIR, pg['audio']))
    if not fp.startswith(os.path.realpath(PICTUREBOOK_DIR) + os.sep) or not os.path.isfile(fp):
        return jsonify({'error': 'no audio'}), 404
    resp = send_file(fp, mimetype='audio/mpeg')
    resp.headers['Cache-Control'] = 'no-store'
    return resp


@app.route('/api/picturebook/<date_str>/video')
def api_picturebook_video(date_str):
    if not check_auth():
        return jsonify({'error': 'unauthorized'}), 401
    fp = os.path.realpath(os.path.join(PICTUREBOOK_DIR, 'video', f'{date_str}.mp4'))
    if not fp.startswith(os.path.realpath(PICTUREBOOK_DIR) + os.sep) or not os.path.isfile(fp):
        return jsonify({'error': 'no video'}), 404
    resp = send_file(fp, mimetype='video/mp4')
    resp.headers['Cache-Control'] = 'no-store'
    return resp


@app.route('/pbpub/<date_str>/<int:seq>.png')
def pbpub_image(date_str, seq):
    """公网首帧图（供 agnes video 服务器拉取）：token 校验同哭声预览，只读单图"""
    if not _preview_auth_ok():
        return jsonify({'error': 'unauthorized'}), 401
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', date_str):
        return jsonify({'error': 'bad date'}), 400
    fp = os.path.realpath(os.path.join(PICTUREBOOK_DIR, 'images', f'{date_str}_p{seq}.png'))
    if not fp.startswith(os.path.realpath(PICTUREBOOK_DIR) + os.sep) or not os.path.isfile(fp):
        return jsonify({'error': 'not found'}), 404
    resp = send_file(fp, mimetype='image/png')
    resp.headers['Cache-Control'] = 'no-store'
    return resp


@app.route('/pbpub/tv/version.json')
def pbpub_tv_version():
    """TV 壳 App 版本信息（供自动更新检查，读构建产物元数据）"""
    if not _preview_auth_ok():
        return jsonify({'error': 'unauthorized'}), 401
    meta = '/Users/mac/picturebook-tv-apk/app/build/outputs/apk/debug/output-metadata.json'
    apk = '/Users/mac/picturebook-tv-apk/picturebook-tv.apk'
    try:
        with open(meta) as f:
            v = json.load(f)['elements'][0]
    except Exception:
        return jsonify({'error': 'unavailable'}), 404
    return jsonify({'versionCode': int(v.get('versionCode', 0)),
                    'versionName': v.get('versionName', ''),
                    'size': os.path.getsize(apk) if os.path.isfile(apk) else 0})


@app.route('/pbpub/picturebook-tv.apk')
def pbpub_tv_apk():
    """TV 壳 APK 下载（免 token：长 token URL 易被电视浏览器截断导致存下错误页装不上；
    仅暴露签名安装包本身，无敏感信息）"""
    apk = '/Users/mac/picturebook-tv-apk/picturebook-tv.apk'
    if not os.path.isfile(apk):
        return jsonify({'error': 'not found'}), 404
    return send_file(apk, mimetype='application/vnd.android.package-archive',
                     as_attachment=True, download_name='picturebook-tv.apk')


@app.route('/pbpub/tmp/<name>')
def pbpub_tmp(name):
    """临时试听文件（音色样本等，仅允许字母数字下划线连字符 + .mp3）"""
    import re as _re
    if not _re.fullmatch(r'[A-Za-z0-9_\-]+\.mp3', name):
        return jsonify({'error': 'bad name'}), 400
    path = os.path.join('/Users/mac/asr-server/english_enlightenment/picturebook/pbpub', name)
    if not os.path.isfile(path):
        return jsonify({'error': 'not found'}), 404
    return send_file(path, mimetype='audio/mpeg')


@app.route('/')
def index():
    if not check_auth():
        return redirect(url_for('login'))
    return render_template('index.html')

if __name__ == "__main__":
    try:
        # 初始化数据库连接池
        logger_web.info("初始化数据库连接池...")
        if not init_pool(CONFIG["DATABASE_URL"]):
            logger_web.warning("⚠️ 数据库连接池初始化失败！服务将继续运行，数据库功能暂不可用，连接恢复后自动生效")
        else:
            # 初始化数据库表结构
            logger_web.info("初始化数据库表结构...")
            init_db()
        
        args = parse_args()
        update_config(args)
        
        # 启动后台状态监控
        start_status_monitor()
        # 启动 Pixel 待补救数后台扫描（事件驱动+每小时兜底，供「Pixel 转录补救」卡亮黄提醒）
        threading.Thread(target=_pixel_backlog_loop, daemon=True).start()

        logger_web.info(f"🌐 [Web Viewer] 启动在端口 {CONFIG['WEB_PORT']}")
        app.run(host='0.0.0.0', port=CONFIG["WEB_PORT"], debug=False)
    except BaseException as e:
        logger_web.error(f"启动失败 (BaseException): {e}")
        import traceback
        traceback.print_exc()
