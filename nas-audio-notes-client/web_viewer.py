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
        source_dir = CONFIG["SOURCE_DIR"]
        if os.path.exists(source_dir) and os.path.isdir(source_dir):
            count = 0
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
        else:
            pending_count = 0
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
    '是不', '的是', '了吗', '了吧', '去吧', '对不', '不能'
}

def _growth_text_terms(text):
    """Extract lightweight Chinese terms without adding a tokenizer dependency."""
    text = re.sub(r'\s+', '', text or '')
    terms = []

    for block in re.findall(r'[\u4e00-\u9fff]{2,}', text):
        max_n = min(3, len(block))
        for n in range(2, max_n + 1):
            for i in range(len(block) - n + 1):
                term = block[i:i + n]
                if term not in GROWTH_STOP_TERMS:
                    terms.append(term)

    for token in re.findall(r'[A-Za-z0-9]{2,}', text):
        terms.append(token.lower())

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
            'speaker_name': request.form.get('speaker_name', '')
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

@app.route('/speaker/list', methods=['GET'])
@login_required
def proxy_list_speakers():
    """转发获取说话人列表请求"""
    try:
        response = requests.get(f"{ASR_SERVER_URL}/speaker/list", timeout=10)
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

@app.route('/baby_cry_page')
@login_required
def proxy_baby_cry_page():
    """转发宝宝哭闹分析页面"""
    try:
        response = requests.get(f"{ASR_SERVER_URL}/baby_cry", timeout=10)
        html = response.text
        html = html.replace('/api/', '/api/') # 本地代理也是 /api/
        # 修改静态资源和导航链接
        html = html.replace('href="/manage"', 'href="/"')
        return html
    except Exception as e:
        return f"<h1>Error loading baby cry page</h1><p>{str(e)}</p>", 500

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
        source_dir = CONFIG["SOURCE_DIR"]
        for cand in (os.path.join(source_dir, "processed", media_path),
                     os.path.join(source_dir, media_path)):
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
        
        # 获取所有数据
        all_items = db_get_transcripts(offset=0, limit=10000)
        
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
        total_count = len(filtered_items)
        paginated_items = filtered_items[offset:offset+limit]
        
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
    """提供音频片段文件"""
    try:
        segments_dir = os.path.join(CONFIG["SOURCE_DIR"], "audio_segments")
        full_path = os.path.join(segments_dir, filepath)
        
        # 调试日志
        logger_web.info(f"[Audio] 请求: {filepath}")
        logger_web.info(f"[Audio] SOURCE_DIR: {CONFIG['SOURCE_DIR']}")
        logger_web.info(f"[Audio] 完整路径: {full_path}")
        logger_web.info(f"[Audio] 文件存在: {os.path.exists(full_path)}")
        
        # 安全检查：确保路径在segments目录内
        if not os.path.abspath(full_path).startswith(os.path.abspath(segments_dir)):
            return jsonify({"error": "Invalid path"}), 403
        
        return send_file(full_path, mimetype='audio/wav')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/audio/<path:filepath>')
@login_required
def serve_original_audio(filepath):
    """提供原始或已处理的录音文件回放"""
    try:
        source_dir = CONFIG["SOURCE_DIR"]
        # 【2026-09-20】本地持久音频（temp_cry 暂存区）：audio_urls 形如
        # /api/audio/Users/mac/asr-server/temp_cry/cry_xxx.wav，此前只在 NAS 目录找 → 404 无声
        if filepath.startswith("Users/mac/asr-server/temp_cry/"):
            local_path = "/" + filepath
            if os.path.isfile(local_path):
                return send_file(local_path)
        # 先尝试在 processed 目录下找
        processed_path = os.path.join(source_dir, "processed", filepath)
        if os.path.exists(processed_path) and os.path.isfile(processed_path):
            return send_file(processed_path)
            
        # 再尝试直接在根目录下找
        root_path = os.path.join(source_dir, filepath)
        if os.path.exists(root_path) and os.path.isfile(root_path):
            return send_file(root_path)
            
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
                
                search_path = os.path.join(source_dir, "processed", date_found, filepath)
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
    return send_file('templates/mobile_monitor.html')

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

        logger_web.info(f"🌐 [Web Viewer] 启动在端口 {CONFIG['WEB_PORT']}")
        app.run(host='0.0.0.0', port=CONFIG["WEB_PORT"], debug=False)
    except BaseException as e:
        logger_web.error(f"启动失败 (BaseException): {e}")
        import traceback
        traceback.print_exc()
