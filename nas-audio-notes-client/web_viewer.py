#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import sys
import threading
import time
from flask import Flask, render_template, jsonify, request, Response, send_file
import datetime
import requests
import subprocess
import argparse
from db_manager import init_pool, get_transcripts as db_get_transcripts

# --- 配置 ---
# 获取脚本自身所在的目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 跨平台路径配置
import platform
if platform.system() == "Darwin":
    # macOS 路径 - 与实际 SMB 挂载路径保持一致
    DEFAULT_SOURCE_DIR = "/Volumes/download/records/Sony-2"
    DEFAULT_LOG_FILE_PATH = os.path.expanduser("~/asr-server/log/asr-server.log")
else:
    # Windows 路径
    DEFAULT_SOURCE_DIR = "V:\\Sony-2"
    DEFAULT_LOG_FILE_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "log", "asr-server.log")

DEFAULT_ASR_API_URL = "http://localhost:5008/transcribe"
DEFAULT_WEB_PORT = 5009 

# 全局配置变量
CONFIG = {
    "SOURCE_DIR": DEFAULT_SOURCE_DIR,
    "ASR_API_URL": DEFAULT_ASR_API_URL,
    "LOG_FILE_PATH": DEFAULT_LOG_FILE_PATH,
    "WEB_PORT": DEFAULT_WEB_PORT,
    "DATABASE_URL": "postgresql://cnncn:74123698cN@cncn.postgres.database.azure.com:5432/postgres?sslmode=require"

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
    if args.source_path:
        base_path = args.source_path
        CONFIG["SOURCE_DIR"] = base_path
        logger_web.info(f"[配置] 使用自定义源路径: {base_path}")
    
    if args.port:
        CONFIG["WEB_PORT"] = args.port
    
    if args.asr_url:
        CONFIG["ASR_API_URL"] = args.asr_url
        logger_web.info(f"[配置] 使用自定义ASR服务地址: {args.asr_url}")

# -----------------

app = Flask(__name__)

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
    
    # 写入与 asr_server 相同的系统日志文件
    log_path = os.path.join(log_dir, "asr-web.log")
    handler = TimedRotatingFileHandler(
        log_path, when='M', interval=10, backupCount=144, encoding='utf-8'
    )
    handler.setFormatter(formatter)
    l.addHandler(handler)
    
    # 控制台输出
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    l.addHandler(console)
    
    return l

import logging
from logging.handlers import TimedRotatingFileHandler
logger_web = setup_web_logger()

def update_system_status():
    """后台更新系统状态"""
    global g_status_cache
    
    # 1. 检查 ASR Server
    asr_status = "offline"
    try:
        requests.get(CONFIG["ASR_API_URL"].replace("/transcribe", "/"), timeout=2)
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
        logs_a = [line.split(' | ', 2)[2] if ' | ' in line else line for line in lines]
    
    # B 轨日志
    path_b = os.path.join(log_dir, "asr-b.log")
    if os.path.exists(path_b):
        lines = read_last_lines(path_b, 50)
        logs_b = [line.split(' | ', 2)[2] if ' | ' in line else line for line in lines]
    
    # Web/系统日志
    path_web = os.path.join(log_dir, "asr-web.log")
    if os.path.exists(path_web):
        lines = read_last_lines(path_web, 50)
        logs_web = [line.split(' | ', 2)[2] if ' | ' in line else line for line in lines]
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

# --- HTML 模板 ---



# =================== 声纹管理API转发 ===================
ASR_SERVER_URL = "http://localhost:5008"

@app.route('/speaker/register', methods=['POST'])
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
            timeout=30
        )
        
        return Response(response.content, status=response.status_code, content_type=response.headers.get('Content-Type'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/speaker/list', methods=['GET'])
def proxy_list_speakers():
    """转发获取说话人列表请求"""
    try:
        response = requests.get(f"{ASR_SERVER_URL}/speaker/list", timeout=10)
        return Response(response.content, status=response.status_code, content_type=response.headers.get('Content-Type'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/speaker/<speaker_name>', methods=['DELETE'])
def proxy_delete_speaker(speaker_name):
    """转发删除说话人请求"""
    try:
        response = requests.delete(f"{ASR_SERVER_URL}/speaker/{speaker_name}", timeout=10)
        return Response(response.content, status=response.status_code, content_type=response.headers.get('Content-Type'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/speaker/audio/<path:filename>', methods=['GET'])
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
def proxy_trigger_reprocess():
    try:
        date_param = request.args.get('date', '')
        start_time = request.args.get('start_time', '')
        end_time = request.args.get('end_time', '')
        replace_param = request.args.get('replace', 'false')
        url = f"{ASR_SERVER_URL}/api/trigger_reprocess?date={date_param}&start_time={start_time}&end_time={end_time}&replace={replace_param}"
        response = requests.post(url, timeout=10)
        return Response(response.content, status=response.status_code, content_type=response.headers.get('Content-Type'))
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/stop_reprocess', methods=['POST'])
def proxy_stop_reprocess():
    try:
        url = f"{ASR_SERVER_URL}/api/stop_reprocess"
        response = requests.post(url, timeout=10)
        return Response(response.content, status=response.status_code, content_type=response.headers.get('Content-Type'))
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/cry_events', methods=['GET'])
def proxy_cry_events():
    try:
        limit = request.args.get('limit', 100)
        offset = request.args.get('offset', 0)
        response = requests.get(f"{ASR_SERVER_URL}/api/cry_events?limit={limit}&offset={offset}", timeout=10)
        return Response(response.content, status=response.status_code, content_type=response.headers.get('Content-Type'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/reprocess_logs', methods=['GET'])
def proxy_reprocess_logs():
    try:
        response = requests.get(f"{ASR_SERVER_URL}/api/reprocess_logs", timeout=5)
        return Response(response.content, status=response.status_code, content_type=response.headers.get('Content-Type'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/status')
def api_status():
    return jsonify(get_system_status())

@app.route('/api/data')
def api_data():
    """获取转录数据，支持分页"""
    offset = request.args.get('offset', 0, type=int)
    limit = request.args.get('limit', 20, type=int)
    return jsonify(get_transcripts(offset=offset, limit=limit))

@app.route('/api/data/range')
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

@app.route('/api/emotion-timeline')
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

@app.route('/api/config', methods=['GET'])
def api_get_config():
    return jsonify(CONFIG)

@app.route('/api/config', methods=['POST'])
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
def serve_original_audio(filepath):
    """提供原始或已处理的录音文件回放"""
    try:
        source_dir = CONFIG["SOURCE_DIR"]
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

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    try:
        # 初始化数据库连接池
        logger_web.info("初始化数据库连接池...")
        if not init_pool(CONFIG["DATABASE_URL"]):
            logger_web.error("数据库连接池初始化失败，程序退出")
            sys.exit(1)
        
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