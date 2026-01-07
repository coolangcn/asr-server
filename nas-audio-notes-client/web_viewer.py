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
        print(f"[配置] 使用自定义源路径: {base_path}")
    
    if args.port:
        CONFIG["WEB_PORT"] = args.port
    
    if args.asr_url:
        CONFIG["ASR_API_URL"] = args.asr_url
        print(f"[配置] 使用自定义ASR服务地址: {args.asr_url}")

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
    "updated_at": 0
}
g_status_lock = threading.Lock()

def read_last_lines(filepath, line_count=20, encoding='utf-8', errors='ignore'):
    """高效读取文件最后几行"""
    try:
        with open(filepath, 'rb') as f:
            # 移动到文件末尾
            try:
                f.seek(-2048, os.SEEK_END) # 假设最后20行大约2KB,可根据实际情况调整
            except IOError:
                # 文件太小
                f.seek(0)
            
            lines = f.readlines()
            decoded_lines = [line.decode(encoding, errors).strip() for line in lines]
            return decoded_lines[-line_count:]
    except Exception:
        return []

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
    try:
        source_dir = CONFIG["SOURCE_DIR"]
        if os.path.exists(source_dir) and os.path.isdir(source_dir):
            files = [f for f in os.listdir(source_dir) 
                     if f.lower().endswith(('.m4a', '.acc', '.aac', '.mp3', '.wav', '.ogg'))
                     and 'TEMP' not in f]
            pending_count = len(files)
        else:
            # 目录不存在时显示0而不是-1
            pending_count = 0
    except Exception as e:
        print(f"[StatusMonitor] 检查待处理文件失败: {e}")
        pending_count = -1

    # 3. 读取日志
    last_log = "读取日志失败"
    try:
        log_path = CONFIG["LOG_FILE_PATH"]
        if not os.path.exists(log_path):
            # macOS 备用路径
            log_path = os.path.expanduser("~/asr-server/log/asr-server.log")

        if os.path.exists(log_path):
            lines = read_last_lines(log_path, 20)
            # 去掉每行的时间戳和日志级别，只保留消息内容
            # 格式: "2025-12-22 09:26:34,414 | INFO | 消息内容"
            cleaned_lines = []
            for line in lines:
                # 尝试分割日志行，提取消息部分
                parts = line.split(' | ', 2)  # 最多分割2次
                if len(parts) >= 3:
                    # 第三部分是消息内容
                    cleaned_lines.append(parts[2])
                else:
                    # 如果格式不匹配，保留原始行
                    cleaned_lines.append(line)
            last_log = "\n".join(cleaned_lines)
        else:
            last_log = f"找不到日志文件: {log_path}"
    except Exception as e:
        last_log = f"读取日志异常: {str(e)}"

    # 更新缓存
    with g_status_lock:
        g_status_cache = {
            "asr_server": asr_status,
            "pending_files": pending_count,
            "last_log": last_log,
            "updated_at": time.time()
        }

def status_monitor_loop():
    """状态监控循环"""
    print("[StatusMonitor] 启动后台状态监控线程...")
    while True:
        try:
            update_system_status()
        except Exception as e:
            print(f"[StatusMonitor] 更新失败: {e}")
        time.sleep(3) # 每3秒更新一次

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
        items = db_get_transcripts(offset=offset, limit=limit)
        
        results = []
        for data in items:
            # 格式化时间戳
            for seg in data.get('segments', []):
                seg['start_fmt'] = format_timestamp(seg.get('start', 0))
                seg['spk_id'] = seg.get('spk', 0)
            
            # 解析时间
            filename = data['filename']
            dt = None
            time_patterns = [
                r'^\s*(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})\s*',
                r'^\s*recording-(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})\s*'
            ]
            
            for pattern in time_patterns:
                match = re.match(pattern, os.path.splitext(filename)[0])
                if match:
                    try:
                        if pattern == time_patterns[0]:
                            date_part = match.group(1)
                            time_part = match.group(2)
                            dt_str = f"{date_part} {time_part.replace('-', ':')}"
                            dt = datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
                        else:
                            year, month, day, hour, minute, second = match.groups()
                            dt_str = f"{year}-{month}-{day} {hour}:{minute}:{second}"
                            dt = datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
                        break
                    except ValueError:
                        continue
                        
            if dt is None:
                try:
                    dt = datetime.datetime.fromisoformat(data['created_at'])
                except:
                    pass

            if dt is not None:
                now = datetime.datetime.now()
                data['is_new'] = (now - dt).total_seconds() < 300
                data['date_group'] = dt.strftime('%Y-%m-%d')
                data['time_simple'] = dt.strftime('%H:%M')
                data['time_full'] = dt.strftime('%Y-%m-%d %H:%M:%S')
            else:
                data['is_new'] = False
                data['date_group'] = "Unknown"
                data['time_simple'] = ""
                data['time_full'] = ""
            results.append(data)
        return results
    except Exception as e:
        print(f"[Error] 获取转录记录失败: {e}")
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
        print(f"[Audio] 请求: {filepath}")
        print(f"[Audio] SOURCE_DIR: {CONFIG['SOURCE_DIR']}")
        print(f"[Audio] 完整路径: {full_path}")
        print(f"[Audio] 文件存在: {os.path.exists(full_path)}")
        
        # 安全检查：确保路径在segments目录内
        if not os.path.abspath(full_path).startswith(os.path.abspath(segments_dir)):
            return jsonify({"error": "Invalid path"}), 403
        
        if not os.path.exists(full_path):
            return jsonify({"error": f"Audio segment not found: {full_path}"}), 404
        
        return send_file(full_path, mimetype='audio/wav')
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
        # 先初始化数据库连接池
        print("初始化数据库连接池...", flush=True)
        if not init_pool():
            print("数据库连接池初始化失败，程序退出", flush=True)
            exit(1)

        args = parse_args()
        update_config(args)
        
        # 启动后台状态监控
        start_status_monitor()

        print(f"[Web Viewer] 启动在端口 {CONFIG['WEB_PORT']}", flush=True)
        app.run(host='0.0.0.0', port=CONFIG["WEB_PORT"], debug=False)
    except BaseException as e:
        print(f"启动失败 (BaseException): {e}", flush=True)
        import traceback
        traceback.print_exc()