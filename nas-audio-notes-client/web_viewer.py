#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import sys
from flask import Flask, render_template, jsonify, request, Response, send_file
import datetime
import requests
import subprocess
import argparse
from db_manager import init_pool, get_transcripts as db_get_transcripts

# --- 配置 ---
# 获取脚本自身所在的目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_SOURCE_DIR = "V:\\Sony-2"
DEFAULT_ASR_API_URL = "http://localhost:5008/transcribe"
DEFAULT_LOG_FILE_PATH = os.path.join(SCRIPT_DIR, "transcribe.log")
DEFAULT_WEB_PORT = 5009 

# 全局配置变量
CONFIG = {
    "SOURCE_DIR": DEFAULT_SOURCE_DIR,
    "ASR_API_URL": DEFAULT_ASR_API_URL,
    "LOG_FILE_PATH": DEFAULT_LOG_FILE_PATH,
    "WEB_PORT": DEFAULT_WEB_PORT,
    "DATABASE_URL": "postgresql://postgres:difyai123456@192.168.1.188:5432/postgres"
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

def get_system_status():
    status = {
        "asr_server": "unknown",
        "pending_files": 0,
        "last_log": "等待日志..."
    }
    try:
        try:
            requests.get(CONFIG["ASR_API_URL"].replace("/transcribe", "/"), timeout=3)
            status["asr_server"] = "online"
        except requests.exceptions.RequestException:
             status["asr_server"] = "offline"
    except:
        status["asr_server"] = "offline"

    try:
        if os.path.exists(CONFIG["SOURCE_DIR"]):
            files = [f for f in os.listdir(CONFIG["SOURCE_DIR"]) 
                     if f.lower().endswith(('.m4a', '.acc', '.aac', '.mp3', '.wav', '.ogg'))]
            status["pending_files"] = len(files)
        else:
            status["pending_files"] = -1
    except:
        status["pending_files"] = -1

    try:
        # 优先读取本地目录下的日志，或者配置里的日志
        log_path = CONFIG["LOG_FILE_PATH"]
        # 如果配置的日志不存在，尝试在当前目录找
        if not os.path.exists(log_path):
             log_path = "transcribe.log"

        if os.path.exists(log_path):
            # 读取最后 20 行
            try:
                if sys.platform == "win32":
                    # On Windows, read file directly
                    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        status["last_log"] = f"[{current_time}] " + "".join(lines[-20:])
                else:
                    # On other systems, use tail command
                    cmd = f"tail -n 20 {log_path}"
                    result = subprocess.check_output(cmd, shell=True).decode('utf-8')
                    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    status["last_log"] = f"[{current_time}] " + result
            except Exception:
                # Fallback for any error (e.g., tail not found even on non-Windows)
                with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    status["last_log"] = f"[{current_time}] " + "".join(lines[-20:])
        else:
            status["last_log"] = f"找不到日志文件: {log_path}"
    except Exception as e:
        status["last_log"] = f"读取日志失败: {e}"

    return status

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
        
        # 安全检查：确保路径在segments目录内
        if not os.path.abspath(full_path).startswith(os.path.abspath(segments_dir)):
            return jsonify({"error": "Invalid path"}), 403
        
        if not os.path.exists(full_path):
            return jsonify({"error": "Audio segment not found"}), 404
        
        return send_file(full_path, mimetype='audio/wav')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/long_sentences/<path:filename>')
def serve_long_sentence_audio(filename):
    """提供ASR服务器保存的长句音频文件"""
    try:
        # Long sentences are saved in the ASR server directory
        asr_server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        long_sentences_dir = os.path.join(asr_server_dir, "asr-server", "long_sentences")
        
        # Fallback: try relative path from current directory
        if not os.path.exists(long_sentences_dir):
            long_sentences_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "long_sentences")
        
        # Another fallback: absolute path
        if not os.path.exists(long_sentences_dir):
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
        
        print(f"[Web Viewer] 启动在端口 {CONFIG['WEB_PORT']}", flush=True)
        app.run(host='0.0.0.0', port=CONFIG["WEB_PORT"], debug=False)
    except BaseException as e:
        print(f"启动失败 (BaseException): {e}", flush=True)
        import traceback
        traceback.print_exc()