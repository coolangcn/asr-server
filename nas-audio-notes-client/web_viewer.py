#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sqlite3
import json
from flask import Flask, render_template_string, jsonify, request, Response
import datetime
import requests
import subprocess
import argparse
import time
import threading

# --- 配置 ---
# 获取脚本自身所在的目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DB_PATH = "/volume2/download/records/Sony-2/transcripts.db"
DEFAULT_SOURCE_DIR = "/volume2/download/records/Sony-2"
DEFAULT_ASR_API_URL = "http://192.168.1.111:5008/transcribe"
DEFAULT_ASR_LOG_STREAM_URL = "http://192.168.1.111:5008/logs/stream"
DEFAULT_LOG_FILE_PATH = os.path.join(SCRIPT_DIR, "transcribe.log")
DEFAULT_WEB_PORT = 5009 

# 全局配置变量
CONFIG = {
    "DB_PATH": DEFAULT_DB_PATH,
    "SOURCE_DIR": DEFAULT_SOURCE_DIR,
    "ASR_API_URL": DEFAULT_ASR_API_URL,
    "ASR_LOG_STREAM_URL": DEFAULT_ASR_LOG_STREAM_URL,
    "LOG_FILE_PATH": DEFAULT_LOG_FILE_PATH,
    "WEB_PORT": DEFAULT_WEB_PORT
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
    parser.add_argument('--asr-log-url', type=str, help='ASR服务日志流地址', default=DEFAULT_ASR_LOG_STREAM_URL)
    return parser.parse_args()

def update_config(args):
    """根据命令行参数更新配置"""
    if args.source_path:
        base_path = args.source_path
        CONFIG["SOURCE_DIR"] = base_path
        CONFIG["DB_PATH"] = os.path.join(base_path, "transcripts.db")
        print(f"[配置] 使用自定义源路径: {base_path}")
    
    if args.port:
        CONFIG["WEB_PORT"] = args.port
    
    if args.asr_url:
        CONFIG["ASR_API_URL"] = args.asr_url
        print(f"[配置] 使用自定义ASR服务地址: {args.asr_url}")
    
    if args.asr_log_url:
        CONFIG["ASR_LOG_STREAM_URL"] = args.asr_log_url
        print(f"[配置] 使用自定义ASR日志流地址: {args.asr_log_url}")

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
            requests.get(CONFIG["ASR_API_URL"].replace("/transcribe", "/"), timeout=1)
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
                # 使用 tail 命令 (Linux/Mac)
                cmd = f"tail -n 20 {log_path}" 
                result = subprocess.check_output(cmd, shell=True).decode('utf-8')
                # 添加当前时间戳
                current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                status["last_log"] = f"[{current_time}] " + result
            except:
                # Windows 兼容或者是读文件失败，用 Python 读取
                with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    # 添加当前时间戳
                    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    status["last_log"] = f"[{current_time}] " + "".join(lines[-20:])
        else:
            status["last_log"] = f"找不到日志文件: {log_path}"
    except Exception as e:
        status["last_log"] = f"读取日志失败: {e}"

    return status

def get_transcripts():
    if not os.path.exists(CONFIG["DB_PATH"]):
        return []
    try:
        db = sqlite3.connect(CONFIG["DB_PATH"])
        db.row_factory = sqlite3.Row
        cursor = db.cursor()
        # 获取最近 100 条记录
        cursor.execute("SELECT id, filename, created_at, full_text, segments_json FROM transcriptions ORDER BY created_at DESC LIMIT 100")
        rows = cursor.fetchall()
        db.close()
        
        results = []
        for row in rows:
            data = dict(row)
            try:
                data['segments'] = json.loads(data['segments_json'])
            except:
                data['segments'] = []
            
            for seg in data['segments']:
                seg['start_fmt'] = format_timestamp(seg.get('start', 0))
                # 兼容后端传来的 spk 字段 (可能是数字，可能是字符串"爸爸")
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
                try: dt = datetime.datetime.fromisoformat(data['created_at'])
                except: pass

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
    except:
        return []

# ---------------- 对话历史功能 ----------------
def init_chat_history_db():
    """初始化对话历史数据库表"""
    try:
        db = sqlite3.connect(CONFIG["DB_PATH"])
        cursor = db.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            speaker_id TEXT NOT NULL,
            speaker_name TEXT NOT NULL,
            message_text TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        ''')
        # 创建索引以提高查询性能
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_history(session_id);
        ''')
        db.commit()
        db.close()
        print("对话历史数据库表初始化成功")
    except Exception as e:
        print(f"对话历史数据库表初始化失败: {e}")

def save_chat_message(session_id, speaker_id, speaker_name, message_text, timestamp):
    """保存单条聊天消息到数据库"""
    try:
        db = sqlite3.connect(CONFIG["DB_PATH"])
        cursor = db.cursor()
        cursor.execute(
            "INSERT INTO chat_history (session_id, speaker_id, speaker_name, message_text, timestamp) VALUES (?, ?, ?, ?, ?)",
            (session_id, speaker_id, speaker_name, message_text, timestamp)
        )
        db.commit()
        db.close()
        return True
    except Exception as e:
        print(f"保存聊天消息失败: {e}")
        return False

def save_chat_session(session_id, chat_data):
    """保存整个聊天会话到数据库"""
    try:
        # 首先删除已存在的会话数据
        db = sqlite3.connect(CONFIG["DB_PATH"])
        cursor = db.cursor()
        cursor.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
        
        # 然后插入新的聊天数据
        for item in chat_data:
            if item.get('segments'):
                for seg in item['segments']:
                    if seg.get('text') and seg.get('text').strip():
                        speaker_id = seg.get('spk_id', 0)
                        speaker_name = seg.get('speaker_name', f"说话人 {speaker_id}")
                        timestamp = seg.get('start', 0)
                        message_text = seg.get('text', '').strip()
                        
                        cursor.execute(
                            "INSERT INTO chat_history (session_id, speaker_id, speaker_name, message_text, timestamp) VALUES (?, ?, ?, ?, ?)",
                            (session_id, speaker_id, speaker_name, message_text, timestamp)
                        )
            elif item.get('full_text') and item.get('full_text').strip():
                # 处理没有分段的情况
                speaker_id = 0
                speaker_name = "系统"
                timestamp = 0
                message_text = item.get('full_text', '').strip()
                
                cursor.execute(
                    "INSERT INTO chat_history (session_id, speaker_id, speaker_name, message_text, timestamp) VALUES (?, ?, ?, ?, ?)",
                    (session_id, speaker_id, speaker_name, message_text, timestamp)
                )
        
        db.commit()
        db.close()
        return True
    except Exception as e:
        print(f"保存聊天会话失败: {e}")
        return False

def get_chat_sessions():
    """获取所有聊天会话列表"""
    try:
        db = sqlite3.connect(CONFIG["DB_PATH"])
        db.row_factory = sqlite3.Row
        cursor = db.cursor()
        cursor.execute('''
        SELECT session_id, MIN(created_at) as created_at, COUNT(*) as message_count
        FROM chat_history
        GROUP BY session_id
        ORDER BY created_at DESC
        ''')
        rows = cursor.fetchall()
        db.close()
        
        sessions = []
        for row in rows:
            sessions.append({
                'session_id': row['session_id'],
                'created_at': row['created_at'],
                'message_count': row['message_count']
            })
        return sessions
    except Exception as e:
        print(f"获取聊天会话列表失败: {e}")
        return []

def get_chat_session_messages(session_id):
    """获取特定聊天会话的所有消息"""
    try:
        db = sqlite3.connect(CONFIG["DB_PATH"])
        db.row_factory = sqlite3.Row
        cursor = db.cursor()
        cursor.execute('''
        SELECT speaker_id, speaker_name, message_text, timestamp
        FROM chat_history
        WHERE session_id = ?
        ORDER BY timestamp ASC
        ''', (session_id,))
        rows = cursor.fetchall()
        db.close()
        
        messages = []
        for row in rows:
            messages.append({
                'speaker_id': row['speaker_id'],
                'speaker_name': row['speaker_name'],
                'message_text': row['message_text'],
                'timestamp': row['timestamp'],
                'start_fmt': format_timestamp(row['timestamp'])
            })
        return messages
    except Exception as e:
        print(f"获取聊天会话消息失败: {e}")
        return []

def delete_chat_session(session_id):
    """删除特定聊天会话"""
    try:
        db = sqlite3.connect(CONFIG["DB_PATH"])
        cursor = db.cursor()
        cursor.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
        db.commit()
        db.close()
        return True
    except Exception as e:
        print(f"删除聊天会话失败: {e}")
        return False

# --- HTML 模板 ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI 录音存档</title>
    <style>
        :root { --primary: #007bff; --bg: #f0f2f5; --card-bg: #ffffff; --text: #333; --console-bg: #1e1e1e; --console-text: #00ff00; --chat-me: #d9fdd3; --chat-other: #ffffff; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: var(--bg); color: var(--text); margin: 0; padding: 0; height: 100vh; display: flex; flex-direction: column; }
        
        /* 顶部导航 Tab */
        .nav-header { background: var(--card-bg); padding: 10px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); z-index: 100; display: flex; justify-content: center; gap: 20px; }
        .nav-btn { padding: 8px 20px; border: none; background: none; font-size: 1em; font-weight: 600; color: #666; cursor: pointer; border-bottom: 3px solid transparent; transition: all 0.3s; }
        .nav-btn.active { color: var(--primary); border-bottom-color: var(--primary); }
        .nav-btn:hover { color: var(--primary); }

        /* 主内容区域 */
        .view-container { flex: 1; overflow-y: auto; padding: 20px; display: none; }
        .view-container.active { display: block; }

        /* === 视图 1: 仪表盘样式 === */
        .dashboard-panel { display: grid; grid-template-columns: 1fr 2fr; grid-template-rows: 1fr; gap: 20px; margin-bottom: 20px; max-width: 1000px; margin-left: auto; margin-right: auto; align-items: stretch; }
        .status-card { background: var(--card-bg); padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
        .status-item { margin-bottom: 15px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #eee; padding-bottom: 10px; }
        .status-item:last-child { border-bottom: none; }
        .badge { padding: 4px 8px; border-radius: 4px; font-size: 0.9em; color: white; font-weight: bold; }
        .bg-green { background-color: #28a745; } .bg-red { background-color: #dc3545; } .bg-blue { background-color: #17a2b8; }
        
        .console-window { background: var(--console-bg); color: var(--console-text); padding: 15px; border-radius: 8px; font-family: monospace; font-size: 0.85em; height: 150px; overflow-y: auto; white-space: pre-wrap; }
        
        .transcript-card { background: var(--card-bg); border-radius: 8px; margin-bottom: 15px; padding: 20px; max-width: 960px; margin-left: auto; margin-right: auto; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .transcript-card.new-item { border-left: 4px solid #28a745; background-color: #f8fff9; }
        .card-meta { display: flex; justify-content: space-between; color: #888; font-size: 0.85em; margin-bottom: 10px; border-bottom: 1px solid #eee; padding-bottom: 5px; }
        .filename { font-weight: 600; color: #444; }
        .segment { display: flex; gap: 10px; margin-bottom: 4px; }
        .timestamp { font-family: monospace; color: #999; font-size: 0.8em; min-width: 80px; }

        /* === 视图 2: 时光对话样式 (Chat) === */
        .chat-container { max-width: 800px; margin: 0 auto; }
        .chat-date-separator { text-align: center; margin: 20px 0; }
        .chat-date-label { background-color: rgba(0,0,0,0.05); color: #666; padding: 4px 12px; border-radius: 12px; font-size: 0.85em; }
        .file-separator { text-align: center; margin: 15px 0; font-size: 0.8em; color: #aaa; display: flex; align-items: center; gap: 10px; }
        .file-separator::before, .file-separator::after { content: ""; flex: 1; height: 1px; background: #ddd; }

        .chat-bubble-row { display: flex; margin-bottom: 15px; gap: 10px; }
        .avatar { width: 40px; height: 40px; background-color: #ccc; border-radius: 50%; display: flex; justify-content: center; align-items: center; font-weight: bold; color: white; font-size: 0.9em; flex-shrink: 0; }
        
        .bubble-content { max-width: 70%; display: flex; flex-direction: column; }
        .speaker-name { font-size: 0.75em; color: #888; margin-bottom: 2px; margin-left: 5px; }
        .bubble { background-color: var(--chat-other); padding: 10px 14px; border-radius: 0 12px 12px 12px; position: relative; box-shadow: 0 1px 2px rgba(0,0,0,0.1); font-size: 1em; line-height: 1.5; }
        .chat-time { font-size: 0.7em; color: #999; text-align: right; margin-top: 4px; margin-right: 5px; }

        /* === 视图 3: 统计分析样式 (新增) === */
        .analysis-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 12px; margin-bottom: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
        .analysis-card h3 { margin: 0 0 10px 0; font-size: 1.5em; }
        .analysis-card p { margin: 0; opacity: 0.9; }
        .speaker-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 20px; }
        .speaker-card { background: var(--card-bg); border-radius: 12px; padding: 20px; display: flex; align-items: center; gap: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); transition: transform 0.2s; border: 1px solid #eee; }
        .speaker-card:hover { transform: translateY(-3px); box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
        .speaker-icon { width: 60px; height: 60px; border-radius: 50%; display: flex; justify-content: center; align-items: center; font-size: 1.5em; font-weight: bold; color: white; flex-shrink: 0; }
        .speaker-info { flex: 1; }
        .speaker-info h4 { margin: 0 0 10px 0; color: #333; font-size: 1.1em; }
        .speaker-stats-detail { display: flex; justify-content: space-between; background: #f8f9fa; padding: 8px 12px; border-radius: 8px; text-align: center; }
        
        /* 头像颜色 (限定为 5 种高对比度色) */
        .avatar-0 { background: #1A53E0; } /* 亮蓝色 */
        .avatar-1 { background: #28A745; } /* 鲜绿色 */
        .avatar-2 { background: #FF7733; } /* 亮橙色 */
        .avatar-3 { background: #8E44AD; } /* 深紫色 */
        .avatar-4 { background: #DC3545; } /* 鲜红色 */

        /* === 视图 4: 实时日志样式 === */
        .logs-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 1px solid #eee; }
        .logs-controls { display: flex; gap: 10px; align-items: center; }
        .logs-status { display: flex; gap: 10px; align-items: center; }
        .logs-container { height: calc(100vh - 200px); }
        .logs-window { 
            background: var(--console-bg); 
            color: var(--console-text); 
            padding: 15px; 
            border-radius: 8px; 
            font-family: monospace; 
            font-size: 0.85em; 
            height: 100%; 
            overflow-y: auto; 
            white-space: pre-wrap; 
        }
        .btn { padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; }
        .btn-primary { background-color: var(--primary); color: white; }
        .btn-secondary { background-color: #6c757d; color: white; }

    </style>
</head>
<body>

    <div class="nav-header">
        <button class="nav-btn active" onclick="switchTab('dashboard')">️ 仪表盘</button>
        <button class="nav-btn" onclick="switchTab('chat')"> 时光对话</button>
        <button class="nav-btn" onclick="switchTab('analysis')">📊 统计分析</button>
        <button class="nav-btn" onclick="switchTab('logs')">📄 实时日志</button>
        <button class="nav-btn" onclick="switchTab('config')">⚙️ 配置管理</button>
    </div>

    <div id="view-dashboard" class="view-container active">
        <div class="dashboard-panel">
            <div class="status-card">
                <div class="status-item"><span class="status-label">PC 服务状态</span><span id="status-asr" class="badge bg-red">检测中...</span></div>
                <div class="status-item"><span class="status-label">排队文件数</span><span id="status-files" class="badge bg-blue">0</span></div>
                <div class="status-item"><span class="status-label">Web 界面</span><span class="badge bg-green">在线</span></div>
            </div>
            <div class="console-window">
                <div style="border-bottom:1px solid #444; margin-bottom:5px; color:#888;">root@NAS: monitor_logs (实时)</div>
                <div id="log-display">正在连接日志流...</div>
            </div>
        </div>
        <div id="dashboard-content">
            <div style="text-align: center; color: #999;">加载中...</div>
        </div>
    </div>

    <div id="view-chat" class="view-container">
        <div style="max-width: 800px; margin: 0 auto; padding: 10px; display: flex; justify-content: space-between; align-items: center;">
            <h3 style="margin: 0; color: #007bff;">时光对话</h3>
            <div>
                <select id="chat-session-select" style="margin-right: 10px; padding: 5px;">
                    <option value="">选择历史会话...</option>
                </select>
                <button id="save-chat-btn" class="btn btn-primary" style="padding: 5px 15px;">保存对话</button>
                <button id="new-chat-btn" class="btn btn-secondary" style="padding: 5px 15px; margin-left: 5px;">新建会话</button>
            </div>
        </div>
        <div class="chat-container" id="chat-content">
            <div style="text-align: center; color: #999; margin-top: 50px;">正在生成对话流...</div>
        </div>
    </div>
    
    <div id="view-analysis" class="view-container">
        <div id="analysis-content" style="max-width: 1000px; margin: 0 auto;">
            <div style="text-align: center; color: #999; margin-top: 50px;">正在分析声纹数据...</div>
        </div>
    </div>

    <div id="view-logs" class="view-container">
        <div class="logs-header">
            <h2>📄 实时日志</h2>
            <div class="logs-controls">
                <button id="toggle-logs-btn" class="btn btn-primary">开始接收日志</button>
                <button id="clear-logs-btn" class="btn btn-secondary">清空日志</button>
                <div class="logs-status">
                    <span id="logs-connection-status" class="badge bg-red">未连接</span>
                    <span id="logs-lines-count">0 行</span>
                </div>
            </div>
        </div>
        <div class="logs-container">
            <div id="logs-display" class="logs-window"></div>
        </div>
    </div>

    <div id="view-config" class="view-container">
        <div id="config-content" style="max-width: 1000px; margin: 0 auto; padding: 20px;">
            <h2 style="color: #007bff; margin-bottom: 20px;">系统配置</h2>
            <div id="config-form">
                <div style="text-align: center; color: #999;">正在加载配置...</div>
            </div>
            <div style="margin-top: 20px; text-align: center;">
                <button id="save-config-btn" class="btn btn-primary" style="padding: 8px 30px; font-size: 16px;">保存配置</button>
                <div id="save-status" style="margin-top: 10px; color: #28a745; font-weight: bold;"></div>
            </div>
        </div>
    </div>

    <script>
        let lastDataFingerprint = "";
        const speakerColorMap = {};
        let nextColorIndex = 0;

        function switchTab(tabName) {
            document.querySelectorAll('.view-container').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));
            document.getElementById('view-' + tabName).classList.add('active');
            
            const btns = document.querySelectorAll('.nav-btn');
            if(tabName === 'dashboard') btns[0].classList.add('active');
            else if(tabName === 'chat') btns[1].classList.add('active');
            else if(tabName === 'analysis') btns[2].classList.add('active');
            else if(tabName === 'logs') btns[3].classList.add('active');
            else if(tabName === 'config') btns[4].classList.add('active');
            
            // Load config when switching to config tab
            if (tabName === 'config') {
                loadConfig();
            }
            
            // Initialize logs when switching to logs tab
            if (tabName === 'logs') {
                initLogsView();
            }
        }

        // Load configuration from API
        function loadConfig() {
            fetch('/api/config')
                .then(response => response.json())
                .then(config => {
                    const form = document.getElementById('config-form');
                    form.innerHTML = '';
                    
                    for (const key in config) {
                        const value = config[key];
                        const div = document.createElement('div');
                        div.style.marginBottom = '15px';
                        
                        const label = document.createElement('label');
                        label.textContent = key;
                        label.style.display = 'block';
                        label.style.marginBottom = '5px';
                        label.style.fontWeight = 'bold';
                        
                        const input = document.createElement('input');
                        input.type = typeof value === 'number' ? 'number' : 'text';
                        input.value = value;
                        input.id = 'config-' + key;
                        input.style.width = '100%';
                        input.style.padding = '8px';
                        input.style.border = '1px solid #ccc';
                        input.style.borderRadius = '4px';
                        input.style.fontSize = '14px';
                        
                        div.appendChild(label);
                        div.appendChild(input);
                        form.appendChild(div);
                    }
                })
                .catch(error => {
                    const form = document.getElementById('config-form');
                    form.innerHTML = `<div style="text-align: center; color: #dc3545;">加载配置失败: ${error.message}</div>`;
                });
        }

        // 对话历史相关功能
        let currentChatData = [];
        let currentSessionId = null;

        // 加载聊天会话列表
        function loadChatSessions() {
            fetch('/api/chat/sessions')
                .then(response => response.json())
                .then(sessions => {
                    const select = document.getElementById('chat-session-select');
                    select.innerHTML = '<option value="">选择历史会话...</option>';
                    
                    sessions.forEach(session => {
                        const option = document.createElement('option');
                        option.value = session.session_id;
                        // 格式化日期显示
                        const createdDate = new Date(session.created_at).toLocaleString();
                        option.textContent = `${session.session_id} (${createdDate})`;
                        select.appendChild(option);
                    });
                })
                .catch(error => {
                    console.error('加载聊天会话失败:', error);
                });
        }

        // 保存当前对话
        function saveChatSession() {
            if (!currentChatData || currentChatData.length === 0) {
                alert('没有可保存的对话内容');
                return;
            }

            const sessionName = prompt('请输入会话名称:', `对话_${new Date().toLocaleDateString()}`);
            if (!sessionName) return;

            const sessionId = sessionName.replace(/\s+/g, '_').replace(/[^\w\u4e00-\u9fa5]/g, '');
            
            // 准备符合后端期望的数据结构
            const chatData = [];
            currentChatData.forEach(item => {
                if (item.segments) {
                    chatData.push({
                        segments: item.segments
                    });
                } else if (item.message_text) {
                    // 如果是已加载的历史消息，转换为segments格式
                    chatData.push({
                        segments: [{
                            text: item.message_text,
                            spk_id: item.speaker_id,
                            speaker_name: item.speaker_name,
                            start: item.timestamp
                        }]
                    });
                }
            });
            
            fetch('/api/chat/session', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: sessionId,
                    chat_data: chatData
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('对话保存成功!');
                    loadChatSessions(); // 重新加载会话列表
                } else {
                    alert('保存失败: ' + data.message);
                }
            })
            .catch(error => {
                alert('保存失败: ' + error.message);
            });
        }

        // 加载特定会话的对话
        function loadChatSession(sessionId) {
            if (!sessionId) {
                // 如果没有选择会话，则加载当前转录数据
                updateLoop();
                return;
            }

            fetch(`/api/chat/session/${sessionId}`)
                .then(response => response.json())
                .then(messages => {
                    if (messages && messages.length > 0) {
                        currentSessionId = sessionId;
                        // 将消息转换为前端渲染需要的格式
                        currentChatData = messages.map(msg => ({
                            message_text: msg.message_text,
                            speaker_id: msg.speaker_id,
                            speaker_name: msg.speaker_name,
                            timestamp: msg.timestamp,
                            start_fmt: msg.start_fmt
                        }));
                        renderChat(currentChatData);
                    } else {
                        alert('没有找到该会话的对话内容');
                    }
                })
                .catch(error => {
                    alert('加载对话失败: ' + error.message);
                });
        }

        // 新建会话
        function newChatSession() {
            currentSessionId = null;
            document.getElementById('chat-session-select').value = '';
            updateLoop(); // 加载当前转录数据
        }

        // 添加事件监听器
        document.addEventListener('DOMContentLoaded', function() {
            // 保存对话按钮
            document.getElementById('save-chat-btn').addEventListener('click', saveChatSession);
            
            // 新建会话按钮
            document.getElementById('new-chat-btn').addEventListener('click', newChatSession);
            
            // 会话选择下拉框
            document.getElementById('chat-session-select').addEventListener('change', function() {
                loadChatSession(this.value);
            });
            
            // 初始加载会话列表
            loadChatSessions();
        });

        // Save configuration to API
        document.getElementById('save-config-btn')?.addEventListener('click', () => {
            const form = document.getElementById('config-form');
            const inputs = form.querySelectorAll('input');
            const newConfig = {};
            
            inputs.forEach(input => {
                const key = input.id.replace('config-', '');
                const value = input.value;
                
                if (input.type === 'number') {
                    newConfig[key] = parseInt(value) || parseFloat(value) || value;
                } else {
                    newConfig[key] = value;
                }
            });
            
            fetch('/api/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(newConfig)
            })
            .then(response => {
                if (!response.ok) {
                    return response.text().then(text => {
                        throw new Error(`HTTP error! status: ${response.status}, message: ${text}`);
                    });
                }
                return response.json();
            })
            .then(data => {
                const status = document.getElementById('save-status');
                if (data.success) {
                    status.textContent = '配置保存成功!';
                    status.style.color = '#28a745';
                    setTimeout(() => status.textContent = '', 2000);
                } else {
                    status.textContent = '保存失败: ' + data.message;
                    status.style.color = '#dc3545';
                    setTimeout(() => status.textContent = '', 2000);
                }
            })
            .catch(err => {
                const status = document.getElementById('save-status');
                status.textContent = `保存失败: ${err.message}`;
                status.style.color = '#dc3545';
                setTimeout(() => status.textContent = '', 2000);
            });
        });
        
        // --- 核心辅助函数 ---

        // 1. 深度文本清洗：去除标签、标点、空格
        function cleanText(text) {
            if (!text) return "";
            // 去除 SenseVoice 标签
            let clean = text.replace(/<\|.*?\|>/g, "");
            return clean;
        }

        // 2. 检查是否包含有效内容 (过滤掉只有标点符号的情况)
        function hasMeaningfulContent(text) {
            if (!text) return false;
            const clean = cleanText(text);
            // 去除所有标点符号、空格、换行
            // 匹配：英文标点, 中文标点, 空白符
            const stripped = clean.replace(/[.,\/#!$%\^&\*;:{}=\-_`~()。，、？！：；“”‘’\s]/g, "");
            return stripped.length > 0;
        }

        // 3. 获取头像颜色索引 (使用映射表确保颜色稳定)
        function getAvatarIndex(spkId) {
            if (spkId in speakerColorMap) {
                return speakerColorMap[spkId];
            }
            const colorIndex = nextColorIndex % 5;
            speakerColorMap[spkId] = colorIndex;
            nextColorIndex++;
            return colorIndex;
        }

        // 4. 预处理统计数据
        function processStats(items) {
            items.forEach(item => {
                const stats = {};
                if (item.segments && item.segments.length > 0) {
                    item.segments.forEach(seg => {
                        // 只有有效内容才计入统计
                        if (!hasMeaningfulContent(seg.text)) return;

                        const spkId = seg.spk_id !== undefined ? seg.spk_id : 'unknown';
                        const spkName = typeof seg.spk_id === 'number' ? `说话人 ${seg.spk_id}` : (seg.spk_id || "未知");
                        
                        const key = String(spkId); 
                        if (!stats[key]) {
                            stats[key] = {
                                original_id: spkId,
                                speaker_name: spkName,
                                count: 0,
                                total_duration: 0
                            };
                        }
                        stats[key].count += 1;
                        const dur = (seg.end && seg.start) ? (seg.end - seg.start) : 0;
                        stats[key].total_duration += dur;
                    });
                }
                item.speaker_stats = stats;
            });
            return items;
        }

        async function updateLoop() {
            try {
                const statusRes = await fetch('/api/status');
                const statusData = await statusRes.json();
                const asrBadge = document.getElementById('status-asr');
                if (statusData.asr_server === 'online') {
                    asrBadge.innerText = "在线"; asrBadge.className = "badge bg-green";
                } else {
                    asrBadge.innerText = "离线"; asrBadge.className = "badge bg-red";
                }
                document.getElementById('status-files').innerText = statusData.pending_files;
                document.getElementById('log-display').innerText = statusData.last_log;
                const consoleWin = document.querySelector('.console-window');
                consoleWin.scrollTop = consoleWin.scrollHeight;

                const dataRes = await fetch('/api/data');
                let items = await dataRes.json();
                
                // 更新当前对话数据
                currentChatData = items;
                
                items = processStats(items);
                
                if (items.length === 0) return;
                const currentFingerprint = items.length + "_" + items[0].id;
                if (currentFingerprint === lastDataFingerprint) return;
                lastDataFingerprint = currentFingerprint;

                renderDashboard(items);
                renderChat(items);
                renderAnalysis(items);

            } catch (e) { console.error(e); }
        }

        function renderDashboard(items) {
            const container = document.getElementById('dashboard-content');
            let html = "";
            items.forEach(item => {
                // === 严格过滤 ===
                // 如果全文都没有有效内容(去标点后为空)，直接跳过整张卡片
                let hasValidContent = false;
                if (item.segments && item.segments.length > 0) {
                    hasValidContent = item.segments.some(seg => hasMeaningfulContent(seg.text));
                } else {
                    hasValidContent = hasMeaningfulContent(item.full_text);
                }
                if (!hasValidContent) return; // 跳过无效文件
                // ===============

                let segHtml = "";
                if (item.segments && item.segments.length > 0) {
                    item.segments.forEach(seg => {
                        if (!hasMeaningfulContent(seg.text)) return; // 跳过无效片段
                        const txt = cleanText(seg.text);
                        segHtml += `<div class="segment"><span class="timestamp">[${seg.start_fmt}]</span><span>${txt}</span></div>`;
                    });
                } else {
                    const txt = cleanText(item.full_text);
                    if (txt) segHtml = `<div class="segment"><span>${txt}</span></div>`;
                }
                
                // 二次检查：如果过滤后没有 segHtml 了，也不渲染
                if (!segHtml) return;

                html += `
                    <div class="transcript-card ${item.is_new ? 'new-item' : ''}">
                        <div class="card-meta"><span class="filename">${item.filename}</span><span>${item.time_full}</span></div>
                        <div>${segHtml}</div>
                    </div>`;
            });
            container.innerHTML = html;
        }

        function renderChat(items) {
            const container = document.getElementById('chat-content');
            let html = "";
            let currentDay = "";

            items.forEach(item => {
                // === 严格过滤 ===
                let hasValidContent = false;
                if (item.segments && item.segments.length > 0) {
                    hasValidContent = item.segments.some(seg => hasMeaningfulContent(seg.text));
                } else {
                    hasValidContent = hasMeaningfulContent(item.full_text);
                }
                if (!hasValidContent) return; 
                // ===============

                if (item.date_group !== currentDay) {
                    html += `<div class="chat-date-separator"><span class="chat-date-label">${item.date_group}</span></div>`;
                    currentDay = item.date_group;
                }
                html += `<div class="file-separator">来源: ${item.filename} (${item.time_simple})</div>`;

                if (item.segments && item.segments.length > 0) {
                    item.segments.forEach(seg => {
                        if (!hasMeaningfulContent(seg.text)) return; // 跳过无效气泡
                        
                        const txt = cleanText(seg.text);
                        const spkId = seg.spk_id !== undefined ? seg.spk_id : 0;
                        let spkName = typeof spkId === 'number' ? `说话人 ${spkId}` : spkId;
                        let avatarIdx = getAvatarIndex(spkId);
                        
                        // 截取名字的第一个字作为头像文字
                        let iconText = spkName;
                        if(iconText.length > 0) iconText = iconText.slice(0, 1);

                        html += `
                            <div class="chat-bubble-row">
                                <div class="avatar avatar-${avatarIdx % 5}">${iconText}</div>
                                <div class="bubble-content">
                                    <div class="speaker-name">${spkName}</div>
                                    <div class="bubble">${txt}</div>
                                    <div class="chat-time">${seg.start_fmt}</div>
                                </div>
                            </div>
                        `;
                    });
                } else {
                    const txt = cleanText(item.full_text);
                    // 截取名字的第一个字作为头像文字
                    let iconText = "未知"; // Default for full_text without speaker
                    if (item.speaker_stats && Object.keys(item.speaker_stats).length > 0) {
                        const firstSpkId = Object.keys(item.speaker_stats)[0];
                        const firstSpkName = item.speaker_stats[firstSpkId].speaker_name;
                        if (firstSpkName.length > 0) iconText = firstSpkName.slice(0, 1);
                    }
                    let avatarIdx = getAvatarIndex(0); // Default avatar for full_text
                    html += `
                        <div class="chat-bubble-row">
                             <div class="avatar avatar-${avatarIdx % 5}">${iconText}</div>
                             <div class="bubble-content">
                                <div class="bubble">${txt}</div>
                                <div class="chat-time">来源时间: ${item.time_simple}</div>
                             </div>
                        </div>`;
                }
            });
            container.innerHTML = html;
        }

        function renderAnalysis(items) {
            const container = document.getElementById('analysis-content');
            const globalSpeakerStats = {};
            let totalFiles = items.length;
            
            items.forEach(item => {
                if (item.speaker_stats) {
                    for (const [key, stats] of Object.entries(item.speaker_stats)) {
                        if (!globalSpeakerStats[key]) {
                            globalSpeakerStats[key] = {
                                original_id: stats.original_id,
                                name: stats.speaker_name,
                                totalCount: 0,
                                totalDuration: 0,
                                filesParticipated: new Set()
                            };
                        }
                        globalSpeakerStats[key].totalCount += stats.count;
                        globalSpeakerStats[key].totalDuration += stats.total_duration;
                        globalSpeakerStats[key].filesParticipated.add(item.filename);
                    }
                }
            });
            
            let html = `
                <div class="analysis-card">
                    <h3>📊 声纹识别统计分析</h3>
                    <p>共分析 ${totalFiles} 个录音文件，识别出 ${Object.keys(globalSpeakerStats).length} 位不同的说话人</p>
                </div>
                <div class="speaker-grid">`;
            
            const sortedStats = Object.values(globalSpeakerStats).sort((a, b) => b.totalCount - a.totalCount);

            for (const stats of sortedStats) {
                const avgDuration = stats.totalCount > 0 ? (stats.totalDuration / stats.totalCount / 1000).toFixed(1) : 0;
                const filesCount = stats.filesParticipated.size;
                const avatarIdx = getAvatarIndex(stats.original_id);
                
                // 截取名字的第一个字作为头像文字
                let iconText = stats.name;
                if(iconText.length > 0) iconText = iconText.slice(0, 1);
                
                html += `
                    <div class="speaker-card">
                        <div class="speaker-icon avatar-${avatarIdx % 5}">
                            ${iconText}
                        </div>
                        <div class="speaker-info">
                            <h4>${stats.name}</h4>
                            <div class="speaker-stats-detail">
                                <div><div style="font-weight: bold;">${stats.totalCount}</div><div style="font-size: 0.8em;">发言次数</div></div>
                                <div><div style="font-weight: bold;">${avgDuration}s</div><div style="font-size: 0.8em;">平均时长</div></div>
                                <div><div style="font-weight: bold;">${filesCount}</div><div style="font-size: 0.8em;">参与文件</div></div>
                            </div>
                        </div>
                    </div>`;
            }
            
            html += '</div>';
            container.innerHTML = html;
        }

        // === 实时日志功能 ===
        let logsEventSource = null;
        let logsLineCount = 0;
        let isLogsConnected = false;

        // 初始化日志视图
        function initLogsView() {
            // 如果已经连接，不需要重新初始化
            if (logsEventSource && isLogsConnected) {
                return;
            }
            
            // 设置按钮事件
            const toggleBtn = document.getElementById('toggle-logs-btn');
            const clearBtn = document.getElementById('clear-logs-btn');
            
            toggleBtn.onclick = toggleLogsConnection;
            clearBtn.onclick = clearLogsDisplay;
            
            // 更新状态显示
            updateLogsStatus();
        }

        // 切换日志连接状态
        function toggleLogsConnection() {
            const toggleBtn = document.getElementById('toggle-logs-btn');
            
            if (logsEventSource && isLogsConnected) {
                // 断开连接
                logsEventSource.close();
                logsEventSource = null;
                isLogsConnected = false;
                toggleBtn.textContent = '开始接收日志';
                updateLogsStatus();
            } else {
                // 建立连接
                connectToLogsStream();
                toggleBtn.textContent = '停止接收日志';
            }
        }

        // 连接到SSE日志流
        function connectToLogsStream() {
            try {
                // 使用服务端的SSE日志流URL
                const asrLogStreamUrl = '{{ CONFIG["ASR_LOG_STREAM_URL"] }}';
                logsEventSource = new EventSource(asrLogStreamUrl);
                
                logsEventSource.onopen = function() {
                    isLogsConnected = true;
                    updateLogsStatus();
                    addLogMessage('系统', '正在连接到日志流...', 'info');
                };
                
                logsEventSource.onmessage = function(event) {
                    try {
                        const data = JSON.parse(event.data);
                        
                        if (data.type === 'log') {
                            addLogMessage('日志', data.message, 'log');
                        } else if (data.type === 'connected') {
                            addLogMessage('系统', data.message, 'success');
                            isLogsConnected = true;
                            updateLogsStatus();
                        } else if (data.type === 'heartbeat') {
                            // 心跳消息，不显示但更新连接状态
                            isLogsConnected = true;
                        } else if (data.type === 'error') {
                            addLogMessage('错误', data.message, 'error');
                        } else if (data.type === 'info') {
                            addLogMessage('信息', data.message, 'info');
                        }
                    } catch (e) {
                        // 如果不是JSON格式，直接显示为日志
                        addLogMessage('日志', event.data, 'log');
                    }
                };
                
                logsEventSource.onerror = function() {
                    isLogsConnected = false;
                    updateLogsStatus();
                    addLogMessage('系统', '日志流连接错误，5秒后尝试重连...', 'error');
                    
                    // 5秒后尝试重连
                    setTimeout(function() {
                        if (document.getElementById('view-logs').classList.contains('active') && !isLogsConnected) {
                            connectToLogsStream();
                        }
                    }, 5000);
                };
                
            } catch (e) {
                addLogMessage('错误', '无法创建日志流连接: ' + e.message, 'error');
            }
        }

        // 添加日志消息到显示区域
        function addLogMessage(source, message, type) {
            const logsDisplay = document.getElementById('logs-display');
            const timestamp = new Date().toLocaleTimeString();
            
            // 根据类型设置不同的颜色
            let colorClass = '';
            if (type === 'error') {
                colorClass = 'color: #ff6b6b;';
            } else if (type === 'success') {
                colorClass = 'color: #51cf66;';
            } else if (type === 'info') {
                colorClass = 'color: #74c0fc;';
            } else {
                colorClass = 'color: var(--console-text);';
            }
            
            const logEntry = document.createElement('div');
            logEntry.style.marginBottom = '2px';
            logEntry.innerHTML = `<span style="color: #888;">[${timestamp}]</span> <span style="color: #aaa;">[${source}]</span> <span style="${colorClass}">${message}</span>`;
            
            logsDisplay.appendChild(logEntry);
            logsLineCount++;
            
            // 自动滚动到底部
            logsDisplay.scrollTop = logsDisplay.scrollHeight;
            
            // 限制显示的行数，防止内存占用过多
            const maxLines = 1000;
            if (logsDisplay.children.length > maxLines) {
                logsDisplay.removeChild(logsDisplay.firstChild);
            }
            
            updateLogsStatus();
        }

        // 清空日志显示
        function clearLogsDisplay() {
            const logsDisplay = document.getElementById('logs-display');
            logsDisplay.innerHTML = '';
            logsLineCount = 0;
            updateLogsStatus();
        }

        // 更新日志状态显示
        function updateLogsStatus() {
            const statusBadge = document.getElementById('logs-connection-status');
            const linesCount = document.getElementById('logs-lines-count');
            
            if (isLogsConnected) {
                statusBadge.textContent = '已连接';
                statusBadge.className = 'badge bg-green';
            } else {
                statusBadge.textContent = '未连接';
                statusBadge.className = 'badge bg-red';
            }
            
            linesCount.textContent = `${logsLineCount} 行`;
        }

        setInterval(updateLoop, 3000);
        updateLoop();
    </script>
</body>
</html>
"""

@app.route('/api/status')
def api_status():
    return jsonify(get_system_status())

@app.route('/api/data')
def api_data():
    return jsonify(get_transcripts())

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
            log_file.write(log_message + '\n')
        
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
            log_file.write(log_message + '\n')
        
        return jsonify(success=True, message="Configuration updated successfully")
    # 记录无效请求
    log_message = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] API Config Update - Invalid JSON data received"
    with open(CONFIG["LOG_FILE_PATH"], 'a', encoding='utf-8') as log_file:
        log_file.write(log_message + '\n')
    return jsonify(success=False, message="Invalid JSON data"), 400

# 对话历史API端点
@app.route('/api/chat/sessions', methods=['GET'])
def api_get_chat_sessions():
    """获取所有聊天会话列表"""
    try:
        # 确保对话历史表已初始化
        init_chat_history_db()
        sessions = get_chat_sessions()
        return jsonify(sessions)
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/api/chat/session/<session_id>', methods=['GET'])
def api_get_chat_session(session_id):
    """获取特定聊天会话的所有消息"""
    try:
        # 确保对话历史表已初始化
        init_chat_history_db()
        messages = get_chat_session_messages(session_id)
        return jsonify(messages)
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/api/chat/session', methods=['POST'])
def api_save_chat_session():
    """保存聊天会话"""
    try:
        # 确保对话历史表已初始化
        init_chat_history_db()
        
        data = request.get_json(silent=True)
        if not data or 'session_id' not in data or 'chat_data' not in data:
            return jsonify(success=False, message="Missing required fields: session_id, chat_data"), 400
        
        session_id = data['session_id']
        chat_data = data['chat_data']
        
        success = save_chat_session(session_id, chat_data)
        if success:
            return jsonify(success=True, message="Chat session saved successfully")
        else:
            return jsonify(success=False, message="Failed to save chat session"), 500
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500

@app.route('/api/chat/session/<session_id>', methods=['DELETE'])
def api_delete_chat_session(session_id):
    """删除特定聊天会话"""
    try:
        # 确保对话历史表已初始化
        init_chat_history_db()
        
        success = delete_chat_session(session_id)
        if success:
            return jsonify(success=True, message="Chat session deleted successfully")
        else:
            return jsonify(success=False, message="Failed to delete chat session"), 500
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500

# SSE日志流接口
@app.route('/logs/stream')
def stream_logs():
    """SSE日志流接口，实时传递服务端日志"""
    def generate():
        # 确定日志文件路径，先尝试CONFIG中的路径，然后尝试常见位置
        log_paths = [
            CONFIG["LOG_FILE_PATH"],
            "transcribe.log",
            "asr_server.log",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcribe.log"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "asr_server.log")
        ]
        
        log_path = None
        for path in log_paths:
            if os.path.exists(path):
                log_path = path
                break
        
        # 如果没有找到日志文件，使用默认路径
        if not log_path:
            log_path = "transcribe.log"
            # 尝试创建一个空日志文件
            try:
                with open(log_path, 'w', encoding='utf-8') as f:
                    f.write(f"# 日志文件创建于 {datetime.datetime.now()}\n")
            except:
                pass
        
        # 获取日志文件的初始大小
        last_size = 0
        try:
            if os.path.exists(log_path):
                last_size = os.path.getsize(log_path)
        except:
            pass
        
        # 发送连接确认
        yield f"data: {json.dumps({'type': 'connected', 'message': f'已连接到日志流: {log_path}'})}\n\n"
        
        # 读取最后几行作为初始内容
        try:
            if os.path.exists(log_path) and last_size > 0:
                with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    # 只发送最后10行作为初始内容
                    for line in lines[-10:]:
                        if line.strip():  # 只发送非空行
                            yield f"data: {json.dumps({'type': 'log', 'message': line.strip()})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'读取初始日志内容失败: {str(e)}'})}\n\n"
        
        # 持续监控日志文件
        last_heartbeat = time.time()
        while True:
            try:
                current_time = time.time()
                
                # 每30秒发送一次心跳
                if current_time - last_heartbeat > 30:
                    yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': int(current_time)})}\n\n"
                    last_heartbeat = current_time
                
                # 检查文件是否存在
                if os.path.exists(log_path):
                    try:
                        current_size = os.path.getsize(log_path)
                        
                        # 如果文件大小增加，读取新内容
                        if current_size > last_size:
                            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                                f.seek(last_size)
                                new_lines = f.readlines()
                                for line in new_lines:
                                    if line.strip():  # 只发送非空行
                                        yield f"data: {json.dumps({'type': 'log', 'message': line.strip()})}\n\n"
                            last_size = current_size
                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'error', 'message': f'读取新日志内容失败: {str(e)}'})}\n\n"
                else:
                    # 文件不存在，尝试重新创建
                    try:
                        with open(log_path, 'w', encoding='utf-8') as f:
                            f.write(f"# 日志文件重新创建于 {datetime.datetime.now()}\n")
                        last_size = 0
                        yield f"data: {json.dumps({'type': 'info', 'message': '日志文件已重新创建'})}\n\n"
                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'error', 'message': f'无法创建日志文件: {str(e)}'})}\n\n"
                
                # 短暂休眠，减少CPU使用
                time.sleep(1)
                
            except GeneratorExit:
                # 客户端断开连接
                break
            except Exception as e:
                # 发送错误信息
                yield f"data: {json.dumps({'type': 'error', 'message': f'日志监控错误: {str(e)}'})}\n\n"
                time.sleep(5)  # 出错时等待更长时间
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, CONFIG=CONFIG)

if __name__ == "__main__":
    args = parse_args()
    update_config(args)
    
    # 初始化对话历史数据库表
    init_chat_history_db()
    
    app.run(host='0.0.0.0', port=CONFIG["WEB_PORT"], debug=False)