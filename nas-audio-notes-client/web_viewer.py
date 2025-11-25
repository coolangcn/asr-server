#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import sys
from flask import Flask, render_template_string, jsonify, request, Response
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
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Audio Notes | 智能语音笔记</title>
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- FontAwesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    fontFamily: {
                        sans: ['Inter', 'sans-serif'],
                        mono: ['JetBrains Mono', 'monospace'],
                    },
                    colors: {
                        gray: {
                            850: '#1f2937',
                            900: '#111827',
                            950: '#030712',
                        },
                        primary: {
                            400: '#818cf8',
                            500: '#6366f1',
                            600: '#4f46e5',
                        },
                        accent: {
                            400: '#2dd4bf',
                            500: '#14b8a6',
                        }
                    },
                    animation: {
                        'fade-in': 'fadeIn 0.5s ease-out',
                        'slide-up': 'slideUp 0.5s ease-out',
                        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                    },
                    keyframes: {
                        fadeIn: {
                            '0%': { opacity: '0' },
                            '100%': { opacity: '1' },
                        },
                        slideUp: {
                            '0%': { transform: 'translateY(20px)', opacity: '0' },
                            '100%': { transform: 'translateY(0)', opacity: '1' },
                        }
                    }
                }
            }
        }
    </script>

    <style>
        /* 自定义滚动条 */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: #111827; 
        }
        ::-webkit-scrollbar-thumb {
            background: #374151; 
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #4b5563; 
        }

        /* 玻璃拟态效果 */
        .glass {
            background: rgba(17, 24, 39, 0.7);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .glass-card {
            background: rgba(31, 41, 55, 0.4);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.03);
            transition: all 0.3s ease;
        }
        
        .glass-card:hover {
            background: rgba(31, 41, 55, 0.6);
            border-color: rgba(99, 102, 241, 0.3);
            transform: translateY(-2px);
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        }

        /* 霓虹光效 */
        .neon-text {
            text-shadow: 0 0 10px rgba(99, 102, 241, 0.5);
        }
        
        .status-dot {
            box-shadow: 0 0 8px currentColor;
        }

        /* 聊天气泡 */
        .chat-bubble {
            position: relative;
            z-index: 1;
        }
        .chat-bubble::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0) 100%);
            z-index: -1;
            border-radius: inherit;
        }

        body {
            background-color: #030712;
            background-image: 
                radial-gradient(circle at 15% 50%, rgba(99, 102, 241, 0.08) 0%, transparent 25%),
                radial-gradient(circle at 85% 30%, rgba(20, 184, 166, 0.08) 0%, transparent 25%);
            color: #f3f4f6;
        }
        
        .line-clamp-3 {
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
    
        /* 移动端优化 */
        @media (max-width: 768px) {
            /* 隐藏侧边栏,使用全屏布局 */
            aside {
                position: fixed;
                left: -100%;
                top: 0;
                height: 100vh;
                z-index: 1000;
                transition: left 0.3s ease;
            }
            
            aside.mobile-open {
                left: 0;
            }
            
            /* 添加遮罩层 */
            .mobile-overlay {
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0, 0, 0, 0.5);
                z-index: 999;
            }
            
            .mobile-overlay.active {
                display: block;
            }
            
            /* 主内容区占满屏幕 */
            main {
                width: 100%;
            }
            
            /* 添加汉堡菜单按钮 */
            .mobile-menu-btn {
                display: flex !important;
                position: fixed;
                top: 1rem;
                left: 1rem;
                z-index: 100;
                width: 48px;
                height: 48px;
                background: rgba(99, 102, 241, 0.9);
                border-radius: 12px;
                align-items: center;
                justify-content: center;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                cursor: pointer;
            }
            
            /* 卡片网格调整为单列 */
            #dashboard-content {
                grid-template-columns: 1fr !important;
            }
            
            /* 调整字体大小 */
            h2 {
                font-size: 1.5rem !important;
            }
            
            h3 {
                font-size: 1rem !important;
            }
            
            /* 增大触摸目标 */
            .nav-btn {
                min-height: 48px;
            }
            
            button {
                min-height: 44px;
                padding: 0.75rem 1.5rem !important;
            }
            
            /* 优化聊天气泡 */
            .chat-bubble {
                max-width: 90% !important;
            }
            
            /* 调整padding */
            .p-8 {
                padding: 1rem !important;
            }
            
            .p-6 {
                padding: 1rem !important;
            }
        }
        
        /* 桌面端隐藏汉堡菜单 */
        .mobile-menu-btn {
            display: none;
        }

    </style>
</head>
<body class="h-screen flex overflow-hidden selection:bg-primary-500 selection:text-white">

    <!-- 侧边栏 -->
    
    <!-- 移动端汉堡菜单 -->
    <div class="mobile-menu-btn" onclick="toggleMobileMenu()">
        <i class="fa-solid fa-bars text-white text-xl"></i>
    </div>
    
    <!-- 移动端遮罩层 -->
    <div class="mobile-overlay" onclick="toggleMobileMenu()"></div>

    <aside class="w-64 glass flex flex-col border-r border-gray-800 z-20">
        <div class="p-6 flex items-center gap-3">
            <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-purple-600 flex items-center justify-center shadow-lg shadow-primary-500/20">
                <i class="fa-solid fa-wave-square text-white text-lg"></i>
            </div>
            <div>
                <h1 class="font-bold text-lg tracking-tight text-white">Audio Notes</h1>
                <p class="text-xs text-gray-400">AI 智能语音助手</p>
            </div>
        </div>

        <nav class="flex-1 px-4 space-y-2 mt-4">
            <button onclick="switchTab('dashboard')" id="nav-dashboard" class="nav-btn w-full flex items-center gap-3 px-4 py-3 rounded-xl text-gray-400 hover:text-white hover:bg-white/5 transition-all group active">
                <i class="fa-solid fa-grid-2 text-lg group-hover:text-primary-400 transition-colors"></i>
                <span class="font-medium">仪表盘</span>
            </button>
            <button onclick="switchTab('chat')" id="nav-chat" class="nav-btn w-full flex items-center gap-3 px-4 py-3 rounded-xl text-gray-400 hover:text-white hover:bg-white/5 transition-all group">
                <i class="fa-solid fa-comments text-lg group-hover:text-accent-400 transition-colors"></i>
                <span class="font-medium">时光对话</span>
            </button>
            <button onclick="switchTab('analysis')" id="nav-analysis" class="nav-btn w-full flex items-center gap-3 px-4 py-3 rounded-xl text-gray-400 hover:text-white hover:bg-white/5 transition-all group">
                <i class="fa-solid fa-chart-pie text-lg group-hover:text-purple-400 transition-colors"></i>
                <span class="font-medium">统计分析</span>
            </button>
            <button onclick="switchTab('speaker')" id="nav-speaker" class="nav-btn w-full flex items-center gap-3 px-4 py-3 rounded-xl text-gray-400 hover:text-white hover:bg-white/5 transition-all group">
                <i class="fa-solid fa-microphone text-lg group-hover:text-pink-400 transition-colors"></i>
                <span class="font-medium">声纹管理</span>
            </button>
        </nav>

        <div class="p-4 mt-auto">
            <div class="glass-card rounded-xl p-4 space-y-3">
                <div class="flex items-center justify-between">
                    <span class="text-xs font-medium text-gray-400">ASR 服务</span>
                    <span id="status-asr" class="flex h-2 w-2 rounded-full bg-red-500 status-dot"></span>
                </div>
                <div class="flex items-center justify-between">
                    <span class="text-xs font-medium text-gray-400">待处理</span>
                    <span id="status-files" class="text-xs font-bold text-white bg-gray-700 px-2 py-0.5 rounded-full">0</span>
                </div>
                <div class="h-px bg-gray-700/50 my-2"></div>
                <div class="text-[10px] text-gray-500 font-mono truncate" id="log-display">
                    系统就绪...
                </div>
            </div>
        </div>
    </aside>

    <!-- 主内容区 -->
    <main class="flex-1 relative overflow-hidden flex flex-col">
        <!-- 顶部光晕 -->
        <div class="absolute top-0 left-0 w-full h-64 bg-gradient-to-b from-primary-900/10 to-transparent pointer-events-none"></div>

        <!-- 视图容器 -->
        <div id="view-dashboard" class="view-container flex-1 overflow-y-auto p-8 space-y-8 animate-fade-in active">
            <header class="flex justify-between items-end mb-6">
                <div>
                    <h2 class="text-3xl font-bold text-white mb-2">概览</h2>
                    <p class="text-gray-400">最近的录音与转录记录</p>
                </div>
                <button onclick="loadMore()" class="group px-5 py-2.5 rounded-xl bg-gradient-to-r from-primary-600 to-primary-500 hover:from-primary-500 hover:to-primary-400 text-sm font-medium text-white transition-all duration-300 shadow-lg shadow-primary-500/20 hover:shadow-primary-500/40 hover:scale-105 border border-primary-400/20">
                    <i class="fa-solid fa-rotate-right mr-2 group-hover:rotate-180 transition-transform duration-500"></i>刷新数据
                </button>
            </header>
            
            <div id="dashboard-content" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 pb-10">
                <!-- 卡片将通过JS插入 -->
            </div>
            <div id="loading-indicator" class="text-center py-8 hidden">
                <i class="fa-solid fa-circle-notch fa-spin text-primary-500 text-2xl"></i>
            </div>
        </div>

        <div id="view-chat" class="view-container flex-1 overflow-y-auto p-0 hidden">
            <div class="max-w-4xl mx-auto w-full h-full flex flex-col bg-gray-900/30 border-x border-gray-800/50">
                <header class="p-6 border-b border-gray-800/50 glass sticky top-0 z-10 backdrop-blur-xl">
                    <h2 class="text-xl font-bold text-white flex items-center gap-2">
                        <i class="fa-regular fa-clock text-accent-400"></i> 时光轴
                    </h2>
                </header>
                <div id="chat-content" class="flex-1 p-6 space-y-8 pb-20">
                    <!-- 对话内容 -->
                </div>
            </div>
        </div>

        <div id="view-analysis" class="view-container flex-1 overflow-y-auto p-8 hidden">
            <header class="mb-8">
                <h2 class="text-3xl font-bold text-white mb-2">数据洞察</h2>
                <p class="text-gray-400">说话人统计与趋势分析</p>
            </header>
            <div id="analysis-content" class="grid grid-cols-1 lg:grid-cols-2 gap-8 pb-10">
                <!-- 统计图表 -->
            </div>
        </div>
        
         <div id="view-speaker" class="view-container flex-1 overflow-y-auto p-8 hidden">
            <header class="mb-8">
                <h2 class="text-3xl font-bold text-white mb-2 flex items-center gap-3">
                    <i class="fa-solid fa-microphone text-pink-400"></i>
                    声纹管理
                </h2>
                <p class="text-gray-400">注册和管理说话人声纹</p>
            </header>

            <!-- 注册区域 -->
            <div class="glass-card rounded-2xl p-8 mb-8">
                <h3 class="text-xl font-bold text-white mb-6 flex items-center gap-2">
                    <i class="fa-solid fa-user-plus text-primary-400"></i>
                    注册新说话人
                </h3>
                
                <form id="register-form" class="space-y-6">
                    <div>
                        <label class="block text-sm font-medium text-gray-300 mb-2">
                            <i class="fa-solid fa-signature mr-2"></i>说话人姓名
                        </label>
                        <input 
                            type="text" 
                            id="speaker-name" 
                            placeholder="请输入姓名"
                            class="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20 transition-all"
                            required
                        >
                    </div>
                    
                    <div>
                        <label class="block text-sm font-medium text-gray-300 mb-2">
                            <i class="fa-solid fa-file-audio mr-2"></i>音频文件 (3-10秒清晰语音)
                        </label>
                        <div class="relative">
                            <input 
                                type="file" 
                                id="audio-file" 
                                accept="audio/*"
                                class="hidden"
                                required
                            >
                            <button 
                                type="button"
                                onclick="document.getElementById('audio-file').click()"
                                class="w-full px-4 py-3 bg-gray-800 border-2 border-dashed border-gray-600 rounded-xl text-gray-400 hover:border-primary-500 hover:text-primary-400 transition-all flex items-center justify-center gap-2"
                            >
                                <i class="fa-solid fa-cloud-arrow-up text-2xl"></i>
                                <span id="file-name">点击选择音频文件</span>
                            </button>
                        </div>
                    </div>
                    
                    <button 
                        type="submit"
                        class="w-full px-6 py-4 bg-gradient-to-r from-primary-600 to-primary-500 hover:from-primary-500 hover:to-primary-400 text-white font-medium rounded-xl transition-all duration-300 shadow-lg shadow-primary-500/20 hover:shadow-primary-500/40 hover:scale-105 flex items-center justify-center gap-2"
                    >
                        <i class="fa-solid fa-fingerprint"></i>
                        <span>注册声纹</span>
                    </button>
                </form>
                
                <div id="register-status" class="mt-4 hidden"></div>
            </div>

            <!-- 已注册说话人列表 -->
            <div class="glass-card rounded-2xl p-8">
                <h3 class="text-xl font-bold text-white mb-6 flex items-center gap-2">
                    <i class="fa-solid fa-users text-accent-400"></i>
                    已注册说话人
                </h3>
                
                <div id="speaker-list" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    <!-- 说话人卡片将通过JS插入 -->
                </div>
            </div>
        </div>

        <div id="view-speaker" class="view-container flex-1 overflow-hidden p-0 hidden">
            <iframe src="/register_page" style="width:100%; height:100%; border:none;"></iframe>
        </div>
    </main>

    <script>
        // 状态管理
        let lastDataFingerprint = "";
        const speakerColorMap = {};
        const colors = [
            'text-blue-400 border-blue-500/30 bg-blue-500/10',
            'text-emerald-400 border-emerald-500/30 bg-emerald-500/10',
            'text-purple-400 border-purple-500/30 bg-purple-500/10',
            'text-amber-400 border-amber-500/30 bg-amber-500/10',
            'text-rose-400 border-rose-500/30 bg-rose-500/10',
            'text-cyan-400 border-cyan-500/30 bg-cyan-500/10',
        ];
        let nextColorIndex = 0;
        
        // 懒加载状态
        let currentOffset = 0;
        const pageSize = 20;
        let isLoading = false;
        let hasMore = true;
        let allItems = [];

        // 切换视图
        function switchTab(tabName) {
            document.querySelectorAll('.view-container').forEach(el => el.classList.add('hidden'));
            document.getElementById('view-' + tabName).classList.remove('hidden');
            
            document.querySelectorAll('.nav-btn').forEach(el => {
                el.classList.remove('bg-white/10', 'text-white', 'shadow-lg');
                el.classList.add('text-gray-400');
            });
            
            const activeBtn = document.getElementById('nav-' + tabName);
            activeBtn.classList.remove('text-gray-400');
            activeBtn.classList.add('bg-white/10', 'text-white', 'shadow-lg');
        }
        
        // 初始化导航状态
        switchTab('dashboard');

        function getSpeakerStyle(spk) {
            if (!speakerColorMap[spk]) {
                speakerColorMap[spk] = colors[nextColorIndex % colors.length];
                nextColorIndex++;
            }
            return speakerColorMap[spk];
        }

        function formatTime(isoString) {
            if (!isoString) return '';
            const date = new Date(isoString);
            return date.toLocaleString('zh-CN', {
                month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
            });
        }

        function parseFilenameTime(filename) {
            // 尝试从文件名解析时间
            // 支持格式: TermuxAudioRecording_2025-11-20_14-33-08
            // 支持格式: recording-20251115-131250
            
            // 格式1: TermuxAudioRecording_YYYY-MM-DD_HH-mm-ss
            let match = filename.match(/(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})/);
            if (match) {
                return `${match[1]}-${match[2]}-${match[3]}T${match[4]}:${match[5]}:${match[6]}`;
            }
            
            // 格式2: recording-YYYYMMDD-HHmmss
            match = filename.match(/recording-(\d{8})-(\d{6})/);
            if (match) {
                const date = match[1];  // YYYYMMDD
                const time = match[2];  // HHmmss
                return `${date.substring(0,4)}-${date.substring(4,6)}-${date.substring(6,8)}T${time.substring(0,2)}:${time.substring(2,4)}:${time.substring(4,6)}`;
            }
            
            return null;
        }

        function processStats(items) {
            // 简单的数据预处理
            return items.map(item => {
                item.date_group = item.created_at ? item.created_at.split('T')[0] : 'Unknown';
                return item;
            });
        }

        // 渲染仪表盘
        function renderDashboard(items) {
            const container = document.getElementById('dashboard-content');
            container.innerHTML = ""; // 清空并重新渲染所有项
            
            let html = "";
            items.forEach((item, index) => {
                // 提取摘要 (前100字)
                let summary = item.full_text || "无内容";
                if (summary.length > 100) summary = summary.substring(0, 100) + "...";
                
                // 提取参与者
                const speakers = new Set();
                item.segments.forEach(s => speakers.add(s.spk));
                const speakerTags = Array.from(speakers).map(s => {
                    const style = getSpeakerStyle(s);
                    // 提取颜色类名中的 text-xxx
                    const colorClass = style.split(' ')[0]; 
                    return `<span class="text-xs font-medium ${colorClass} bg-gray-800/50 px-2 py-1 rounded-md border border-gray-700/50">${s}</span>`;
                }).join('');

                // 统计情感分布
                const emotionStats = {};
                item.segments.forEach(s => {
                    const emo = s.emotion || 'neutral';
                    emotionStats[emo] = (emotionStats[emo] || 0) + 1;
                });
                
                // 生成情感标签 (显示所有情感包括neutral，用于测试)
                const emotionTags = Object.entries(emotionStats)
                    // .filter(([emo]) => emo !== 'neutral')  // 临时注释以测试
                    .map(([emo, count]) => {
                        const icon = getEmotionIcon(emo);
                        return `<span class="text-xs px-2 py-1 rounded-md bg-purple-500/10 text-purple-400 border border-purple-500/30 flex items-center gap-1">
                            <span>${icon}</span>
                            <span>${emo}</span>
                            <span class="text-[10px] opacity-70">×${count}</span>
                        </span>`;
                    }).join('');

                html += `
                <div class="glass-card rounded-2xl p-6 flex flex-col h-full animate-slide-up hover:scale-105 transition-transform duration-300" style="animation-delay: ${index * 50}ms">
                    <div class="flex justify-between items-start mb-4">
                        <div class="flex items-center gap-3">
                            <div class="w-10 h-10 rounded-full bg-gray-800 flex items-center justify-center border border-gray-700">
                                <i class="fa-solid fa-file-audio text-primary-400"></i>
                            </div>
                            <div>
                                <h3 class="font-semibold text-white text-sm truncate w-40" title="${item.filename}">${item.filename}</h3>
                                <span class="text-xs text-gray-500">${formatTime(parseFilenameTime(item.filename) || item.created_at)}</span>
                            </div>
                        </div>
                        <span class="text-xs font-mono text-gray-600 bg-gray-900 px-2 py-1 rounded">ID: ${item.id}</span>
                    </div>
                    
                    <div class="flex-1 mb-4">
                        <p class="text-gray-400 text-sm leading-relaxed line-clamp-3">${summary}</p>
                    </div>
                    
                    <div class="space-y-2 mt-auto pt-4 border-t border-gray-800/50">
                        <div class="flex flex-wrap gap-2">
                            ${speakerTags || '<span class="text-xs text-gray-600">无说话人</span>'}
                        </div>
                        ${emotionTags ? `<div class="flex flex-wrap gap-2 pt-2 border-t border-gray-800/30">
                            ${emotionTags}
                        </div>` : ''}
                    </div>
                </div>`;
            });
            
            container.innerHTML = html;
        }

        // 渲染聊天视图
        function renderChat(items) {
            const container = document.getElementById('chat-content');
            if (currentOffset === 0) container.innerHTML = "";
            
            // 按日期分组
            const groups = {};
            items.slice(currentOffset).forEach(item => {
                if (!groups[item.date_group]) groups[item.date_group] = [];
                groups[item.date_group].push(item);
            });

            let html = "";
            
            // 如果是追加加载，不需要重新渲染日期头（简化处理，这里可能需要优化，但暂且如此）
            // 实际上，为了保持时间线正确，我们应该重新渲染整个列表或者精细控制。
            // 鉴于 allItems 包含了所有数据，我们这里简单地重新渲染整个 allItems 会更安全，
            // 但为了性能，我们只渲染新增部分。
            // 修正：renderChat 接收的是 allItems，所以我们可以全量渲染，或者优化。
            // 为了简单且保证顺序正确，我们清空并全量渲染 allItems。
            
            container.innerHTML = ""; // 清空重绘，确保顺序
            
            // 重新分组 allItems
            const allGroups = {};
            allItems.forEach(item => {
                if (!allGroups[item.date_group]) allGroups[item.date_group] = [];
                allGroups[item.date_group].push(item);
            });

            Object.keys(allGroups).sort().reverse().forEach(date => {
                // 对同一天内的记录按文件名时间严格倒序排列(最新的在上面)
                allGroups[date].sort((a, b) => {
                    const timeA = parseFilenameTime(a.filename) || a.created_at;
                    const timeB = parseFilenameTime(b.filename) || b.created_at;
                    return new Date(timeB) - new Date(timeA);  // 倒序: 新 - 旧
                });
                
                html += `
                <div class="relative flex items-center justify-center my-8">
                    <div class="absolute inset-0 flex items-center">
                        <div class="w-full border-t border-gray-800"></div>
                    </div>
                    <div class="relative bg-gray-900 px-4 py-1 rounded-full border border-gray-800 text-xs font-medium text-gray-500">
                        ${date}
                    </div>
                </div>`;

                allGroups[date].forEach(item => {
                    // 文件头
                    html += `
                    <div class="mb-8 animate-fade-in">
                        <div class="flex items-center gap-2 mb-4 px-4">
                            <i class="fa-solid fa-record-vinyl text-gray-600 text-xs"></i>
                            <span class="text-xs font-mono text-gray-500">${item.filename}</span>
                            <span class="text-xs text-gray-600 ml-auto">${formatTime(parseFilenameTime(item.filename) || item.created_at)}</span>
                        </div>
                    `;

                    let lastSpk = null;
                    item.segments.forEach(seg => {
                        const style = getSpeakerStyle(seg.spk);
                        // 提取颜色
                        const textColor = style.match(/text-(\w+)-400/)[1]; // e.g. blue
                        const isMe = seg.spk === 'Me'; // 假设有 'Me'，目前没有，预留
                        
                        const showAvatar = seg.spk !== lastSpk;
                        lastSpk = seg.spk;

                        html += `
                        <div class="flex gap-4 mb-2 ${showAvatar ? 'mt-4' : ''} px-2 hover:bg-white/5 rounded-lg transition-colors p-2 -mx-2">
                            <div class="w-10 flex-shrink-0 flex flex-col items-center">
                                ${showAvatar ? `
                                <div class="w-10 h-10 rounded-full bg-gradient-to-br from-${textColor}-500 to-${textColor}-700 border-2 border-${textColor}-400/30 flex items-center justify-center shadow-lg shadow-${textColor}-500/20">
                                    <span class="text-sm font-bold text-white">${seg.spk.substring(0,1).toUpperCase()}</span>
                                </div>
                                ` : ''}
                            </div>
                            <div class="flex-1 min-w-0">
                                ${showAvatar ? `
                                <div class="flex items-baseline gap-2 mb-1">
                                    <span class="text-sm font-bold text-gray-200">${seg.spk}</span>
                                    <span class="text-[10px] text-gray-600">${formatTime(item.created_at)}</span>
                                </div>
                                ` : ''}
                                <div class="text-gray-300 leading-relaxed text-sm flex items-start gap-2">
                                    <span class="flex-1">${seg.text}</span>
                                    ${seg.segment_audio_path ? 
                                        `<button onclick="playAudio('${seg.segment_audio_path}')" class="flex-shrink-0 w-6 h-6 rounded-full bg-primary-500/20 hover:bg-primary-500/40 border border-primary-400/30 flex items-center justify-center transition-all duration-200 hover:scale-110 group" title="播放音频">
                                            <i class="fa-solid fa-play text-[10px] text-primary-300 group-hover:text-primary-200"></i>
                                        </button>` : ''}
                                    ${seg.emotion ? 
                                        `<span class="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium bg-purple-500/10 text-purple-300 ml-2 border border-purple-500/30">
                                            <span>${getEmotionIcon(seg.emotion)}</span>
                                            <span>${seg.emotion}</span>
                                        </span>` : ''}
                                </div>
                                ${seg.whisper_text ? 
                                    `<div class="text-gray-500 text-xs mt-1 pl-4 border-l-2 border-gray-700/50">
                                        <span class="text-gray-600">↪️ Whisper: </span>${seg.whisper_text}
                                    </div>` : ''}
                                ${seg.sensevoice_text ? 
                                    `<div class="text-purple-500 text-xs mt-1 pl-4 border-l-2 border-purple-700/50">
                                        <span class="text-purple-400">🎭 SenseVoice: </span>${seg.sensevoice_text}
                                    </div>` : ''}
                            </div>
                        </div>`;
                    });
                    html += `</div>`;
                });
            });
            
            container.innerHTML = html;
        }

        // 渲染分析视图
        function renderAnalysis(items) {
            const container = document.getElementById('analysis-content');
            
            // 统计数据
            const stats = {};
            allItems.forEach(item => {
                item.segments.forEach(seg => {
                    if (!stats[seg.spk]) {
                        stats[seg.spk] = { count: 0, words: 0, emotions: {} };
                    }
                    stats[seg.spk].count++;
                    stats[seg.spk].words += seg.text.length;
                    
                    const emo = seg.emotion || 'neutral';
                    stats[seg.spk].emotions[emo] = (stats[seg.spk].emotions[emo] || 0) + 1;
                });
            });

            // 转换为数组并排序
            const sortedStats = Object.entries(stats)
                .sort((a, b) => b[1].count - a[1].count);

            let html = "";
            
            // 概览卡片
            html += `
            <div class="col-span-1 lg:col-span-2 glass-card rounded-2xl p-6 mb-6">
                <h3 class="text-lg font-bold text-white mb-4">说话人活跃度排行</h3>
                <div class="space-y-4">
            `;
            
            sortedStats.forEach(([spk, data], index) => {
                const style = getSpeakerStyle(spk);
                const colorName = style.match(/text-(\w+)-400/)[1];
                const percent = Math.min(100, (data.count / sortedStats[0][1].count) * 100);
                
                html += `
                <div class="relative">
                    <div class="flex justify-between text-sm mb-1">
                        <span class="font-medium text-gray-300 flex items-center gap-2">
                            <span class="w-2 h-2 rounded-full bg-${colorName}-500"></span>
                            ${spk}
                        </span>
                        <span class="text-gray-500 font-mono">${data.count} 句 / ${data.words} 字</span>
                    </div>
                    <div class="h-2 bg-gray-800 rounded-full overflow-hidden">
                        <div class="h-full bg-${colorName}-500 rounded-full transition-all duration-1000" style="width: ${percent}%"></div>
                    </div>
                </div>
                `;
            });
            html += `</div></div>`;

            // 情感分布 (模拟图表，实际可以用Chart.js)
            html += `
            <div class="glass-card rounded-2xl p-6">
                <h3 class="text-lg font-bold text-white mb-4">情感分布</h3>
                <div class="grid grid-cols-2 gap-4">
            `;
            
            // 汇总所有情感
            const allEmotions = {};
            Object.values(stats).forEach(s => {
                Object.entries(s.emotions).forEach(([e, c]) => {
                    allEmotions[e] = (allEmotions[e] || 0) + c;
                });
            });
            
            Object.entries(allEmotions).forEach(([emo, count]) => {
                html += `
                <div class="bg-gray-800/50 rounded-xl p-3 border border-gray-700/50 text-center">
                    <div class="text-2xl mb-1">${getEmotionIcon(emo)}</div>
                    <div class="text-xs text-gray-400 uppercase tracking-wider">${emo}</div>
                    <div class="text-lg font-bold text-white">${count}</div>
                </div>
                `;
            });
            
            html += `</div></div>`;
            
            container.innerHTML = html;
        }
        
        function getEmotionIcon(emo) {
            const map = {
                'happy': '😊', 'sad': '😔', 'angry': '😡', 'neutral': '😐',
                'laughter': '🤣', 'fearful': '😨', 'surprised': '😲'
            };
            return map[emo] || '😶';
        }

        // 加载更多数据
        async function loadMore() {
            if (isLoading || !hasMore) return;
            
            isLoading = true;
            document.getElementById('loading-indicator').classList.remove('hidden');
            
            try {
                const dataRes = await fetch(`/api/data?offset=${currentOffset}&limit=${pageSize}`);
                let newItems = await dataRes.json();
                
                if (newItems.length === 0) {
                    hasMore = false;
                    document.getElementById('loading-indicator').classList.add('hidden');
                    return;
                }
                
                if (newItems.length < pageSize) {
                    hasMore = false;
                }
                
                newItems = processStats(newItems);
                
                // 追加新数据
                allItems = allItems.concat(newItems);
                currentOffset += newItems.length;
                
                // 重新渲染所有视图
                renderDashboard(allItems); // 仪表盘支持追加
                renderChat(allItems);      // 聊天全量重绘
                renderAnalysis(allItems);  // 分析全量重绘
                
            } catch (e) { 
                console.error('加载数据失败:', e); 
            } finally {
                isLoading = false;
                document.getElementById('loading-indicator').classList.add('hidden');
            }
        }

        async function updateLoop() {
            try {
                const statusRes = await fetch('/api/status');
                const statusData = await statusRes.json();
                
                const asrDot = document.getElementById('status-asr');
                if (statusData.asr_server === 'online') {
                    asrDot.classList.remove('text-red-500');
                    asrDot.classList.add('text-green-500', 'shadow-[0_0_8px_#22c55e]');
                } else {
                    asrDot.classList.remove('text-green-500', 'shadow-[0_0_8px_#22c55e]');
                    asrDot.classList.add('text-red-500');
                }
                
                document.getElementById('status-files').innerText = statusData.pending_files;
                document.getElementById('log-display').innerText = statusData.last_log || "无日志";

                // 只检查是否有新数据（检查最新一条）
                const checkRes = await fetch('/api/data?offset=0&limit=1');
                const latestItem = await checkRes.json();
                
                if (latestItem.length > 0) {
                    const latestId = latestItem[0].id;
                    const currentLatestId = allItems.length > 0 ? allItems[0].id : null;
                    
                    // 如果有新数据，重新加载所有数据
                    if (latestId !== currentLatestId) {
                        console.log('检测到新数据，重新加载');
                        currentOffset = 0;
                        allItems = [];
                        hasMore = true;
                        await loadMore();
                    }
                }

            } catch (e) { console.error(e); }
        }
        
        // 添加滚动监听 - 滚动到底部时加载更多
        ['view-dashboard', 'view-chat', 'view-analysis'].forEach(containerId => {
            const container = document.getElementById(containerId);
            if (container) {
                container.addEventListener('scroll', () => {
                    if (container.scrollTop + container.clientHeight >= container.scrollHeight - 100) {
                        loadMore();
                    }
                });
            }
        });

        // 初始加载
        loadMore();
        // 定时检查新数据
        setInterval(updateLoop, 5000);
    
        // 移动端菜单切换
        function toggleMobileMenu() {
            const aside = document.querySelector('aside');
            const overlay = document.querySelector('.mobile-overlay');
            
            aside.classList.toggle('mobile-open');
            overlay.classList.toggle('active');
        }
        
        // 点击导航按钮后自动关闭移动端菜单
        const originalSwitchTab = switchTab;
        switchTab = function(tabName) {
            originalSwitchTab(tabName);
            
            // 如果是移动端,关闭菜单
            if (window.innerWidth <= 768) {
                toggleMobileMenu();
            }
        };


        // 声纹管理相关函数
        document.getElementById('audio-file')?.addEventListener('change', function(e) {
            const fileName = e.target.files[0]?.name || '点击选择音频文件';
            document.getElementById('file-name').textContent = fileName;
        });

        // 注册声纹
        document.getElementById('register-form')?.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const name = document.getElementById('speaker-name').value;
            const file = document.getElementById('audio-file').files[0];
            const statusDiv = document.getElementById('register-status');
            
            if (!name || !file) {
                showStatus('请填写完整信息', 'error');
                return;
            }
            
            const formData = new FormData();
            formData.append('speaker_name', name);
            formData.append('audio_file', file);
            
            showStatus('正在注册...', 'loading');
            
            try {
                const response = await fetch('/speaker/register', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    showStatus('✅ 注册成功!', 'success');
                    document.getElementById('register-form').reset();
                    document.getElementById('file-name').textContent = '点击选择音频文件';
                    loadSpeakers();
                } else {
                    showStatus('❌ ' + (result.error || '注册失败'), 'error');
                }
            } catch (error) {
                showStatus('❌ 网络错误: ' + error.message, 'error');
            }
        });

        // 显示状态消息
        function showStatus(message, type) {
            const statusDiv = document.getElementById('register-status');
            statusDiv.className = 'mt-4 p-4 rounded-xl ' + 
                (type === 'success' ? 'bg-green-500/10 border border-green-500/30 text-green-400' :
                 type === 'error' ? 'bg-red-500/10 border border-red-500/30 text-red-400' :
                 'bg-blue-500/10 border border-blue-500/30 text-blue-400');
            statusDiv.textContent = message;
            statusDiv.classList.remove('hidden');
            
            if (type !== 'loading') {
                setTimeout(() => statusDiv.classList.add('hidden'), 5000);
            }
        }

        // 加载说话人列表
        async function loadSpeakers() {
            try {
                const response = await fetch('/speaker/list');
                const data = await response.json();
                
                const container = document.getElementById('speaker-list');
                if (!data.speakers || data.speakers.length === 0) {
                    container.innerHTML = '<p class="text-gray-500 col-span-full text-center py-8">暂无已注册说话人</p>';
                    return;
                }
                
                container.innerHTML = data.speakers.map(speaker => `
                    <div class="glass-card p-6 rounded-xl hover:scale-105 transition-transform">
                        <div class="flex items-center justify-between mb-4">
                            <div class="flex items-center gap-3">
                                <div class="w-12 h-12 rounded-full bg-gradient-to-br from-pink-500 to-purple-600 flex items-center justify-center">
                                    <i class="fa-solid fa-user text-white text-xl"></i>
                                </div>
                                <div>
                                    <h4 class="font-bold text-white">${speaker.name}</h4>
                                    <p class="text-xs text-gray-500">${speaker.sample_count || 0} 个样本</p>
                                </div>
                            </div>
                        </div>
                        <button 
                            onclick="deleteSpeaker('${speaker.name}')"
                            class="w-full px-4 py-2 bg-red-500/10 hover:bg-red-500/20 border border-red-500/30 text-red-400 rounded-lg transition-all flex items-center justify-center gap-2"
                        >
                            <i class="fa-solid fa-trash"></i>
                            <span>删除</span>
                        </button>
                    </div>
                `).join('');
            } catch (error) {
                console.error('加载说话人列表失败:', error);
            }
        }

        // 删除说话人
        async function deleteSpeaker(name) {
            if (!confirm(`确定要删除说话人 "${name}" 吗?`)) return;
            
            try {
                const response = await fetch(`/speaker/${encodeURIComponent(name)}`, {
                    method: 'DELETE'
                });
                
                if (response.ok) {
                    loadSpeakers();
                } else {
                    alert('删除失败');
                }
            } catch (error) {
                alert('网络错误: ' + error.message);
            }
        }

        // 切换到声纹管理时加载列表
        const originalSwitchTab2 = switchTab;
        switchTab = function(tabName) {
            originalSwitchTab2(tabName);
            if (tabName === 'speaker') {
                loadSpeakers();
            }
        };


    </script>
</body>
</html>
"""


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

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

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