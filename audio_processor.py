#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import logging
import threading
import shutil
import requests
import traceback
from datetime import datetime
from db_manager import parse_recording_time

logger = logging.getLogger('AudioProcessor')

class FileMonitorConfig:
    ENABLED = True
    SOURCE_DIR = "/Volumes/download/records/Sony-2"
    PROCESSED_DIR = "processed"
    FAILED_DIR = "failed"
    SCAN_INTERVAL = 3
    SUPPORTED_FORMATS = ['.m4a', '.mp3', '.wav', '.aac', '.flac', '.ogg', '.acc']

def start_monitor():
    """启动文件监控并自动处理新音频"""
    if not FileMonitorConfig.ENABLED:
        logger.info("📂 文件监控功能已禁用")
        return
        
    thread = threading.Thread(target=_monitor_loop, daemon=True)
    thread.start()
    return thread

def _monitor_loop():
    logger.info("📂 文件监控线程已启动")
    logger.info(f"   监控目录: {FileMonitorConfig.SOURCE_DIR}")
    
    # 确保必要的目录存在
    processed_dir = os.path.join(FileMonitorConfig.SOURCE_DIR, FileMonitorConfig.PROCESSED_DIR)
    os.makedirs(processed_dir, exist_ok=True)
    failed_dir = os.path.join(FileMonitorConfig.SOURCE_DIR, FileMonitorConfig.FAILED_DIR)
    os.makedirs(failed_dir, exist_ok=True)
    
    processed_files = set()
    
    while True:
        try:
            if not os.path.exists(FileMonitorConfig.SOURCE_DIR):
                logger.warning(f"⚠️ 源目录不存在: {FileMonitorConfig.SOURCE_DIR}")
                time.sleep(FileMonitorConfig.SCAN_INTERVAL)
                continue
            
            files_to_process = []
            for item in os.listdir(FileMonitorConfig.SOURCE_DIR):
                item_path = os.path.join(FileMonitorConfig.SOURCE_DIR, item)
                
                if item in [FileMonitorConfig.PROCESSED_DIR, FileMonitorConfig.FAILED_DIR, "audio_segments", "logs"] or item.startswith('.'):
                    continue
                
                items_to_check = []
                if os.path.isfile(item_path):
                    items_to_check.append(item_path)
                elif os.path.isdir(item_path):
                    try:
                        for subitem in os.listdir(item_path):
                            if not subitem.startswith('.'):
                                subp = os.path.join(item_path, subitem)
                                if os.path.isfile(subp):
                                    items_to_check.append(subp)
                    except Exception as e:
                        logger.error(f"读取子目录 {item_path} 失败: {e}")
                                
                for filepath in items_to_check:
                    filename = os.path.basename(filepath)
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in FileMonitorConfig.SUPPORTED_FORMATS and 'TEMP' not in filename and filename not in processed_files:
                        files_to_process.append((filename, filepath))
            
            # 按文件名排序
            files_to_process.sort(key=lambda x: x[0])
            
            if files_to_process:
                logger.info(f"🔍 发现 {len(files_to_process)} 个待处理文件")
                for filename, filepath in files_to_process:
                    try:
                        _process_one_file(filename, filepath, processed_dir, failed_dir)
                        processed_files.add(filename)
                    except Exception as e:
                        logger.error(f"处理文件 {filename} 失败: {e}")
            
        except Exception as e:
            logger.error(f"监控循环异常: {e}")
            
        time.sleep(FileMonitorConfig.SCAN_INTERVAL)

def _process_one_file(filename, filepath, processed_dir, failed_dir):
    """处理单个音频文件"""
    # 1. 检查录音时间，跳过凌晨 1-6 点
    recording_time = parse_recording_time(filename)
    if recording_time:
        hour = recording_time.hour
        if 1 <= hour < 6:
            logger.info(f"⏭️ 跳过凌晨录音: {filename}")
            _move_file(filepath, filename, processed_dir, recording_time)
            return

    logger.info(f"📤 开始处理: {filename}")
    
    # 2. 发起转录请求
    try:
        with open(filepath, 'rb') as f:
            files_data = {'audio_file': (filename, f, 'audio/mpeg')}
            # 直接调用本地 ASR 接口 (假设 asr_server 运行在 5008 端口)
            response = requests.post('http://localhost:5008/transcribes', files=files_data, timeout=7200)
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ 转录完成: {filename} ({len(result.get('full_text', ''))} 字)")
            _move_file(filepath, filename, processed_dir, recording_time)
        else:
            logger.error(f"❌ 转录失败: {filename} (HTTP {response.status_code})")
            _move_file(filepath, filename, failed_dir, recording_time)
            
    except Exception as e:
        logger.error(f"❌ 处理文件 {filename} 时发生异常: {e}")
        _move_file(filepath, filename, failed_dir, recording_time)

def _move_file(src_path, filename, base_dest_dir, recording_time=None):
    """根据日期子目录移动文件"""
    date_subdir = recording_time.strftime("%Y-%m-%d") if recording_time else datetime.now().strftime("%Y-%m-%d")
    target_dir = os.path.join(base_dest_dir, date_subdir)
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, filename)
    
    try:
        shutil.move(src_path, target_path)
        logger.info(f"📦 已移动至: {os.path.basename(base_dest_dir)}/{date_subdir}/{filename}")
    except Exception as e:
        logger.warning(f"⚠️ 移动文件失败: {e}")
