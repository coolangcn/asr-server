#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一ASR服务 - 整合文件监控和转录功能

import os
import sys
import time
import threading
import subprocess
import shutil
import re
import json
import traceback
import requests  # 用于调用 /transcribe API
from datetime import datetime

# 导入ASR服务模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asr_server
from asr_server import Config as ASRConfig, logger
from db_manager import init_pool, init_db, save_to_db, close_pool

# =================【文件监控配置】=================
class FileMonitorConfig:
    ENABLE = True
    SOURCE_DIR = r"V:\Sony-2"
    TRANSCRIPT_DIR = r"V:\Sony-2\transcripts"
    PROCESSED_DIR = r"V:\Sony-2\processed"
    MONITOR_INTERVAL = 3  # 秒
    SUPPORTED_EXTENSIONS = ('.m4a', '.acc', '.aac', '.mp3', '.wav', '.ogg', '.flac')

# =================【辅助函数】=================
def clean_sensevoice_tags(text):
    """清理SenseVoice标签"""
    if not text:
        return ""
    cleaned = re.sub(r'<\|.*?\|>', '', text)
    return cleaned.strip()

def format_time(ms):
    """格式化时间（毫秒转hh:mm:ss）"""
    seconds = ms / 1000
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02}:{int(m):02}:{int(s):02}"

def convert_audio_to_wav(audio_path, wav_path):
    """将音频转换为WAV格式"""
    FFMPEG_PATH = "ffmpeg"
    command = [
        FFMPEG_PATH, '-y', '-i', audio_path, '-vn', '-map', '0:a',
        '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', wav_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8', 'ignore').strip() if e.stderr else "Unknown error"
        if "moov atom not found" in error_msg:
            logger.error(f"  [Convert Error] 文件已损坏或未完成录制 (moov atom not found)")
        elif "Decoding requested, but no decoder found" in error_msg:
            logger.error(f"  [Convert Error] 文件不包含有效的音频流")
        else:
            logger.error(f"  [Convert Error] ffmpeg转换失败: ... {error_msg[-500:]}")
        return False
    except Exception as e:
        logger.error(f"  [Convert Error] {e}")
        return False

def save_transcript_txt(full_text, segments, txt_path):
    """保存转录结果为TXT文件"""
    try:
        content_lines = []
        emo_map = {
            "happy": "😊开心", "sad": "😔悲伤", "angry": "😡生气",
            "laughter": "🤣大笑", "fearful": "😨害怕", "surprised": "😲惊讶",
            "neutral": ""
        }
        content_lines.append(f"=== 全文摘要 ===\n{full_text}\n")
        content_lines.append("=== 对话记录 (按说话人) ===")
        for seg in segments:
            start_str = format_time(seg.get('start', 0))
            spk_label = str(seg.get('spk', 'Unknown'))
            emotion_key = seg.get('emotion', 'neutral')
            emo_str = emo_map.get(emotion_key, "")
            if emo_str:
                emo_str = f" {emo_str}"
            text = clean_sensevoice_tags(seg.get('text', '').strip())
            if not text:
                continue
            line = f"[{start_str}] [{spk_label}]{emo_str}: {text}"
            content_lines.append(line)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(content_lines))
        return True
    except Exception as e:
        logger.error(f"  [Save TXT Error] {e}")
        return False

# =================【文件监控循环】=================
def process_one_file(filename):
    """处理单个音频文件"""
    source_path = os.path.join(FileMonitorConfig.SOURCE_DIR, filename)
    
    logger.info(f"\n>>> 处理: {filename}")
    
    # 创建临时文件目录（使用asr-server的temp目录）
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # 转换为WAV（使用temp目录）
    base_filename = os.path.basename(filename)
    wav_path = os.path.join(temp_dir, base_filename + "_TEMP.wav")
    if not convert_audio_to_wav(source_path, wav_path):
        logger.error("  音频转换失败，跳过")
        return False
    
    try:
        # 调用 /transcribe API
        api_url = f"http://localhost:{ASRConfig.PORT}/transcribe"
        
        with open(wav_path, 'rb') as f:
            files = {'audio_file': (os.path.basename(wav_path), f, 'audio/wav')}
            response = requests.post(api_url, files=files, timeout=300)
        
        if response.status_code != 200:
            logger.error(f"  API调用失败: {response.status_code}")
            return False
        
        result = response.json()
        full_text = result.get("full_text", "")
        segments = result.get("segments", [])
        
        if not segments:
            logger.warning(f"  未检测到有效语音分段")
            return False
        
        logger.info(f"  转录成功: {len(segments)} 个分段")
        
        # 解析录音时间
        from db_manager import parse_recording_time
        recording_time = parse_recording_time(filename)
        
        # 保存到数据库
        try:
            from db_manager import connection_pool
            if not connection_pool:
                logger.error(f"  数据库连接池未初始化，无法保存")
            else:
                success = save_to_db(filename, full_text, segments, recording_time)
                if success:
                    logger.info(f"  数据库保存成功 (recording_time: {recording_time})")
                else:
                    logger.error(f"  数据库保存失败: save_to_db返回False")
        except Exception as e:
            logger.error(f"  数据库保存失败: {e}")
            logger.error(traceback.format_exc())
        
        # 保存TXT文件
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(FileMonitorConfig.TRANSCRIPT_DIR, txt_filename)
        os.makedirs(FileMonitorConfig.TRANSCRIPT_DIR, exist_ok=True)
        
        if save_transcript_txt(full_text, segments, txt_path):
            logger.info(f"  TXT已保存")
        else:
            logger.warning(f"  TXT保存失败")
        
        # 移动到processed目录
        os.makedirs(FileMonitorConfig.PROCESSED_DIR, exist_ok=True)
        dest_path = os.path.join(FileMonitorConfig.PROCESSED_DIR, filename)
        shutil.move(source_path, dest_path)
        logger.info(f"  已移动到: processed/{filename}")
        
        return True
        
    finally:
        # 清理临时WAV文件
        if os.path.exists(wav_path):
            try:
                os.remove(wav_path)
                logger.info(f"  [Cleanup] 已删除临时文件: {os.path.basename(wav_path)}")
            except Exception as e:
                logger.warning(f"  [Cleanup] 删除临时文件失败: {e}")


def file_monitor_loop():
    """文件监控主循环"""
    logger.info(f"\n📁 文件监控已启动")
    logger.info(f"   监控目录: {FileMonitorConfig.SOURCE_DIR}")
    logger.info(f"   扫描间隔: {FileMonitorConfig.MONITOR_INTERVAL}秒\n")
    
    while True:
        try:
            if not os.path.exists(FileMonitorConfig.SOURCE_DIR):
                logger.warning(f"源目录不存在: {FileMonitorConfig.SOURCE_DIR}")
                time.sleep(FileMonitorConfig.MONITOR_INTERVAL)
                continue
            
            files = [
                f for f in os.listdir(FileMonitorConfig.SOURCE_DIR)
                if f.lower().endswith(FileMonitorConfig.SUPPORTED_EXTENSIONS)
            ]
            
            if files:
                logger.info(f"发现 {len(files)} 个待处理文件")
                for filename in files:
                    process_one_file(filename)
            
            time.sleep(FileMonitorConfig.MONITOR_INTERVAL)
            
        except KeyboardInterrupt:
            logger.info("文件监控停止")
            break
        except Exception as e:
            logger.error(f"文件监控出错: {e}")
            time.sleep(10)

# =================【主函数】=================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 统一ASR服务启动中...")
    print("="*60 + "\n")
    
    # 1. 初始化数据库
    if FileMonitorConfig.ENABLE:
        logger.info("初始化数据库连接...")
        if not init_pool():
            logger.error("数据库连接池初始化失败")
            sys.exit(1)
        init_db()
    
    # 2. 加载AI模型
    asr_server.load_models()
    
    # 3. 启动文件监控线程
    if FileMonitorConfig.ENABLE:
        monitor_thread = threading.Thread(target=file_monitor_loop, daemon=True)
        monitor_thread.start()
        logger.info("✅ 文件监控线程已启动\n")
    
    # 4. 启动Flask服务
    try:
        logger.info(f"🌐 启动HTTP服务: http://{ASRConfig.HOST}:{ASRConfig.PORT}\n")
        asr_server.app.run(
            host=ASRConfig.HOST,
            port=ASRConfig.PORT,
            debug=False,
            threaded=True
        )
    except KeyboardInterrupt:
        logger.info("\n正在关闭服务...")
    finally:
        if FileMonitorConfig.ENABLE:
            close_pool()
        logger.info("服务已停止")
