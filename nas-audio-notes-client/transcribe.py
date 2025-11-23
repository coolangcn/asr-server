#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import requests
import json
import datetime
import time
import argparse
import re
import shutil
import sys
from db_manager import init_pool, init_db, save_to_db, close_pool, parse_recording_time

# --- Logging Setup ---
class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode, encoding='utf-8', buffering=1)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()

# Redirect stdout and stderr to log file
log_file = "transcribe.log"
# Only redirect if not already redirected (to avoid recursion if script reloads)
if not isinstance(sys.stdout, Tee):
    sys.stdout = Tee(log_file, "a")
    sys.stderr = sys.stdout
# ---------------------

# ---------------- 配置 ----------------
CONFIG_FILE = "config.json"

DEFAULT_CONFIG = {
    "ASR_API_URL": "http://192.168.1.111:5008/transcribe",
    "DIARIZE_API_URL": "http://192.168.1.111:5008/transcribe",
    "USE_DIARIZE": False,
    "SOURCE_DIR": "V:\\Sony-2",
    "TRANSCRIPT_DIR": "V:\\Sony-2\\transcripts",
    "PROCESSED_DIR": "V:\\Sony-2\\processed",
    "N8N_WEBHOOK_URL": "https://n8n.moco.fun/webhook/bea45d47-d1fc-498e-bf69-d48dc079f04a",
    "DATABASE_URL": "postgresql://postgres:difyai123456@192.168.1.188:5432/postgres",
    "LOG_FILE_PATH": "transcribe.log",
    "WEB_PORT": 5010
}

# Load config from JSON file
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        loaded_config = json.load(f)
    DEFAULT_CONFIG.update(loaded_config)

CONFIG = DEFAULT_CONFIG.copy()
SUPPORTED_EXTENSIONS = ('.m4a', '.acc', '.aac', '.mp3', '.wav', '.ogg', '.flac')

# ---------------- 命令行参数 ----------------
def parse_args():
    parser = argparse.ArgumentParser(description='音频转录脚本')
    parser.add_argument('--source-path', type=str, help='源音频文件路径')
    parser.add_argument('--use-diarize', action='store_true', help='启用说话人分离功能')
    return parser.parse_args()

def update_config(args):
    global CONFIG
    if args.source_path:
        base_path = args.source_path
        CONFIG["SOURCE_DIR"] = base_path
        CONFIG["TRANSCRIPT_DIR"] = os.path.join(base_path, "transcripts")
        CONFIG["PROCESSED_DIR"] = os.path.join(base_path, "processed")
        print(f"[配置] 使用自定义源路径: {base_path}")
    
    if args.use_diarize:
        CONFIG["USE_DIARIZE"] = True
        print(f"[配置] 已启用说话人分离功能")

# ---------------- 工具函数 ----------------
def format_time(ms):
    seconds = ms / 1000
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02}:{int(m):02}:{int(s):02}"

def clean_sensevoice_tags(text):
    if not text: return ""
    cleaned = re.sub(r'<\|.*?\|>', '', text)
    return cleaned.strip()

# ---------------- 数据库 ----------------
# 数据库功能已迁移到 db_manager.py

def notify_n8n(status, filename, details):
    if not CONFIG["N8N_WEBHOOK_URL"]: return
    payload = {
        "status": status, 
        "filename": filename, 
        "details": details, 
        "timestamp": datetime.datetime.now().isoformat()
    }
    try:
        requests.post(CONFIG["N8N_WEBHOOK_URL"], json=payload, timeout=5)
    except:
        pass

# ---------------- 临时文件清理 ----------------
def cleanup_temp_files():
    """Remove any orphaned temporary WAV files from previous runs"""
    if not os.path.exists(CONFIG["SOURCE_DIR"]):
        return
    
    temp_files = [f for f in os.listdir(CONFIG["SOURCE_DIR"]) 
                  if "_TEMP" in f.upper() or f.lower().endswith('.wav')]
    
    if temp_files:
        print(f"[Startup] 发现 {len(temp_files)} 个临时文件，正在清理...")
        for filename in temp_files:
            try:
                file_path = os.path.join(CONFIG["SOURCE_DIR"], filename)
                os.remove(file_path)
                print(f"  已删除: {filename}")
            except Exception as e:
                print(f"  清理失败 {filename}: {e}")

def is_file_ready(filepath, stable_duration=2):
    """Check if file is stable and not being written to"""
    try:
        if not os.path.exists(filepath):
            return False
        
        # Check if file size is stable
        size1 = os.path.getsize(filepath)
        time.sleep(stable_duration)
        size2 = os.path.getsize(filepath)
        
        return size1 == size2 and size1 > 0
    except:
        return False

# ---------------- 音频处理 ----------------
def convert_audio_to_wav(audio_path, wav_path):
    # Windows下使用系统PATH中的ffmpeg
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
            print("  [Convert Error] ffmpeg 转换失败. 原因: 文件已损坏或未完成录制 (moov atom not found).")
        elif "Decoding requested, but no decoder found" in error_msg:
            print("  [Convert Error] ffmpeg 转换失败. 原因: 文件不包含有效的音频流.")
        else:
            print(f"  [Convert Error] ffmpeg 转换失败. 详细错误: ... {error_msg[-500:]}")
        return False
    except Exception as e:
        print(f"  [Convert Error] {e}")
        return False

# ---------------- TXT 保存 ----------------
def save_transcript_with_spk(full_text, segments, txt_path):
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
            if emo_str: emo_str = f" {emo_str}"
            text = clean_sensevoice_tags(seg.get('text', '').strip())
            if not text: continue
            line = f"[{start_str}] [{spk_label}]{emo_str}: {text}"
            content_lines.append(line)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(content_lines))
        return True
    except Exception as e:
        print(f"  [Save TXT Error] {e}")
        return False

def save_diarize_result(diarization_data, txt_path):
    """保存diarize结果到文本文件"""
    try:
        content_lines = []
        content_lines.append("=== 说话人分离结果 ===\n")
        
        diarization = diarization_data.get('diarization', [])
        if not diarization:
            content_lines.append("未检测到说话人分段数据")
        else:
            # 按时间排序
            diarization_sorted = sorted(diarization, key=lambda x: x.get('start_ms', 0))
            
            # 统计每个说话人的发言次数
            speaker_stats = {}
            for seg in diarization_sorted:
                speaker = seg.get('speaker', 'Unknown')
                speaker_stats[speaker] = speaker_stats.get(speaker, 0) + 1
            
            # 显示说话人统计
            content_lines.append("=== 说话人统计 ===")
            for speaker, count in speaker_stats.items():
                content_lines.append(f"{speaker}: {count} 段发言")
            content_lines.append("")
            
            # 显示详细分段
            content_lines.append("=== 详细分段 ===")
            for seg in diarization_sorted:
                speaker = seg.get('speaker', 'Unknown')
                text = seg.get('text', '').strip()
                start_ms = seg.get('start_ms', 0)
                end_ms = seg.get('end_ms', 0)
                
                start_str = format_time(start_ms)
                end_str = format_time(end_ms)
                
                if text:
                    line = f"[{start_str} - {end_str}] [{speaker}]: {text}"
                    content_lines.append(line)
                else:
                    line = f"[{start_str} - {end_str}] [{speaker}]: [无声/非语音]"
                    content_lines.append(line)
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(content_lines))
        return True
    except Exception as e:
        print(f"  [Save Diarize TXT Error] {e}")
        return False

# ---------------- 调用服务端 ----------------
def transcribe_wav(wav_path):
    url = CONFIG["ASR_API_URL"]
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with open(wav_path, 'rb') as f:
                files = {'audio_file': (os.path.basename(wav_path), f, 'audio/wav')}
                if attempt > 0:
                    print(f"  网络波动，正在重试 ({attempt+1}/{max_retries})...")
                else:
                    print(f"  正在上传并等待转录结果 (超时: 3600s)...")
                print(f"  [DEBUG] 请求URL: {url}")
                print(f"  [DEBUG] 请求参数: files={{audio_file: '{os.path.basename(wav_path)}'}}")
                response = requests.post(url, files=files, timeout=3600)
            print(f"  [DEBUG] 响应状态码: {response.status_code}")
            print(f"  [DEBUG] 响应头: {dict(response.headers)}")
            print(f"  [DEBUG] 响应体(前1000字符): {response.text[:1000]}")
            response.raise_for_status()
            data = response.json()
            print(f"  [DEBUG] 解析后的JSON: {json.dumps(data, ensure_ascii=False, indent=2)[:2000]}")
            print(f"  [Info] 服务端返回 {len(data.get('segments', []))} 个语音分段。")
            if "error" in data:
                print(f"  [Server Error] {data['error']}")
                return None
            return data if "full_text" in data else None
        except requests.exceptions.ConnectionError:
            print(f"  [Connection Error] 无法连接服务端 ({url})，等待 5秒 后重试...")
            time.sleep(5)
        except requests.exceptions.Timeout:
            print(f"  [Timeout] 请求超时，服务端仍在处理。")
            return None
        except Exception as e:
            print(f"  [Request Error] {e}")
            print(f"  [DEBUG] 异常详情: {type(e).__name__}: {str(e)}")
            return None
    print("  [Failed] 重试次数耗尽，跳过此文件")
    return None

def diarize_wav(wav_path):
    """调用服务端的/transcribe接口进行说话人分离"""
    url = CONFIG["DIARIZE_API_URL"]
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with open(wav_path, 'rb') as f:
                files = {'audio_file': (os.path.basename(wav_path), f, 'audio/wav')}
                if attempt > 0:
                    print(f"  [Diarize] 网络波动，正在重试 ({attempt+1}/{max_retries})...")
                else:
                    print(f"  [Diarize] 正在上传并等待说话人分离结果...")
                print(f"  [DIARIZE DEBUG] 请求URL: {url}")
                print(f"  [DIARIZE DEBUG] 请求参数: files={{audio_file: '{os.path.basename(wav_path)}'}}")
                response = requests.post(url, files=files, timeout=3600)
            print(f"  [DIARIZE DEBUG] 响应状态码: {response.status_code}")
            print(f"  [DIARIZE DEBUG] 响应头: {dict(response.headers)}")
            print(f"  [DIARIZE DEBUG] 响应体(前1000字符): {response.text[:1000]}")
            response.raise_for_status()
            data = response.json()
            print(f"  [DIARIZE DEBUG] 解析后的JSON: {json.dumps(data, ensure_ascii=False, indent=2)[:2000]}")
            if "error" in data:
                print(f"  [Diarize Server Error] {data['error']}")
                return None
            
            # 将 /transcribe 返回的 segments 转换为 diarization 格式
            segments = data.get('segments', [])
            if segments:
                diarization = []
                for seg in segments:
                    diarization.append({
                        'speaker': seg.get('spk', 'Unknown'),
                        'text': seg.get('text', ''),
                        'start_ms': seg.get('start', 0),
                        'end_ms': seg.get('end', 0)
                    })
                converted_data = {'diarization': diarization}
                print(f"  [Diarize Info] 服务端返回 {len(diarization)} 个说话人分段。")
                return converted_data
            else:
                print(f"  [Diarize Info] 服务端返回 0 个说话人分段。")
                return None
        except requests.exceptions.ConnectionError:
            print(f"  [Diarize Connection Error] 无法连接服务端 ({url})，等待 5秒 后重试...")
            time.sleep(5)
        except requests.exceptions.Timeout:
            print(f"  [Diarize Timeout] 请求超时，服务端仍在处理。")
            return None
        except Exception as e:
            print(f"  [Diarize Request Error] {e}")
            print(f"  [DIARIZE DEBUG] 异常详情: {type(e).__name__}: {str(e)}")
            return None
    print("  [Diarize Failed] 重试次数耗尽，跳过此文件")
    return None

# ---------------- 处理循环 ----------------
# ---------------- 处理循环 ----------------
def process_one_loop():
    processed_count = 0
    if not os.path.exists(CONFIG["SOURCE_DIR"]):
        print(f"源目录不存在: {CONFIG['SOURCE_DIR']}")
        return 0
    files = [f for f in os.listdir(CONFIG["SOURCE_DIR"]) 
             if f.lower().endswith(SUPPORTED_EXTENSIONS) 
             and "_TEMP" not in f.upper()
             and not f.lower().endswith('.wav')]  # WAV files are only temp files
    
    if not files: return 0
    
    # 按录音时间排序，优先处理最早的录音
    def get_recording_time_for_sort(filename):
        recording_time = parse_recording_time(filename)
        if recording_time:
            return recording_time
        else:
            # 如果无法解析时间，使用一个很晚的时间，让它排在后面
            # 同时使用文件名作为次要排序
            return datetime(9999, 12, 31, 23, 59, 59)
    
    files.sort(key=get_recording_time_for_sort)
    print(f"发现 {len(files)} 个新文件，按录音时间排序处理...")
    
    os.makedirs(CONFIG["TRANSCRIPT_DIR"], exist_ok=True)
    os.makedirs(CONFIG["PROCESSED_DIR"], exist_ok=True)
    for filename in files:
        print(f"\n>>> 处理: {filename}")
        audio_path = os.path.join(CONFIG["SOURCE_DIR"], filename)

        # 在处理前再次确认文件是否存在,防止文件被移动或删除
        if not os.path.exists(audio_path):
            print(f"  [Error] 文件在处理前消失: {filename}。可能已被移动或删除,已跳过。")
            continue
        
        # 检查文件是否稳定(不在写入中)
        if not is_file_ready(audio_path):
            print(f"  [跳过] 文件可能正在写入中: {filename}")
            continue

        base_name = os.path.splitext(filename)[0]
        wav_path = os.path.join(CONFIG["SOURCE_DIR"], f"{base_name}_TEMP.wav")
        txt_path = os.path.join(CONFIG["TRANSCRIPT_DIR"], f"{base_name}.txt")
        processed_audio_path = os.path.join(CONFIG["PROCESSED_DIR"], filename)
        
        # 从文件名解析录音时间
        recording_time = parse_recording_time(filename)
        if recording_time:
            print(f"  [时间] 解析到录音时间: {recording_time.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"  [时间] 无法从文件名解析时间，将使用当前时间")
        try:
            if not convert_audio_to_wav(audio_path, wav_path): continue
            
            # 根据配置选择使用普通转录还是说话人分离
            if CONFIG["USE_DIARIZE"]:
                print(f"  使用说话人分离模式处理音频")
                diarize_data = diarize_wav(wav_path)
                
                # 说话人分离成功后的处理
                if diarize_data and diarize_data.get("diarization"):
                    diarization = diarize_data.get("diarization", [])
                    
                    # 额外统计命名说话人，增强日志
                    named_speakers = set(seg.get("speaker") for seg in diarization if seg.get("speaker") != "Unknown")
                    print(f"  [Diarize Success] 共识别 {len(diarization)} 个分段，其中命名说话人: {', '.join(named_speakers) if named_speakers else '无'}")

                    # === Normal Diarization Success Logic ===
                    # 保存diarize结果
                    save_diarize_result(diarize_data, txt_path)
                    
                    # 获取所有文本用于数据库存储
                    full_text = " ".join([seg.get("text", "").strip() for seg in diarization if seg.get("text", "").strip()])
                    
                    # 转换格式以兼容数据库存储
                    segments = []
                    for seg in diarization:
                        if seg.get("text", "").strip():
                            segments.append({
                                "text": seg.get("text", "").strip(),
                                # 客户端信任服务端返回的 start_ms/end_ms
                                "start": seg.get("start_ms", 0), 
                                "end": seg.get("end_ms", 0),
                                "spk": seg.get("speaker", "Unknown"),
                                "emotion": seg.get("emotion", "neutral")  # 从服务器读取真实emotion
                            })
                    
                    # 保存到数据库
                    save_to_db(filename, full_text, segments, recording_time)
                    
                    print(f"  [完成] 说话人分离结果已保存 -> {txt_path}")
                    notify_n8n("success", filename, f"说话人分离完成，共{len(segments)}个分段 ({len(named_speakers)}个命名说话人)")
                
                # 说话人分离失败时的降级处理
                else:
                    print("  [Info] 服务端没有返回有效的说话人分离结果 (0分段)，尝试降级为普通转录...")
                    
                    # === Fallback Logic ===
                    result_data = transcribe_wav(wav_path)
                    if not result_data or not result_data.get("segments"):
                        print("  [Failed] 普通转录也失败，跳过此文件")
                        notify_n8n("failed", filename, "说话人分离和普通转录均失败")
                        continue
                        
                    # Construct fallback segments from ASR result
                    full_text = result_data.get("full_text", "")
                    asr_segments = result_data.get("segments", [])
                    
                    # If ASR has no segments but has full_text, create a dummy segment
                    if not asr_segments and full_text:
                        asr_segments = [{"start": 0, "end": 0, "text": full_text}]
                    
                    # Convert ASR segments to the format expected by our DB and TXT saver
                    fallback_segments = []
                    for seg in asr_segments:
                        fallback_segments.append({
                            "start": seg.get("start", 0),
                            "end": seg.get("end", 0),
                            "text": seg.get("text", "").strip(),
                            "spk": seg.get("spk", "Unknown"),  # 也从服务器读取spk
                            "emotion": seg.get("emotion", "neutral")  # 从服务器读取真实emotion
                        })
                    
                    save_transcript_with_spk(full_text, fallback_segments, txt_path)
                    save_to_db(filename, full_text, fallback_segments, recording_time)
                    
                    print(f"  [完成] (降级模式) 转录结果已保存 -> {txt_path}")
                    notify_n8n("success", filename, f"[降级] {full_text[:100]}")
            
            # 标准转录模式 (USE_DIARIZE=False)
            else:
                print(f"  使用标准转录模式处理音频")
                result_data = transcribe_wav(wav_path)
                if not result_data or not result_data.get("segments"):
                    print("  [Info] 服务端没有返回有效的语音分段，已跳过。")
                    notify_n8n("skipped", filename, "服务端没有返回有效的语音分段")
                    continue

                full_text = result_data.get("full_text", "")
                segments = result_data.get("segments", [])
                filtered_segments = [seg for seg in segments if seg.get("text","").strip()]
                save_transcript_with_spk(full_text, filtered_segments, txt_path)
                save_to_db(filename, full_text, filtered_segments, recording_time)
                print(f"  [完成] 转录结果已保存 -> {txt_path}")
                notify_n8n("success", filename, full_text[:100])
            
            # 移动已处理的音频文件
            if os.path.exists(processed_audio_path): os.remove(processed_audio_path)
            os.rename(audio_path, processed_audio_path)
            print(f"  [完成] 音频已归档 -> {processed_audio_path}")
            processed_count += 1
        except Exception as e:
            print(f"  [异常] {e}")
        finally:
            # 增强的临时文件清理 - 清理所有相关的临时文件
            if wav_path:
                try:
                    # 1. 清理主临时文件 (_TEMP.wav)
                    if os.path.exists(wav_path):
                        os.remove(wav_path)
                        print(f"  [清理] 已删除临时文件: {os.path.basename(wav_path)}")
                    
                    # 2. 清理所有以该临时文件名为前缀的文件 (如 .processed.wav, .seg_*.wav 等)
                    base_name = os.path.basename(wav_path)
                    source_dir = os.path.dirname(wav_path)
                    
                    for file in os.listdir(source_dir):
                        if file.startswith(base_name):
                            related_file = os.path.join(source_dir, file)
                            try:
                                os.remove(related_file)
                                print(f"  [清理] 已删除关联文件: {file}")
                            except Exception as e:
                                print(f"  [清理警告] 关联文件删除失败 {file}: {e}")
                                
                except Exception as e:
                    print(f"  [清理警告] 临时文件清理失败: {e}")
    return processed_count

# ---------------- 主函数 ----------------
def main():
    args = parse_args()
    update_config(args)
    print("--- 启动实时监控模式 (SenseVoice 适配版) ---")
    print(f"监控目录: {CONFIG['SOURCE_DIR']}")
    
    # Clean up any orphaned temporary files from previous runs
    cleanup_temp_files()
    
    print("初始化数据库连接池...")
    if not init_pool():
        print("数据库连接池初始化失败，程序退出")
        return
    init_db()
    while True:
        try:
            process_one_loop()
            time.sleep(3)
        except KeyboardInterrupt:
            print("停止监控。")
            close_pool()
            break
        except Exception as e:
            print(f"主循环发生错误: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
