#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import requests
import json
import datetime
import sqlite3
import time
import argparse
import re

# ---------------- 配置 ----------------
CONFIG_FILE = "config.json"

DEFAULT_CONFIG = {
    "ASR_API_URL": "http://192.168.1.111:5009/transcribe",
    "SOURCE_DIR": "/volume2/download/records/Sony-2",
    "TRANSCRIPT_DIR": "/volume2/download/records/Sony-2/transcripts",
    "PROCESSED_DIR": "/volume2/download/records/Sony-2/processed",
    "N8N_WEBHOOK_URL": "https://n8n.moco.fun/webhook/bea45d47-d1fc-498e-bf69-d48dc079f04a",
    "DB_PATH": "/volume2/download/records/Sony-2/transcripts.db",
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
    return parser.parse_args()

def update_config(args):
    global CONFIG
    if args.source_path:
        base_path = args.source_path
        CONFIG["SOURCE_DIR"] = base_path
        CONFIG["TRANSCRIPT_DIR"] = os.path.join(base_path, "transcripts")
        CONFIG["PROCESSED_DIR"] = os.path.join(base_path, "processed")
        CONFIG["DB_PATH"] = os.path.join(base_path, "transcripts.db")
        print(f"[配置] 使用自定义源路径: {base_path}")

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
def init_db():
    try:
        conn = sqlite3.connect(CONFIG["DB_PATH"])
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS transcriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            full_text TEXT,
            segments_json TEXT
        );
        ''')
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"数据库初始化失败: {e}")

def save_to_db(filename, full_text, segments_list):
    try:
        conn = sqlite3.connect(CONFIG["DB_PATH"])
        cursor = conn.cursor()
        segments_json = json.dumps(segments_list, ensure_ascii=False)
        cursor.execute(
            "INSERT INTO transcriptions (filename, full_text, segments_json) VALUES (?, ?, ?)",
            (filename, full_text, segments_json)
        )
        conn.commit()
        conn.close()
        print(f"  [DB] Saved {filename}")
        return True
    except Exception as e:
        print(f"  [DB Error] {e}")
        return False

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

# ---------------- 音频处理 ----------------
def convert_audio_to_wav(audio_path, wav_path):
    FFMPEG_PATH = "/usr/local/bin/ffmpeg"
    command = [
        FFMPEG_PATH, '-y', '-i', audio_path, '-vn', '-map', '0:a',
        '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', wav_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode().strip() if e.stderr else "Unknown error"
        print(f"  [Convert Error] ffmpeg 转换失败: {error_msg[:200]}...")
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
                response = requests.post(url, files=files, timeout=3600)
            response.raise_for_status()
            data = response.json()
            if "error" in data:
                print(f"  [Server Error] {data['error']}")
                return None
            return data if "full_text" in data else None
        except requests.exceptions.ConnectionError:
            print(f"  [Connection Error] 无法连接服务端，等待 5秒 后重试...")
            time.sleep(5)
        except requests.exceptions.Timeout:
            print(f"  [Timeout] 请求超时，服务端仍在处理。")
            return None
        except Exception as e:
            print(f"  [Request Error] {e}")
            return None
    print("  [Failed] 重试次数耗尽，跳过此文件")
    return None

# ---------------- 处理循环 ----------------
def process_one_loop():
    processed_count = 0
    if not os.path.exists(CONFIG["SOURCE_DIR"]):
        print(f"源目录不存在: {CONFIG['SOURCE_DIR']}")
        return 0
    files = [f for f in os.listdir(CONFIG["SOURCE_DIR"]) if f.lower().endswith(SUPPORTED_EXTENSIONS)]
    if not files: return 0
    print(f"发现 {len(files)} 个新文件，开始处理...")
    os.makedirs(CONFIG["TRANSCRIPT_DIR"], exist_ok=True)
    os.makedirs(CONFIG["PROCESSED_DIR"], exist_ok=True)
    for filename in files:
        print(f"\n>>> 处理: {filename}")
        audio_path = os.path.join(CONFIG["SOURCE_DIR"], filename)
        base_name = os.path.splitext(filename)[0]
        wav_path = os.path.join(CONFIG["SOURCE_DIR"], f"{base_name}_TEMP.wav")
        txt_path = os.path.join(CONFIG["TRANSCRIPT_DIR"], f"{base_name}.txt")
        processed_audio_path = os.path.join(CONFIG["PROCESSED_DIR"], filename)
        try:
            if not convert_audio_to_wav(audio_path, wav_path): continue
            result_data = transcribe_wav(wav_path)
            if not result_data: continue
            full_text = result_data.get("full_text", "")
            segments = result_data.get("segments", [])
            filtered_segments = [seg for seg in segments if seg.get("text","").strip()]
            save_transcript_with_spk(full_text, filtered_segments, txt_path)
            save_to_db(filename, full_text, filtered_segments)
            if os.path.exists(processed_audio_path): os.remove(processed_audio_path)
            os.rename(audio_path, processed_audio_path)
            print(f"  [完成] 已归档 -> {processed_audio_path}")
            notify_n8n("success", filename, full_text[:100])
            processed_count += 1
        except Exception as e:
            print(f"  [异常] {e}")
        finally:
            if os.path.exists(wav_path): os.remove(wav_path)
    return processed_count

# ---------------- 主函数 ----------------
def main():
    args = parse_args()
    update_config(args)
    print("--- 启动实时监控模式 (SenseVoice 适配版) ---")
    print(f"监控目录: {CONFIG['SOURCE_DIR']}")
    init_db()
    while True:
        try:
            process_one_loop()
            time.sleep(3)
        except KeyboardInterrupt:
            print("停止监控。")
            break
        except Exception as e:
            print(f"主循环发生错误: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
