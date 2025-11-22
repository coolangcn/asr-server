#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一ASR服务 - 整合文件监控和转录功能

这个脚本整合了：
1. ASR服务端（asr_server.py）- 提供转录API和模型加载
2. 文件监控（transcribe.py）- 自动处理音频文件

优势：
- 在同一进程中运行，直接函数调用（无HTTP开销）
- 共享AI模型内存
- 简化部署
"""

import os
import sys
import time
import threading
import subprocess
import shutil
import re
import json
from datetime import datetime

# 导入ASR服务模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asr_server
from asr_server import Config as ASRConfig, logger, preprocess_audio
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

# =================【内部转录函数】=================
def transcribe_internal(wav_path):
    """
    内部转录函数 - 直接调用asr_server的转录逻辑
    
    Args:
        wav_path: WAV音频文件路径
    
    Returns:
        dict: 转录结果 {"full_text": str, "segments": list, "meta": dict}
        None: 转录失败
    """
    try:
        logger.info(f"📥 收到转录任务: {os.path.basename(wav_path)}")
        
        # 调用asr_server模块的转录endpoint的核心逻辑
        # 这里我们需要从asr_server.py的transcribe_audio函数中提取核心逻辑
        # 为了简化，我们直接导入必要的变量并复制部分逻辑
        
        from asr_server import asr_pipeline, identify_speaker_fusion, gpu_lock, EMOTION_TAGS, INVALID_TAGS
        
        if not os.path.exists(wav_path):
            logger.error(f"文件不存在: {wav_path}")
            return None
        
        # 预处理音频
        processed_path = wav_path + ".processed.wav"
        if not preprocess_audio(wav_path, processed_path):
            logger.error("音频预处理失败")
            return None
        
        # 获取音频时长
        import torchaudio
        waveform, sr = torchaudio.load(processed_path)
        audio_duration = waveform.shape[1] / sr
        
        logger.info("  [生命周期: 1. ASR识别] 开始...")
        start_time = time.time()
        
        with gpu_lock:
            res = asr_pipeline.generate(
                input=processed_path,
                batch_size_s=300,
                hotword='魔都'
            )
        
        process_time = time.time() - start_time
        
        segments = []
        full_text = ""
        
        if res and isinstance(res, list) and len(res) > 0:
            item = res[0]
            full_text = item.get("text", "")
            
            raw_segments = item.get("sentence_info", [])
            logger.info(f"  [生命周期: 2. VAD & ASR] 完成, VAD检出 {len(raw_segments)} 个分段。")
            
            if not raw_segments and full_text:
                raw_segments = [{"text": full_text, "start": 0, "end": int(audio_duration * 1000)}]
            
            processed_segments = []
            
            if raw_segments:
                logger.info("  [生命周期: 3. 逐段声纹识别] 开始...")
                for i, seg in enumerate(raw_segments):
                    seg_text = seg.get("text", "").strip()
                    start = seg.get("start", 0)
                    end = seg.get("end", 0)
                    
                    # 清理emotion/event标签
                    emotion = "neutral"
                    for tag, emo in EMOTION_TAGS.items():
                        if tag in seg_text:
                            emotion = emo
                            seg_text = seg_text.replace(tag, "")
                    
                    for tag in INVALID_TAGS:
                        seg_text = seg_text.replace(tag, "")
                    
                    clean_text = seg_text.strip()
                    
                    if not clean_text:
                        logger.info(f"      [3.{i+1}] 分段文本在清洗后为空，已跳过。")
                        continue
                    
                    duration_ms = end - start
                    if duration_ms < ASRConfig.MIN_SPEAKER_DURATION_MS:
                        logger.info(f"      [3.{i+1}] 分段时长不足 {ASRConfig.MIN_SPEAKER_DURATION_MS}ms，已跳过。")
                        continue
                    
                    # 声纹识别
                    identity = None
                    confidence = 0.0
                    recognition_details = {}
                    
                    segment_path = processed_path + f".seg_{i}.wav"
                    try:
                        from asr_server import extract_segment
                        if extract_segment(processed_path, start, end, segment_path):
                            result = identify_speaker_fusion(segment_path)
                            if result:
                                identity, confidence, recognition_details = result
                                
                    except Exception as e:
                        logger.warning(f"      [3.{i+1}] 声纹识别出错: {e}")
                    finally:
                        if os.path.exists(segment_path):
                            try:
                                os.remove(segment_path)
                            except:
                                pass
                    
                    processed_segments.append({
                        "text": clean_text,
                        "start": start,
                        "end": end,
                        "spk": identity or "Unknown",
                        "emotion": emotion,
                        "confidence": float(f"{confidence:.3f}"),
                        "recognition_details": recognition_details
                    })
                
                logger.info("  [生命周期: 3. 逐段声纹识别] 完成。")
            
            segments = processed_segments
            full_text = "".join([s["text"] for s in segments])
        
        # 清理临时文件
        if os.path.exists(processed_path):
            try:
                os.remove(processed_path)
            except:
                pass
        
        rtf = process_time / audio_duration if audio_duration > 0 else 0
        
        result = {
            "full_text": full_text,
            "segments": segments,
            "meta": {
                "audio_duration": audio_duration,
                "process_time": process_time,
                "rtf": rtf,
                "rtf_description": "Real-Time Factor(实时因子)，处理时间/音频时长，RTF < 1表示可实时处理，值越低性能越好"
            }                        }
        
        logger.info(f"✅ 转录完成: {len(segments)} 个分段, RTF={rtf:.3f}")
        return result
        
    except Exception as e:
        logger.error(f"转录失败: {e}")
        logger.error(traceback.format_exc())
        return None

# =================【文件监控循环】=================
def process_one_file(filename):
    """处理单个音频文件"""
    source_path = os.path.join(FileMonitorConfig.SOURCE_DIR, filename)
    
    logger.info(f"\n>>> 处理: {filename}")
    
    # 转换为WAV
    wav_path = source_path + "_TEMP.wav"
    if not convert_audio_to_wav(source_path, wav_path):
        logger.error("  音频转换失败，跳过")
        return False
    
    try:
        # 内部转录（直接函数调用）
        result = transcribe_internal(wav_path)
        
        if not result:
            logger.error("  转录失败")
            return False
        
        full_text = result.get("full_text", "")
        segments = result.get("segments", [])
        
        if not segments:
            logger.warning(f"  未检测到有效语音分段")
            return False
        
        logger.info(f"  转录成功: {len(segments)} 个分段")
        
        # 保存到数据库
        try:
            save_to_db(filename, full_text, segments)
        except Exception as e:
            logger.error(f"  数据库保存失败: {e}")
        
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
            except:
                pass

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
