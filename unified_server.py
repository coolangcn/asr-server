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
import traceback
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
        
        # 预处理音频 - 确保在temp目录创建
        # 从wav_path提取文件名，在temp目录创建processed文件
        temp_dir = os.path.dirname(wav_path)  # wav_path已经在temp目录中
        wav_basename = os.path.basename(wav_path)
        processed_path = os.path.join(temp_dir, wav_basename.replace("_TEMP.wav", "_TEMP.processed.wav"))
        
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
                    emotion_source = "funasr"  # 默认来源
                    original_emotion_tag = None
                    
                    for tag, emo in EMOTION_TAGS.items():
                        if tag in seg_text:
                            emotion = emo
                            emotion_source = "funasr"
                            original_emotion_tag = tag
                            seg_text = seg_text.replace(tag, "")
                    
                    for tag in INVALID_TAGS:
                        seg_text = seg_text.replace(tag, "")
                    
                    clean_text = seg_text.strip()
                    
                    if not clean_text:
                        # 跳过空文本（静默）
                        logger.debug(f"  #{i+1} 跳过: 文本为空")
                        continue
                    
                    duration_ms = end - start
                    if duration_ms < ASRConfig.MIN_SPEAKER_DURATION_MS:
                        # 跳过时长不足（静默）
                        logger.debug(f"  #{i+1} 跳过: 时长{duration_ms}ms < {ASRConfig.MIN_SPEAKER_DURATION_MS}ms")
                        continue
                    
                    # 声纹识别
                    identity = None
                    confidence = 0.0
                    recognition_details = {}
                    whisper_text = None
                    sensevoice_text = None
                    
                    segment_path = processed_path + f".seg_{i}.wav"
                    try:
                        from asr_server import extract_segment, transcribe_with_whisper, transcribe_with_sensevoice
                        if extract_segment(processed_path, start, end, segment_path):
                            result = identify_speaker_fusion(segment_path)
                            if result:
                                identity, confidence, recognition_details = result
                            
                            # 性能优化: 只有识别出的说话人才进行Whisper和SenseVoice处理
                            if identity is not None:
                                # Whisper对比识别
                                whisper_text = transcribe_with_whisper(segment_path)
                                
                                # SenseVoice识别和情感检测
                                sensevoice_result = transcribe_with_sensevoice(segment_path)
                                if sensevoice_result:
                                    sensevoice_text, sensevoice_emotion = sensevoice_result
                                    # 使用SenseVoice的情感结果(如果检测到)
                                    if sensevoice_emotion is not None:
                                        emotion = sensevoice_emotion
                                        emotion_source = "sensevoice"
                                        original_emotion_tag = f"<|{sensevoice_emotion}|>"
                                
                                # 识别成功（静默）
                                logger.debug(f"  #{i+1} 识别: {identity} ({confidence:.3f})")
                            else:
                                # 未识别（静默）
                                logger.debug(f"  #{i+1} 未识别")
                                
                    except Exception as e:
                        logger.warning(f"      [3.{i+1}] 声纹识别出错: {e}")
                    finally:
                        if os.path.exists(segment_path):
                            try:
                                os.remove(segment_path)
                            except:
                                pass
                    
                    # 检测是否为噪音(重复字符过多或填充词)
                    def is_noise(text):
                        if not text:
                            return True
                        # 检测单字符重复率
                        from collections import Counter
                        char_counts = Counter(text)
                        most_common_char, most_common_count = char_counts.most_common(1)[0]
                        repeat_ratio = most_common_count / len(text)
                        # 如果某个字符占比超过40%,认为是噪音
                        if repeat_ratio > 0.4:
                            return True
                        
                        # 检测填充词(嗯、啊、呃等)
                        filler_words = ['嗯', '啊', '呃', '额', '哦', '唔']
                        # 移除标点后检查
                        text_no_punct = re.sub(r'[，。、！？,.!?]', '', text)
                        if not text_no_punct:
                            return True
                        # 计算填充词占比
                        filler_count = sum(text_no_punct.count(w) for w in filler_words)
                        filler_ratio = filler_count / len(text_no_punct)
                        # 如果填充词占比超过60%,认为是噪音
                        return filler_ratio > 0.6
                    
                    # 计算文本质量指标
                    def calculate_text_quality(text):
                        """计算文本质量评估指标"""
                        from collections import Counter
                        if not text:
                            return {
                                "is_noise": True,
                                "noise_score": 1.0,
                                "repeat_ratio": 0.0,
                                "filler_ratio": 0.0
                            }
                        
                        # 计算重复字符占比
                        char_counts = Counter(text)
                        most_common_char, most_common_count = char_counts.most_common(1)[0]
                        repeat_ratio = most_common_count / len(text)
                        
                        # 计算填充词占比
                        filler_words = ['嗯', '啊', '呃', '额', '哦', '唔']
                        text_no_punct = re.sub(r'[，。、！？,.!?]', '', text)
                        filler_count = sum(text_no_punct.count(w) for w in filler_words) if text_no_punct else 0
                        filler_ratio = filler_count / len(text_no_punct) if text_no_punct else 0
                        
                        # 综合噪音评分 (0-1, 越高越可能是噪音)
                        noise_score = (repeat_ratio * 0.6 + filler_ratio * 0.4)
                        is_noise_flag = repeat_ratio > 0.4 or filler_ratio > 0.6
                        
                        return {
                            "is_noise": is_noise_flag,
                            "noise_score": round(noise_score, 3),
                            "repeat_ratio": round(repeat_ratio, 3),
                            "filler_ratio": round(filler_ratio, 3)
                        }
                    
                    # 过滤噪音
                    if is_noise(clean_text):
                        # 跳过噪音（静默）
                        logger.debug(f"  #{i+1} 跳过: 噪音")
                        continue
                    
                    # 只保留已注册说话人,丢弃Unknown
                    if ASRConfig.ONLY_REGISTERED_SPEAKERS and identity is None:
                        # 跳过未识别（静默）
                        logger.debug(f"  #{i+1} 跳过: 未识别说话人")
                        continue
                    
                    # 计算语速指标
                    duration_seconds = duration_ms / 1000.0
                    word_count = len(clean_text)  # 中文按字符数计算
                    speech_rate = word_count / duration_seconds if duration_seconds > 0 else 0
                    
                    # 计算文本质量
                    text_quality = calculate_text_quality(clean_text)
                    
                    processed_segments.append({
                        # === 原有字段（保持不变）===
                        "text": clean_text,
                        "start": start,
                        "end": end,
                        "spk": identity or "Unknown",
                        "emotion": emotion,
                        "whisper_text": whisper_text,
                        "sensevoice_text": sensevoice_text,
                        "confidence": float(f"{confidence:.3f}"),
                        "recognition_details": recognition_details,
                        
                        # === 新增字段：语速指标 ===
                        "speech_metrics": {
                            "duration_seconds": round(duration_seconds, 2),
                            "word_count": word_count,
                            "speech_rate": round(speech_rate, 2)  # 字/秒
                        },
                        
                        # === 新增字段：文本质量评估 ===
                        "text_quality": text_quality,
                        
                        # === 新增字段：情感详细信息 ===
                        "emotion_info": {
                            "emotion": emotion,
                            "source": emotion_source,  # "funasr" 或 "sensevoice"
                            "original_tag": original_emotion_tag,  # 原始情感标签，如 "<|happy|>"
                            "detected_by_sensevoice": emotion_source == "sensevoice"
                        }
                    })
                
                logger.info("  [生命周期: 3. 逐段声纹识别] 完成。")
            
            segments = processed_segments
            full_text = "".join([s["text"] for s in segments])
        
        # 清理临时文件
        if os.path.exists(processed_path):
            processed_basename = os.path.basename(processed_path)
            try:
                os.remove(processed_path)
                logger.debug(f"  [Cleanup] 已删除处理后文件: {processed_basename}")
            except Exception as e:
                logger.warning(f"  [Cleanup] 删除处理后文件失败: {processed_basename}, 错误: {e}")
        
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
        
        # 解析录音时间
        from db_manager import parse_recording_time
        recording_time = parse_recording_time(filename)
        
        # 提取音频片段并添加segment_audio_path
        base_name = os.path.splitext(filename)[0]
        segments_dir = os.path.join(FileMonitorConfig.SOURCE_DIR, "audio_segments", base_name)
        os.makedirs(segments_dir, exist_ok=True)
        logger.info(f"  [Audio Segments] 创建片段目录: {segments_dir}")
        
        # 导入音频提取函数
        from asr_server import extract_segment
        
        updated_segments = []
        for i, seg in enumerate(segments):
            # 提取音频片段
            seg_filename = f"seg_{i}.wav"
            seg_path = os.path.join(segments_dir, seg_filename)
            seg_audio_path = f"/audio_segments/{base_name}/{seg_filename}"
            
            start_ms = seg.get("start", 0)
            end_ms = seg.get("end", 0)
            
            # 尝试提取音频片段
            if extract_segment(wav_path, start_ms, end_ms, seg_path):
                logger.info(f"  [Audio Segments] 提取片段 {i}: {start_ms}ms - {end_ms}ms → {seg_path}")
            else:
                logger.warning(f"  [Audio Segments] 片段 {i} 提取失败")
                seg_audio_path = None
            
            # 保留所有原始字段并添加segment_audio_path
            original_path = seg.get("segment_audio_path")
            segment_data = seg.copy()
            segment_data["segment_audio_path"] = seg_audio_path
            
            # 日志追踪路径变化
            if original_path:
                logger.info(f"  [Path Override] 片段 {i}: '{original_path}' → '{seg_audio_path}'")
            else:
                logger.info(f"  [Path Set] 片段 {i}: '{seg_audio_path}'")
            
            updated_segments.append(segment_data)
        
        segments = updated_segments
        
        # 保存到数据库
        try:
            # 检查数据库连接池是否初始化
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
        # 清理所有临时文件（从temp目录）
        if os.path.exists(wav_path):
            try:
                os.remove(wav_path)
                logger.info(f"  [Cleanup] 已删除临时文件: {os.path.basename(wav_path)}")
            except Exception as e:
                logger.warning(f"  [Cleanup] 删除临时文件失败: {e}")
        
        # 清理.processed.wav文件（使用正确的路径）
        temp_dir = os.path.dirname(wav_path)
        wav_basename = os.path.basename(wav_path)
        processed_path = os.path.join(temp_dir, wav_basename.replace("_TEMP.wav", "_TEMP.processed.wav"))
        
        if os.path.exists(processed_path):
            try:
                os.remove(processed_path)
                logger.info(f"  [Cleanup] 已删除处理后文件: {os.path.basename(processed_path)}")
            except Exception as e:
                logger.warning(f"  [Cleanup] 删除处理后文件失败: {e}")


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
