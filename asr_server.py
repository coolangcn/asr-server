#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, logging, json, threading, subprocess, time, traceback, tempfile, argparse
import numpy as np
from scipy.spatial.distance import cosine
from flask import Flask, request, jsonify, render_template, send_file, send_from_directory, Response
from funasr import AutoModel  # ASR 用 FunASR
from modelscope.pipelines import pipeline  # SV 用 ModelScope
from modelscope.utils.constant import Tasks
import torch
import torchaudio
import shutil
import re
from collections import Counter
from db_manager import save_to_db, update_topics, parse_recording_time, init_pool, init_db
from logging.handlers import TimedRotatingFileHandler
import whisper
import requests
import hashlib
from datetime import datetime

# =================【 配置 】=================
class Config:
    DEVICE = "cuda:0"
    HOST = '0.0.0.0'
    PORT = 5008
    SPEAKER_DB_FILE = "speaker_db_multi.json"    
    # 长句音频保存配置
    SAVE_LONG_SENTENCES = True  # 是否保存长句音频
    MIN_TEXT_LENGTH_TO_SAVE = 15  # 最少字数
    LONG_SENTENCES_DIR = "long_sentences"  # 保存目录
    TEMP_DIR = "temp"  # 临时文件目录
    
    ONLY_REGISTERED_SPEAKERS = True  # 只保留已注册说话人,丢弃Unknown
    # ASR模型配置 - Paraformer (支持VAD分段和说话人分离)
    ASR_MODEL = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"  # 从 SenseVoiceSmall 切换到 Paraformer
    VAD_MODEL = "fsmn-vad"       # VAD模型
    SPK_MODEL = "cam++"          # 说话人分离模型  
    PUNC_MODEL = "ct-punc"       # 标点恢复模型
    
    # VAD参数配置(为Paraformer优化)
    VAD_MAX_SINGLE_SEGMENT = 15000  # ms - 单段最长时间
    VAD_MAX_END_SILENCE = 300       # ms - 段尾静音阈值
    VAD_SIL_TO_SPEECH = 50          # ms - 静音到语音阈值
    VAD_SPEECH_TO_SIL = 80          # ms - 语音到静音阈值
    
    SV_MODELS = {
        "eres2net_large": {
            "id": "iic/speech_eres2net_large_200k_sv_zh-cn_16k-common",
            "rev": "v1.0.0",
            "threshold": 0.50,  # 提高阈值以提高准确率
            "gap": 0.08         # 降低置信度间隔要求
        },
        "rdino_ecapa": {
            "id": "iic/speech_rdino_ecapa_tdnn_sv_zh-cn_cnceleb_16k",
            "rev": "v1.0.0",
            "threshold": 0.50,  # 提高阈值以提高准确率
            "gap": 0.08         # 降低置信度间隔要求
        },
        "camplusplus": {
            "id": "iic/speech_campplus_sv_zh-cn_16k-common",
            "rev": "v1.0.0",
            "threshold": 0.50,  # 提高阈值以提高准确率
            "gap": 0.08         # 降低置信度间隔要求
        }
    }
    
    MIN_SPEAKER_DURATION_MS = 800
    NORMALIZE_AUDIO = True
    DENOISE_AUDIO = False  # 启用高级降噪
    
    # 可选功能开关
    ENABLE_EMOTION_DETECTION = True  # 是否启用情感检测(需要SenseVoice模型)
    ENABLE_WHISPER_COMPARISON = True  # 是否启用Whisper对比(需要Whisper模型)
    
    # SenseVoice配置 (情感检测)
    SENSEVOICE_MODEL = "iic/SenseVoiceSmall"
    ENABLE_SENSEVOICE = True  # 是否启用SenseVoice(情感检测+第三转录)

# 文件监控配置
class FileMonitorConfig:
    SOURCE_DIR = "V:\\Sony-2"
    PROCESSED_DIR = "processed"
    FAILED_DIR = "failed"
    TRANSCRIPTS_DIR = "transcripts"
    SCAN_INTERVAL = 3  # 秒
    SUPPORTED_FORMATS = ['.m4a', '.mp3', '.wav', '.aac', '.flac', '.ogg', '.acc']

# LLM 配置
class LLMConfig:
    USE_GEMINI_LLM = False
    GEMINI_API_KEY = "cncncncn"
    GEMINI_API_BASE_URL = "https://gl.moco.fun/proxy/gemini"
    GEMINI_MODEL_NAME = "gemini-2.5-flash"
    
    # 批量处理配置
    LLM_BATCH_MODE = True
    LLM_BATCH_SIZE = 20
    LLM_BATCH_TIMEOUT = 600  # 10分钟
    
    # 过滤条件
    LLM_MIN_TEXT_LENGTH = 50
    LLM_MIN_SEGMENTS = 3
    LLM_CACHE_SIZE = 100
    LLM_REQUEST_TIMEOUT = 30
# ==========================================

EMOTION_TAGS = {
    "<|happy|>": "happy", "<|sad|>": "sad", "<|angry|>": "angry",
    "<|neutral|>": "neutral", "<|laughter|>": "laughter", "<|fearful|>": "fearful",
    "<|disgusted|>": "disgusted", "<|surprised|>": "surprised", "<|EMO_UNKNOWN|>": "neutral"
}
INVALID_TAGS = {"<|nospeech|>", "<|BGM|>", "<|Event_UNK|>", "<|music|>"}

# 新增：定义说话人数据结构
# {
#   "speaker_name": {
#     "samples": [
#       {
#         "id": "sample_id",
#         "filename": "file_name.wav",
#         "timestamp": "2023-01-01 12:00:00",
#         "embeddings": {
#           "eres2net_large": [...],
#           "rdino_ecapa": [...]
#         }
#       }
#     ],
#     "avg_embeddings": {
#       "eres2net_large": [...],
#       "rdino_ecapa": [...]
#     }
#   }
# }

# 创建日志队列用于SSE
export_logger = logging.getLogger('export_logger')
export_logger.setLevel(logging.INFO)

# 自定义日志处理器，将日志消息发送到SSE连接
class SSEHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.clients = set()
    
    def add_client(self, client):
        self.clients.add(client)
    
    def remove_client(self, client):
        self.clients.remove(client)
    
    def emit(self, record):
        msg = self.format(record)
        for client in list(self.clients):
            try:
                client.write(f"data: {json.dumps({'message': msg, 'level': record.levelname})}\n\n")
            except Exception:
                self.remove_client(client)

# 创建日志处理器
log_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

# 创建并配置SSE处理器
sse_handler = SSEHandler()
sse_handler.setFormatter(log_formatter)
sse_handler.setLevel(logging.INFO)

# 配置根日志记录器
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 创建文件处理器，用于将日志写入文件（每10分钟轮转一次）
# 确保日志目录存在
os.makedirs("log", exist_ok=True)
file_handler = TimedRotatingFileHandler(
    'log/asr-server.log', 
    when='M',           # 按分钟轮转
    interval=10,        # 每10分钟
    backupCount=144,    # 保留144个文件（24小时）
    encoding='utf-8'
)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler) # 添加文件处理器

logger.addHandler(console_handler)
logger.addHandler(sse_handler)

app = Flask(__name__)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

asr_pipeline = None
sv_pipelines = {}
speaker_db = {}
emotion_pipeline = None  # 可选: 情感检测模型
whisper_model = None     # 可选: Whisper对比模型
sensevoice_pipeline = None  # 可选: SenseVoice模型(情感+转录)
gpu_lock = threading.Lock()
db_lock = threading.Lock()

# =================【 LLM 批量处理全局变量 】=================
llm_batch_queue = []
llm_batch_lock = threading.Lock()
llm_last_batch_time = time.time()
llm_cache = {}  # 缓存 LLM 响应
llm_cache_lock = threading.Lock()
# =========================================================

# =================== 模型加载 ===================
def load_models():
    global asr_pipeline, sv_pipelines, whisper_model, sensevoice_pipeline
    print("\n====== 🚀 启动 SOTA 融合服务 ======")
    
    load_speaker_db()

    # 2. 加载 ASR (FunASR)
    print(f"🧠 加载 ASR: {Config.ASR_MODEL} ...")
    # 2. 加载 ASR (FunASR Paraformer + VAD + 说话人分离)
    print(f"🧠 加载 ASR: {Config.ASR_MODEL} (支持VAD分段和说话人分离) ...")
    asr_pipeline = AutoModel(
        model=Config.ASR_MODEL,       # paraformer-zh
        vad_model=Config.VAD_MODEL,   # fsmn-vad
        punc_model=Config.PUNC_MODEL, # ct-punc (标点恢复)
        spk_model=Config.SPK_MODEL,   # cam++ (说话人分离)
        vad_kwargs={
            "max_single_segment_time": Config.VAD_MAX_SINGLE_SEGMENT,
            "max_end_silence_time": Config.VAD_MAX_END_SILENCE,
            "sil_to_speech_time_thres": Config.VAD_SIL_TO_SPEECH,
            "speech_to_sil_time_thres": Config.VAD_SPEECH_TO_SIL
        },
        device=Config.DEVICE, 
        disable_update=True
    )
    print("✅ Paraformer模型加载完成，已启用VAD分段和说话人分离功能")

    # 3. 加载 SV 模型
    for name, conf in Config.SV_MODELS.items():
        print(f"🔍 加载 SV [{name}] : {conf['id']} ...")
        sv_pipelines[name] = pipeline(
            task=Tasks.speaker_verification,
            model=conf['id'], 
            model_revision=conf['rev'], 
            device=Config.DEVICE.split(':')[0]
        )
    print(f"✅ 服务就绪 | ASR: SenseVoice | SV: {list(sv_pipelines.keys())}\n")

    # 4. 加载 Whisper 模型 (可选)
    if Config.ENABLE_WHISPER_COMPARISON:
        print(f"🎤 加载 Whisper large-v3 模型 (最新最佳版本,需要~10GB显存)...")

        try:
            whisper_model = whisper.load_model("large-v3", device=Config.DEVICE.split(':')[0])
            print("✅ Whisper large-v3 模型加载完成")

        except Exception as e:
            logger.warning(f"⚠️ Whisper模型加载失败: {e}，将禁用Whisper对比功能")
            whisper_model = None

    # 5. 加载 SenseVoice 模型 (情感检测)
    if Config.ENABLE_SENSEVOICE:
        print(f"🎭 加载 SenseVoice 模型 (情感检测+第三转录)...")
        try:
            sensevoice_pipeline = AutoModel(
                model=Config.SENSEVOICE_MODEL,
                device=Config.DEVICE
            )
            print("✅ SenseVoice 模型加载完成")
        except Exception as e:
            logger.warning(f"⚠️ SenseVoice模型加载失败: {e}，将禁用SenseVoice功能")
            sensevoice_pipeline = None

# =================【 智能摘要和 LLM 函数 】=================

def generate_conversation_summary(segments, audio_duration):
    """生成对话智能摘要"""
    if not segments:
        return None
    
    # 统计说话人
    speaker_stats = {}
    for seg in segments:
        speaker = seg.get('spk', 'Unknown')
        if speaker not in speaker_stats:
            speaker_stats[speaker] = {'count': 0, 'total_duration': 0, 'word_count': 0}
        speaker_stats[speaker]['count'] += 1
        speaker_stats[speaker]['total_duration'] += (seg.get('end', 0) - seg.get('start', 0)) / 1000.0
        speaker_stats[speaker]['word_count'] += len(seg.get('text', ''))
    
    # 提取高频词
    stop_words = {'的', '了', '是', '在', '我', '你', '他', '她', '它', '们', '这', '那', '有', '个', '就', '不', '和', '与'}
    all_text = ''.join([seg.get('text', '') for seg in segments])
    words = [all_text[i:i+2] for i in range(len(all_text)-1)]
    word_freq = Counter([w for w in words if w not in stop_words and len(w) == 2])
    top_keywords = [word for word, count in word_freq.most_common(5)]
    
    # 情感统计
    emotion_stats = Counter([seg.get('emotion') for seg in segments if seg.get('emotion')])
    
    return {
        'total_segments': len(segments),
        'total_duration': round(audio_duration, 2),
        'speaker_count': len(speaker_stats),
        'speakers': speaker_stats,
        'keywords': top_keywords,
        'emotions': dict(emotion_stats),
        'avg_segment_duration': round(audio_duration / len(segments), 2) if segments else 0
    }

def call_gemini_api(prompt):
    """调用 Gemini API"""
    if not LLMConfig.USE_GEMINI_LLM:
        return None
        
    try:
        # 检查缓存
        cache_key = hashlib.md5(prompt.encode()).hexdigest()
        with llm_cache_lock:
            if cache_key in llm_cache:
                logger.info(f"  [LLM] 使用缓存响应")
                return llm_cache[cache_key]
        
        url = f"{LLMConfig.GEMINI_API_BASE_URL}/v1beta/models/{LLMConfig.GEMINI_MODEL_NAME}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": LLMConfig.GEMINI_API_KEY
        }
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 500}
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=LLMConfig.LLM_REQUEST_TIMEOUT)
        response.raise_for_status()
        result = response.json()
        
        if 'candidates' in result and len(result['candidates']) > 0:
            text = result['candidates'][0]['content']['parts'][0]['text']
            
            # 保存到缓存
            with llm_cache_lock:
                if len(llm_cache) >= LLMConfig.LLM_CACHE_SIZE:
                    llm_cache.pop(next(iter(llm_cache)))
                llm_cache[cache_key] = text
            
            return text
        return None
    except Exception as e:
        logger.error(f"  [LLM] API 调用失败: {e}")
        return None

def extract_conversation_topics(full_text, segments):
    """提取对话主题"""
    try:
        speakers = list(set([seg.get('spk', 'Unknown') for seg in segments]))
        speaker_text = ', '.join(speakers[:3])
        
        prompt = f"""分析以下对话内容，提取关键信息：

对话内容：
{full_text[:500]}

说话人：{speaker_text}

请以JSON格式返回：
{{
  "topics": ["主题1", "主题2"],
  "keywords": ["关键词1", "关键词2", "关键词3"],
  "sentiment": "positive/neutral/negative",
  "summary": "一句话总结"
}}"""
        
        response_text = call_gemini_api(prompt)
        if not response_text:
            return None
        
        # 解析 JSON
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return None
    except Exception as e:
        logger.warning(f"  [LLM] 主题提取失败: {e}")
        return None

def add_to_llm_queue(filename, full_text, segments):
    """添加到 LLM 批量处理队列"""
    global llm_last_batch_time
    
    with llm_batch_lock:
        llm_batch_queue.append({
            'filename': filename,
            'full_text': full_text,
            'segments': segments
        })
        
        queue_size = len(llm_batch_queue)
        time_since_last = time.time() - llm_last_batch_time
        
        logger.info(f"  [LLM队列] 已添加，当前队列: {queue_size}/{LLMConfig.LLM_BATCH_SIZE}")
        
        # 触发批量处理
        if queue_size >= LLMConfig.LLM_BATCH_SIZE or time_since_last >= LLMConfig.LLM_BATCH_TIMEOUT:
            logger.info(f"  [LLM队列] 触发批量处理 (队列={queue_size}, 超时={time_since_last:.0f}s)")
            threading.Thread(target=process_llm_batch, daemon=True).start()

def process_llm_batch():
    """批量处理 LLM 任务"""
    global llm_last_batch_time
    
    if not LLMConfig.USE_GEMINI_LLM:
        with llm_batch_lock:
            llm_batch_queue.clear()
        return
        
    with llm_batch_lock:
        if not llm_batch_queue:
            return
        
        batch = llm_batch_queue.copy()
        llm_batch_queue.clear()
        llm_last_batch_time = time.time()
    
    logger.info(f"  [LLM批处理] 开始处理 {len(batch)} 条记录")
    
    for item in batch:
        try:
            topics = extract_conversation_topics(item['full_text'], item['segments'])
            if topics:
                update_topics(item['filename'], topics)
                logger.info(f"  [LLM] {item['filename']}: 主题={topics.get('topics', [])}")
        except Exception as e:
            logger.error(f"  [LLM] 处理失败 {item['filename']}: {e}")
    
    logger.info(f"  [LLM批处理] 完成")

# =========================================================

def cleanup_temp_dir():
    """清理超过1小时的临时文件"""
    try:
        temp_dir = Config.TEMP_DIR
        if not os.path.exists(temp_dir):
            return
        
        current_time = time.time()
        cleaned_count = 0
        
        for filename in os.listdir(temp_dir):
            filepath = os.path.join(temp_dir, filename)
            if os.path.isfile(filepath):
                try:
                    file_age = current_time - os.path.getmtime(filepath)
                    if file_age > 3600:  # 1小时
                        os.remove(filepath)
                        cleaned_count += 1
                        logger.debug(f"清理旧临时文件: {filename}")
                except Exception as e:
                    logger.warning(f"清理文件失败 {filename}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"临时文件清理完成，删除了 {cleaned_count} 个文件")
        
        # 每小时执行一次
        threading.Timer(3600, cleanup_temp_dir).start()
    except Exception as e:
        logger.error(f"临时文件清理失败: {e}")
        # 即使失败也要继续定时任务
        threading.Timer(3600, cleanup_temp_dir).start()

def load_speaker_db():
    global speaker_db
    with db_lock:
        if os.path.exists(Config.SPEAKER_DB_FILE):
            try:
                with open(Config.SPEAKER_DB_FILE, 'r', encoding='utf-8') as f:
                    loaded_db = json.load(f)
                
                # 兼容旧数据结构
                converted_db = {}
                for name, data in loaded_db.items():
                    if "samples" in data and "avg_embeddings" in data:
                        # 新数据结构，直接使用
                        converted_db[name] = data
                    else:
                        # 旧数据结构，转换为新结构
                        logger.info(f"🔄 转换旧数据结构 for speaker: {name}")
                        converted_db[name] = {
                            "samples": [],  # 旧数据结构没有样本信息
                            "avg_embeddings": data  # 旧数据结构直接是嵌入字典
                        }
                
                speaker_db = converted_db
                logger.info(f"📚 声纹库已挂载: {len(speaker_db)} 人")
            except Exception as e:
                logger.error(f"声纹库损坏: {e}")
                speaker_db = {}
        else:
            logger.warning(f"⚠️ 未找到 {Config.SPEAKER_DB_FILE}，将创建新的数据库。")
            speaker_db = {}

# =================== 音频预处理 ===================
def preprocess_audio(input_path, output_path):
    # 如果启用了高级降噪，先进行降噪处理
    if Config.DENOISE_AUDIO:
        denoised_path = input_path + ".denoised.wav"
        if advanced_denoise(input_path, denoised_path):
            input_path = denoised_path
        else:
            logger.warning("高级降噪处理失败，使用原始音频")
    
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", input_path]
    filters = ["loudnorm=I=-14:TP=-1.5:LRA=11"] if Config.NORMALIZE_AUDIO else []
    if filters: cmd.extend(["-af", ",".join(filters)])
    cmd.extend(["-ac", "1", "-ar", "16000", output_path])
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        # 清理临时降噪文件
        if Config.DENOISE_AUDIO and input_path.endswith(".denoised.wav"):
            try:
                os.remove(input_path)
            except:
                pass
        return True
    except Exception as e:
        logger.error(f"FFmpeg 预处理失败: {e}")
        return False

def advanced_denoise(input_path, output_path):
    """使用谱减法进行高级降噪"""
    try:
        # 加载音频
        waveform, sample_rate = torchaudio.load(input_path)
        
        # 如果采样率不是16kHz，先进行重采样
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        # 转换为单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 简化的谱减法降噪
        # 这里我们使用一个简化的实现，实际应用中可以使用更复杂的算法
        audio_np = waveform.numpy()[0]
        
        # 计算短时傅里叶变换
        from scipy import signal
        frequencies, times, Zxx = signal.stft(audio_np, fs=sample_rate, nperseg=512)
        
        # 估计噪声谱（假设前100ms为噪声）
        noise_seg_len = min(int(0.1 * sample_rate), len(audio_np))
        noise_segment = audio_np[:noise_seg_len]
        _, _, noise_stft = signal.stft(noise_segment, fs=sample_rate, nperseg=512)
        noise_spectrum = np.mean(np.abs(noise_stft), axis=1)
        
        # 应用谱减法
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)
        
        # 减去噪声谱的估计值
        noise_factor = 1.5
        magnitude_denoised = np.maximum(magnitude - noise_factor * noise_spectrum[:, np.newaxis], 0)
        
        # 重构信号
        Zxx_denoised = magnitude_denoised * np.exp(1j * phase)
        _, audio_denoised = signal.istft(Zxx_denoised, fs=sample_rate)
        
        # 裁剪到原始长度
        audio_denoised = audio_denoised[:len(audio_np)]
        
        # 保存降噪后的音频
        waveform_denoised = torch.tensor(audio_denoised).unsqueeze(0)
        torchaudio.save(output_path, waveform_denoised, sample_rate)
        
        return True
    except Exception as e:
        logger.error(f"高级降噪处理失败: {e}")
        return False

def extract_segment(source_path, start_ms, end_ms, output_path):
    if start_ms >= end_ms: return False
    start_sec = start_ms / 1000.0
    duration = (end_ms - start_ms) / 1000.0
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start_sec:.3f}", "-t", f"{duration:.3f}", "-i", source_path, "-ac", "1", "-ar", "16000", output_path]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        return True
    except:
        return False



def transcribe_with_whisper(audio_path):
    """
    使用Whisper识别音频片段（作为FunASR的对比参考）
    
    Args:
        audio_path: 音频片段路径
        
    Returns:
        str: Whisper识别的文本，如果失败返回None
    """
    if not Config.ENABLE_WHISPER_COMPARISON or whisper_model is None:
        return None
    
    try:
        result = whisper_model.transcribe(
            audio_path,
            language='zh',
            fp16=True,  # GPU加速
            verbose=False
        )
        whisper_text = result['text'].strip()
        logger.info(f"      [Whisper对比] {whisper_text}")
        return whisper_text
    except Exception as e:
        logger.warning(f"      [Whisper对比] 识别失败: {e}")
        return None


def transcribe_with_sensevoice(audio_path):
    """
    使用SenseVoice识别音频并检测情感
    
    Returns:
        tuple: (text, emotion) - 识别文本和情感
    """
    if not Config.ENABLE_SENSEVOICE or sensevoice_pipeline is None:
        return None, None  # 未识别到情感返回None
    
    try:
        result = sensevoice_pipeline.generate(
            input=audio_path,
            language="auto",
            use_itn=True
        )
        
        if not result or len(result) == 0:
            return None, None  # 未识别到情感返回None
        
        raw_text = result[0].get("text", "")
        
        # 提取情感
        emotion = None  # 未识别到情感时为None,不使用neutral
        for tag, emo_code in EMOTION_TAGS.items():
            if tag.lower() in raw_text.lower():
                emotion = emo_code
                break
        
        # 移除情感标签
        clean_text = re.sub(r'<\|.*?\|>', '', raw_text).strip()
        
        logger.info(f"      [SenseVoice] {clean_text} (情感: {emotion})")
        return clean_text, emotion
        
    except Exception as e:
        logger.warning(f"      [SenseVoice] 识别失败: {e}")
        return None, None  # 未识别到情感返回None

def detect_emotion_for_segment(audio_path):
    """使用SenseVoice检测音频段的情感"""
    if not Config.ENABLE_EMOTION_DETECTION or emotion_pipeline is None:
        return "neutral"
    
    try:
        result = emotion_pipeline(
            audio_in=audio_path,
            language="auto",
            use_itn=True
        )
        
        if not result or len(result) == 0:
            return "neutral"
        
        raw_text = result[0].get("text", "")
        logger.info(f"      [SenseVoice情感] 原始输出: {raw_text}")
        
        # 提取情感标签
        emotion = None  # 未识别到情感时为None,不使用neutral
        raw_text_lower = raw_text.lower()
        
        EMOTION_MAP = {
            '<|happy|>': 'happy',
            '<|sad|>': 'sad', 
            '<|angry|>': 'angry',
            '<|neutral|>': 'neutral',
            '<|fearful|>': 'fearful',
            '<|disgusted|>': 'disgusted',
            '<|surprised|>': 'surprised'
        }
        
        for tag, emo in EMOTION_MAP.items():
            if tag in raw_text_lower:
                emotion = emo
                logger.info(f"      [SenseVoice情感] 检测到情感: {emotion}")
                break
        
        return emotion
    except Exception as e:
        logger.warning(f"      [SenseVoice情感] 检测失败: {e}")
        return "neutral"


# =================== 提取 embedding ===================
def extract_embedding_from_file(sv_pipe, wav_path):
    try:
        model = sv_pipe.model
        audio, sr = torchaudio.load(wav_path)
        if sr != 16000:
            resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            audio = resample(audio)
        
        audio = audio.mean(dim=0, keepdim=True) # [C, T] -> [1, T]

        with torch.no_grad():
            out = model(audio)
            if isinstance(out, dict):
                emb = out.get("spk_embedding")
            else:
                emb = out
        return emb.squeeze().cpu().numpy()

    except Exception as e:
        logger.error(f"❌ extract_embedding 失败: {e}")
        return None

# =================== 多模型交叉验证 ===================
def identify_speaker_fusion(segment_path):
    if not speaker_db: 
        logger.info("🤷‍♂️ 声纹数据库为空，无法进行识别")
        return None, 0.0, []

    model_votes = {}
    model_scores = {}

    logger.info(f"🎯 开始声纹识别: 音频段路径={segment_path}")
    logger.info(f"📋 声纹数据库包含 {len(speaker_db)} 个说话人")

    for model_name, sv_pipe in sv_pipelines.items():
        # 模型处理（静默）
        
        emb_a = extract_embedding_from_file(sv_pipe, segment_path)
        if emb_a is None:
            logger.error(f"❌ 模型 {model_name} 特征提取失败")
            model_votes[model_name] = "Failed"
            continue

        scores = []
        conf = Config.SV_MODELS[model_name]
        threshold = conf['threshold']
        gap = conf['gap']
        # 配置已在启动时显示，无需重复

        for name, speaker_data in speaker_db.items():
            # 使用平均嵌入进行比较
            if "avg_embeddings" not in speaker_data or model_name not in speaker_data["avg_embeddings"]: 
                continue
            emb_b = np.array(speaker_data["avg_embeddings"][model_name]).flatten()
            score = 1 - cosine(emb_a.flatten(), emb_b)
            scores.append((name, score))
            # DEBUG: 详细评分
            logger.debug(f"  {model_name}: {name}={score:.3f}")

        if not scores:
            logger.warning(f"⚠️ 模型 {model_name} 未找到匹配的说话人数据")
            model_votes[model_name] = "NoDB"
            continue

        scores.sort(key=lambda x: x[1], reverse=True)
        top1_name, top1_score = scores[0]
        top2_name, top2_score = scores[1] if len(scores) > 1 else (None, 0.0)
        score_gap = top1_score - top2_score
        
        # DEBUG: 模型识别结果
        logger.debug(f"  {model_name}: {top1_name}={top1_score:.3f} (gap={score_gap:.3f})")

        if top1_score >= threshold and score_gap >= gap:
            model_votes[model_name] = top1_name
            model_scores[model_name] = top1_score
            # 验证通过（静默）
            pass
        else:
            model_votes[model_name] = "Unknown"
            model_scores[model_name] = top1_score
            reason = []
            if top1_score < threshold:
                reason.append(f"得分 {top1_score:.6f} < 阈值 {threshold}")
            if score_gap < gap:
                reason.append(f"差距 {score_gap:.6f} < 置信度间隔 {gap}")
            # 验证失败（静默）
            pass

    # DEBUG: 投票结果
    logger.debug(f"  投票: {model_votes}")
    
    # 2/3投票逻辑
    votes = [v for v in model_votes.values() if v not in ["Unknown", "Failed", "NoDB"]]
    if not votes:
        # 识别失败（由上层记录）
        logger.debug("  未识别: 所有模型均未通过")
        return None, 0.0, []

    vote_counts = Counter(votes)
    most_common_vote = vote_counts.most_common(1)[0]
    winner, count = most_common_vote
    
    # 至少需要2票
    if count >= 2:
        # 计算获胜者的平均置信度
        winning_scores = [model_scores[model] for model, vote in model_votes.items() if vote == winner]
        avg_confidence = np.mean(winning_scores)
        
        # 识别成功（由上层记录）
        logger.debug(f"  识别: {winner} (置信度={avg_confidence:.3f}, 票数={count})")
        
        # 生成详细信息
        recognition_details = []
        for model_name, result in model_votes.items():
            if result in ["Unknown", "Failed", "NoDB"]:
                recognition_details.append(f"模型 {model_name}: {result}")
            else:
                recognition_details.append(f"模型 {model_name}: 识别为 {result} (相似度: {model_scores.get(model_name, 0):.6f})")
        recognition_details.append(f"最终识别结果: {winner} (多数票: {count} 票, 平均置信度: {avg_confidence:.3f})")
        
        return winner, avg_confidence, recognition_details
    else:
        # 生成识别失败的详细信息
        recognition_details = []
        for model_name, result in model_votes.items():
            recognition_details.append(f"模型 {model_name}: {result} (相似度: {model_scores.get(model_name, 0):.6f})")
        recognition_details.append("最终识别结果: 识别失败，没有候选人获得足够票数 (多数票 ≥ 2)")
        
        # 识别失败（由上层记录）
        logger.debug(f"  未识别: 票数不足 ({winner}={count}<2)")
        return None, 0.0, []

# =================== Flask 接口 ===================
@app.route("/")
def home():
    return render_template("register.html")

@app.route("/register_page")
def register_page():
    return render_template("register.html")

@app.route("/manage")
def manage_page():
    return render_template("manage.html")

@app.route("/speakers", methods=["GET"])
def get_speakers():
    """获取所有说话人列表"""
    try:
        # 重新加载声纹数据库以确保数据是最新的
        load_speaker_db()
        # 返回说话人列表（不包含具体的embedding数据）
        speakers_summary = {}
        for name, data in speaker_db.items():
            sample_count = len(data.get("samples", []))
            model_names = list(data.get("avg_embeddings", {}).keys())
            speakers_summary[name] = {
                "sample_count": sample_count,
                "models": model_names
            }
        return jsonify({"speakers": speakers_summary})
    except Exception as e:
        logger.error(f"获取说话人列表失败: {str(e)}")
        return jsonify({"error": "Failed to retrieve speakers"}), 500

@app.route("/speaker/<speaker_name>", methods=["GET"])
def get_speaker_samples(speaker_name):
    """获取指定说话人的样本列表"""
    try:
        # 重新加载声纹数据库以确保数据是最新的
        load_speaker_db()
        if speaker_name not in speaker_db:
            return jsonify({"error": f"Speaker '{speaker_name}' not found."}), 404
            
        speaker_data = speaker_db[speaker_name]
        # 返回样本信息（不包含具体的embedding数据）
        samples_info = []
        for sample in speaker_data.get("samples", []):
            samples_info.append({
                "id": sample["id"],
                "filename": sample["filename"],
                "timestamp": sample["timestamp"]
            })
        
        return jsonify({
            "speaker_name": speaker_name,
            "sample_count": len(samples_info),
            "samples": samples_info,
            "models": list(speaker_data.get("avg_embeddings", {}).keys())
        })
    except Exception as e:
        logger.error(f"获取说话人样本列表失败: {str(e)}")
        return jsonify({"error": "Failed to retrieve speaker samples"}), 500

@app.route("/speaker/<speaker_name>", methods=["DELETE"])
def delete_speaker(speaker_name):
    """删除指定说话人"""
    try:
        with db_lock:
            if speaker_name in speaker_db:
                del speaker_db[speaker_name]
                # 保存更新后的数据库
                with open(Config.SPEAKER_DB_FILE, 'w', encoding='utf-8') as f:
                    json.dump(speaker_db, f, indent=2, ensure_ascii=False)
                logger.info(f"✅ 成功删除说话人: {speaker_name}")
                return jsonify({"message": f"Speaker '{speaker_name}' deleted successfully."})
            else:
                return jsonify({"error": f"Speaker '{speaker_name}' not found."}), 404
    except Exception as e:
        logger.error(f"删除说话人失败: {str(e)}")
        return jsonify({"error": "Failed to delete speaker"}), 500

@app.route("/speaker/<speaker_name>/sample/<sample_id>", methods=["DELETE"])
def delete_speaker_sample(speaker_name, sample_id):
    """删除指定说话人的特定样本"""
    try:
        with db_lock:
            if speaker_name not in speaker_db:
                return jsonify({"error": f"Speaker '{speaker_name}' not found."}), 404
                
            speaker_data = speaker_db[speaker_name]
            if "samples" not in speaker_data:
                return jsonify({"error": f"No samples found for speaker '{speaker_name}'."}), 404
                
            # 查找并删除指定样本
            samples = speaker_data["samples"]
            sample_to_remove = None
            sample_index = -1
            for i, sample in enumerate(samples):
                if sample["id"] == sample_id:
                    sample_to_remove = sample
                    sample_index = i
                    break
                    
            if sample_to_remove is None:
                return jsonify({"error": f"Sample '{sample_id}' not found for speaker '{speaker_name}'."}), 404
                
            # 删除样本的音频文件
            if "audio_path" in sample_to_remove and os.path.exists(sample_to_remove["audio_path"]):
                try:
                    os.remove(sample_to_remove["audio_path"])
                    logger.info(f"🗑️ 删除了音频文件: {sample_to_remove['audio_path']}")
                except Exception as e:
                    logger.warning(f"⚠️ 删除音频文件失败: {sample_to_remove['audio_path']}, 错误: {str(e)}")
            
            # 从数据库中移除样本记录
            del samples[sample_index]
                
            # 如果删除样本后没有剩余样本，则删除整个说话人
            if len(samples) == 0:
                del speaker_db[speaker_name]
                # 删除说话人的目录
                speaker_dir = os.path.join("speaker_samples", speaker_name)
                if os.path.exists(speaker_dir):
                    try:
                        shutil.rmtree(speaker_dir)
                        logger.info(f"🗑️ 删除了说话人目录: {speaker_dir}")
                    except Exception as e:
                        logger.warning(f"⚠️ 删除说话人目录失败: {speaker_dir}, 错误: {str(e)}")
                
                with open(Config.SPEAKER_DB_FILE, 'w', encoding='utf-8') as f:
                    json.dump(speaker_db, f, indent=2, ensure_ascii=False)
                logger.info(f"🗑️ 删除了说话人 {speaker_name}（最后一个样本已删除）")
                return jsonify({"message": f"Speaker '{speaker_name}' deleted (last sample removed)."})

            # 重新计算平均嵌入
            all_model_embeddings = {model_name: [] for model_name in sv_pipelines.keys()}
            for sample in samples:
                for model_name, emb in sample["embeddings"].items():
                    all_model_embeddings[model_name].append(np.array(emb))
            
            # 计算新的平均嵌入
            new_avg_embeddings = {}
            for model_name, emb_list in all_model_embeddings.items():
                if emb_list:
                    avg_emb = np.mean(emb_list, axis=0)
                    new_avg_embeddings[model_name] = avg_emb.tolist()
            
            speaker_db[speaker_name]["avg_embeddings"] = new_avg_embeddings
            
            # 保存更新后的数据库
            with open(Config.SPEAKER_DB_FILE, 'w', encoding='utf-8') as f:
                json.dump(speaker_db, f, indent=2, ensure_ascii=False)
                
            logger.info(f"🗑️ 删除了说话人 {speaker_name} 的样本 {sample_id}")
            return jsonify({
                "message": f"Sample '{sample_id}' deleted from speaker '{speaker_name}'.",
                "remaining_samples": len(samples)
            })
    except Exception as e:
        logger.error(f"删除说话人样本失败: {str(e)}")
        return jsonify({"error": "Failed to delete speaker sample"}), 500


@app.route("/speaker/list", methods=["GET"])
def list_speakers():
    """获取所有说话人列表 (Web Viewer格式)"""
    try:
        # 重新加载声纹数据库以确保数据是最新的
        load_speaker_db()
        # 返回说话人列表数组格式
        speakers_list = []
        for name, data in speaker_db.items():
            sample_count = len(data.get("samples", []))
            speakers_list.append({
                "name": name,
                "sample_count": sample_count
            })
        return jsonify({"speakers": speakers_list})
    except Exception as e:
        logger.error(f"获取说话人列表失败: {str(e)}")
        return jsonify({"error": "Failed to retrieve speakers"}), 500

@app.route("/speaker/register", methods=["POST"])
def register_speaker_web():
    """注册声纹 (Web Viewer格式) - 适配器端点"""
    # 确保临时目录存在
    os.makedirs(Config.TEMP_DIR, exist_ok=True)
    temp_files = []
    with gpu_lock:
        try:
            if 'speaker_name' not in request.form or not request.form['speaker_name']:
                return jsonify({"error": "Speaker name is required"}), 400
            
            speaker_name = request.form['speaker_name']
            
            # Web viewer发送单个audio_file，需要转换为audio_files列表
            if 'audio_file' not in request.files:
                return jsonify({"error": "Audio file is required"}), 400
            
            audio_file = request.files['audio_file']
            
            # 自动检测是否需要增强模式
            enhance_mode = speaker_name in speaker_db

            action = "增强" if enhance_mode else "注册"
            logger.info(f"📥 开始{action}新声纹: {speaker_name} | 文件: {audio_file.filename}")
            
            # 创建说话人样本目录
            speaker_dir = os.path.join("speaker_samples", speaker_name)
            if not os.path.exists(speaker_dir):
                os.makedirs(speaker_dir)
            
            # 收集新样本数据
            new_samples = []
            model_embeddings = {model_name: [] for model_name in sv_pipelines.keys()}

            # 处理音频文件
            raw_temp = os.path.join(Config.TEMP_DIR, f"reg_raw_{int(time.time())}_{audio_file.filename}")
            audio_file.save(raw_temp)
            temp_files.append(raw_temp)
            
            proc_temp = os.path.join(Config.TEMP_DIR, f"reg_proc_{int(time.time())}.wav")
            temp_files.append(proc_temp)

            if not preprocess_audio(raw_temp, proc_temp):
                return jsonify({"error": f"Audio preprocessing failed for {audio_file.filename}"}), 500

            # 为每个模型提取嵌入
            sample_embeddings = {}
            for model_name, sv_pipe in sv_pipelines.items():
                emb = extract_embedding_from_file(sv_pipe, proc_temp)
                if emb is not None:
                    sample_embeddings[model_name] = emb.tolist()
                    model_embeddings[model_name].append(emb)
                else:
                    logger.warning(f"⚠️ 从 {audio_file.filename} 提取 {model_name} embedding 失败。")

            # 保存样本信息和音频文件
            if not sample_embeddings:
                return jsonify({"error": "Failed to extract embeddings from audio file"}), 500
            
            # 生成唯一的样本ID
            sample_id = f"{int(time.time())}_{hash(audio_file.filename) % 10000}"
            
            # 保存处理后的音频文件
            sample_audio_path = os.path.join(speaker_dir, f"{sample_id}.wav")
            shutil.copy2(proc_temp, sample_audio_path)
            
            sample_info = {
                "id": sample_id,
                "filename": audio_file.filename,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "audio_path": sample_audio_path,
                "embeddings": sample_embeddings
            }
            new_samples.append(sample_info)

            # 计算每个模型的平均嵌入
            avg_embeddings = {}
            for model_name, emb_list in model_embeddings.items():
                if not emb_list:
                    continue
                avg_emb = np.mean(emb_list, axis=0)
                avg_embeddings[model_name] = avg_emb.tolist()
                logger.info(f"  - 模型 [{model_name}] 处理了 {len(emb_list)} 个样本")

            if not avg_embeddings:
                return jsonify({"error": "Failed to extract embeddings from any samples"}), 500

            with db_lock:
                # 如果说话人已存在，则添加新样本并更新平均嵌入
                if enhance_mode and speaker_name in speaker_db:
                    # 添加新样本到现有样本列表
                    if "samples" not in speaker_db[speaker_name]:
                        speaker_db[speaker_name]["samples"] = []
                    speaker_db[speaker_name]["samples"].extend(new_samples)
                    
                    # 重新计算所有样本的平均嵌入
                    all_model_embeddings = {model_name: [] for model_name in sv_pipelines.keys()}
                    
                    # 添加现有样本的嵌入
                    for sample in speaker_db[speaker_name]["samples"]:
                        for model_name, emb in sample["embeddings"].items():
                            all_model_embeddings[model_name].append(np.array(emb))
                    
                    # 重新计算平均嵌入
                    new_avg_embeddings = {}
                    for model_name, emb_list in all_model_embeddings.items():
                        if emb_list:
                            new_avg_embeddings[model_name] = np.mean(emb_list, axis=0).tolist()
                    
                    speaker_db[speaker_name]["avg_embeddings"] = new_avg_embeddings
                    total_samples = len(speaker_db[speaker_name]["samples"])
                    logger.info(f"✅ 成功增强说话人 [{speaker_name}]，当前共 {total_samples} 个样本")
                else:
                    # 新建说话人
                    speaker_db[speaker_name] = {
                        "samples": new_samples,
                        "avg_embeddings": avg_embeddings
                    }
                    logger.info(f"✅ 成功注册新说话人 [{speaker_name}]，共 {len(new_samples)} 个样本")

                # 保存到数据库文件
                with open(Config.SPEAKER_DB_FILE, 'w', encoding='utf-8') as f:
                    json.dump(speaker_db, f, indent=2, ensure_ascii=False)

            return jsonify({
                "message": f"Speaker '{speaker_name}' {'enhanced' if enhance_mode else 'registered'} successfully.",
                "sample_count": len(speaker_db[speaker_name]["samples"])
            })

        except Exception as e:
            logger.error(f"注册声纹失败: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"error": str(e)}), 500
        finally:
            # 清理临时文件
            for tmp in temp_files:
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except:
                    pass

@app.route("/register", methods=["POST"])
def register_speaker():
    # 确保临时目录存在
    os.makedirs(Config.TEMP_DIR, exist_ok=True)
    temp_files = []
    with gpu_lock:
        try:
            if 'speaker_name' not in request.form or not request.form['speaker_name']:
                return jsonify({"error": "Speaker name is required"}), 400
            
            speaker_name = request.form['speaker_name']
            audio_files = request.files.getlist('audio_files')
            
            # 自动检测是否需要增强模式
            enhance_mode = speaker_name in speaker_db

            if not audio_files:
                return jsonify({"error": "At least one audio file is required"}), 400

            action = "增强" if enhance_mode else "注册"
            logger.info(f"📥 开始{action}新声纹: {speaker_name} | 文件数: {len(audio_files)}")
            
            # 创建说话人样本目录
            speaker_dir = os.path.join("speaker_samples", speaker_name)
            if not os.path.exists(speaker_dir):
                os.makedirs(speaker_dir)
            
            # 收集新样本数据
            new_samples = []
            model_embeddings = {model_name: [] for model_name in sv_pipelines.keys()}

            for file in audio_files:
                raw_temp = os.path.join(Config.TEMP_DIR, f"reg_raw_{int(time.time())}_{file.filename}")
                file.save(raw_temp)
                temp_files.append(raw_temp)
                
                proc_temp = os.path.join(Config.TEMP_DIR, f"reg_proc_{int(time.time())}.wav")
                temp_files.append(proc_temp)

                if not preprocess_audio(raw_temp, proc_temp):
                    logger.warning(f"⚠️ 文件 {file.filename} 预处理失败，已跳过。")
                    continue

                # 为每个模型提取嵌入
                sample_embeddings = {}
                for model_name, sv_pipe in sv_pipelines.items():
                    emb = extract_embedding_from_file(sv_pipe, proc_temp)
                    if emb is not None:
                        sample_embeddings[model_name] = emb.tolist()
                        model_embeddings[model_name].append(emb)
                    else:
                        logger.warning(f"⚠️ 从 {file.filename} 提取 {model_name} embedding 失败。")

                # 保存样本信息和音频文件
                if sample_embeddings:  # 只有当至少有一个模型成功提取嵌入时才保存样本
                    # 生成唯一的样本ID
                    sample_id = f"{int(time.time())}_{hash(file.filename) % 10000}"
                    
                    # 保存处理后的音频文件
                    sample_audio_path = os.path.join(speaker_dir, f"{sample_id}.wav")
                    shutil.copy2(proc_temp, sample_audio_path)
                    
                    sample_info = {
                        "id": sample_id,
                        "filename": file.filename,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "audio_path": sample_audio_path,
                        "embeddings": sample_embeddings
                    }
                    new_samples.append(sample_info)

            # 计算每个模型的平均嵌入
            avg_embeddings = {}
            for model_name, emb_list in model_embeddings.items():
                if not emb_list:
                    continue
                avg_emb = np.mean(emb_list, axis=0)
                avg_embeddings[model_name] = avg_emb.tolist()
                logger.info(f"  - 模型 [{model_name}] 处理了 {len(emb_list)} 个样本")

            if not avg_embeddings:
                return jsonify({"error": "Failed to extract embeddings from any samples"}), 500

            with db_lock:
                # 如果说话人已存在，则添加新样本并更新平均嵌入
                if enhance_mode and speaker_name in speaker_db:
                    # 添加新样本到现有样本列表
                    if "samples" not in speaker_db[speaker_name]:
                        speaker_db[speaker_name]["samples"] = []
                    speaker_db[speaker_name]["samples"].extend(new_samples)
                    
                    # 重新计算所有样本的平均嵌入
                    all_model_embeddings = {model_name: [] for model_name in sv_pipelines.keys()}
                    
                    # 添加现有样本的嵌入
                    for sample in speaker_db[speaker_name]["samples"]:
                        for model_name, emb in sample["embeddings"].items():
                            all_model_embeddings[model_name].append(np.array(emb))
                    
                    # 重新计算平均嵌入
                    new_avg_embeddings = {}
                    for model_name, emb_list in all_model_embeddings.items():
                        if emb_list:
                            avg_emb = np.mean(emb_list, axis=0)
                            new_avg_embeddings[model_name] = avg_emb.tolist()
                    
                    speaker_db[speaker_name]["avg_embeddings"] = new_avg_embeddings
                    logger.info(f"🔄 增强了说话人 {speaker_name} 的声纹，新增 {len(new_samples)} 个样本")
                else:
                    # 创建新的说话人条目
                    speaker_db[speaker_name] = {
                        "samples": new_samples,
                        "avg_embeddings": avg_embeddings
                    }
                    logger.info(f"🆕 创建了新说话人 {speaker_name} 的声纹，包含 {len(new_samples)} 个样本")
                    
                # 保存更新后的数据库
                with open(Config.SPEAKER_DB_FILE, 'w', encoding='utf-8') as f:
                    json.dump(speaker_db, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ 声纹{action}成功: {speaker_name}")
            return jsonify({
                "message": f"Speaker '{speaker_name}' {action} successfully.",
                "samples_added": len(new_samples)
            })

        except Exception as e:
            logger.error(f"❌ 注册异常: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"error": "An internal error occurred during registration."} ), 500
        finally:
            # 清理临时文件，但保留语音片段文件供web端预览使用
            for f in temp_files:
                if os.path.exists(f):
                    # 不删除语音片段文件 (seg_*.wav)，这些文件需要保留供web端预览
                    if os.path.basename(f).startswith("seg_"):
                        logger.info(f"  [保留] 语音片段文件供预览使用: {os.path.basename(f)}")
                        continue
                    try: os.remove(f)
                    except: pass


@app.route("/transcribes", methods=["POST"])
def transcribe_audio():
    # 确保临时目录存在
    os.makedirs(Config.TEMP_DIR, exist_ok=True)
    request_start = time.time()
    temp_files = []

    with gpu_lock:
        try:
            if 'audio_file' not in request.files: return jsonify({"error": "No file uploaded"}), 400
            
            file = request.files['audio_file']
            
            # 忽略包含 TEMP 的文件名 (静默跳过,不处理)
            if 'TEMP' in file.filename:
                logger.info(f"⏭️ 忽略临时文件: {file.filename}")
                return jsonify({
                    "message": "Temporary file ignored",
                    "filename": file.filename,
                    "full_text": "",
                    "segments": [],
                    "meta": {"ignored": True}
                }), 200
            
            raw_temp = os.path.join(Config.TEMP_DIR, f"raw_{int(time.time())}_{file.filename}")
            file.save(raw_temp)
            temp_files.append(raw_temp)
            proc_temp = os.path.join(Config.TEMP_DIR, f"proc_{int(time.time())}.wav")
            temp_files.append(proc_temp)
            
            logger.info(f"📥 收到转录任务: {file.filename}")
            
            logger.info("  [生命周期: 1. 音频预处理] 开始 (FFmpeg降噪、重采样、归一化)...")
            if not preprocess_audio(raw_temp, proc_temp):
                return jsonify({"error": "Audio preprocessing failed"}), 500
            logger.info("  [生命周期: 1. 音频预处理] 完成。")

            audio_duration = 0
            try:
                probe = subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', proc_temp])
                audio_duration = float(probe)
            except: pass

            logger.info("  [生命周期: 2. VAD & ASR] 开始 (FunASR语音检测与文字转录)...")
            res = asr_pipeline.generate(input=proc_temp, language="auto", use_itn=True, use_punc=True)
            # logger.info(f"  [VAD 调试] FunASR generate() 原始返回: {json.dumps(res, ensure_ascii=False, indent=2)}")
            full_text = ""
            segments = []
            processed_segments = []

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
                        raw_text = seg.get("text", "")
                        start, end = seg.get("start", 0), seg.get("end", 0)
                        logger.info(f"    [3.{i+1}] 处理分段 {start}ms - {end}ms...")
                        
                        if any(tag in raw_text for tag in INVALID_TAGS): continue

                        # Case-insensitive emotion detection
                        emotion = None  # 未识别到情感时为None,不使用neutral
                        raw_text_lower = raw_text.lower()
                        for tag, emo_code in EMOTION_TAGS.items():
                            if tag.lower() in raw_text_lower:
                                emotion = emo_code
                                if "laughter" in tag.lower():
                                    emotion = "laughter" # Prioritize laughter
                                    break
                        if "<|cry|>" in raw_text_lower:
                            emotion = "sad"

                        # Case-insensitive, universal tag removal
                        clean_text = re.sub(r'<\|.*?\|>', '', raw_text).replace(" ", "").strip()
                        if not clean_text: 
                            logger.info(f"      [3.{i+1}] 分段文本在清洗后为空，已跳过。")
                            continue

                        identity, confidence = None, 0.0
                        recognition_details = []

                        segment_audio_path = None

                        if (end - start) > Config.MIN_SPEAKER_DURATION_MS:
                            # 创建持久化的音频片段目录
                            # 使用原始文件名（不含扩展名和_TEMP后缀）作为子目录
                            original_filename = file.filename.replace('_TEMP', '')  # 移除_TEMP后缀
                            base_filename = os.path.splitext(original_filename)[0]
                            segments_dir = os.path.join(FileMonitorConfig.SOURCE_DIR, "audio_segments", base_filename)
                            os.makedirs(segments_dir, exist_ok=True)
                            
                            # 临时文件用于处理
                            seg_wav_temp = os.path.join(Config.TEMP_DIR, f"seg_{start}_{i}_{int(time.time())}.wav")
                            # 持久化文件
                            seg_filename = f"seg_{i}.wav"
                            seg_wav_persistent = os.path.join(segments_dir, seg_filename)
                            
                            if extract_segment(proc_temp, start, end, seg_wav_temp):
                                temp_files.append(seg_wav_temp)
                                
                                # 复制到持久化目录
                                try:
                                    shutil.copy2(seg_wav_temp, seg_wav_persistent)
                                    # 只有成功复制后才保存路径
                                    segment_audio_path = f"/audio_segments/{base_filename}/{seg_filename}"
                                    logger.debug(f"      [音频片段] 已保存: {seg_wav_persistent}")
                                except Exception as copy_error:
                                    logger.error(f"      [音频片段] 复制失败: {copy_error}")
                                    segment_audio_path = None  # 如果复制失败,不设置路径

                                identity, confidence, recognition_details = identify_speaker_fusion(seg_wav_temp)
                                
                                # 性能优化: 只有识别出的说话人才进行Whisper和SenseVoice处理
                                if identity is not None:
                                    # 情感检测
                                    emotion = detect_emotion_for_segment(seg_wav_temp)
                                    # Whisper对比识别
                                    whisper_text = transcribe_with_whisper(seg_wav_temp)
                                    
                                    # SenseVoice识别和情感检测
                                    sensevoice_text, sensevoice_emotion = transcribe_with_sensevoice(seg_wav_temp)
                                    
                                    # 使用SenseVoice的情感结果(如果检测到)
                                    if sensevoice_emotion is not None:
                                        emotion = sensevoice_emotion
                                    
                                    logger.info(f"      [性能] 已识别说话人 {identity}, 完整处理")
                                else:
                                    # Unknown说话人跳过额外处理
                                    logger.info(f"      [性能] 未识别说话人, 跳过Whisper/SenseVoice处理")
                                    whisper_text = None
                                    sensevoice_text = None
                                    emotion = None  # 未识别到情感时为None,不使用neutral
                                
                                # 保存超过15个字的语句音频
                                # 检测是否为噪音(重复字符过多)
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


                                # 只保存已识别说话人的长句子(跳过Unknown)
                                if Config.SAVE_LONG_SENTENCES and identity is not None and len(clean_text) >= Config.MIN_TEXT_LENGTH_TO_SAVE and not is_noise(clean_text):
                                    try:
                                        os.makedirs(Config.LONG_SENTENCES_DIR, exist_ok=True)
                                        timestamp = int(time.time())
                                        speaker_name = identity  # 已确保identity不为None
                                        saved_filename = f"{timestamp}_{speaker_name}_{len(clean_text)}chars.wav"
                                        saved_path = os.path.join(Config.LONG_SENTENCES_DIR, saved_filename)
                                        shutil.copy2(seg_wav_temp, saved_path)
                                        
                                        # 同时保存文本信息
                                        txt_path = saved_path.replace('.wav', '.txt')
                                        with open(txt_path, 'w', encoding='utf-8') as f:
                                            f.write(f"说话人: {speaker_name}\n")
                                            f.write(f"文本长度: {len(clean_text)} 字\n")
                                            f.write(f"时间: {start}ms - {end}ms\n")
                                            f.write(f"情感: {emotion}\n")
                                            f.write(f"置信度: {confidence:.3f}\n")
                                            f.write(f"\n=== FunASR 识别结果 ===\n{clean_text}\n")
                                            if whisper_text:
                                                f.write(f"\n=== Whisper 识别结果 ===\n{whisper_text}\n")
                                            if sensevoice_text:
                                                f.write(f"\n=== SenseVoice 识别结果 ===\n{sensevoice_text}\n")
                                        
                                        logger.info(f"      [长句保存] 已保存 {len(clean_text)} 字音频: {saved_filename}")
                                    except Exception as e:
                                        logger.warning(f"      [长句保存] 保存失败: {e}")
                        else:
                            logger.info(f"      [3.{i+1}] 分段时长过短({end-start}ms)，跳过声纹识别。")
                            # 即使跳过声纹识别，也要初始化这些变量
                            emotion = None  # 未识别到情感时为None,不使用neutral
                            whisper_text = None


                        if Config.ONLY_REGISTERED_SPEAKERS and identity is None: continue
                        
                        # 计算语速指标
                        duration_seconds = (end - start) / 1000.0
                        word_count = len(clean_text)  # 中文按字符数计算
                        speech_rate = word_count / duration_seconds if duration_seconds > 0 else 0
                        
                        # 计算文本质量
                        from collections import Counter
                        char_counts = Counter(clean_text)
                        most_common_char, most_common_count = char_counts.most_common(1)[0] if clean_text else ('', 0)
                        repeat_ratio = most_common_count / len(clean_text) if clean_text else 0
                        
                        filler_words = ['嗯', '啊', '呃', '额', '哦', '唔']
                        text_no_punct = re.sub(r'[，。、！？,.!?]', '', clean_text)
                        filler_count = sum(text_no_punct.count(w) for w in filler_words) if text_no_punct else 0
                        filler_ratio = filler_count / len(text_no_punct) if text_no_punct else 0
                        noise_score = (repeat_ratio * 0.6 + filler_ratio * 0.4)
                        is_noise_flag = repeat_ratio > 0.4 or filler_ratio > 0.6
                        
                        text_quality = {
                            "is_noise": is_noise_flag,
                            "noise_score": round(noise_score, 3),
                            "repeat_ratio": round(repeat_ratio, 3),
                            "filler_ratio": round(filler_ratio, 3)
                        }
                        
                        # 确定情感来源
                        emotion_source = "funasr"  # 默认
                        original_emotion_tag = None
                        
                        # 检查是否有原始情感标签
                        for tag, emo_code in EMOTION_TAGS.items():
                            if tag.lower() in raw_text.lower():
                                original_emotion_tag = tag
                                break
                        
                        # 如果有 sensevoice_text，说明使用了 SenseVoice
                        if sensevoice_text and emotion:
                            emotion_source = "sensevoice"
                            if not original_emotion_tag:
                                original_emotion_tag = f"<|{emotion}|>"
                        
                        processed_segments.append({
                            "text": clean_text, "start": start, "end": end,
                            "spk": identity or "Unknown", "emotion": emotion,
                            "whisper_text": whisper_text,
                            "sensevoice_text": sensevoice_text,
                            "confidence": float(f"{confidence:.3f}"),
                            "recognition_details": recognition_details,
                            "segment_audio_path": segment_audio_path,
                            
                            # 语速指标
                            "speech_metrics": {
                                "duration_seconds": round(duration_seconds, 2),
                                "word_count": word_count,
                                "speech_rate": round(speech_rate, 2)
                            },
                            
                            # 文本质量评估
                            "text_quality": text_quality,
                            
                            # 情感详细信息
                            "emotion_info": {
                                "emotion": emotion,
                                "source": emotion_source,
                                "original_tag": original_emotion_tag,
                                "detected_by_sensevoice": emotion_source == "sensevoice"
                            }
                        })
                    logger.info("  [生命周期: 3. 逐段声纹识别] 完成。")

                segments = processed_segments
                full_text = "".join([s["text"] for s in segments]) # Reconstruct from clean segments

            process_time = time.time() - request_start
            rtf = process_time / audio_duration if audio_duration > 0 else 0
            # RTF(Real-Time Factor)是实时因子，评估系统处理速度与音频时长的比率
            # RTF < 1表示可以实时处理，RTF越低系统性能越好
            logger.info(f"✅ 完成! 音频:{audio_duration:.1f}s | 耗时:{process_time:.2f}s | RTF:{rtf:.3f} (RTF < 1表示可实时处理，值越低性能越好)")

            logger.info("  [生命周期: 4. 组装响应] 开始...")
            response_data = {
                "full_text": full_text,
                "segments": segments,
                "meta": {
                    "process_time": process_time,
                    "audio_duration": audio_duration,
                    "rtf": rtf,
                    "rtf_description": "Real-Time Factor(实时因子)，处理时间/音频时长，RTF < 1表示可实时处理，值越低性能越好"
                }
            }
            logger.info(f"📤  [生命周期: 4. 组装响应] 完成, 返回 /transcribe 结果: {json.dumps(response_data, ensure_ascii=False, indent=2)}")
            
            # =================【 数据库保存和 LLM 处理 】=================
            if processed_segments:
                try:
                    # 生成智能摘要
                    summary = generate_conversation_summary(processed_segments, audio_duration)
                    
                    # 解析录音时间
                    recording_time = parse_recording_time(file.filename)
                    
                    # 保存到数据库
                    success = save_to_db(file.filename, full_text, processed_segments, recording_time, summary)
                    
                    if success:
                        logger.info(f"✅ 数据库保存成功 (recording_time: {recording_time})")
                        if summary:
                            logger.info(f"  智能摘要: {summary['speaker_count']}位说话人, {summary['total_segments']}个分段")
                        
                        # 添加到 LLM 批量处理队列
                        if LLMConfig.USE_GEMINI_LLM:
                            has_identified_speakers = any(seg.get('spk') != 'Unknown' for seg in processed_segments)
                            if (len(full_text) >= LLMConfig.LLM_MIN_TEXT_LENGTH and 
                                len(processed_segments) >= LLMConfig.LLM_MIN_SEGMENTS and 
                                has_identified_speakers):
                                add_to_llm_queue(file.filename, full_text, processed_segments)
                    else:
                        logger.error(f"❌ 数据库保存失败")
                except Exception as e:
                    logger.error(f"❌ 数据库保存异常: {e}")
                    logger.error(traceback.format_exc())
            # =========================================================

            return jsonify(response_data)

        except Exception as e:
            logger.error(f"❌ 处理异常: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"error": str(e)}), 500
        finally:
            for f in temp_files:
                if os.path.exists(f):
                    try: os.remove(f)
                    except: pass

@app.route("/speaker/<speaker_name>/sample/<sample_id>/audio")
def get_sample_audio(speaker_name, sample_id):
    """获取指定说话人样本的音频文件"""
    try:
        # 重新加载声纹数据库以确保数据是最新的
        load_speaker_db()
        if speaker_name not in speaker_db:
            return jsonify({"error": f"Speaker '{speaker_name}' not found."}), 404
            
        speaker_data = speaker_db[speaker_name]
        if "samples" not in speaker_data:
            return jsonify({"error": f"No samples found for speaker '{speaker_name}'."}), 404
            
        # 查找指定样本
        for sample in speaker_data["samples"]:
            if sample["id"] == sample_id:
                if "audio_path" in sample and os.path.exists(sample["audio_path"]):
                    return send_file(sample["audio_path"], as_attachment=True, download_name=sample["filename"])
                else:
                    return jsonify({"error": f"Audio file for sample '{sample_id}' not found."}), 404
        
        return jsonify({"error": f"Sample '{sample_id}' not found for speaker '{speaker_name}'."}), 404
    except Exception as e:
        logger.error(f"获取样本音频文件失败: {str(e)}")
        return jsonify({"error": "Failed to retrieve sample audio"}), 500


@app.route('/audio_segments/<path:filename>')
def serve_audio_segment(filename):
    """提供音频片段静态文件服务"""
    try:
        audio_segments_dir = os.path.join(FileMonitorConfig.SOURCE_DIR, 'audio_segments')
        return send_from_directory(audio_segments_dir, filename)
    except Exception as e:
        logger.error(f"获取音频片段失败: {str(e)}")
        return jsonify({"error": "Audio segment not found"}), 404


@app.route("/logs/stream")
def stream_logs():
    """SSE endpoint for real-time log streaming"""
    def generate_logs():
        # 创建一个新的客户端连接
        client = type('Client', (), {'write': lambda self, msg: print(msg, end='', flush=True) or msg})
        
        # 添加客户端到SSE处理器
        sse_handler.add_client(client)
        try:
            # 保持连接打开
            while True:
                time.sleep(1)
        except GeneratorExit:
            # 客户端断开连接时移除客户端
            sse_handler.remove_client(client)
    
    return Response(generate_logs(), mimetype='text/event-stream')

# =================== 文件监控 ===================
def monitor_files():
    """监控源目录中的新音频文件并自动转录"""
    logger.info("📂 文件监控线程已启动")
    logger.info(f"   监控目录: {FileMonitorConfig.SOURCE_DIR}")
    logger.info(f"   扫描间隔: {FileMonitorConfig.SCAN_INTERVAL}秒")
    logger.info(f"   支持格式: {', '.join(FileMonitorConfig.SUPPORTED_FORMATS)}")
    
    # 确保必要的目录存在
    os.makedirs(FileMonitorConfig.SOURCE_DIR, exist_ok=True)
    processed_dir = os.path.join(FileMonitorConfig.SOURCE_DIR, FileMonitorConfig.PROCESSED_DIR)
    os.makedirs(processed_dir, exist_ok=True)
    failed_dir = os.path.join(FileMonitorConfig.SOURCE_DIR, FileMonitorConfig.FAILED_DIR)
    os.makedirs(failed_dir, exist_ok=True)
    
    processed_files = set()  # 记录已处理的文件，避免重复处理
    
    while True:
        try:
            # 扫描源目录
            logger.info(f"🔍 正在扫描: {FileMonitorConfig.SOURCE_DIR}")
            if not os.path.exists(FileMonitorConfig.SOURCE_DIR):
                logger.warning(f"⚠️ 源目录不存在: {FileMonitorConfig.SOURCE_DIR}")
                time.sleep(FileMonitorConfig.SCAN_INTERVAL)
                continue
            
            files = []
            for filename in os.listdir(FileMonitorConfig.SOURCE_DIR):
                filepath = os.path.join(FileMonitorConfig.SOURCE_DIR, filename)
                
                # 只处理文件，跳过目录
                if not os.path.isfile(filepath):
                    continue
                
                # 检查文件扩展名
                ext = os.path.splitext(filename)[1].lower()
                if ext not in FileMonitorConfig.SUPPORTED_FORMATS:
                    continue
                
                # 跳过包含TEMP的文件
                if 'TEMP' in filename or '_TEMP' in filename:
                    continue
                
                # 跳过已处理的文件
                if filename in processed_files:
                    continue
                
                files.append((filename, filepath))
            
            # 按文件名排序，确保时间戳早的文件先处理
            # 文件名格式如: TermuxAudioRecording_2025-11-18_00-34-27.m4a
            # 或: recording-20251115-131250.m4a
            files.sort(key=lambda x: x[0])  # 按文件名字母顺序排序，时间戳早的在前
            
            # 处理找到的文件
            if files:
                logger.info(f"🔍 发现 {len(files)} 个待处理文件")
                logger.info(f"   处理顺序: {files[0][0]} → ... → {files[-1][0]}")
                
                for filename, filepath in files:
                    try:
                        logger.info(f"📤 开始处理: {filename}")
                        
                        # 调用本地转录API
                        with open(filepath, 'rb') as f:
                            files_data = {'audio_file': (filename, f, 'audio/mpeg')}
                            response = requests.post(
                                'http://localhost:5008/transcribes',
                                files=files_data,
                                timeout=7200  # 2小时超时
                            )
                        
                        if response.status_code == 200:
                            result = response.json()
                            logger.info(f"✅ 转录完成: {filename}")
                            logger.info(f"   文本长度: {len(result.get('full_text', ''))} 字")
                            logger.info(f"   分段数: {len(result.get('segments', []))}")
                            
                            # 移动到已处理目录
                            processed_path = os.path.join(processed_dir, filename)
                            try:
                                shutil.move(filepath, processed_path)
                                logger.info(f"📦 已移动到: {FileMonitorConfig.PROCESSED_DIR}/{filename}")
                            except Exception as move_error:
                                logger.warning(f"⚠️ 移动文件失败: {move_error}")
                            
                            # 标记为已处理
                            processed_files.add(filename)
                            
                        else:
                            logger.error(f"❌ 转录失败: {filename} (HTTP {response.status_code})")
                            logger.error(f"   响应: {response.text[:200]}")
                            
                            # 移动到失败目录
                            failed_path = os.path.join(failed_dir, filename)
                            try:
                                shutil.move(filepath, failed_path)
                                logger.info(f"🚫 已移动到失败目录: {FileMonitorConfig.FAILED_DIR}/{filename}")
                            except Exception as move_error:
                                logger.warning(f"⚠️ 移动到失败目录失败: {move_error}")
                            
                    except requests.exceptions.Timeout:
                        logger.error(f"⏱️ 转录超时: {filename}")
                        # 超时也认为是失败，移动到失败目录
                        failed_path = os.path.join(failed_dir, filename)
                        try:
                            shutil.move(filepath, failed_path)
                            logger.info(f"🚫 超时已移动到失败目录: {FileMonitorConfig.FAILED_DIR}/{filename}")
                        except Exception as move_error:
                            logger.warning(f"⚠️ 移动到失败目录失败: {move_error}")
                            
                    except Exception as e:
                        logger.error(f"❌ 处理文件失败: {filename}")
                        logger.error(f"   错误: {str(e)}")
                        logger.error(traceback.format_exc())
                        
                        # 其他异常也移动到失败目录
                        failed_path = os.path.join(failed_dir, filename)
                        try:
                            if os.path.exists(filepath):
                                shutil.move(filepath, failed_path)
                                logger.info(f"🚫 发生异常已移动到失败目录: {FileMonitorConfig.FAILED_DIR}/{filename}")
                        except Exception as move_error:
                            logger.warning(f"⚠️ 移动到失败目录失败: {move_error}")
            
            # 等待下一次扫描
            time.sleep(FileMonitorConfig.SCAN_INTERVAL)
            
        except Exception as e:
            logger.error(f"❌ 文件监控异常: {str(e)}")
            logger.error(traceback.format_exc())
            time.sleep(FileMonitorConfig.SCAN_INTERVAL)

# =================== 启动 ===================
def parse_args():
    parser = argparse.ArgumentParser(description='ASR Service')
    parser.add_argument('--source-path', type=str, help='Source directory for audio files')
    parser.add_argument('--port', type=int, help='Port to run the server on')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Update config from args
    if args.source_path:
        FileMonitorConfig.SOURCE_DIR = args.source_path
        print(f"配置更新: 源目录 -> {args.source_path}")
        
    if args.port:
        Config.PORT = args.port
        print(f"配置更新: 端口 -> {args.port}")

    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        logger.critical("❌ 系统未安装 FFmpeg！")
        sys.exit(1)

    # 初始化数据库连接池和表结构
    print("初始化数据库连接池...")
    if not init_pool():
        logger.critical("❌ 数据库连接池初始化失败！")
        sys.exit(1)
    
    print("初始化数据库表结构...")
    if not init_db():
        logger.warning("⚠️ 数据库表结构初始化失败，但服务将继续运行")

    load_models()
    
    # 启动临时文件清理定时任务
    cleanup_temp_dir()
    logger.info("临时文件清理定时任务已启动")
    
    # 启动文件监控线程
    monitor_thread = threading.Thread(target=monitor_files, daemon=True)
    monitor_thread.start()
    logger.info("文件监控线程已启动")
    
    print("🎉 服务启动成功！")
    print("📌 声纹注册页面: http://127.0.0.1:5008/register_page")
    print("📌 语音转录API: http://127.0.0.1:5008/transcribes (本地监控专用)")
    print("📌 外部调用API: http://127.0.0.1:5008/transcribe (保留给NAS使用)")
    print(f"📂 文件监控目录: {FileMonitorConfig.SOURCE_DIR}")
    print(f"⏱️  扫描间隔: {FileMonitorConfig.SCAN_INTERVAL}秒")
    print("🔧 API使用方法: POST请求，参数名 'audio_file'，上传音频文件")
    print("🔍 示例命令: curl -X POST -F \"audio_file=@your_audio.wav\" http://127.0.0.1:5008/transcribes")
    app.run(host=Config.HOST, port=Config.PORT, debug=False, threaded=True)