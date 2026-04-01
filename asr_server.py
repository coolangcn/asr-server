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
from datetime import datetime, timezone, timedelta

# 东八区时区 (UTC+8) - 上海时间
UTC_PLUS_8 = timezone(timedelta(hours=8))
from email_utils import send_cry_alert_email   # 增加邮件通知支持
import audio_processor
from dotenv import load_dotenv

load_dotenv()

# =================【 配置 】=================
import platform

def get_device():
    """自动检测可用设备: CUDA (NVIDIA GPU) → MPS (Apple Silicon) → CPU"""
    if torch.cuda.is_available():
        return "cuda:0"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def get_modelscope_device():
    """ModelScope pipeline 设备 (不支持 MPS，仅支持 cuda/cpu)"""
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def get_whisper_model():
    """根据设备选择合适的 Whisper 模型大小
    
    - CUDA: large-v3 (~10GB VRAM)
    - MPS/CPU: medium (~5GB RAM) 避免内存溢出
    """
    if torch.cuda.is_available():
        return "large-v3"
    else:
        return "medium"

class Config:
    DEVICE = get_device()
    MODELSCOPE_DEVICE = get_modelscope_device()
    WHISPER_MODEL = get_whisper_model()
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
            "threshold": 0.60,  # 提高阈值以减少误识别
            "gap": 0.10         # 提高置信度间隔要求以增强区分度
        },
        "rdino_ecapa": {
            "id": "iic/speech_rdino_ecapa_tdnn_sv_zh-cn_cnceleb_16k",
            "rev": "v1.0.0",
            "threshold": 0.60,  # 提高阈值以减少误识别
            "gap": 0.10         # 提高置信度间隔要求以增强区分度
        },
        "camplusplus": {
            "id": "iic/speech_campplus_sv_zh-cn_16k-common",
            "rev": "v1.0.0",
            "threshold": 0.60,  # 提高阈值以减少误识别
            "gap": 0.10         # 提高置信度间隔要求以增强区分度
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

# 文件监控配置 (已迁移至 audio_processor)
FileMonitorConfig = audio_processor.FileMonitorConfig

# LLM 配置
class LLMConfig:
    USE_GEMINI_LLM = True
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyABpNzAb90t6EpIsJtbF1UbekDTGlLaKTE")
    GEMINI_API_BASE_URL = os.getenv("GEMINI_API_BASE_URL", "https://generativelanguage.googleapis.com")
    GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")
    
    # 批量处理配置
    LLM_BATCH_MODE = True
    LLM_BATCH_SIZE = 20
    LLM_BATCH_TIMEOUT = 600  # 10分钟
    
    # 过滤条件
    LLM_MIN_TEXT_LENGTH = 50
    LLM_MIN_SEGMENTS = 3
    LLM_CACHE_SIZE = 100
    LLM_REQUEST_TIMEOUT = 120
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

# 创建通用控制台和SSE处理器
# 创建通用控制台和SSE处理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

sse_handler = SSEHandler()
sse_handler.setFormatter(log_formatter)
sse_handler.setLevel(logging.INFO)

# 确保日志目录存在
os.makedirs("log", exist_ok=True)

def create_sub_logger(name, filename, level=logging.INFO):
    l = logging.getLogger(name)
    l.setLevel(level)
    l.handlers = [] # 清除可能存在的旧处理器
    handler = TimedRotatingFileHandler(
        filename, when='M', interval=10, backupCount=144, encoding='utf-8'
    )
    handler.setFormatter(log_formatter)
    l.addHandler(handler)
    l.addHandler(console_handler)
    l.addHandler(sse_handler)
    l.propagate = False 
    return l

# 定义三个物理隔离的日志记录器
logger_a = create_sub_logger("track_a", "log/asr-a.log")
logger_b = create_sub_logger("track_b", "log/asr-b.log")
logger_sys = create_sub_logger("system", "log/asr-web.log")

# 轨道 logger 初始化完成，彻底移除全局 logger 别名
# logger = logger_sys 

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

# =================【 历史分析锁 】=================
_history_reprocess_lock = threading.Lock()
_history_reprocess_running = False
_history_reprocess_proc = None # 用于存储当前运行的进程对象
_track_b_paused = False # 是否暂停 B 轨实时扫描
_track_b_running = False # B 轨是否正在运行
_monitor_thread = None # B 轨实时监听线程

# =================【 轨道A: 独立哭声检测配置 】=================
# 与语音识别参数完全隔离，仅用于原始音频的哭声声纹匹配
class CryDetectionConfig:
    """哭声检测专用参数 - 与语音识别 (Config.SV_MODELS) 完全隔离"""
    ENABLED = True
    MIN_DURATION_SEC = 3            # 最短有效哭声时长 (秒)
    
    # 声纹阈值 (远低于语音识别的 0.60)
    # 基于 3月20日已知哭声数据: Baby 全局得分约 0.52
    VOICEPRINT_THRESHOLD = 0.65     # 极致严格门槛 (用户倾向严格)
    VOICEPRINT_GAP = 0.15           # 严格置信度间隔 (原为0.02/0.10)
    
    # 二票放行: 必须 2 个模型命中才通过 (三选二)
    MIN_VOTES = 2
    
    # 目标声纹名 (大小写不敏感)
    TARGET_SPEAKERS = ["baby", "宝宝"]
    
    # 冷却机制
    COOLDOWN_SEC = 600              # 10分钟冷却

_last_cry_trigger_time = 0.0
_cry_cooldown_lock = threading.Lock()
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
            device=Config.MODELSCOPE_DEVICE
        )
    print(f"✅ 服务就绪 | ASR: SenseVoice | SV: {list(sv_pipelines.keys())}\n")

    # 4. 加载 Whisper 模型 (可选)
    if Config.ENABLE_WHISPER_COMPARISON:
        print(f"🎤 加载 Whisper {Config.WHISPER_MODEL} 模型...")

        try:
            whisper_model = whisper.load_model(Config.WHISPER_MODEL, device=Config.DEVICE.split(':')[0])
            print(f"✅ Whisper {Config.WHISPER_MODEL} 模型加载完成")

        except Exception as e:
            # 针对 macOS MPS 环境下的特殊报错进行友好提示
            error_msg = str(e)
            if "aten::_sparse_coo_tensor_with_dims_and_tensors" in error_msg:
                logger_sys.warning("⚠️ Whisper 在 macOS MPS 环境下检测到张量不兼容，已自动回退到 SenseVoice 引擎进行高精度识别。")
            else:
                logger_sys.warning(f"⚠️ Whisper模型加载受限: {error_msg}，将使用主引擎进行音频转录。")
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
            logger_sys.warning(f"⚠️ SenseVoice模型加载失败: {e}，将禁用SenseVoice功能")
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
                logger_sys.info(f"  [LLM] 使用缓存响应")
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
        logger_sys.error(f"  [LLM] API 调用失败: {e}")
        return None

def call_gemini_audio_api(audio_paths, prompt):
    """调用 Gemini API 并上传一个或多个音频(支持上下文合并)"""
    import base64
    if not LLMConfig.USE_GEMINI_LLM:
        logger_a.warning("  [BabyCry LLM] LLM 未启用，跳过")
        return None
        
    try:
        logger_a.info(f"  [BabyCry LLM] 开始调用，音频文件数: {len(audio_paths) if isinstance(audio_paths, list) else 1}")
        
        url = f"{LLMConfig.GEMINI_API_BASE_URL}/v1beta/models/{LLMConfig.GEMINI_MODEL_NAME}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": LLMConfig.GEMINI_API_KEY
        }
        
        parts_list = [{"text": prompt}]
        
        if isinstance(audio_paths, str):
            audio_paths = [audio_paths]
            
        total_size = 0
        MAX_INLINE_SIZE = 15 * 1024 * 1024 # 防止超出 Gemini Inline Base64 限制 (20MB)
        
        logger_a.info(f"  [BabyCry LLM] 准备上传音频文件...")
        for i, path in enumerate(audio_paths):
            if not os.path.exists(path):
                logger_a.warning(f"  [BabyCry LLM] 文件不存在，跳过: {path}")
                continue
            file_size = os.path.getsize(path)
            if total_size + file_size > MAX_INLINE_SIZE:
                logger_sys.warning(f"  [BabyCry LLM] 上下文音频片段过大，截断。已加载: {total_size/1024/1024:.1f}MB")
                break
                
            logger_a.info(f"  [BabyCry LLM] 加载文件 {i+1}/{len(audio_paths)}: {os.path.basename(path)} ({file_size/1024:.1f}KB)")
            with open(path, "rb") as f:
                audio_data = base64.b64encode(f.read()).decode("utf-8")
                
            ext = os.path.splitext(path)[1].lower()
            mime_type = "audio/wav" if ext == ".wav" else "audio/m4a"
            parts_list.append({
                "inline_data": {
                    "mime_type": mime_type,
                    "data": audio_data
                }
            })
            total_size += file_size

        logger_a.info(f"  [BabyCry LLM] 共加载 {len(parts_list)-1} 个音频文件，总大小: {total_size/1024/1024:.2f}MB")

        if len(parts_list) == 1:
            logger_a.error(f"  [BabyCry LLM] 没有任何有效音频")
            return None
            
        data = {
            "contents": [{
                "parts": parts_list
            }],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 32768,  # 32K token，足以容纳思考+输出
                "responseModalities": ["TEXT"]
            }
        }
        
        logger_a.info(f"  [BabyCry LLM] 发送请求到 Gemini API...")
        response = requests.post(url, headers=headers, json=data, timeout=LLMConfig.LLM_REQUEST_TIMEOUT + 40)
        
        try:
            result = response.json()
        except Exception as e:
            logger_sys.error(f"  [BabyCry LLM] 返回非JSON数据: {e}, 原始响应: {response.text}")
            return None

        if not response.ok:
            logger_sys.error(f"  [BabyCry LLM] API HTTP 错误 {response.status_code}: {result}")
            return None
            
        logger_a.info(f"  [BabyCry LLM] 收到 HTTP 响应，状态码: {response.status_code}")
        
        if 'candidates' in result and len(result['candidates']) > 0:
            content = result['candidates'][0].get('content', {})
            logger_a.info(f"  [BabyCry LLM] 响应内容结构: {result}")
            
            # 即使 finishReason 是 MAX_TOKENS，也尝试提取已有内容
            if 'parts' in content and len(content['parts']) > 0:
                response_text = content['parts'][0].get('text', '')
                logger_a.info(f"  [BabyCry LLM] 收到响应，长度: {len(response_text)} 字符")
                logger_a.info(f"  [BabyCry LLM] 响应内容: {response_text[:300]}...")
                if response_text:
                    return response_text
            
            # 如果没有内容但有 MAX_TOKENS，记录警告
            if result['candidates'][0].get('finishReason') == 'MAX_TOKENS':
                logger_a.warning(f"  [BabyCry LLM] 生成 Token 超限，但尝试返回已生成内容")
            
            logger_a.error(f"  [BabyCry LLM] 安全拦截或空回复: {result}")
            return None
        else:
            logger_a.error(f"  [BabyCry LLM] API 返回异常结构: {result}")
            return None
    except Exception as e:
        logger_a.error(f"  [BabyCry LLM] 发送请求异常：{e}")
        import traceback
        logger_a.error(f"  [BabyCry LLM] 异常堆栈: {traceback.format_exc()}")
        return None

def call_gemini_image_api(prompt):
    """调用 Gemini API 生成插图（文生图）"""
    if not LLMConfig.USE_GEMINI_LLM:
        logger_a.warning("🎨 [插图生成] LLM 未启用，跳过")
        return None
        
    try:
        logger_a.info(f"🎨 [插图生成] 开始调用插图生成 API")
        logger_a.info(f"🎨 [插图生成] Prompt 长度: {len(prompt)} 字符")
        logger_a.info(f"🎨 [插图生成] Prompt 开头: {prompt[:100]}...")
        
        # 使用 Gemini 2.5 Flash Image 支持图像生成
        url = f"{LLMConfig.GEMINI_API_BASE_URL}/v1beta/models/gemini-2.5-flash-image:generateContent"
        logger_a.info(f"🎨 [插图生成] API URL: {url}")
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": LLMConfig.GEMINI_API_KEY
        }
        
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 4096,
                "responseModalities": ["TEXT", "IMAGE"]  # 请求图像响应
            }
        }
        
        logger_a.info(f"🎨 [插图生成] 发送请求到 Gemini API...")
        logger_a.info(f"🎨 [插图生成] 超时设置: 60秒")
        
        response = requests.post(url, headers=headers, json=data, timeout=60)
        
        logger_a.info(f"🎨 [插图生成] HTTP 状态码: {response.status_code}")
        
        try:
            result = response.json()
        except Exception as e:
            logger_a.error(f"🎨 [插图生成] 返回非 JSON 数据: {e}")
            logger_a.error(f"🎨 [插图生成] 原始响应: {response.text}")
            return None
        
        if not response.ok:
            logger_a.error(f"🎨 [插图生成] API HTTP 错误 {response.status_code}")
            logger_a.error(f"🎨 [插图生成] 错误详情: {json.dumps(result, indent=2, ensure_ascii=False)}")
            return None
        
        logger_a.info(f"🎨 [插图生成] 收到响应，正在解析...")
        logger_a.info(f"🎨 [插图生成] 完整响应结构: {json.dumps(result, indent=2, ensure_ascii=False)}")
        
        # 解析图像数据
        if 'candidates' in result and len(result['candidates']) > 0:
            logger_a.info(f"🎨 [插图生成] 找到 {len(result['candidates'])} 个 candidates")
            content = result['candidates'][0].get('content', {})
            logger_a.info(f"🎨 [插图生成] Content 结构: {content}")
            
            if 'parts' in content:
                logger_a.info(f"🎨 [插图生成] 找到 {len(content['parts'])} 个 parts")
                for i, part in enumerate(content['parts']):
                    logger_a.info(f"🎨 [插图生成] Part {i} 类型: {list(part.keys())}")
                    if 'inlineData' in part:
                        logger_a.info(f"🎨 [插图生成] 找到图像数据!")
                        image_data = part['inlineData']
                        # 返回 base64 编码的图像 URL（数据 URL 格式）
                        mime_type = image_data.get('mimeType', 'image/png')
                        data = image_data.get('data', '')
                        logger_a.info(f"🎨 [插图生成] 图像 MIME 类型: {mime_type}")
                        logger_a.info(f"🎨 [插图生成] 图像数据长度: {len(data)} 字符")
                        if data:
                            logger_a.info(f"🎨 [插图生成] 图像生成成功!")
                            return f"data:{mime_type};base64,{data}"
                        else:
                            logger_a.warning(f"🎨 [插图生成] 图像数据为空")
            else:
                logger_a.warning(f"🎨 [插图生成] 响应中没有 parts")
        else:
            logger_a.warning(f"🎨 [插图生成] 响应中没有 candidates")
        
        logger_a.warning(f"🎨 [插图生成] 未返回图像数据")
        return None
        
    except Exception as e:
        logger_a.error(f"🎨 [插图生成] 发送请求异常: {e}")
        import traceback
        logger_a.error(f"🎨 [插图生成] 异常堆栈: {traceback.format_exc()}")
        return None

def process_baby_cry_async(filename, audio_path, start_time, end_time, placeholder_id=None):
    """异步处理宝宝哭声分析，支持加载前后5分钟录音作为上下文"""
    import time, re, json
    time.sleep(1) # 等待文件落盘
    if not os.path.exists(audio_path):
        return None, None
        
    logger_a.info(f"👶 [BabyCry] 开始收集上下文音频并发送分析... ({start_time}ms - {end_time}ms)")
    
    # 搜集前后 5 分钟 (300秒) 的同目录相关录音
    audio_paths_to_send = []
    context_files_before = []
    context_files_after = []
    
    from db_manager import parse_recording_time
    record_dt = parse_recording_time(filename)
    
    if record_dt:
        date_dir = os.path.join(FileMonitorConfig.SOURCE_DIR, FileMonitorConfig.PROCESSED_DIR, record_dt.strftime("%Y-%m-%d"))
        if os.path.exists(date_dir):
            for f in os.listdir(date_dir):
                if f.endswith(tuple(FileMonitorConfig.SUPPORTED_FORMATS)):
                    f_dt = parse_recording_time(f)
    
    context_len = len(audio_paths_to_send) - 1
    logger_a.info(f"👶 [BabyCry] 收集完毕，共附带 {context_len} 个相邻时段记录作为多模态上下文...")
    
    prompt = "以下是多段连续的录音（时间顺序排列），其中包含了两岁半宝宝的哭泣声（位于中间的某段）。请结合完整的上下文音频（前后高达5分钟的情境），综合推理宝宝在这段时间哭泣的真正原因（如困倦Sleepy、饥饿Hungry、情绪发泄Frustration、疼痛Pain、要求未被满足等），并给出针对此时情境的安抚建议。请严格按如下JSON格式返回：{\"category\": \"核心原因简短分类(如：困倦/饥饿/疼痛/情绪等)\", \"reason\": \"结合上下文的深度分析原因\", \"advice\": \"针对此时情境的安抚建议\"}"
    response_text = call_gemini_audio_api(audio_paths_to_send, prompt)
    
    if not response_text:
        return None, None
        
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            category = result.get("category", "未知")
            reason = result.get("reason", "未知")
            advice = result.get("advice", "无")
            from db_manager import save_cry_analysis, update_cry_analysis
            
            # 转换 absolute path 为相对于 SOURCE_DIR 的路径，以便前端使用
            rel_audio_path = audio_path
            if audio_path.startswith(FileMonitorConfig.SOURCE_DIR):
                rel_audio_path = "/" + os.path.relpath(audio_path, FileMonitorConfig.SOURCE_DIR)
            
            if placeholder_id:
                update_cry_analysis(placeholder_id, reason, advice, 
                                  reason_category=category, event_files=audio_paths_to_send)
            else:
                save_cry_analysis(filename, start_time/1000.0, end_time/1000.0, reason, advice, 
                                  reason_category=category, event_files=audio_paths_to_send, 
                                  audio_path=rel_audio_path)
            logger_a.info(f"👶 [宝宝哭声深度分析] 分类: {category}, 原因: {reason[:50]}..., 路径: {rel_audio_path}")
            return reason, advice
    except Exception as e:
        logger_a.error(f"  [BabyCry 解析错误] {e}")
    return None, None

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
        logger_b.warning(f"  [LLM] 主题提取失败: {e}")
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
        
        logger_b.info(f"  [LLM队列] 已添加，当前队列: {queue_size}/{LLMConfig.LLM_BATCH_SIZE}")
        
        # 触发批量处理
        if queue_size >= LLMConfig.LLM_BATCH_SIZE or time_since_last >= LLMConfig.LLM_BATCH_TIMEOUT:
            logger_b.info(f"  [LLM队列] 触发批量处理 (队列={queue_size}, 超时={time_since_last:.0f}s)")
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
    
    logger_b.info(f"  [LLM批处理] 开始处理 {len(batch)} 条记录")
    
    for item in batch:
        try:
            topics = extract_conversation_topics(item['full_text'], item['segments'])
            if topics:
                update_topics(item['filename'], topics)
                logger_b.info(f"  [LLM] {item['filename']}: 主题={topics.get('topics', [])}")
        except Exception as e:
            logger_b.error(f"  [LLM] 处理失败 {item['filename']}: {e}")
    
    logger_b.info(f"  [LLM批处理] 完成")

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
                        logger_sys.debug(f"清理旧临时文件: {filename}")
                except Exception as e:
                    logger_sys.warning(f"清理文件失败 {filename}: {e}")
        
        if cleaned_count > 0:
            logger_sys.info(f"临时文件清理完成，删除了 {cleaned_count} 个文件")
        
        # 每小时执行一次
        threading.Timer(3600, cleanup_temp_dir).start()
    except Exception as e:
        logger_sys.error(f"临时文件清理失败: {e}")
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
                        logger_sys.info(f"🔄 转换旧数据结构 for speaker: {name}")
                        converted_db[name] = {
                            "samples": [],  # 旧数据结构没有样本信息
                            "avg_embeddings": data  # 旧数据结构直接是嵌入字典
                        }
                
                speaker_db = converted_db
                logger_sys.info(f"📚 声纹库已挂载: {len(speaker_db)} 人")
            except Exception as e:
                logger_sys.error(f"声纹库损坏: {e}")
                speaker_db = {}
        else:
            logger_sys.warning(f"⚠️ 未找到 {Config.SPEAKER_DB_FILE}，将创建新的数据库。")
            speaker_db = {}

# =================== 音频预处理 ===================
def preprocess_audio(input_path, output_path):
    # 如果启用了高级降噪，先进行降噪处理
    if Config.DENOISE_AUDIO:
        denoised_path = input_path + ".denoised.wav"
        if advanced_denoise(input_path, denoised_path):
            input_path = denoised_path
        else:
            logger_b.warning("高级降噪处理失败，使用原始音频")
    
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
        logger_b.error(f"FFmpeg 预处理失败: {e}")
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
        logger_sys.error(f"高级降噪处理失败: {e}")
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
        logger_b.info(f"      [Whisper对比] {whisper_text}")
        return whisper_text
    except Exception as e:
        logger_b.warning(f"      [Whisper对比] 识别失败: {e}")
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
        
        logger_b.info(f"      [SenseVoice] {clean_text} (情感: {emotion})")
        return clean_text, emotion
        
    except Exception as e:
        logger_b.warning(f"      [SenseVoice] 识别失败: {e}")
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
        logger_b.info(f"      [SenseVoice情感] 原始输出: {raw_text}")
        
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
                logger_b.info(f"      [SenseVoice情感] 检测到情感: {emotion}")
                break
        
        return emotion
    except Exception as e:
        logger_sys.warning(f"      [SenseVoice情感] 检测失败: {e}")
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
        logger_sys.error(f"❌ extract_embedding 失败: {e}")
        return None

# =================== 多模型交叉验证 ===================
def identify_speaker_fusion(segment_path):
    """【轨道B: 语音识别专用】声纹融合识别 - 标准参数，不受哭声检测影响"""
    if not speaker_db: 
        logger_sys.info("🤷‍♂️ 声纹数据库为空，无法进行识别")
        return None, 0.0, []

    model_votes = {}
    model_scores = {}

    logger_sys.info(f"🎯 开始声纹识别: 音频段路径={segment_path}")
    logger_sys.info(f"📋 声纹数据库包含 {len(speaker_db)} 个说话人")

    for model_name, sv_pipe in sv_pipelines.items():
        emb_a = extract_embedding_from_file(sv_pipe, segment_path)
        if emb_a is None:
            logger_sys.error(f"❌ 模型 {model_name} 特征提取失败")
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
            
            # 【轨道B】语音识别模式：不跳过 Baby，所有说话人都参与比对

            emb_b = np.array(speaker_data["avg_embeddings"][model_name]).flatten()
            score = 1 - cosine(emb_a.flatten(), emb_b)
            scores.append((name, score))
            logger_sys.debug(f"  {model_name}: {name}={score:.3f}")

        if not scores:
            logger_sys.warning(f"⚠️ 模型 {model_name} 未找到匹配的说话人数据")
            model_votes[model_name] = "NoDB"
            continue

        scores.sort(key=lambda x: x[1], reverse=True)
        top1_name, top1_score = scores[0]
        top2_name, top2_score = scores[1] if len(scores) > 1 else (None, 0.0)
        score_gap = top1_score - top2_score
        
        logger_sys.debug(f"  {model_name}: {top1_name}={top1_score:.3f} (gap={score_gap:.3f})")

        # 【轨道B】使用标准阈值，不做任何哭声补偿
        if top1_score >= threshold and score_gap >= gap:
            model_votes[model_name] = top1_name
            model_scores[model_name] = top1_score
        else:
            model_votes[model_name] = "Unknown"
            model_scores[model_name] = top1_score

    # DEBUG: 投票结果
    logger_b.debug(f"  投票: {model_votes}")
    
    # 2/3投票逻辑
    votes = [v for v in model_votes.values() if v not in ["Unknown", "Failed", "NoDB"]]
    if not votes:
        # 识别失败（由上层记录）
        logger_b.debug("  未识别: 所有模型均未通过")
        return None, 0.0, []

    vote_counts = Counter(votes)
    most_common_vote = vote_counts.most_common(1)[0]
    winner, count = most_common_vote
    
    # 【轨道B】标准 2/3 投票，不做任何哭声特许
    if count >= 2:
        # 计算获胜者的平均置信度
        winning_scores = [model_scores[model] for model, vote in model_votes.items() if vote == winner]
        avg_confidence = np.mean(winning_scores)
        
        # 识别成功（由上层记录）
        logger_b.debug(f"  识别: {winner} (置信度={avg_confidence:.3f}, 票数={count})")
        
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
        logger_b.debug(f"  未识别: 票数不足 ({winner}={count}<2)")
        return None, 0.0, []

def detect_cry_from_full_audio(audio_path):
    """
    【轨道A: 独立哭声检测】
    直接对 60s 原始音频进行声纹匹配，使用 CryDetectionConfig 参数。
    与 identify_speaker_fusion (轨道B) 完全独立，互不影响。
    
    Returns:
        (is_cry: bool, confidence: float, details: list[str])
    """
    if not CryDetectionConfig.ENABLED or not speaker_db:
        return False, 0.0, []
    
    logger_a.info(f"🔍 [轨道A: 哭声检测] 开始对完整音轨进行独立声纹分析...")
    logger_a.info(f"   参数: threshold={CryDetectionConfig.VOICEPRINT_THRESHOLD}, gap={CryDetectionConfig.VOICEPRINT_GAP}, min_votes={CryDetectionConfig.MIN_VOTES}")
    
    target_speakers = CryDetectionConfig.TARGET_SPEAKERS
    cry_threshold = CryDetectionConfig.VOICEPRINT_THRESHOLD
    cry_gap = CryDetectionConfig.VOICEPRINT_GAP
    min_votes = CryDetectionConfig.MIN_VOTES
    
    model_results = {}  # {model_name: (top_target_name, top_target_score, gap_to_others)}
    all_details = []
    
    for model_name, sv_pipe in sv_pipelines.items():
        emb_a = extract_embedding_from_file(sv_pipe, audio_path)
        if emb_a is None:
            all_details.append(f"  {model_name}: 特征提取失败")
            continue
        
        # 对所有已注册说话人打分
        all_scores = []
        for name, speaker_data in speaker_db.items():
            if "avg_embeddings" not in speaker_data or model_name not in speaker_data["avg_embeddings"]:
                continue
            emb_b = np.array(speaker_data["avg_embeddings"][model_name]).flatten()
            score = 1 - cosine(emb_a.flatten(), emb_b)
            all_scores.append((name, score))
        
        if not all_scores:
            all_details.append(f"  {model_name}: 无可比对数据")
            continue
        
        all_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 找出目标说话人 (Baby/宝宝) 的最高分
        target_hits = [(n, s) for n, s in all_scores if n.lower() in target_speakers]
        non_target_scores = [(n, s) for n, s in all_scores if n.lower() not in target_speakers]
        
        # 日志：输出所有说话人得分供调试
        scores_str = ", ".join([f"{n}={s:.3f}" for n, s in all_scores])
        logger_a.info(f"   {model_name}: [{scores_str}]")
        
        if target_hits:
            best_target_name, best_target_score = target_hits[0]
            best_other_score = non_target_scores[0][1] if non_target_scores else 0.0
            gap = best_target_score - best_other_score
            
            passed = best_target_score >= cry_threshold and gap >= cry_gap
            status = "✅ PASS" if passed else "❌ FAIL"
            all_details.append(f"  {model_name}: {best_target_name}={best_target_score:.3f} (gap={gap:.3f}) {status}")
            
            if passed:
                model_results[model_name] = (best_target_name, best_target_score, gap)
        else:
            all_details.append(f"  {model_name}: 未找到目标说话人")
    
    # 投票判定
    vote_count = len(model_results)
    is_cry = vote_count >= min_votes
    
    if is_cry:
        avg_conf = np.mean([v[1] for v in model_results.values()])
        winner_name = list(model_results.values())[0][0]
        logger_a.info(f"   🍼 [轨道A 结论] 检出哭声! 说话人={winner_name}, 票数={vote_count}/{len(sv_pipelines)}, 平均置信度={avg_conf:.3f}")
        all_details.append(f"结论: 哭声检出 ({winner_name}, {vote_count}票, conf={avg_conf:.3f})")
        return True, avg_conf, all_details
    else:
        logger_sys.info(f"   ℹ️ [轨道A 结论] 未检出哭声 (命中模型数={vote_count} < 所需={min_votes})")
        all_details.append(f"结论: 未检出哭声 (票数不足: {vote_count}<{min_votes})")
        return False, 0.0, all_details

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

@app.route("/baby_cry")
def baby_cry_page():
    return render_template("baby_cry.html")

@app.route("/api/cry_events", methods=["GET"])
def api_get_cry_events():
    from db_manager import get_baby_cry_events
    try:
        limit = int(request.args.get('limit', 100))
        offset = int(request.args.get('offset', 0))
        date_filter = request.args.get('date')
        start_time_filter = request.args.get('start_time')
        end_time_filter = request.args.get('end_time')
        
        events = get_baby_cry_events(
            offset=offset, 
            limit=limit, 
            date_filter=date_filter,
            start_time_filter=start_time_filter,
            end_time_filter=end_time_filter
        )
        return jsonify({"events": events})
    except Exception as e:
        logger_a.error(f"获取宝宝哭声记录失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/analyze_cry", methods=["POST"])
def api_analyze_cry():
    """
    主动触发单次哭声 Gemini 深度分析（供 reprocess 脚本在合并事件后调用）。

    Body JSON:
      {
        "filename":    "代表文件名（用于保存数据库记录）",
        "audio_path":  "代表文件绝对路径",
        "start_ms":    0,
        "end_ms":      60000,
        "audio_paths": ["/path/a.m4a", "/path/b.m4a", ...]   ← 可选，事件全部文件（已排序）
      }

    若 audio_paths 非空，直接将这些文件发给 Gemini，跳过自动上下文搜索。
    若未提供 audio_paths，则降级为旧的 process_baby_cry_async 自动搜索逻辑。
    """
    import re as _re, json as _json
    try:
        logger_a.info(f"👶 [analyze_cry API] 收到请求")
        body = request.get_json(force=True)
        logger_a.info(f"👶 [analyze_cry API] 请求体: {body}")
        
        filename   = body.get("filename", "")
        audio_path = body.get("audio_path", "")
        start_ms   = int(body.get("start_ms", 0))
        end_ms     = int(body.get("end_ms", start_ms + 60000))
        audio_paths = body.get("audio_paths")   # 可选：事件文件列表

        logger_a.info(f"👶 [analyze_cry API] 解析参数 - filename: {filename}, audio_path: {audio_path}")
        
        if not filename or not audio_path:
            logger_a.error(f"👶 [analyze_cry API] 参数缺失: filename={filename}, audio_path={audio_path}")
            return jsonify({"error": "filename 和 audio_path 为必填项"}), 400
        if not os.path.exists(audio_path):
            logger_a.error(f"👶 [analyze_cry API] 文件不存在: {audio_path}")
            return jsonify({"error": f"音频文件不存在: {audio_path}"}), 404

        # ── 先运行声纹检测获取投票详情 ──
        import tempfile
        cry_confidence = 0.0
        cry_details = []
        try:
            logger_a.info(f"👶 [analyze_cry API] 开始运行声纹检测...")
            # 预处理音频
            temp_dir = tempfile.mkdtemp()
            proc_temp = os.path.join(temp_dir, "processed.wav")
            if preprocess_audio(audio_path, proc_temp):
                _, cry_confidence, cry_details = detect_cry_from_full_audio(proc_temp)
                logger_a.info(f"👶 [analyze_cry API] 声纹检测完成 - 置信度: {cry_confidence}, 详情数: {len(cry_details)}")
            # 清理临时文件
            try:
                os.remove(proc_temp)
                os.rmdir(temp_dir)
            except:
                pass
        except Exception as e:
            logger_a.warning(f"👶 [analyze_cry API] 声纹检测异常: {e}")

        # ── 模式1：调用方提供了完整事件文件列表，直接送 Gemini ──
        if audio_paths and isinstance(audio_paths, list) and len(audio_paths) > 0:
            valid_paths = [p for p in audio_paths if os.path.exists(p)]
            logger_a.info(
                f"👶 [analyze_cry API] 事件模式：直接发送 {len(valid_paths)} 个文件给 Gemini "
                f"(代表文件: {filename})"
            )
            logger_a.info(f"👶 [analyze_cry API] 有效文件列表: {valid_paths}")
            prompt = (
                "以下是多段连续的录音（时间顺序排列），其中包含了两岁半宝宝的哭泣声。"
                "请结合完整的上下文音频（前后高达10分钟的情境），综合推理宝宝在这段时间哭泣的真正原因"
                "（如困倦Sleepy、饥饿Hungry、情绪发泄Frustration、疼痛Pain、要求未被满足等），"
                "并给出针对此时情境的安抚建议。"
                "请严格按如下JSON格式返回：{\"category\": \"核心原因简短分类(如：困倦/饥饿/疼痛/情绪等)\", \"reason\": \"结合上下文的深度分析原因\", \"advice\": \"针对此时情境的安抚建议\"}"
            )
            logger_a.info(f"👶 [analyze_cry API] 开始调用 Gemini API...")
            response_text = call_gemini_audio_api(valid_paths, prompt)
            logger_a.info(f"👶 [analyze_cry API] Gemini 返回: {response_text[:100] if response_text else 'None'}...")
            
            if response_text:
                # 移除可能的 markdown 代码块标记
                cleaned_text = response_text.strip()
                
                # 移除 markdown 代码块标记
                if cleaned_text.startswith('```json'):
                    cleaned_text = cleaned_text[7:]
                elif cleaned_text.startswith('```'):
                    cleaned_text = cleaned_text[3:]
                if cleaned_text.endswith('```'):
                    cleaned_text = cleaned_text[:-3]
                cleaned_text = cleaned_text.strip()
                
                # 尝试解析 JSON（可能不完整）
                category = "未知"
                reason = "未知"
                advice = "无"
                parse_success = False
                
                # 方法1：直接尝试解析整个清理后的文本
                try:
                    result = _json.loads(cleaned_text)
                    category = result.get("category", "未知")
                    reason = result.get("reason", "未知")
                    advice = result.get("advice", "无")
                    parse_success = True
                    logger_a.info(f"👶 [analyze_cry API] 直接解析成功")
                except Exception as e1:
                    # 方法2：尝试找到 JSON 对象
                    try:
                        # 找到第一个 { 和最后一个 }
                        start_idx = cleaned_text.find('{')
                        end_idx = cleaned_text.rfind('}') + 1
                        if start_idx != -1 and end_idx > start_idx:
                            json_str = cleaned_text[start_idx:end_idx]
                            result = _json.loads(json_str)
                            category = result.get("category", "未知")
                            reason = result.get("reason", "未知")
                            advice = result.get("advice", "无")
                            parse_success = True
                            logger_a.info(f"👶 [analyze_cry API] 提取解析成功")
                    except Exception as e2:
                        logger_a.warning(f"👶 [analyze_cry API] JSON 解析失败: {e1}, {e2}")
                        logger_a.info(f"👶 [analyze_cry API] 原始响应前200字符: {cleaned_text[:200]}")
                
                if parse_success:
                    logger_a.info(f"👶 [analyze_cry API] 解析结果 - category: {category}, reason: {reason[:50]}...")
                    
                    from db_manager import save_cry_analysis
                    # 保存分析结果（包含文生图提示和声纹详情）
                    save_cry_analysis(
                        filename, start_ms / 1000.0, end_ms / 1000.0,
                        reason, advice,
                        reason_category=category,
                        event_files=valid_paths,
                        audio_path=audio_path,
                        confidence=cry_confidence,
                        details=cry_details
                    )

                    # 异步生成插图（不阻塞返回）
                    def generate_illustration():
                        try:
                            logger_a.info(f"🎨 [插图生成] ================ 开始生成场景插图 ================")
                            logger_a.info(f"🎨 [插图生成] 文件名: {filename}")
                            logger_a.info(f"🎨 [插图生成] 原因描述: {reason}")

                            # 根据原因的详细描述生成场景插图
                            # 宝宝信息：2023 年 8 月 9 日生日，现在约 2 岁半
                            image_prompt = (
                                f"创作一幅温暖治愈的儿童绘本风格卡通插图。"
                                f"主角：一个可爱的2岁半中国宝宝（2023年8月出生）。"
                                f"场景描述：{reason}。"
                                f"展现这个宝宝在此情境下的真实日常生活场景。"
                                f"风格：柔和的粉彩色调、柔和的灯光、可爱的卡通形象、情感丰富、表情生动，"
                                f"绘本插画风格、温馨氛围、细节丰富的背景。"
                                f"重要：请在画面中添加中文文字（如对话框、场景标注等），使用中文。"
                            )

                            logger_a.info(f"🎨 [插图生成] Image Prompt 已构建，准备调用 API...")
                            image_url = call_gemini_image_api(image_prompt)

                            if image_url:
                                logger_a.info(f"🎨 [插图生成] API 返回成功，正在更新数据库...")
                                from db_manager import update_cry_event_image
                                update_result = update_cry_event_image(filename, image_url)
                                logger_a.info(f"🎨 [插图生成] 数据库更新结果: {update_result}")
                                logger_a.info(f"🎨 [插图生成] ================ 插图生成完成 ================")
                            else:
                                logger_a.warning(f"🎨 [插图生成] API 返回空，未生成插图")
                                logger_a.info(f"🎨 [插图生成] ================ 插图生成失败 ================")
                        except Exception as e:
                            logger_a.error(f"🎨 [插图生成] 异常: {e}")
                            import traceback
                            logger_a.error(f"🎨 [插图生成] 异常堆栈: {traceback.format_exc()}")
                            logger_a.info(f"🎨 [插图生成] ================ 插图生成异常 ================")

                    logger_a.info(f"🎨 [插图生成] 启动后台线程生成插图...")
                    threading.Thread(target=generate_illustration, daemon=True).start()

                    logger_a.info(f"👶 [analyze_cry API] 分析完成：[{category}] {reason[:50]}...")
                    return jsonify({"category": category, "reason": reason, "advice": advice})
                else:
                    logger_a.error(f"👶 [analyze_cry API] 无法解析 JSON 响应")
                    logger_a.info(f"👶 [analyze_cry API] 原始响应: {response_text[:500]}")
            else:
                logger_a.error(f"👶 [analyze_cry API] Gemini 未返回响应")
            return jsonify({"reason": None, "advice": None, "message": "Gemini 未返回有效分析结果"}), 200

        # ── 模式2：降级为旧的自动上下文搜索 ──
        logger_a.info(f"👶 [analyze_cry API] 自动搜索模式: {filename} ({start_ms}ms-{end_ms}ms)")
        reason, advice = process_baby_cry_async(filename, audio_path, start_ms, end_ms)
        if reason:
            return jsonify({"reason": reason, "advice": advice or ""})
        return jsonify({"reason": None, "advice": None, "message": "Gemini 未返回有效分析结果"}), 200

    except Exception as e:
        logger_a.error(f"[analyze_cry API] 异常: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/trigger_reprocess", methods=["POST"])
def trigger_reprocess():
    """触发重新处理历史音频任务"""
    global _history_reprocess_running
    import subprocess
    
    # 尝试加锁
    if not _history_reprocess_lock.acquire(blocking=False):
        return jsonify({"error": "另一个分析任务正在运行中，请等待其完成。"}), 409
    
    if _history_reprocess_running:
        _history_reprocess_lock.release()
        return jsonify({"error": "分析任务已在执行中。"}), 409

    try:
        _history_reprocess_running = True
        date_param = request.args.get('date', '')
        start_time = request.args.get('start_time', '')
        end_time = request.args.get('end_time', '')
        force_replace = request.args.get('replace', 'false').lower() == 'true'
        
        # 不提前删除记录，等分析完成后再处理（避免记录消失）
            
        # 暂停 B 轨处理，直到 A 轨结束
        global _track_b_paused
        _track_b_paused = True
        logger_sys.warning("⚠️ 由于 A 轨历史分析启动，B 轨实时分析已自动暂停。")
        
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reprocess_history_cries.py")
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "history_process.log")
        
        # 写入一条启动信息
        is_targeted = bool(date_param or start_time or end_time)
        display_replace = "True (Targeted)" if is_targeted else force_replace
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🚀 开始执行历史音频重分析(过滤={date_param} {start_time}~{end_time}, 替换={display_replace})...\n")
            
        args = [sys.executable, "-u", script_path]
        args.append(date_param if date_param else "")
        args.append(start_time if start_time else "")
        args.append(end_time if end_time else "")
        if force_replace:
            args.append("--replace")
            
        def run_and_cleanup():
            global _history_reprocess_running, _history_reprocess_proc
            try:
                with open(log_file, "a", encoding="utf-8") as f:
                    _history_reprocess_proc = subprocess.Popen(
                        args, 
                        stdout=f, 
                        stderr=subprocess.STDOUT,
                        cwd=os.path.dirname(os.path.abspath(__file__))
                    )
                    _history_reprocess_proc.wait()
            finally:
                global _track_b_paused
                _history_reprocess_running = False
                _history_reprocess_proc = None
                _track_b_paused = False
                _history_reprocess_lock.release()
                logger_sys.info("✅ A 轨任务结束，B 轨分析已恢复。")
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✅ 历史音频重分析任务已结束。\n")

        threading.Thread(target=run_and_cleanup, daemon=True).start()
        
        logger_a.info(f"✅ 历史音频重分析线程已启动，输出重定向至: {log_file}")
        return jsonify({"message": "任务已在后台启动，请查看下方实时日志。"})
    except Exception as e:
        _history_reprocess_running = False
        _history_reprocess_lock.release()
        logger_a.error(f"❌ 运行历史分析脚本失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/stop_reprocess", methods=["POST"])
def stop_reprocess():
    """停止正在运行的历史音频处理任务"""
    global _history_reprocess_proc, _history_reprocess_running

    if not _history_reprocess_running or _history_reprocess_proc is None:
        return jsonify({"message": "当前没有正在运行的任务", "status": "info"})

    try:
        # 终止进程
        if _history_reprocess_proc:
            _history_reprocess_proc.terminate()
            # 等待确保释放
            time.sleep(0.5)
            # 检查进程是否还在运行
            if _history_reprocess_proc.poll() is None:
                _history_reprocess_proc.kill()
            logger_a.info("✅ 后台分析任务已停止")
        else:
            logger_a.warning("⚠️ 没有正在运行的分析任务")

        return jsonify({"message": "后台分析任务已手动停止", "status": "success"})
    except AttributeError:
        # 进程对象不存在或已释放
        logger_a.info("ℹ️ 后台任务已自然结束，无需手动停止")
        return jsonify({"message": "后台任务已结束", "status": "success"})
    except Exception as e:
        logger_a.error(f"⚠️ 停止任务时遇到问题：{type(e).__name__}")
        return jsonify({"message": "任务已在后台结束", "status": "success", "note": str(e)}), 200

# 刷盘任务状态存储
refresh_cache_task = {
    "running": False,
    "start_time": None,
    "count": 0,
    "status": "idle",  # idle, running, completed, error
    "message": ""
}

def update_refresh_progress(count, current_dir):
    """更新刷盘进度"""
    global refresh_cache_task
    refresh_cache_task["count"] = count
    refresh_cache_task["message"] = f"正在扫描目录: {current_dir}"

def log_to_process(msg):
    """写入日志到 process logs"""
    import time as time_module
    try:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "history_process.log")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{time_module.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
            f.flush()  # 立即刷新到磁盘
    except Exception as e:
        logger_a.error(f"[刷盘] 写入process log失败: {e}")

def run_refresh_file_cache(target_dir):
    """后台执行刷盘任务"""
    global refresh_cache_task
    refresh_cache_task["running"] = True
    refresh_cache_task["status"] = "running"
    refresh_cache_task["count"] = 0
    refresh_cache_task["start_time"] = time.time()
    refresh_cache_task["message"] = "正在扫描..."
    
    # 清空并初始化日志文件
    import time as time_module
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "history_process.log")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"[{time_module.strftime('%Y-%m-%d %H:%M:%S')}] 🔄 开始刷盘任务，扫描目录: {target_dir}\n")
        f.flush()
    
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nas-audio-notes-client'))
        from db_manager import refresh_file_cache as do_refresh, init_pool
        
        init_pool()
        logger_a.info(f"[刷盘] 后台任务开始，扫描目录: {target_dir}")
        log_to_process(f"🔄 刷盘任务开始，目标目录: {target_dir}")
        
        # 使用回调函数实时更新进度和日志
        count = do_refresh(target_dir, progress_callback=update_refresh_progress, log_callback=log_to_process)
        
        if count >= 0:
            refresh_cache_task["count"] = count
            refresh_cache_task["status"] = "completed"
            refresh_cache_task["message"] = f"完成，共 {count} 个文件"
            elapsed = time.time() - refresh_cache_task["start_time"]
            logger_a.info(f"[刷盘] 后台任务完成，共 {count} 个文件")
            log_to_process(f"✅ 刷盘完成！共 {count} 个文件，耗时 {elapsed:.1f}秒")
        else:
            refresh_cache_task["status"] = "error"
            refresh_cache_task["message"] = "刷新失败"
            logger_a.error(f"[刷盘] 后台任务失败")
            log_to_process("❌ 刷盘失败")
    except Exception as e:
        refresh_cache_task["status"] = "error"
        refresh_cache_task["message"] = str(e)
        logger_a.error(f"[刷盘] 后台任务异常: {e}")
        log_to_process(f"❌ 刷盘异常: {e}")
    finally:
        refresh_cache_task["running"] = False

@app.route("/api/refresh_file_cache", methods=["POST"])
def refresh_file_cache():
    """启动刷盘任务（异步后台执行）"""
    global refresh_cache_task
    
    # 如果已有任务在运行，返回状态
    if refresh_cache_task["running"]:
        return jsonify({
            "message": "刷盘任务已在运行中",
            "status": "running",
            "task": refresh_cache_task
        })
    
    # 启动后台线程
    target_dir = FileMonitorConfig.SOURCE_DIR
    thread = threading.Thread(target=run_refresh_file_cache, args=(target_dir,))
    thread.daemon = True
    thread.start()
    
    logger_a.info(f"[刷盘] 启动后台任务，扫描目录: {target_dir}")
    return jsonify({
        "message": "刷盘任务已启动",
        "status": "started",
        "task": refresh_cache_task
    })

@app.route("/api/refresh_file_cache/status", methods=["GET"])
def refresh_file_cache_status():
    """获取刷盘任务状态"""
    return jsonify({
        "status": "success",
        "task": refresh_cache_task
    })

@app.route("/api/file_cache_status", methods=["GET"])
def file_cache_status():
    """获取文件缓存状态"""
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nas-audio-notes-client'))
        from db_manager import get_file_count_from_redis

        count = get_file_count_from_redis()

        return jsonify({"count": count, "status": "success"})
    except Exception as e:
        logger_a.error(f"❌ 获取文件缓存状态失败: {e}")
        return jsonify({"message": str(e), "status": "error"}), 500

@app.route("/api/reprocess_logs", methods=["GET"])
def get_reprocess_logs():
    """获取历史分析进程的实时日志"""
    try:
        log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log", "history_process.log")
        if not os.path.exists(log_file):
            return jsonify({"logs": "尚未开始处理，或日志文件不存在...", "a_running": False})

        # 检查进程是否还在运行
        a_running = _history_reprocess_proc is not None and _history_reprocess_proc.poll() is None

        with open(log_file, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(size - 20000, 0), 0) # 读取最后大概 20KB
            logs_bytes = f.read()
            logs = logs_bytes.decode('utf-8', errors='replace')
            return jsonify({"logs": logs, "a_running": a_running, "pid": _history_reprocess_proc.pid if _history_reprocess_proc else None})
    except Exception as e:
        return jsonify({"error": str(e), "a_running": False}), 500

@app.route("/api/live_status", methods=["GET"])
def get_live_status():
    """获取 A/B 轨状态"""
    try:
        global _track_b_running, _track_b_paused, _history_reprocess_proc
        # 检查 A 轨进程是否还在运行
        a_running = _history_reprocess_proc is not None and _history_reprocess_proc.poll() is None
        return jsonify({
            "a_running": a_running,
            "b_running": _track_b_running,
            "b_paused": _track_b_paused if _track_b_running else True,
            "a_reason": "A 轨历史分析中" if (a_running and _track_b_running and _track_b_paused) else None,
            "b_reason": "A 轨历史分析中" if (a_running and _track_b_running) else None,
            "message": "B 轨已暂停" if (_track_b_running and _track_b_paused) else ("B 轨正常运行中" if _track_b_running else "B 轨未启动")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/live_logs", methods=["GET"])
def get_live_logs():
    """获取实时分析（B 轨）的日志"""
    try:
        log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log", "asr-b.log")
        if not os.path.exists(log_file):
            return jsonify({"logs": "暂无日志", "b_running": False})

        global _track_b_running
        with open(log_file, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(size - 20000, 0), 0)
            logs_bytes = f.read()
            logs = logs_bytes.decode('utf-8', errors='replace')
            return jsonify({"logs": logs, "b_running": _track_b_running})
    except Exception as e:
        return jsonify({"error": str(e), "b_running": False}), 500

@app.route("/api/start_live", methods=["POST"])
def start_live():
    """启动实时监听（B 轨）"""
    try:
        import threading
        global _monitor_thread, _track_b_running, _track_b_paused
        if _track_b_running and _monitor_thread and _monitor_thread.is_alive():
            return jsonify({"message": "B 轨已在运行中", "status": "already_running"})
        _track_b_running = True
        _track_b_paused = False
        _monitor_thread = threading.Thread(target=audio_processor.start_monitor, daemon=True)
        _monitor_thread.start()
        return jsonify({"message": "B 轨实时监听已启动", "status": "started"})
    except Exception as e:
        logger.error(f"启动 B 轨失败: {e}")
        return jsonify({"message": f"启动失败: {e}", "status": "error"}), 500

@app.route("/api/pause_live", methods=["POST"])
def pause_live():
    """暂停实时监听（B 轨）"""
    try:
        global _track_b_paused
        if not _track_b_running:
            return jsonify({"message": "B 轨未启动", "status": "not_running"})
        _track_b_paused = True
        return jsonify({"message": "B 轨已暂停", "status": "paused"})
    except Exception as e:
        return jsonify({"message": f"暂停失败: {e}", "status": "error"}), 500

@app.route("/api/stop_live", methods=["POST"])
def stop_live():
    """停止实时监听（B 轨）"""
    try:
        global _track_b_running, _track_b_paused, _monitor_thread
        _track_b_running = False
        _track_b_paused = True
        _monitor_thread = None
        return jsonify({"message": "B 轨已停止", "status": "stopped"})
    except Exception as e:
        return jsonify({"message": f"停止失败: {e}", "status": "error"}), 500

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
        logger_b.error(f"获取说话人列表失败: {str(e)}")
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
        logger_b.error(f"获取说话人样本列表失败: {str(e)}")
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
                logger_b.info(f"✅ 成功删除说话人: {speaker_name}")
                return jsonify({"message": f"Speaker '{speaker_name}' deleted successfully."})
            else:
                return jsonify({"error": f"Speaker '{speaker_name}' not found."}), 404
    except Exception as e:
        logger_b.error(f"删除说话人失败: {str(e)}")
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
                    logger_b.info(f"🗑️ 删除了音频文件: {sample_to_remove['audio_path']}")
                except Exception as e:
                    logger_b.warning(f"⚠️ 删除音频文件失败: {sample_to_remove['audio_path']}, 错误: {str(e)}")
            
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
                        logger_b.info(f"🗑️ 删除了说话人目录: {speaker_dir}")
                    except Exception as e:
                        logger_b.warning(f"⚠️ 删除说话人目录失败: {speaker_dir}, 错误: {str(e)}")
                
                with open(Config.SPEAKER_DB_FILE, 'w', encoding='utf-8') as f:
                    json.dump(speaker_db, f, indent=2, ensure_ascii=False)
                logger_b.info(f"🗑️ 删除了说话人 {speaker_name}（最后一个样本已删除）")
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
                
            logger_b.info(f"🗑️ 删除了说话人 {speaker_name} 的样本 {sample_id}")
            return jsonify({
                "message": f"Sample '{sample_id}' deleted from speaker '{speaker_name}'.",
                "remaining_samples": len(samples)
            })
    except Exception as e:
        logger_b.error(f"删除说话人样本失败: {str(e)}")
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
        logger_b.error(f"获取说话人列表失败: {str(e)}")
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
            logger_b.info(f"📥 开始{action}新声纹: {speaker_name} | 文件: {audio_file.filename}")
            
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
                    logger_b.warning(f"⚠️ 从 {audio_file.filename} 提取 {model_name} embedding 失败。")

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
                logger_b.info(f"  - 模型 [{model_name}] 处理了 {len(emb_list)} 个样本")

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
                    logger_b.info(f"✅ 成功增强说话人 [{speaker_name}]，当前共 {total_samples} 个样本")
                else:
                    # 新建说话人
                    speaker_db[speaker_name] = {
                        "samples": new_samples,
                        "avg_embeddings": avg_embeddings
                    }
                    logger_b.info(f"✅ 成功注册新说话人 [{speaker_name}]，共 {len(new_samples)} 个样本")

                # 保存到数据库文件
                with open(Config.SPEAKER_DB_FILE, 'w', encoding='utf-8') as f:
                    json.dump(speaker_db, f, indent=2, ensure_ascii=False)

            return jsonify({
                "message": f"Speaker '{speaker_name}' {'enhanced' if enhance_mode else 'registered'} successfully.",
                "sample_count": len(speaker_db[speaker_name]["samples"])
            })

        except Exception as e:
            logger_b.error(f"注册声纹失败: {str(e)}")
            logger_b.error(traceback.format_exc())
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
            logger_b.info(f"📥 开始{action}新声纹: {speaker_name} | 文件数: {len(audio_files)}")
            
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
                    logger_b.warning(f"⚠️ 文件 {file.filename} 预处理失败，已跳过。")
                    continue

                # 为每个模型提取嵌入
                sample_embeddings = {}
                for model_name, sv_pipe in sv_pipelines.items():
                    emb = extract_embedding_from_file(sv_pipe, proc_temp)
                    if emb is not None:
                        sample_embeddings[model_name] = emb.tolist()
                        model_embeddings[model_name].append(emb)
                    else:
                        logger_b.warning(f"⚠️ 从 {file.filename} 提取 {model_name} embedding 失败。")

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
                logger_b.info(f"  - 模型 [{model_name}] 处理了 {len(emb_list)} 个样本")

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
                    logger_b.info(f"🔄 增强了说话人 {speaker_name} 的声纹，新增 {len(new_samples)} 个样本")
                else:
                    # 创建新的说话人条目
                    speaker_db[speaker_name] = {
                        "samples": new_samples,
                        "avg_embeddings": avg_embeddings
                    }
                    logger_b.info(f"🆕 创建了新说话人 {speaker_name} 的声纹，包含 {len(new_samples)} 个样本")
                    
                # 保存更新后的数据库
                with open(Config.SPEAKER_DB_FILE, 'w', encoding='utf-8') as f:
                    json.dump(speaker_db, f, indent=2, ensure_ascii=False)
            
            logger_b.info(f"✅ 声纹{action}成功: {speaker_name}")
            return jsonify({
                "message": f"Speaker '{speaker_name}' {action} successfully.",
                "samples_added": len(new_samples)
            })

        except Exception as e:
            logger_b.error(f"❌ 注册异常: {str(e)}")
            logger_b.error(traceback.format_exc())
            return jsonify({"error": "An internal error occurred during registration."} ), 500
        finally:
            # 清理临时文件，但保留语音片段文件供web端预览使用
            for f in temp_files:
                if os.path.exists(f):
                    # 不删除语音片段文件 (seg_*.wav)，这些文件需要保留供web端预览
                    if os.path.basename(f).startswith("seg_"):
                        logger_b.info(f"  [保留] 语音片段文件供预览使用: {os.path.basename(f)}")
                        continue
                    try: os.remove(f)
                    except: pass


@app.route("/transcribes", methods=["POST"])
def transcribe_audio():
    # 检查 B 轨是否暂停
    if _track_b_paused:
        if request.form.get('is_history', 'false').lower() != 'true':
            return jsonify({
                "status": "paused",
                "error": "Service temporarily paused for historical analysis",
                "message": "B 轨当前已暂停，请稍后自动重试"
            }), 503

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
                logger_b.info(f"⏭️ 忽略临时文件: {file.filename}")
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
            
            logger_b.info(f"📥 收到转录任务: {file.filename}")
            
            logger_b.info("  [生命周期: 1. 音频预处理] 开始 (FFmpeg降噪、重采样、归一化)...")
            if not preprocess_audio(raw_temp, proc_temp):
                return jsonify({"error": "Audio preprocessing failed"}), 500
            logger_b.info("  [生命周期: 1. 音频预处理] 完成。")

            audio_duration = 0
            try:
                probe = subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', proc_temp])
                audio_duration = float(probe)
            except: pass

            logger_b.info("  [生命周期: 2. VAD & ASR] 开始 (FunASR语音检测与文字转录)...")
            res = asr_pipeline.generate(input=proc_temp, language="auto", use_itn=True, use_punc=True)
            
            # 【轨道A: 独立哭声检测】直接对完整 60s 原始音频做声纹匹配
            # 使用 CryDetectionConfig 独立参数，与轨道B (VAD+语音识别) 完全隔离
            cry_detected = False
            skip_cry_flag = request.form.get('skip_cry', 'false').lower() == 'true'
            try:
                cry_detected, cry_confidence, cry_details = detect_cry_from_full_audio(proc_temp)
                
                if cry_detected:
                    logger_a.info(f"  🍼 [轨道A] 哭声确认! 置信度={cry_confidence:.3f}, 启动报警流程...")
                    
                    if skip_cry_flag:
                        logger_a.info(f"      [skip_cry] 哭声已标记，历史模式不发送即时邮件")
                    else:
                        # 冷却机制
                        global _last_cry_trigger_time
                        now = time.time()
                        with _cry_cooldown_lock:
                            in_cooldown = (now - _last_cry_trigger_time) < CryDetectionConfig.COOLDOWN_SEC
                            if not in_cooldown:
                                _last_cry_trigger_time = now
                        
                        if in_cooldown:
                            elapsed = int(now - _last_cry_trigger_time)
                            logger_a.info(f"      [冷却中] 距上次哭声分析 {elapsed}s，冷却期 {CryDetectionConfig.COOLDOWN_SEC}s 内跳过")
                        else:
                            # ── 正式报警 ──
                            time_str = datetime.now(UTC_PLUS_8).strftime("%Y-%m-%d %H:%M:%S")
                            send_cry_alert_email(file.filename, time_str)
                            
                            # [即时占位] 立即写入数据库（包含声纹投票详情）
                            from db_manager import save_cry_analysis
                            placeholder_id = save_cry_analysis(
                                file.filename, 0, audio_duration, 
                                "深度分析中 (5分钟观察期)...", "请稍候内容更新", 
                                reason_category="analyzing", event_files=[], 
                                audio_path=proc_temp,
                                confidence=cry_confidence,
                                details=cry_details
                            )

                            # 启动后台延迟分析任务
                            def start_delayed_analysis(fname, a_path, dur, p_id):
                                logger_a.info(f"⏳ [BabyCry] 已启动延迟分析线程，等待 300s 后更新占位 (ID={p_id})...")
                                time.sleep(300)
                                process_baby_cry_async(fname, a_path, 0, dur * 1000, placeholder_id=p_id)
                            
                            threading.Thread(
                                target=start_delayed_analysis, 
                                args=(file.filename, proc_temp, audio_duration, placeholder_id), 
                                daemon=True
                            ).start()
            except Exception as ex:
                logger_a.warning(f"    ⚠️ 哭声检测异常: {ex}")
            
            # 【轨道B: 语音识别】VAD 分段 + 转录 + 说话人标注 (参数不变)
            full_text = ""
            segments = []
            processed_segments = []

            if res and isinstance(res, list) and len(res) > 0:
                item = res[0]
                full_text = item.get("text", "")
                
                raw_segments = item.get("sentence_info", [])
                logger_b.info(f"  [生命周期: 2. VAD & ASR] 完成, VAD检出 {len(raw_segments)} 个分段。")

                if not raw_segments and full_text:
                    raw_segments = [{"text": full_text, "start": 0, "end": int(audio_duration * 1000)}]

                processed_segments = []
                
                if raw_segments:
                    logger_b.info("  [生命周期: 3. 逐段声纹识别] 开始...")
                    for i, seg in enumerate(raw_segments):
                        raw_text = seg.get("text", "")
                        start, end = seg.get("start", 0), seg.get("end", 0)
                        logger_b.info(f"    [3.{i+1}] 处理分段 {start}ms - {end}ms...")
                        
                        if any(tag in raw_text for tag in INVALID_TAGS): continue

                        # Case-insensitive emotion detection
                        emotion = None  # 未识别到情感时为None,不使用neutral
                        original_emotion_tag = None  # 初始化以防未定义引用
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
                        
                        # 核心优化：如果包含哭声标签，即使没有识别出文字，也不应跳过！
                        has_cry_tag = "<|cry|>" in raw_text_lower
                        if not clean_text and not has_cry_tag: 
                            logger_b.info(f"      [3.{i+1}] 分段既无文本也无哭声标签，已跳过。")
                            continue

                        identity, confidence = None, 0.0
                        recognition_details = []

                        segment_audio_path = None

                        if (end - start) > Config.MIN_SPEAKER_DURATION_MS:
                            # 创建持久化的音频片段目录
                            # 使用原始文件名（不含扩展名和_TEMP后缀）作为子目录
                            original_filename = file.filename.replace('_TEMP', '')  # 移除_TEMP后缀
                            base_filename = os.path.splitext(original_filename)[0]
                            
                            # 从文件名解析日期，或使用当前日期
                            date_str = datetime.now(UTC_PLUS_8).strftime("%Y-%m-%d")
                            # 尝试从文件名提取日期 (格式: YYYY-MM-DD 或 YYYYMMDD 或 recording-YYYYMMDD)
                            date_match = re.search(r'(\d{4})-?(\d{2})-?(\d{2})', base_filename)
                            if date_match:
                                date_str = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
                            
                            # 按日期分类的目录结构: audio_segments/YYYY-MM-DD/filename/
                            segments_dir = os.path.join(FileMonitorConfig.SOURCE_DIR, "audio_segments", date_str, base_filename)
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
                                    # 只有成功复制后才保存路径 (包含日期子目录)
                                    segment_audio_path = f"/audio_segments/{date_str}/{base_filename}/{seg_filename}"
                                    logger_b.debug(f"      [音频片段] 已保存: {seg_wav_persistent}")
                                except Exception as copy_error:
                                    logger_b.error(f"      [音频片段] 复制失败: {copy_error}")
                                    segment_audio_path = None  # 如果复制失败,不设置路径

                                # 1. 深度状态探测 (SenseVoice)
                                sensevoice_text, sensevoice_emotion = transcribe_with_sensevoice(seg_wav_temp)
                                
                                # 2. 【轨道B】纯语音识别声纹 (标准参数，不做哭声补偿)
                                identity, confidence, recognition_details = identify_speaker_fusion(seg_wav_temp)
                                
                                # 3. 补全其他信息 (Whisper/Emotion)
                                whisper_text = None
                                emotion = sensevoice_emotion
                                
                                if identity is not None:
                                    if emotion is None:
                                        emotion = detect_emotion_for_segment(seg_wav_temp)
                                    whisper_text = transcribe_with_whisper(seg_wav_temp)
                                    logger_b.info(f"      [性能] 已识别说话人 {identity}")
                                else:
                                    logger_b.info(f"      [性能] 未识别说话人，跳过后续处理")
                                    whisper_text = None
                                
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
                                        
                                        logger_b.info(f"      [长句保存] 已保存 {len(clean_text)} 字音频: {saved_filename}")
                                    except Exception as e:
                                        logger_b.warning(f"      [长句保存] 保存失败: {e}")
                        else:
                            logger_b.info(f"      [3.{i+1}] 分段时长过短({end-start}ms)，跳过声纹识别。")
                            # 即使跳过声纹识别，也要初始化这些变量
                            emotion = None  # 未识别到情感时为None,不使用neutral
                            whisper_text = None


                        # 核心优化：如果 SenseVoice 已经判定为 <|CRY|>，则豁免 ONLY_REGISTERED_SPEAKERS 检查
                        # 这通过防止由于声纹稍有偏差而抛弃真实的哭闹事件
                        has_confirmed_cry = (emotion == "sad" or "<|cry|>" in (original_emotion_tag or "").lower())
                        
                        if Config.ONLY_REGISTERED_SPEAKERS and identity is None and not has_confirmed_cry:
                            continue
                        
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
                        
                        segment_info = {
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
                        }
                        
                        # 【轨道A】如果全局哭声检测已确认，补充标记到每个分段
                        if cry_detected:
                            segment_info["is_baby_cry"] = True
                            segment_info["emotion"] = "sad"
                            segment_info["emotion_info"]["emotion"] = "sad"
                            segment_info["emotion_info"]["source"] = "cry_detection_track_a"

                        processed_segments.append(segment_info)

                segments = processed_segments
                
                # 【强力补充】如果轨道A检出哭声但轨道B(VAD)没有任何分段，则手动补入一个全局哭声片段
                # 这样可以确保重分析脚本(reprocess_history_cries.py)能正确感知并进入详情分析阶段
                if cry_detected and not segments:
                    logger_a.info("  🍼 [轨道A] 补偿机制启动: VAD未命中，手动添加全局哭声片段。")
                    segments = [{
                        "text": "[Baby Cry Detected]",
                        "start": 0,
                        "end": int(audio_duration * 1000),
                        "spk": "Baby",
                        "emotion": "sad",
                        "is_baby_cry": True,
                        "confidence": cry_confidence,
                        "emotion_info": {
                            "emotion": "sad",
                            "source": "cry_detection_track_a",
                            "detected_by_sensevoice": False
                        }
                    }]

                full_text = "".join([s.get("text", "") for s in segments]) 

            process_time = time.time() - request_start
            rtf = process_time / audio_duration if audio_duration > 0 else 0
            logger_b.info(f"✅ 完成! 音频:{audio_duration:.1f}s | 耗时:{process_time:.2f}s | RTF:{rtf:.3f}")

            logger_b.info("  [生命周期: 4. 组装响应] 开始...")
            response_data = {
                "full_text": full_text,
                "segments": segments,
                "duration": audio_duration,  # 补全根节点字段供重分析脚本使用
                "meta": {
                    "process_time": process_time,
                    "audio_duration": audio_duration,
                    "rtf": rtf,
                    "rtf_description": "Real-Time Factor(实时因子)，处理时间/音频时长，RTF < 1表示可实时处理，值越低性能越好"
                }
            }
            logger_b.info(f"📤  [生命周期: 4. 组装响应] 完成, 返回 /transcribe 结果: {json.dumps(response_data, ensure_ascii=False, indent=2)}")
            
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
                        logger_b.info(f"✅ 数据库保存成功 (recording_time: {recording_time})")
                        if summary:
                            logger_b.info(f"  智能摘要: {summary['speaker_count']}位说话人, {summary['total_segments']}个分段")
                        
                        # 添加到 LLM 批量处理队列
                        if LLMConfig.USE_GEMINI_LLM:
                            has_identified_speakers = any(seg.get('spk') != 'Unknown' for seg in processed_segments)
                            if (len(full_text) >= LLMConfig.LLM_MIN_TEXT_LENGTH and 
                                len(processed_segments) >= LLMConfig.LLM_MIN_SEGMENTS and 
                                has_identified_speakers):
                                add_to_llm_queue(file.filename, full_text, processed_segments)
                    else:
                        logger_b.error(f"❌ 数据库保存失败")
                except Exception as e:
                    logger_b.error(f"❌ 数据库保存异常: {e}")
                    logger_b.error(traceback.format_exc())
            # =========================================================

            return jsonify(response_data)

        except Exception as e:
            logger_b.error(f"❌ 处理异常: {str(e)}")
            logger_b.error(traceback.format_exc())
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
        logger_sys.error(f"获取样本音频文件失败: {str(e)}")
        return jsonify({"error": "Failed to retrieve sample audio"}), 500


@app.route('/audio_segments/<path:filename>')
def serve_audio_segment(filename):
    """提供音频片段静态文件服务"""
    try:
        audio_segments_dir = os.path.join(FileMonitorConfig.SOURCE_DIR, 'audio_segments')
        return send_from_directory(audio_segments_dir, filename)
    except Exception as e:
        logger_sys.error(f"获取音频片段失败: {str(e)}")
        return jsonify({"error": "Audio segment not found"}), 404


@app.route('/api/audio/<filename>')
def serve_audio_file(filename):
    """提供原始音频文件服务（如 TermuxAudioRecording_xxx.m4a）"""
    try:
        # 清理路径（移除开头的 /）
        clean_filename = filename.lstrip('/')
        return send_from_directory(FileMonitorConfig.SOURCE_DIR, clean_filename)
    except Exception as e:
        logger_sys.error(f"获取音频文件失败: {str(e)}")
        return jsonify({"error": "Audio file not found"}), 404


@app.route('/api/event_audio/<int:event_id>')
def serve_event_audio(event_id):
    """获取事件的完整上下文音频文件列表"""
    try:
        logger_sys.info(f"🔍 [上下文音频] 收到请求 event_id={event_id}")
        from db_manager import get_baby_cry_event_by_id
        event = get_baby_cry_event_by_id(event_id)
        logger_sys.info(f"🔍 [上下文音频] 查询结果: {event}")
        if not event:
            logger_sys.warning(f"⚠️ [上下文音频] 事件 {event_id} 未找到")
            return jsonify({"error": "Event not found"}), 404

        event_files = event.get('event_files_json', [])
        logger_sys.info(f"🔍 [上下文音频] event_files: {event_files}")
        audio_urls = []
        for f in event_files:
            # 转换为相对于 SOURCE_DIR 的路径
            if f.startswith(FileMonitorConfig.SOURCE_DIR):
                f = os.path.relpath(f, FileMonitorConfig.SOURCE_DIR)
            elif f.startswith('/'):
                f = f.lstrip('/')
            audio_urls.append(f"/api/audio/{f}")

        return jsonify({
            "event_id": event_id,
            "audio_count": len(audio_urls),
            "audio_urls": audio_urls
        })
    except Exception as e:
        logger_sys.error(f"获取事件音频列表失败: {str(e)}")
        import traceback
        logger_sys.error(f"异常堆栈: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


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
        logger_sys.critical("❌ 系统未安装 FFmpeg！")
        sys.exit(1)

    # 初始化数据库连接池和表结构
    print("初始化数据库连接池...")
    if not init_pool():
        logger_sys.critical("❌ 数据库连接池初始化失败！")
        sys.exit(1)
    
    print("初始化数据库表结构...")
    if not init_db():
        logger_sys.warning("⚠️ 数据库表结构初始化失败，但服务将继续运行")

    load_models()
    
    # 启动临时文件清理定时任务
    cleanup_temp_dir()
    logger_sys.info("临时文件清理定时任务已启动")
    
    # 启动文件监控模块 (已解耦)
    _track_b_running = True
    audio_processor.start_monitor()
    
    print("🎉 服务启动成功！")
    print("📌 声纹注册页面: http://127.0.0.1:5008/register_page")
    print("📌 语音转录API: http://127.0.0.1:5008/transcribes (本地监控专用)")
    print("📌 外部调用API: http://127.0.0.1:5008/transcribe (保留给NAS使用)")
    print(f"📂 文件监控目录: {FileMonitorConfig.SOURCE_DIR}")
    print(f"⏱️  扫描间隔: {FileMonitorConfig.SCAN_INTERVAL}秒")
    print("🔧 API使用方法: POST请求，参数名 'audio_file'，上传音频文件")
    print("🔍 示例命令: curl -X POST -F \"audio_file=@your_audio.wav\" http://127.0.0.1:5008/transcribes")
    app.run(host=Config.HOST, port=Config.PORT, debug=False, threaded=True)