#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import functools
import hashlib
import os, sys, logging, json, threading, subprocess, time, traceback, tempfile, argparse, uuid, glob
from dotenv import load_dotenv

load_dotenv()

import numpy as np
from scipy.spatial.distance import cosine
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify, render_template, send_file, send_from_directory, Response
from funasr import AutoModel  # ASR 用 FunASR
from modelscope.pipelines import pipeline  # SV 用 ModelScope
from modelscope.utils.constant import Tasks
import torch
import torchaudio
import shutil
import re
from collections import Counter, OrderedDict
from db_manager import save_to_db, update_topics, parse_recording_time, init_pool, init_db
from logging.handlers import RotatingFileHandler
import requests
import hashlib
from datetime import datetime, timezone, timedelta

# 东八区时区 (UTC+8) - 上海时间
UTC_PLUS_8 = timezone(timedelta(hours=8))
from email_utils import send_cry_alert_email, send_cry_webhook, send_cry_analysis_webhook   # 邮件通知 + 哭声报警/分析报告 Webhook
import audio_processor
import recovery_monitor

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

class Config:
    DEVICE = get_device()
    MODELSCOPE_DEVICE = get_modelscope_device()
    HOST = os.getenv('ASR_SERVER_HOST', '0.0.0.0')
    PORT = int(os.getenv('ASR_SERVER_PORT', '5008'))
    SPEAKER_DB_FILE = "speaker_db_multi.json"
    # 长句音频保存配置
    SAVE_LONG_SENTENCES = True  # 是否保存长句音频
    MIN_TEXT_LENGTH_TO_SAVE = 15  # 最少字数
    LONG_SENTENCES_DIR = "long_sentences"  # 保存目录
    TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")  # 临时文件目录（绝对路径，防 cwd 歧义）

    ONLY_REGISTERED_SPEAKERS = False  # 只保留已注册说话人,丢弃Unknown（2026-10-04 True→False：清库重注册期间库为空，True 会把所有切片连文字一起丢弃导致记录页全空；注册完成后想过滤陌生人噪声可再开回 True）
    # ASR模型配置 - Paraformer (支持VAD分段和说话人分离)
    ASR_MODEL = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"  # 从 SenseVoiceSmall 切换到 Paraformer
    ASR_HOTWORD = os.getenv('ASR_HOTWORD', '大可').strip()  # SeACo 热词定制: 提升人名/专有词识别(空格分隔多个词, 置空禁用)
    VAD_MODEL = "fsmn-vad"       # VAD模型
    SPK_MODEL = "cam++"          # 说话人分离模型
    PUNC_MODEL = "ct-punc"       # 标点恢复模型

    # VAD参数配置(为Paraformer优化)
    VAD_MAX_SINGLE_SEGMENT = 15000  # ms - 单段最长时间
    VAD_MAX_END_SILENCE = 1000      # ms - 段尾静音阈值（2026-10-03 300→800→1000：家人快节奏对话间隙常 <1s，放宽减少碎切片）
    VAD_SIL_TO_SPEECH = 50          # ms - 静音到语音阈值
    VAD_SPEECH_TO_SIL = 80          # ms - 语音到静音阈值

    # ---- VAD 切分引擎（2026-10-03 起默认 Silero v5，fsmn 一键回退）----
    # silero: 婴儿声泛化更好、段间静音阈值放宽（宝宝停顿不再切碎）、CPU ONNX 推理不占 GPU；
    #         段级转写仍由 Paraformer 完成（管线去内嵌 vad_model）
    VAD_ENGINE = os.getenv("VAD_ENGINE", "silero").lower()          # silero | fsmn
    VAD_MIN_SILENCE_MS = int(os.getenv("VAD_MIN_SILENCE_MS", "800"))  # 段间静音阈值（fsmn 为 300ms，宝宝停顿常被切碎）
    VAD_MIN_SPEECH_MS = int(os.getenv("VAD_MIN_SPEECH_MS", "250"))    # 最短语音段
    VAD_MAX_SPEECH_S = int(os.getenv("VAD_MAX_SPEECH_S", "15"))       # 单段上限（与原 fsmn 上限对齐）
    VAD_SPEECH_PAD_MS = int(os.getenv("VAD_SPEECH_PAD_MS", "30"))     # 段两端留白防截字
    # Silero 语音判定阈值（0~1）。默认 0.5 对轻声哄睡讲故事过严——实测妈妈轻声整段被漏切
    # （10-03 21:31 讲故事 1 分钟仅 1 段 vs fsmn 23 段）。降到 0.35 增强轻声/远场灵敏度。
    VAD_SPEECH_THRESHOLD = float(os.getenv("VAD_SPEECH_THRESHOLD", "0.35"))

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
    # Whisper 对比转写已移除（2026-10-04 用户确认效果不佳：远场家庭录音识别质量差，
    # 且每段多一次 Groq API/本地推理拖慢处理；whisper_text 字段保留恒为 null 以兼容历史数据）

    # SenseVoice配置 (情感检测)
    SENSEVOICE_MODEL = "iic/SenseVoiceSmall"
    ENABLE_SENSEVOICE = True  # 是否启用SenseVoice(情感检测+第三转录)

# 文件监控配置 (已迁移至 audio_processor)
FileMonitorConfig = audio_processor.FileMonitorConfig

# =================【 多源路径工具（双 Pixel 双源架构）】=================
# 【2026-10-04 rsync 本地镜像】实时管线主根 = 本地镜像（mirror_sync.sh 每分钟从 NAS 拉取、
# 每5分钟把 processed/failed 推回 NAS）。NAS 根保留用于：历史数据/旧切片回退读取。
RECORDS_ROOT = os.getenv("RECORDS_ROOT", "/Volumes/download/records")
RECORDS_ROOT_NAS = os.getenv("RECORDS_ROOT_NAS", "/Volumes/download/records")
RECORDS_ROOTS = [RECORDS_ROOT]
if os.path.normpath(RECORDS_ROOT_NAS) != os.path.normpath(RECORDS_ROOT):
    RECORDS_ROOTS.append(RECORDS_ROOT_NAS)
SOURCE_DEVICES = [os.path.basename(s.rstrip("/")) for s in FileMonitorConfig.SOURCES]
# Sony 设备保留在兼容候选尾部：历史录音/旧标记仍可能落到 Sony 的 processed/
_LEGACY_DEVICES = [d for d in ("Sony-2", "Sony-1", "Sony-3") if d not in SOURCE_DEVICES]
_DEVICE_PATH_RE = re.compile(r"records/([^/]+)/")


def _device_from_path(p):
    """从 NAS 路径推导设备名（.../records/Pixel-6/2026-10-02/x.m4a → Pixel-6）"""
    m = _DEVICE_PATH_RE.search(str(p).replace("\\", "/"))
    return m.group(1) if m else ""


def _rel_to_records(p):
    """绝对路径 → 相对 records 根的路径（Pixel-6/2026-10-02/x.m4a，含设备级）；
    无法识别时退化为纯文件名"""
    p = str(p).replace("\\", "/")
    i = p.find("/records/")
    if i >= 0:
        return p[i + len("/records/"):]
    return os.path.basename(p)


def _resolve_under_records(p):
    """records 相对路径 → 绝对路径。
    按 RECORDS_ROOTS 优先级逐根尝试：本地镜像（新数据）→ NAS（历史回退）。
    新式（带设备前缀 Pixel-6/...）直接拼；旧式（裸文件名等）先查新源再查 Sony 兼容源。
    找不到返回 None。"""
    p = str(p).lstrip("/").replace("\\", "/")
    if not p or p.startswith(".."):
        return None
    for _root in RECORDS_ROOTS:
        direct = os.path.join(_root, p)
        if os.path.exists(direct):
            return direct
        first = p.split("/", 1)[0]
        # 旧式相对路径（audio_segments/<日期>/... 或纯文件名）逐设备尝试
        if first not in SOURCE_DEVICES and first not in _LEGACY_DEVICES:
            for dev in SOURCE_DEVICES + _LEGACY_DEVICES:
                cand = os.path.join(_root, dev, p)
                if os.path.exists(cand):
                    return cand
    return None


# LLM 配置
class LLMConfig:
    USE_GEMINI_LLM = True
    # 多 Key 轮询支持：优先读 GEMINI_API_KEYS（逗号分隔），兼容旧的 GEMINI_API_KEY
    _keys_str = os.getenv("GEMINI_API_KEYS", "") or os.getenv("GEMINI_API_KEY", "")
    GEMINI_API_KEYS = [k.strip() for k in _keys_str.split(",") if k.strip()]
    GEMINI_API_BASE_URL = os.getenv("GEMINI_API_BASE_URL", "https://generativelanguage.googleapis.com")
    GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-3-flash-preview")
    GEMINI_FALLBACK_MODEL_NAME = os.getenv("GEMINI_FALLBACK_MODEL_NAME", "gemini-3.1-flash-lite-preview")

    # ERNIE 文生图配置 (首选)
    ERNIE_IMAGE_SPACE = os.getenv("ERNIE_IMAGE_SPACE", "baidu/ERNIE-Image-Turbo")
    ERNIE_IMAGE_SIZE = os.getenv("ERNIE_IMAGE_SIZE", "1024x1024")

    # DeepInfra 文生图配置 (备选)
    DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY", "")
    DEEPINFRA_IMAGE_MODEL = os.getenv("DEEPINFRA_IMAGE_MODEL", "black-forest-labs/FLUX-1-schnell")
    DEEPINFRA_IMAGE_SIZE = os.getenv("DEEPINFRA_IMAGE_SIZE", "1024x1024")

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

# Gemini API Key 轮换管理器
class GeminiKeyRotator:
    """多 Key 轮询，遇到 429 自动切换下一个 Key"""
    def __init__(self, keys):
        self._keys = keys
        self._idx = 0
        self._lock = threading.Lock()
        # 记录每个 Key 的 429 冷却时间 {key: cooldown_until_timestamp}
        self._cooldowns = {}

    @property
    def key_count(self):
        return len(self._keys)

    def get_next_key(self):
        """获取下一个可用 Key（跳过冷却中的）"""
        with self._lock:
            if not self._keys:
                return None
            now = time.time()
            # 尝试从当前位置开始找可用的 Key
            for _ in range(len(self._keys)):
                key = self._keys[self._idx]
                self._idx = (self._idx + 1) % len(self._keys)
                cooldown_until = self._cooldowns.get(key, 0)
                if now >= cooldown_until:
                    return key
            # 所有 Key 都在冷却，返回冷却最快结束的
            earliest_key = min(self._keys, key=lambda k: self._cooldowns.get(k, 0))
            return earliest_key

    def mark_rate_limited(self, key, cooldown_seconds=60):
        """标记某个 Key 触发了 429，冷却指定秒数"""
        with self._lock:
            self._cooldowns[key] = time.time() + cooldown_seconds
            logger_sys.warning(f"[KeyRotator] Key ...{key[-6:]} 触发限额，冷却 {cooldown_seconds}s")

    def current_index(self):
        with self._lock:
            return self._idx

_gemini_key_rotator = GeminiKeyRotator(LLMConfig.GEMINI_API_KEYS)

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
        self.clients.discard(client)

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
    # 按大小轮转：单文件 50MB，保留 5 个历史文件
    handler = RotatingFileHandler(
        filename, maxBytes=50*1024*1024, backupCount=5, encoding='utf-8'
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

# =================== 管理接口可选鉴权 ===================
def admin_required(f):
    """可选管理鉴权：.env 配置 ADMIN_TOKEN 后启用，请求需带 X-Admin-Token 头；
    未配置则不拦截（向后兼容）。web_viewer 代理与本地调用方会自动注入该头。

    注意：本装饰器必须定义在所有 @app.route 之前——装饰器在模块加载时即被求值。
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        admin_token = (os.getenv("ADMIN_TOKEN") or "").strip()
        if admin_token and request.headers.get("X-Admin-Token", "") != admin_token:
            logger_sys.warning(f"🚫 管理接口鉴权失败: {request.path} (来自 {request.remote_addr})")
            return jsonify({"error": "Unauthorized: 需要有效的 X-Admin-Token 请求头"}), 401
        return f(*args, **kwargs)
    return wrapper

asr_pipeline = None
sv_pipelines = {}
speaker_db = {}
emotion_pipeline = None  # 可选: 情感检测模型
sensevoice_pipeline = None  # 可选: SenseVoice模型(情感+转录)
gpu_lock = threading.Lock()
db_lock = threading.Lock()

# =================【 LLM 批量处理全局变量 】=================
llm_batch_queue = []
llm_batch_lock = threading.Lock()
llm_last_batch_time = time.time()
llm_cache = OrderedDict()  # 缓存 LLM 响应（LRU 淘汰）
llm_cache_lock = threading.Lock()

# =================【 历史分析锁 】=================
_history_reprocess_lock = threading.Lock()
_history_reprocess_running = False
_history_reprocess_proc = None # 用于存储当前运行的进程对象
_track_b_paused = False # 是否暂停 B 轨实时扫描
_track_b_running = False # B 轨是否正在运行
_monitor_thread = None # B 轨实时监听线程

# B 轨统计状态（由各处理函数更新）
_b_stats = {
    "started_at": None,       # B轨启动时间 ISO格式
    "today_record_count": 0,  # 今日录音数
    "today_cry_count": 0,     # 今日哭声数
    "last_event_time": None,  # 最近事件时间 HH:MM:SS
    "last_cry_time": None,    # 最近哭声时间
    "done_window": [],        # 最近 1 小时各段接收时刻(time.time())——转写速率滑动窗口
}
_b_stats_lock = threading.Lock()

# =================【 系统总览：录音设备健康缓存（后台线程 5 分钟刷新）】=================
# 设计：ffmpeg volumedetect 较重（每台解码最新文件）+ NAS IO 可能挂起，
# 不能跟随前端 30 秒轮询——后台线程每 5 分钟扫一轮，API 只读缓存快照。
# 状态判定对齐 silence_check.sh：全零(<-80dB)红 / 6分钟无新文件红 / 响度偏低(<-50dB)黄 / 间隔偏长(>4分钟)黄
OVERVIEW_DEVICES = ["Pixel-6", "Pixel-5"]  # 2026-10-03 起全面转向双 Pixel，Sony 停录移出总览（历史数据保留可回放）
# 【2026-10-04 rsync 本地镜像】该常量现仅用于 SMB 挂载健康探针（始终指 NAS）；
# 设备录音健康扫描改用 RECORDS_ROOT（本地镜像 watched+processed 双目录取最新文件）
OVERVIEW_RECORDS_ROOT = os.getenv("RECORDS_ROOT_NAS", "/Volumes/download/records")
_overview_cache = {"devices": [], "infra": {}, "updated_at": None,
                   "total_today": None, "prev_total_today": None,  # 今日现存总数及上一轮值→积压趋势
                   "db_stats": None}  # DB 真实入库/哭声统计（进程计数会因重启清零，以此为准）


def _query_db_stats():
    """查 DB 真实统计：近1小时入库、今日入库、今日哭声事件。失败返回 None（沿用上次值）
    注意：created_at 由应用端以 UTC+8 本地时间写入，时间边界必须在 Python 端算好传入，
    不能用 DB 的 NOW()/CURRENT_DATE（连接池会话时区与之不一致，会漂移 8 小时）"""
    try:
        from db_manager import get_connection
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        today_zero = now.replace(hour=0, minute=0, second=0, microsecond=0)
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM transcriptions WHERE created_at >= %s", (hour_ago,))
        ingest_1h = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM transcriptions WHERE created_at >= %s", (today_zero,))
        ingest_today = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM baby_cry_events WHERE created_at >= %s", (today_zero,))
        cry_today = cur.fetchone()[0]
        cur.close()
        conn.close()
        return {"ingest_1h": ingest_1h, "ingest_today": ingest_today, "cry_today": cry_today}
    except Exception as e:
        logger_sys.warning(f"⚠️ 总览 DB 统计查询失败: {e}")
        return None
_overview_cache_lock = threading.Lock()


def _ffmpeg_max_volume(path, timeout=25):
    """返回最新文件的 max_volume（dB，float），失败/超时返回 None"""
    try:
        r = subprocess.run(
            ["ffmpeg", "-i", path, "-af", "volumedetect", "-f", "null", "-"],
            capture_output=True, text=True, timeout=timeout
        )
        m = re.search(r"max_volume:\s*(-?[\d.]+)\s*dB", r.stderr)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def _scan_overview_device(dev, today_str):
    """扫描单台设备今日最新录音（本地镜像 watched+processed 双目录，ffmpeg 解码最新文件，
    必须在带超时的子线程中调用）。
    【2026-10-04 rsync 本地镜像】新录音先落镜像 watched（≤1分钟）→ 处理后移入 processed，
    两目录合并取 mtime 最新者即最新录音；mtime 为 rsync -a 保留的上传时刻，age 语义不变。"""
    info = {"name": dev, "latest_file": None, "latest_time": None, "age_sec": None,
            "max_volume": None, "today_count": 0, "status": "bad", "detail": "今日无录音"}
    dirs = [os.path.join(RECORDS_ROOT, dev, today_str),
            os.path.join(RECORDS_ROOT, dev, FileMonitorConfig.PROCESSED_DIR, today_str)]
    file_items = []  # [(mtime, 绝对路径, 文件名)]
    for ddir in dirs:
        entries = audio_processor._safe_listdir(ddir, timeout=8) or []
        for e in entries:
            if e.endswith(".m4a"):
                p = os.path.join(ddir, e)
                try:
                    file_items.append((os.path.getmtime(p), p, e))
                except OSError:
                    continue
    if not file_items:
        return info
    info["today_count"] = len(file_items)
    file_items.sort(reverse=True)  # mtime 最新在前
    _, latest_path, latest = file_items[0]
    info["latest_file"] = latest
    mt, _, _ = file_items[0]
    info["age_sec"] = int(time.time() - mt)
    info["latest_time"] = datetime.fromtimestamp(mt).strftime("%H:%M")
    vol = _ffmpeg_max_volume(latest_path)
    info["max_volume"] = vol
    age = info["age_sec"]
    if vol is not None and vol < -80:
        info["status"], info["detail"] = "bad", f"全零静音 {vol:.1f}dB（疑未点开 App）"
    elif age is not None and age >= 360:
        info["status"], info["detail"] = "bad", f"{age // 60} 分钟无新录音"
    elif vol is not None and vol < -50:
        info["status"], info["detail"] = "warn", f"响度偏低 {vol:.1f}dB"
    elif age is not None and age >= 240:
        info["status"], info["detail"] = "warn", f"间隔偏长 {age // 60} 分钟"
    else:
        info["status"] = "ok"
        info["detail"] = f"正常 {vol:.1f}dB" if vol is not None else "正常（响度未检出）"
    return info


def _scan_overview_infra():
    """基础设施快照（子线程内执行，全部带硬超时）"""
    infra = {}
    # SMB 挂载：ls 根目录 + 截断检测（<3 项视为异常，对齐 watchdog 规则）
    smb_ok, smb_detail = False, "不可达"
    try:
        r = subprocess.run(["/bin/ls", OVERVIEW_RECORDS_ROOT],
                           capture_output=True, text=True, timeout=8)
        n = len([x for x in r.stdout.splitlines() if x.strip()])
        if r.returncode == 0 and n >= 3:
            smb_ok, smb_detail = True, f"正常（{n} 个设备目录）"
        elif r.returncode == 0:
            smb_detail = f"目录列表异常（仅 {n} 项，疑似截断）"
    except subprocess.TimeoutExpired:
        smb_detail = "目录无响应（挂载假死？）"
    except Exception:
        pass
    infra["smb"] = {"ok": smb_ok, "detail": smb_detail}
    # 5009 面板端口
    try:
        import socket
        with socket.create_connection(("127.0.0.1", 5009), timeout=1):
            infra["panel_5009"] = {"ok": True}
    except Exception:
        infra["panel_5009"] = {"ok": False}
    # 静音巡检：最近一次告警时间（launchd 每 30 分钟跑一次，/tmp 状态文件记录告警时刻）
    last_alert = None
    try:
        last_alert = datetime.fromtimestamp(
            os.path.getmtime("/tmp/silence_check_alert_ts")).strftime("%m-%d %H:%M")
    except Exception:
        pass
    infra["silence_check"] = {"ok": True, "last_alert": last_alert}
    return infra


def _run_overview_scan():
    """完整扫一轮设备 + 基础设施（后台线程调用；单设备套 40s 超时防 NAS 挂起拖死整轮）"""
    today_str = datetime.now().strftime("%Y-%m-%d")
    devices = []
    for dev in OVERVIEW_DEVICES:
        box = {"v": None}

        def _work(dev=dev):
            box["v"] = _scan_overview_device(dev, today_str)

        t = threading.Thread(target=_work, daemon=True)
        t.start()
        t.join(40)
        if t.is_alive() or not isinstance(box["v"], dict):
            logger_sys.warning(f"⚠️ 总览扫描 {dev} 超时，标记扫描超时")
            devices.append({"name": dev, "status": "bad", "detail": "扫描超时（NAS 无响应？）",
                            "latest_file": None, "latest_time": None, "age_sec": None,
                            "max_volume": None, "today_count": 0})
        else:
            devices.append(box["v"])
    infra = _scan_overview_infra()
    db_stats = _query_db_stats() or _overview_cache.get("db_stats")
    total_today = sum(d.get("today_count", 0) for d in devices)
    with _overview_cache_lock:
        _overview_cache["prev_total_today"] = _overview_cache.get("total_today")
        _overview_cache["total_today"] = total_today
        _overview_cache["devices"] = devices
        _overview_cache["infra"] = infra
        _overview_cache["db_stats"] = db_stats
        _overview_cache["updated_at"] = time.time()


def _overview_cache_loop():
    """总览缓存刷新循环：启动缓 15s（让主初始化的 NAS 重活先跑），此后每 300s 一轮"""
    time.sleep(15)
    while True:
        try:
            _run_overview_scan()
        except Exception as e:
            logger_sys.error(f"❌ 系统总览扫描异常: {e}")
        time.sleep(300)

# =================【 轨道A: 独立哭声检测配置 】=================
# 与语音识别参数完全隔离，仅用于原始音频的哭声声纹匹配
class CryDetectionConfig:
    """哭声检测专用参数 - 与语音识别 (Config.SV_MODELS) 完全隔离"""
    ENABLED = True
    MIN_DURATION_SEC = 3            # 最短有效哭声时长 (秒)

    # 声纹阈值 (远低于语音识别的 0.60)
    # 2026-09-18 基于 381 条已知哭声 + 129 条干扰音回放校准 (calibrate_rules.py)
    # 2026-10-06 复核：对当前声纹库重放 12 条标记正样本 + 33 条负样本
    # 发现旧阈值 (0.81/0.72) 对当前 "Baby" 参考向量严重偏严，TPR 仅 1/12；
    # 且 eres2net_large 对正样本最高仅 0.635（永远不投票），2/3 票实际退化为
    # "rdino AND cam++ 同时过阈"。据实测重标：正样本 rdino≥0.678 / cam≥0.575，
    # 负样本 cam≤0.520，故取 0.66/0.55（留安全边际）→ 实测 TPR 8/8、FPR 0/33。
    VOICEPRINT_THRESHOLD = 0.65     # 模型专用阈值未覆盖时的兜底门槛
    VOICEPRINT_GAP = float(os.getenv("CRY_GAP", "0.15"))  # 置信度间隔 (校准结果 gap=0.15 最优)

    # 分模型阈值。全部支持 env 覆盖，便于线上出问题时一键回滚（无需改代码）。
    MODEL_THRESHOLDS = {
        "eres2net_large": float(os.getenv("CRY_TH_ERES2NET", "0.83")),
        "rdino_ecapa": float(os.getenv("CRY_TH_RDINO", "0.66")),
        "camplusplus": float(os.getenv("CRY_TH_CAMPP", "0.55")),
    }

    MIN_VOTES = int(os.getenv("CRY_MIN_VOTES", "2"))

    # 【2026-10-06】历史掉队判定阈值（小时）：录音早于该值的文件按历史处理——不发即时邮件/
    # Webhook/不写实时事件（由补跑链路兜底）。原硬编码 6h 会吃掉手机离线数小时后补传的
    # 真实哭闹；回合合并+冷却已能压制重复告警，故默认放宽到 24h，env 可覆盖便于回滚。
    # 与 audio_processor.FileMonitorConfig.HISTORY_DROP_AGE_HOURS 使用同一环境变量。
    HISTORY_DROP_AGE_HOURS = float(os.getenv("HISTORY_DROP_AGE_HOURS", "24"))
    MIN_AVG_CONFIDENCE = 0.0       # 已停用 (2票制下由分模型阈值把关)
    STRONG_MODEL_SCORE = 0.85      # 仅用于日志统计
    MIN_STRONG_MODELS = 0

    # 目标声纹名 (大小写不敏感)
    TARGET_SPEAKERS = ["baby", "宝宝"]

    # 语音识别用的说话人（在哭声检测时排除，避免干扰）
    VOICE_RECOGNITION_SPEAKERS = ["大可", "妈妈", "婆婆"]

    # 冷却机制
    COOLDOWN_SEC = 600              # 10分钟冷却

_last_cry_trigger_time = 0.0
_cry_cooldown_lock = threading.Lock()
# =========================================================

# =================== 模型加载 ===================
def load_models():
    global asr_pipeline, sv_pipelines, sensevoice_pipeline
    print("\n====== 🚀 启动 SOTA 融合服务 ======")

    load_speaker_db()
    load_negative_samples()
    start_negative_worker()

    # 2. 加载 ASR (FunASR)
    print(f"🧠 加载 ASR: {Config.ASR_MODEL} ...")
    # 2. 加载 ASR (FunASR Paraformer + VAD + 说话人分离)
    print(f"🧠 加载 ASR: {Config.ASR_MODEL} (支持VAD分段和说话人分离) ...")
    _vad_engine = os.getenv("VAD_ENGINE", "silero").lower()
    if _vad_engine == "silero":
        # Silero 模式：管线不带内嵌 VAD，切段由 _generate_with_silero 完成（逐段送转写）
        asr_pipeline = AutoModel(
            model=Config.ASR_MODEL,       # paraformer-zh
            punc_model=Config.PUNC_MODEL, # ct-punc (标点恢复)
            spk_model=Config.SPK_MODEL,   # cam++ (说话人分离)
            device=Config.DEVICE,
            disable_update=True
        )
        print("✅ Paraformer加载完成（VAD引擎=Silero v5，切段在管线外完成）")
    else:
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
        # 【2026-10-04 修复】FunASR 的坑：构造时 vad_kwargs 里的参数被 init 签名捕获存到
        # self.xxx（无人读取），运行时真正生效的是 vad_opts（默认 end_silence=800ms）。
        # 曾导致 Config 的 300/800/1000ms 调整全部无效。此处启动后直写 vad_opts 覆盖。
        try:
            _vad_inner = getattr(asr_pipeline.vad_model, "model", None) or asr_pipeline.vad_model
            _vad_inner.vad_opts.max_end_silence_time = Config.VAD_MAX_END_SILENCE
            _vad_inner.vad_opts.max_single_segment_time = Config.VAD_MAX_SINGLE_SEGMENT
            print(f"✅ VAD参数已直写 vad_opts: end_silence={Config.VAD_MAX_END_SILENCE}ms, "
                  f"max_segment={Config.VAD_MAX_SINGLE_SEGMENT}ms")
        except Exception as _vad_opt_err:
            print(f"⚠️ VAD参数直写失败（将使用 fsmn-vad 默认值 800ms）: {_vad_opt_err}")

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

    # 4. 加载 SenseVoice 模型 (情感检测)
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

def _get_gemini_model_candidates():
    """返回按固定顺序排列的 Gemini 主备模型。"""
    configured_models = [
        ("Gemini 3 Flash", LLMConfig.GEMINI_MODEL_NAME),
        ("Gemini 3.1 Flash Lite", LLMConfig.GEMINI_FALLBACK_MODEL_NAME),
    ]

    candidates = []
    seen = set()
    for display_name, model_name in configured_models:
        model_name = (model_name or "").strip()
        if not model_name or model_name in seen:
            continue
        candidates.append((display_name, model_name))
        seen.add(model_name)
    return candidates


def _extract_gemini_response_text(result):
    """从 Gemini 响应中提取文本内容。"""
    candidates = result.get("candidates") or []
    if not candidates:
        return None, None, "UNKNOWN"

    candidate = candidates[0]
    content = candidate.get("content", {})
    finish_reason = candidate.get("finishReason", "UNKNOWN")

    for part in content.get("parts", []):
        text = part.get("text", "")
        if text:
            return text, candidate, finish_reason

    return None, candidate, finish_reason


def _post_gemini_request_with_fallback(data, timeout, logger_obj, log_prefix):
    """多 Key 轮询 + 多模型降级 + 429 自动退避

    轮询策略：
    1. 从 Key 轮换器获取下一个可用 Key
    2. 遇到 429：标记该 Key 冷却，换下一个 Key 重试
    3. 所有 Key 都 429：等待最短冷却结束后重试
    4. Key 用完仍失败：降级到备用模型
    """
    model_candidates = _get_gemini_model_candidates()
    if not model_candidates:
        logger_obj.error(f"{log_prefix} 未配置任何可用的 Gemini 模型")
        return None

    last_error = None

    for idx, (display_name, model_name) in enumerate(model_candidates):
        url = f"{LLMConfig.GEMINI_API_BASE_URL}/v1beta/models/{model_name}:generateContent"
        logger_obj.info(
            f"{log_prefix} 尝试模型 {idx + 1}/{len(model_candidates)}: {display_name} ({model_name})"
        )

        # 尝试用不同 Key 发送请求
        max_key_attempts = max(_gemini_key_rotator.key_count, 1)
        for key_attempt in range(max_key_attempts):
            api_key = _gemini_key_rotator.get_next_key()
            if not api_key:
                break

            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": api_key
            }
            logger_obj.info(f"{log_prefix} 使用 Key ...{api_key[-6:]} (轮询 {key_attempt + 1}/{max_key_attempts})")

            try:
                response = requests.post(url, headers=headers, json=data, timeout=timeout)

                try:
                    result = response.json()
                except Exception as json_error:
                    last_error = f"返回非 JSON 数据: {json_error}"
                    logger_obj.error(f"{log_prefix} {display_name} 返回非 JSON 数据: {json_error}")
                    result = None

                if result is None:
                    break  # JSON 解析失败，换模型

                if response.ok:
                    response_text, candidate, finish_reason = _extract_gemini_response_text(result)
                    logger_obj.info(f"{log_prefix} {display_name} ✅ API 响应成功 (Key ...{api_key[-6:]})")
                    logger_obj.info(f"{log_prefix} finishReason: {finish_reason}")

                    if response_text:
                        logger_obj.info(f"{log_prefix} 响应文本长度: {len(response_text)} 字符")
                        logger_obj.info(f"{log_prefix} 响应内容预览: {response_text[:500]}...")
                        return response_text

                    if finish_reason == "MAX_TOKENS":
                        logger_obj.warning(f"{log_prefix} 生成 Token 超限，但未提取到有效文本")

                    if candidate and candidate.get("safetyRatings"):
                        logger_obj.warning(f"{log_prefix} 安全评级: {candidate['safetyRatings']}")

                    last_error = f"{display_name} 返回空回复"
                    logger_obj.error(
                        f"{log_prefix} {display_name} 安全拦截或空回复，完整响应: "
                        f"{json.dumps(result, ensure_ascii=False)[:1000]}"
                    )
                    break  # 非 429 错误，换模型

                # 429 / RESOURCE_EXHAUSTED：换 Key 重试
                if response.status_code == 429 or 'RESOURCE_EXHAUSTED' in str(result):
                    _gemini_key_rotator.mark_rate_limited(api_key, cooldown_seconds=60)
                    logger_obj.warning(
                        f"{log_prefix} Key ...{api_key[-6:]} 触发限额 (429)，"
                        f"切换下一个 Key ({key_attempt + 1}/{max_key_attempts})"
                    )
                    last_error = f"HTTP 429 Key...{api_key[-6:]}"
                    continue  # 换下一个 Key

                # 其他 HTTP 错误
                last_error = f"HTTP {response.status_code}"
                logger_obj.error(f"{log_prefix} {display_name} API HTTP 错误 {response.status_code}")
                logger_obj.error(f"{log_prefix} 错误详情: {result}")
                break  # 非 429，换模型

            except Exception as e:
                last_error = str(e)
                logger_obj.error(f"{log_prefix} {display_name} 请求异常：{e}")
                logger_obj.error(f"{log_prefix} 异常堆栈: {traceback.format_exc()}")
                break  # 异常，换模型

        # 所有 Key 都 429 时，等最短冷却后用当前模型再试一次
        if last_error and '429' in last_error and _gemini_key_rotator.key_count > 0:
            logger_obj.warning(f"{log_prefix} 所有 Key 均限额，等待冷却后重试...")
            time.sleep(10)
            api_key = _gemini_key_rotator.get_next_key()
            if api_key:
                headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
                try:
                    response = requests.post(url, headers=headers, json=data, timeout=timeout)
                    if response.ok:
                        result = response.json()
                        retry_text, _, _ = _extract_gemini_response_text(result)
                        if retry_text:
                            logger_obj.info(f"{log_prefix} ✅ 冷却重试成功 (Key ...{api_key[-6:]})")
                            return retry_text
                except Exception:
                    pass

        if idx < len(model_candidates) - 1:
            next_display_name, next_model_name = model_candidates[idx + 1]
            logger_obj.warning(
                f"{log_prefix} {display_name} 调用失败，降级到下一模型: "
                f"{next_display_name} ({next_model_name})"
            )

    logger_obj.error(f"{log_prefix} 所有 Gemini 模型调用均失败，最后错误: {last_error}")
    return None


def call_gemini_api(prompt):
    """调用 Gemini API"""
    if not LLMConfig.USE_GEMINI_LLM:
        return None

    try:
        # 检查缓存（LRU: 命中时移到末尾）
        cache_key = hashlib.md5(prompt.encode()).hexdigest()
        with llm_cache_lock:
            if cache_key in llm_cache:
                llm_cache.move_to_end(cache_key)  # LRU: 标记为最近使用
                logger_sys.info(f"  [LLM] 使用缓存响应")
                return llm_cache[cache_key]

        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            # thinkingLevel low: 3.x 思考模型防思考吃光 500 token 预算, 并加到 800
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 800,
                                 "thinkingConfig": {"thinkingLevel": "low"}}
        }
        text = _post_gemini_request_with_fallback(
            data=data,
            timeout=LLMConfig.LLM_REQUEST_TIMEOUT,
            logger_obj=logger_sys,
            log_prefix="  [LLM]"
        )

        if text:
            with llm_cache_lock:
                if len(llm_cache) >= LLMConfig.LLM_CACHE_SIZE:
                    llm_cache.popitem(last=False)  # LRU: 淘汰最久未使用的
                llm_cache[cache_key] = text

        return text
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
        logger_a.debug(f"  [BabyCry LLM] 请求数据: {data}")

        return _post_gemini_request_with_fallback(
            data=data,
            timeout=LLMConfig.LLM_REQUEST_TIMEOUT + 40,
            logger_obj=logger_a,
            log_prefix="  [BabyCry LLM]"
        )
    except Exception as e:
        logger_a.error(f"  [BabyCry LLM] 发送请求异常：{e}")
        import traceback
        logger_a.error(f"  [BabyCry LLM] 异常堆栈: {traceback.format_exc()}")
        return None

def call_ernie_image_api(prompt):
    """调用百度 ERNIE-Image-Turbo 文生图 API (Gradio HuggingFace Space)"""
    try:
        from gradio_client import Client
        logger_a.info(f"🎨 [ERNIE] 开始调用插图生成 API (Space: {LLMConfig.ERNIE_IMAGE_SPACE})")
        logger_a.info(f"🎨 [ERNIE] Prompt: {prompt[:100]}...")

        # 确保插图目录存在
        _illustration_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "illustrations")
        os.makedirs(_illustration_dir, exist_ok=True)

        # 连接 Gradio Space
        client = Client(LLMConfig.ERNIE_IMAGE_SPACE)

        # 调用生成图片 API
        result = client.predict(
            prompt=prompt,
            size=LLMConfig.ERNIE_IMAGE_SIZE,
            seed=-1,
            use_pe=True,
            api_name="/generate_image",
        )

        logger_a.info(f"🎨 [ERNIE] 响应类型: {type(result)}")

        # Gradio 返回的可能是文件路径或文件对象
        img_path_result = None
        if isinstance(result, str):
            img_path_result = result
        elif isinstance(result, dict):
            img_path_result = result.get("value") or result.get("path")
        elif isinstance(result, (list, tuple)):
            img_path_result = result[0] if result else None

        if not img_path_result:
            logger_a.error(f"🎨 [ERNIE] 无法解析返回的图片数据")
            return None

        # 如果是本地临时文件路径，复制到插图目录
        if os.path.exists(img_path_result):
            img_filename = f"{uuid.uuid4().hex[:12]}.jpg"
            img_path = os.path.join(_illustration_dir, img_filename)
            import shutil
            shutil.copy2(img_path_result, img_path)
            logger_a.info(f"🎨 [ERNIE] 图片已保存到: {img_path} ({os.path.getsize(img_path)} bytes)")
            return f"/api/illustration/{img_filename}"
        else:
            logger_a.error(f"🎨 [ERNIE] 返回的文件路径不存在: {img_path_result}")
            return None

    except ImportError:
        logger_a.warning("🎨 [ERNIE] gradio_client 未安装，跳过")
        return None
    except Exception as e:
        logger_a.error(f"🎨 [ERNIE] 发送请求异常：{e}")
        import traceback
        logger_a.error(f"🎨 [ERNIE] 异常堆栈：{traceback.format_exc()}")
        return None

def call_volcengine_ai(prompt):
    """调用 DeepInfra (FLUX-1-schnell) 文生图 API"""
    try:
        logger_a.info(f"🎨 [DeepInfra] 开始调用插图生成 API")
        logger_a.info(f"🎨 [DeepInfra] Prompt: {prompt[:100]}...")

        api_key = LLMConfig.DEEPINFRA_API_KEY
        model = LLMConfig.DEEPINFRA_IMAGE_MODEL
        size = LLMConfig.DEEPINFRA_IMAGE_SIZE

        if not api_key:
            logger_a.warning("🎨 [DeepInfra] API Key 未配置，跳过")
            return None

        # 确保插图目录存在
        _illustration_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "illustrations")
        os.makedirs(_illustration_dir, exist_ok=True)

        # 调用 DeepInfra OpenAI 兼容接口
        url = "https://api.deepinfra.com/v1/openai/images/generations"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "prompt": prompt,
            "size": size,
            "model": model,
            "n": 1
        }

        logger_a.info(f"🎨 [DeepInfra] 发送请求到 {url} (model: {model})")
        response = requests.post(url, headers=headers, json=payload, timeout=120)

        logger_a.info(f"🎨 [DeepInfra] HTTP 状态码：{response.status_code}")

        if response.status_code != 200:
            logger_a.error(f"🎨 [DeepInfra] 请求失败：{response.status_code}")
            logger_a.error(f"🎨 [DeepInfra] 响应：{response.text[:500]}")
            return None

        try:
            result = response.json()
        except:
            logger_a.error(f"🎨 [DeepInfra] 返回非 JSON 数据：{response.text[:200]}")
            return None

        logger_a.info(f"🎨 [DeepInfra] 响应：{json.dumps(result, ensure_ascii=False)[:200]}...")

        # 解析返回的图片数据（DeepInfra 返回 b64_json 而非 url）
        data_list = result.get("data", [])
        if not data_list:
            logger_a.error(f"🎨 [DeepInfra] 响应中没有图片数据")
            return None

        img_data_item = data_list[0]
        img_url = img_data_item.get("url")
        b64_json = img_data_item.get("b64_json")

        # 确保插图目录存在
        _illustration_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "illustrations")
        os.makedirs(_illustration_dir, exist_ok=True)

        img_filename = f"{uuid.uuid4().hex[:12]}.jpg"
        img_path = os.path.join(_illustration_dir, img_filename)

        if b64_json:
            # 方式 1：直接使用 base64 数据
            import base64
            img_bytes = base64.b64decode(b64_json)
            with open(img_path, 'wb') as f:
                f.write(img_bytes)
            logger_a.info(f"🎨 [DeepInfra] 图片已保存到本地 (b64): {img_path} ({len(img_bytes)} bytes)")
            return f"/api/illustration/{img_filename}"
        elif img_url:
            # 方式 2：下载 URL
            logger_a.info(f"🎨 [DeepInfra] ✅ 图片 URL: {img_url[:100]}...")
            img_response = requests.get(img_url, timeout=60)
            if img_response.status_code == 200:
                with open(img_path, 'wb') as f:
                    f.write(img_response.content)
                logger_a.info(f"🎨 [DeepInfra] 图片已保存到本地 (url): {img_path} ({len(img_response.content)} bytes)")
                return f"/api/illustration/{img_filename}"
            else:
                logger_a.error(f"🎨 [DeepInfra] 下载图片失败：{img_response.status_code}")
                return None
        else:
            logger_a.error(f"🎨 [DeepInfra] 响应中既没有 URL 也没有 b64_json")
            return None

    except Exception as e:
        logger_a.error(f"🎨 [DeepInfra] 发送请求异常：{e}")
        import traceback
        logger_a.error(f"🎨 [DeepInfra] 异常堆栈：{traceback.format_exc()}")
        return None

def call_z_image_turbo_api(prompt, space_name="mrfakename/Z-Image-Turbo"):
    """调用 Z-Image-Turbo API (HuggingFace Spaces)"""
    try:
        logger_a.info(f"🎨 [Z-Image-Turbo:{space_name}] 开始生成插图")
        logger_a.info(f"🎨 [Z-Image-Turbo:{space_name}] Prompt 长度：{len(prompt)} 字符")

        from gradio_client import Client
        
        # 连接到 Z-Image-Turbo
        logger_a.info(f"🎨 [Z-Image-Turbo:{space_name}] 正在连接 API...")
        client = Client(space_name)
        
        logger_a.info(f"🎨 [Z-Image-Turbo:{space_name}] 连接成功，开始生成...")
        result = client.predict(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=9,
            seed=None,  # 使用随机种子
            randomize_seed=True,
            api_name="/generate_image"
        )
        
        if not result or not isinstance(result, tuple) or len(result) == 0:
            logger_a.warning(f"🎨 [Z-Image-Turbo:{space_name}] API 返回空结果")
            return None
        
        image_path = result[0]
        seed_used = result[1] if len(result) > 1 else "unknown"
        
        logger_a.info(f"🎨 [Z-Image-Turbo:{space_name}] ✅ 生成成功！Seed={seed_used}")
        logger_a.info(f"🎨 [Z-Image-Turbo:{space_name}] 临时路径：{image_path}")
        
        # 确保插图目录存在
        _illustration_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "illustrations")
        os.makedirs(_illustration_dir, exist_ok=True)
        
        # 复制文件到插图目录
        img_filename = f"z_image_{space_name.replace('/', '_')}_{uuid.uuid4().hex[:12]}.png"
        img_path = os.path.join(_illustration_dir, img_filename)
        
        shutil.copy2(image_path, img_path)
        logger_a.info(f"🎨 [Z-Image-Turbo:{space_name}] 图片已保存到：{img_path}")
        
        return f"/api/illustration/{img_filename}"
        
    except Exception as e:
        logger_a.error(f"🎨 [Z-Image-Turbo:{space_name}] 发送请求异常：{e}")
        import traceback
        logger_a.error(f"🎨 [Z-Image-Turbo:{space_name}] 异常堆栈：{traceback.format_exc()}")
        return None

def call_flux2_klein_api(prompt):
    """调用 FLUX.2-klein-9B API (HuggingFace Spaces)"""
    try:
        logger_a.info(f"🎨 [FLUX.2-klein-9B] 开始生成插图")
        logger_a.info(f"🎨 [FLUX.2-klein-9B] Prompt 长度：{len(prompt)} 字符")

        from gradio_client import Client, handle_file
        
        # 连接到 FLUX.2-klein-9B
        logger_a.info(f"🎨 [FLUX.2-klein-9B] 正在连接 API...")
        client = Client("black-forest-labs/FLUX.2-klein-9B")
        
        logger_a.info(f"🎨 [FLUX.2-klein-9B] 连接成功，开始生成...")
        result = client.predict(
            prompt=prompt,
            input_images=[],
            mode_choice="Distilled (4 steps)",
            seed=0,
            randomize_seed=True,
            width=1024,
            height=1024,
            num_inference_steps=4,
            guidance_scale=1,
            prompt_upsampling=False,
            api_name="/generate"
        )
        
        if not result or not isinstance(result, tuple) or len(result) == 0:
            logger_a.warning(f"🎨 [FLUX.2-klein-9B] API 返回空结果")
            return None
        
        image_path = result[0]
        seed_used = result[1] if len(result) > 1 else "unknown"
        
        logger_a.info(f"🎨 [FLUX.2-klein-9B] ✅ 生成成功！Seed={seed_used}")
        logger_a.info(f"🎨 [FLUX.2-klein-9B] 临时路径：{image_path}")
        
        # 确保插图目录存在
        _illustration_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "illustrations")
        os.makedirs(_illustration_dir, exist_ok=True)
        
        # 复制文件到插图目录
        img_filename = f"flux2_klein_{uuid.uuid4().hex[:12]}.webp"
        img_path = os.path.join(_illustration_dir, img_filename)
        
        shutil.copy2(image_path, img_path)
        logger_a.info(f"🎨 [FLUX.2-klein-9B] 图片已保存到：{img_path}")
        
        return f"/api/illustration/{img_filename}"
        
    except Exception as e:
        logger_a.error(f"🎨 [FLUX.2-klein-9B] 发送请求异常：{e}")
        import traceback
        logger_a.error(f"🎨 [FLUX.2-klein-9B] 异常堆栈：{traceback.format_exc()}")
        return None

def _save_illustration_bytes(raw_bytes, prefix):
    """把图片字节保存到插图目录，返回 API 相对路径；按文件头自动判定扩展名"""
    _illustration_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "illustrations")
    os.makedirs(_illustration_dir, exist_ok=True)
    ext = ".png"
    if raw_bytes[:3] == b"\xff\xd8\xff":
        ext = ".jpg"
    img_filename = f"{prefix}_{uuid.uuid4().hex[:12]}{ext}"
    img_path = os.path.join(_illustration_dir, img_filename)
    with open(img_path, "wb") as f:
        f.write(raw_bytes)
    logger_a.info(f"🎨 [{prefix}] 图片已保存：{img_path}")
    return f"/api/illustration/{img_filename}"


def call_newapi_image_api(prompt, model=None, prefix="agnes"):
    """NAS new-api 网关（OpenAI 兼容图片接口）；model 缺省取 IMAGE_NEWAPI_MODEL"""
    base_url = os.getenv("IMAGE_NEWAPI_BASE_URL", "http://192.168.1.188:3008/v1").rstrip("/")
    api_key = os.getenv("IMAGE_NEWAPI_API_KEY", "")
    if model is None:
        model = os.getenv("IMAGE_NEWAPI_MODEL", "agnes-image-2.5-flash")
    if not api_key:
        logger_a.warning("🎨 [new-api] 未配置 IMAGE_NEWAPI_API_KEY，跳过")
        return None
    try:
        logger_a.info(f"🎨 [new-api:{model}] 开始生成插图")
        resp = requests.post(
            f"{base_url}/images/generations",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "prompt": prompt, "n": 1, "size": "1024x1024"},
            timeout=120,
        )
        resp.raise_for_status()
        arr = (resp.json() or {}).get("data") or []
        if not arr:
            logger_a.warning(f"🎨 [new-api:{model}] 返回无数据")
            return None
        item = arr[0]
        if item.get("b64_json"):
            import base64 as _b64
            return _save_illustration_bytes(_b64.b64decode(item["b64_json"]), prefix)
        if item.get("url"):
            img_resp = requests.get(item["url"], timeout=60)
            img_resp.raise_for_status()
            return _save_illustration_bytes(img_resp.content, prefix)
        logger_a.warning(f"🎨 [new-api:{model}] 数据项缺少 b64_json/url")
        return None
    except Exception as e:
        logger_a.error(f"🎨 [new-api:{model}] 生成失败：{e}")
        return None


def call_cpa_gemini_image_api(prompt):
    """备选：NAS CPA (CLIProxyAPI) 的 gemini-3.1-flash-image，走 chat 接口返回 base64 图片"""
    base_url = os.getenv("IMAGE_CPA_BASE_URL", "http://192.168.1.188:8317/v1").rstrip("/")
    api_key = os.getenv("IMAGE_CPA_API_KEY", "")
    model = os.getenv("IMAGE_CPA_MODEL", "gemini-3.1-flash-image")
    if not api_key:
        logger_a.warning("🎨 [CPA] 未配置 IMAGE_CPA_API_KEY，跳过")
        return None
    try:
        logger_a.info(f"🎨 [CPA:{model}] 开始生成插图")
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}]},
            timeout=120,
        )
        resp.raise_for_status()
        message = ((resp.json() or {}).get("choices") or [{}])[0].get("message", {})
        images = message.get("images") or []
        if not images:
            logger_a.warning(f"🎨 [CPA:{model}] 响应中无图片")
            return None
        url = images[0].get("image_url", {}).get("url", "")
        if not url.startswith("data:"):
            logger_a.warning(f"🎨 [CPA:{model}] 图片格式非 data URI")
            return None
        import base64 as _b64
        b64_data = url.split(",", 1)[1]
        return _save_illustration_bytes(_b64.b64decode(b64_data), "cpa_gemini")
    except Exception as e:
        logger_a.error(f"🎨 [CPA:{model}] 生成失败：{e}")
        return None


def call_grok2api_image_api(prompt):
    """备选：NAS grok2api 的 grok-imagine-image-2.0（OpenAI 图片接口，返回的容器内 url 需重写为外部地址）"""
    base_url = os.getenv("IMAGE_GROK2API_BASE_URL", "http://192.168.1.188:12323/v1").rstrip("/")
    api_key = os.getenv("IMAGE_GROK2API_API_KEY", "")
    model = os.getenv("IMAGE_GROK2API_MODEL", "grok-imagine-image-2.0")
    if not api_key:
        logger_a.warning("🎨 [grok2api] 未配置 IMAGE_GROK2API_API_KEY，跳过")
        return None
    try:
        logger_a.info(f"🎨 [grok2api:{model}] 开始生成插图")
        resp = requests.post(
            f"{base_url}/images/generations",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "prompt": prompt, "n": 1},
            timeout=120,
        )
        resp.raise_for_status()
        arr = (resp.json() or {}).get("data") or []
        if not arr:
            logger_a.warning(f"🎨 [grok2api:{model}] 返回无数据")
            return None
        item = arr[0]
        import base64 as _b64
        if item.get("b64_json"):
            return _save_illustration_bytes(_b64.b64decode(item["b64_json"]), "grok")
        url = item.get("url", "")
        if not url:
            logger_a.warning(f"🎨 [grok2api:{model}] 数据项缺少 b64_json/url")
            return None
        # 容器内返回的 url 指向 127.0.0.1:8000，重写为 NAS 外部地址
        from urllib.parse import urlparse
        parsed = urlparse(base_url)
        fixed_url = url.replace("http://127.0.0.1:8000", f"{parsed.scheme}://{parsed.netloc}")
        img_resp = requests.get(fixed_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=60)
        img_resp.raise_for_status()
        return _save_illustration_bytes(img_resp.content, "grok")
    except Exception as e:
        logger_a.error(f"🎨 [grok2api:{model}] 生成失败：{e}")
        return None


def call_gemini_image_api(prompt):
    """调用文生图 API (首选 grok2api grok-imagine-image-2.0，备选 NAS new-api agnes-image-2.5-flash → CPA gemini-3.1-flash-image，再回退 HF Space 链: mrfakename/Z-Image-Turbo → laruss5 → FLUX.2-klein-9B → 百度 ERNIE → DeepInfra)"""
    if not LLMConfig.USE_GEMINI_LLM:
        logger_a.warning("🎨 [插图生成] LLM 未启用，跳过")
        return None

    try:
        logger_a.info(f"🎨 [插图生成] 开始生成插图")
        logger_a.info(f"🎨 [插图生成] Prompt 长度：{len(prompt)} 字符")

        # 首选：NAS grok2api 的 grok-imagine-image-2.0（2026-10-07 用户指定提到第一）
        logger_a.info(f"🎨 [插图生成] 尝试使用 Grok ({os.getenv('IMAGE_GROK2API_MODEL', 'grok-imagine-image-2.0')})...")
        image_data = call_grok2api_image_api(prompt)

        if image_data:
            logger_a.info(f"🎨 [插图生成] ✅ Grok 成功!")
            return image_data

        logger_a.warning(f"🎨 [插图生成] Grok 失败，回退到 new-api...")

        # 次选：NAS new-api 网关（agnes-image-2.5-flash，OpenAI 兼容接口）
        logger_a.info(f"🎨 [插图生成] 尝试使用 new-api ({os.getenv('IMAGE_NEWAPI_MODEL', 'agnes-image-2.5-flash')})...")
        image_data = call_newapi_image_api(prompt)

        if image_data:
            logger_a.info(f"🎨 [插图生成] ✅ new-api 成功!")
            return image_data

        logger_a.warning(f"🎨 [插图生成] new-api 失败，回退到 CPA Gemini...")

        # 备选：NAS CPA (CLIProxyAPI) 的 gemini-3.1-flash-image
        image_data = call_cpa_gemini_image_api(prompt)

        if image_data:
            logger_a.info(f"🎨 [插图生成] ✅ CPA Gemini 成功!")
            return image_data

        logger_a.warning(f"🎨 [插图生成] CPA Gemini 失败，回退到 mrfakename/Z-Image-Turbo...")

        # 首选：mrfakename/Z-Image-Turbo (HuggingFace Spaces)
        logger_a.info(f"🎨 [插图生成] 尝试使用 mrfakename/Z-Image-Turbo...")
        image_data = call_z_image_turbo_api(prompt, "mrfakename/Z-Image-Turbo")

        if image_data:
            logger_a.info(f"🎨 [插图生成] ✅ mrfakename/Z-Image-Turbo 成功!")
            return image_data

        logger_a.warning(f"🎨 [插图生成] mrfakename/Z-Image-Turbo 失败，回退到 laruss5/Z-Image-Turbo...")

        # 备选：laruss5/Z-Image-Turbo
        logger_a.info(f"🎨 [插图生成] 尝试使用 laruss5/Z-Image-Turbo...")
        image_data = call_z_image_turbo_api(prompt, "laruss5/Z-Image-Turbo")

        if image_data:
            logger_a.info(f"🎨 [插图生成] ✅ laruss5/Z-Image-Turbo 成功!")
            return image_data

        logger_a.warning(f"🎨 [插图生成] laruss5/Z-Image-Turbo 失败，回退到 FLUX.2-klein-9B...")

        # 备选：black-forest-labs/FLUX.2-klein-9B
        logger_a.info(f"🎨 [插图生成] 尝试使用 black-forest-labs/FLUX.2-klein-9B...")
        image_data = call_flux2_klein_api(prompt)

        if image_data:
            logger_a.info(f"🎨 [插图生成] ✅ FLUX.2-klein-9B 成功!")
            return image_data

        logger_a.warning(f"🎨 [插图生成] FLUX.2-klein-9B 失败，回退到百度 ERNIE...")

        # 备选：百度 ERNIE-Image-Turbo
        logger_a.info(f"🎨 [插图生成] 尝试使用百度 ERNIE-Image-Turbo...")
        image_data = call_ernie_image_api(prompt)

        if image_data:
            logger_a.info(f"🎨 [插图生成] ✅ ERNIE 成功!")
            return image_data

        logger_a.warning(f"🎨 [插图生成] ERNIE 失败，回退到 DeepInfra...")

        # 备选：DeepInfra
        logger_a.info(f"🎨 [插图生成] 使用 DeepInfra (FLUX-1-schnell) 文生图...")
        image_data = call_volcengine_ai(prompt)

        if image_data:
            logger_a.info(f"🎨 [插图生成] ✅ DeepInfra 成功!")
            return image_data
        else:
            logger_a.warning(f"🎨 [插图生成] DeepInfra 也失败了，返回 None")
            return None

    except Exception as e:
        logger_a.error(f"🎨 [插图生成] 发送请求异常：{e}")
        import traceback
        logger_a.error(f"🎨 [插图生成] 异常堆栈：{traceback.format_exc()}")
        return None

def process_baby_cry_async(filename, audio_path, start_time, end_time, placeholder_id=None, cry_conf=None, cry_det=None):
    """异步处理宝宝哭声分析，支持加载前后5分钟录音作为上下文"""
    import time, re, json
    time.sleep(1) # 等待文件落盘
    if not os.path.exists(audio_path):
        logger_a.warning(f"👶 [BabyCry] 音频文件不存在: {audio_path}")
        return None, None, None

    logger_a.info(f"👶 [BabyCry] 开始收集上下文音频并发送分析... ({start_time}ms - {end_time}ms)")

    # 搜集前后 5 分钟 (300秒) 的同目录相关录音
    audio_paths_to_send = []

    from db_manager import parse_recording_time
    record_dt = parse_recording_time(filename)

    # 主文件（哭声文件）
    audio_paths_to_send.append(audio_path)

    if record_dt:
        # 【双源】优先扫描 audio_path 归属设备的 processed/，无命中再依次尝试其余设备
        # 【2026-10-04 多根】本地镜像优先，NAS 历史回退；(设备,文件名) 去重防双根重复注入
        _dev = _device_from_path(audio_path) or (SOURCE_DEVICES[0] if SOURCE_DEVICES else "")
        _dev_order = ([_dev] if _dev else []) + [d for d in SOURCE_DEVICES + _LEGACY_DEVICES if d != _dev]
        _seen_ctx = set()
        _ctx_found = False
        for _scan_dev in _dev_order:
            if _ctx_found:
                break
            for _root in RECORDS_ROOTS:
                date_dir = os.path.join(_root, _scan_dev, FileMonitorConfig.PROCESSED_DIR,
                                        record_dt.strftime("%Y-%m-%d"))
                if not os.path.exists(date_dir):
                    continue
                candidates = []
                for f in os.listdir(date_dir):
                    if f.endswith(tuple(FileMonitorConfig.SUPPORTED_FORMATS)):
                        f_dt = parse_recording_time(f)
                        if f_dt:
                            f_path = os.path.join(date_dir, f)
                            time_diff = (f_dt - record_dt).total_seconds()
                            # 前后5分钟（300秒）
                            if -300 <= time_diff <= 300:
                                candidates.append((time_diff, f_path))

                # 按时间排序：先前的文件 → 主文件 → 后来的文件
                candidates.sort(key=lambda x: x[0])
                for time_diff, f_path in candidates:
                    _key = (_scan_dev, os.path.basename(f_path))
                    if _key in _seen_ctx:
                        continue
                    if f_path not in audio_paths_to_send and os.path.exists(f_path):
                        _seen_ctx.add(_key)
                        audio_paths_to_send.append(f_path)

                logger_a.info(f"👶 [BabyCry] 在 {date_dir} 中找到 {len(candidates)} 个上下文文件")
                if candidates:
                    _ctx_found = True
                    break  # 优先设备命中即止，避免跨设备重复注入上下文

    context_len = len(audio_paths_to_send) - 1
    logger_a.info(f"👶 [BabyCry] 收集完毕，共附带 {context_len} 个相邻时段记录作为多模态上下文 (总计 {len(audio_paths_to_send)} 个文件)")

    prompt = "以下是多段连续的录音（时间顺序排列），其中包含了两岁半宝宝的哭泣声（位于中间的某段）。请结合完整的上下文音频（前后高达5分钟的情境），综合推理宝宝在这段时间哭泣的真正原因（如困倦Sleepy、饥饿Hungry、情绪发泄Frustration、疼痛Pain、要求未被满足等），并给出针对此时情境的安抚建议。请严格按如下JSON格式返回：{\"category\": \"核心原因简短分类(如：困倦/饥饿/疼痛/情绪等)\", \"reason\": \"结合上下文的深度分析原因\", \"advice\": \"针对此时情境的安抚建议\"}"
    response_text = call_gemini_audio_api(audio_paths_to_send, prompt)

    if not response_text:
        logger_a.warning(f"👶 [BabyCry] Gemini Audio API 返回空结果")
        return None, None, None

    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            category = result.get("category", "未知")
            reason = result.get("reason", "未知")
            advice = result.get("advice", "无")
            from db_manager import save_cry_analysis, update_cry_analysis

            # 转换 absolute path 为相对 records 根的路径（含设备级），以便前端使用
            rel_audio_path = "/" + _rel_to_records(audio_path)

            if placeholder_id:
                update_cry_analysis(placeholder_id, reason, advice,
                                  reason_category=category, event_files=audio_paths_to_send,
                                  confidence=cry_conf, details=cry_det)
            else:
                save_cry_analysis(filename, start_time/1000.0, end_time/1000.0, reason, advice,
                                  reason_category=category, event_files=audio_paths_to_send,
                                  audio_path=rel_audio_path)
            logger_a.info(f"👶 [宝宝哭声深度分析] 分类: {category}, 原因: {reason[:50]}..., 路径: {rel_audio_path}")
            return reason, advice, category
    except Exception as e:
        logger_a.error(f"  [BabyCry 解析错误] {e}")
    return None, None, None

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
    global llm_last_batch_time, llm_batch_queue

    with llm_batch_lock:
        # 防止队列无限增长，超过上限时丢弃最旧的条目
        LLM_QUEUE_MAX_SIZE = LLMConfig.LLM_BATCH_SIZE * 5
        if len(llm_batch_queue) >= LLM_QUEUE_MAX_SIZE:
            dropped_count = len(llm_batch_queue) - LLM_QUEUE_MAX_SIZE + 1
            del llm_batch_queue[:dropped_count]
            logger_b.warning(f"  [LLM队列] 队列已满({LLM_QUEUE_MAX_SIZE})，丢弃 {dropped_count} 条旧记录")

        llm_batch_queue.append({
            'filename': filename,
            'full_text': full_text,
            'segments': segments
        })

        queue_size = len(llm_batch_queue)
        time_since_last = time.time() - llm_last_batch_time

        logger_b.info(f"  [LLM队列] 已添加，当前队列: {queue_size}/{LLM_QUEUE_MAX_SIZE}")

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

# =================【 未完成哭声分析自动重试 】=================
RETRY_INCOMPLETE_INTERVAL = 3600  # 每1小时检查一次未完成的分析

def retry_incomplete_cry_analyses():
    """定时扫描并重试未完成的哭声深度分析

    场景：检测到哭声后走 Gemini 分析但失败了（网络超时/API限额等），
    数据库中 category 仍为 analyzing/未分类/未知，需要自动重试。
    """
    try:
        from db_manager import get_incomplete_cry_events, update_cry_analysis

        events = get_incomplete_cry_events()
        if not events:
            logger_a.debug("[重试] 无未完成的哭声分析")
            return

        current_time = time.time()
        retried = 0
        skipped = 0

        for event in events:
            event_id = event.get('id')
            filename = event.get('filename', '')
            category = event.get('reason_category', '')
            audio_path = event.get('audio_path', '')
            event_files = event.get('event_files', [])
            recording_time = event.get('recording_time', '')

            # 不过滤时间，重试所有未完成事件

            # 确定音频文件路径
            # 优先用 event_files（多文件上下文），其次用 audio_path（单文件）
            valid_paths = []
            if event_files:
                for p in event_files:
                    # 兼容相对路径和绝对路径（双源：相对路径按设备前缀/多设备解析）
                    abs_p = p if os.path.isabs(p) else _resolve_under_records(p)
                    if abs_p and os.path.exists(abs_p):
                        valid_paths.append(abs_p)

            # 单文件兜底
            if not valid_paths and audio_path:
                abs_audio = audio_path if os.path.isabs(audio_path) else _resolve_under_records(audio_path)
                if abs_audio and os.path.exists(abs_audio):
                    valid_paths = [abs_audio]

            # 兼容旧的 B 轨占位记录：保存后文件会重命名为 cry_<event_id>_*.wav，
            # 但旧记录的 audio_path 可能还指向重命名前的路径。
            if not valid_paths and event_id:
                cry_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_cry")
                candidates = sorted(glob.glob(os.path.join(cry_dir, f"cry_{event_id}_*.wav")))
                for candidate in candidates:
                    if os.path.exists(candidate):
                        valid_paths = [candidate]
                        logger_a.info(f"[重试] ID={event_id} 通过事件ID找到持久音频: {candidate}")
                        try:
                            from db_manager import update_cry_event_audio_path
                            update_cry_event_audio_path(event_id, candidate)
                        except Exception as path_update_err:
                            logger_a.warning(f"[重试] ID={event_id} 更新持久音频路径失败: {path_update_err}")
                        break

            if not valid_paths:
                logger_a.debug(f"[重试] ID={event_id} 音频文件均不存在，跳过")
                skipped += 1
                continue

            # A轨正在运行时跳过（避免并发API调用冲突）
            if _history_reprocess_running:
                logger_a.info("[重试] A轨任务运行中，跳过本轮重试")
                break

            logger_a.info(f"🔄 [重试] 重新分析 ID={event_id}, category={category}, 文件数={len(valid_paths)}")

            try:
                # 复用 Gemini 分析逻辑
                if len(valid_paths) >= 1:
                    prompt = (
                        "以下是多段连续的录音（时间顺序排列），其中包含了两岁半宝宝的哭泣声。"
                        "请结合完整的上下文音频，综合推理宝宝在这段时间哭泣的真正原因"
                        "（如困倦Sleepy、饥饿Hungry、情绪发泄Frustration、疼痛Pain、要求未被满足等），"
                        "并给出针对此时情境的安抚建议。"
                        "请严格按如下JSON格式返回：{\"category\": \"核心原因简短分类(如：困倦/饥饿/疼痛/情绪等)\", \"reason\": \"结合上下文的深度分析原因\", \"advice\": \"针对此时情境的安抚建议\"}"
                    )
                    response_text = call_gemini_audio_api(valid_paths, prompt)

                    if response_text:
                        import re as _re, json as _json
                        cleaned = response_text.strip()
                        if cleaned.startswith('```json'):
                            cleaned = cleaned[7:]
                        elif cleaned.startswith('```'):
                            cleaned = cleaned[3:]
                        if cleaned.endswith('```'):
                            cleaned = cleaned[:-3]
                        cleaned = cleaned.strip()

                        parse_ok = False
                        result = {}
                        try:
                            result = _json.loads(cleaned, strict=False)
                            parse_ok = True
                        except Exception:
                            match = _re.search(r'\{.*\}', cleaned, _re.DOTALL)
                            if match:
                                try:
                                    result = _json.loads(match.group(), strict=False)
                                    parse_ok = True
                                except Exception:
                                    pass

                        if parse_ok:
                            new_category = result.get("category", "未知")
                            new_reason = result.get("reason", "未知")
                            new_advice = result.get("advice", "无")

                            update_cry_analysis(
                                event_id, new_reason, new_advice,
                                reason_category=new_category,
                                event_files=valid_paths,
                                confidence=event.get('confidence'),
                                details=None
                            )
                            retried += 1
                            logger_a.info(f"🔄 [重试] ID={event_id} 分析完成: category={new_category}")
                        else:
                            logger_a.warning(f"🔄 [重试] ID={event_id} Gemini返回解析失败")
                    else:
                        logger_a.warning(f"🔄 [重试] ID={event_id} Gemini返回空结果")

                # 事件间延迟，避免API限流
                time.sleep(5)

            except Exception as e:
                logger_a.error(f"🔄 [重试] ID={event_id} 重试异常: {e}")

        if retried > 0 or skipped > 0:
            logger_a.info(f"🔄 [重试] 本轮完成: 成功{retried}条, 跳过{skipped}条, 总计{len(events)}条")

    except Exception as e:
        logger_a.error(f"🔄 [重试] 自动重试异常: {e}")
    finally:
        # 调度下一轮
        threading.Timer(RETRY_INCOMPLETE_INTERVAL, retry_incomplete_cry_analyses).start()

# =========================================================

def cleanup_temp_dir():
    """清理超过指定时间的临时文件（覆盖多个临时目录）"""
    # 需要清理的目录列表及其最大保留时间（秒）
    cleanup_dirs = {
        Config.TEMP_DIR: 3600,            # temp/ - 1小时
        "temp_cry": 86400,                 # temp_cry/ - 24小时（保留给未完成哭声事件重试）
        # illustrations/ 已从清理列表中移除 - 插图是永久文件，不应被清理
        Config.LONG_SENTENCES_DIR: 604800,  # long_sentences/ - 7天
    }

    try:
        current_time = time.time()
        total_cleaned = 0

        protected_temp_cry_files = set()
        try:
            from db_manager import get_incomplete_cry_events
            for event in get_incomplete_cry_events():
                audio_path = event.get('audio_path')
                if audio_path:
                    protected_temp_cry_files.add(os.path.abspath(audio_path))
                for event_file in event.get('event_files', []) or []:
                    if event_file:
                        protected_temp_cry_files.add(os.path.abspath(event_file))
        except Exception as e:
            logger_sys.warning(f"读取未完成哭声事件保护列表失败: {e}")

        for dir_name, max_age in cleanup_dirs.items():
            if not os.path.exists(dir_name):
                continue

            cleaned_count = 0

            try:
                for filename in os.listdir(dir_name):
                    filepath = os.path.join(dir_name, filename)
                    try:
                        # 预切哭声候选片段缓存(temp/preview_segments/)是"零GPU秒开预览"的核心，
                        # 按事件ID幂等复用且体积小，曾被1小时清理误删导致每次预览都重新GPU定位——永久保留
                        if dir_name == Config.TEMP_DIR and filename == "preview_segments":
                            continue
                        if os.path.isfile(filepath):
                            file_age = current_time - os.path.getmtime(filepath)
                            if (
                                dir_name == "temp_cry"
                                and os.path.abspath(filepath) in protected_temp_cry_files
                            ):
                                continue
                            if file_age > max_age:
                                os.remove(filepath)
                                cleaned_count += 1
                                logger_sys.debug(f"清理旧临时文件: {dir_name}/{filename}")
                        elif os.path.isdir(filepath):
                            dir_age = current_time - os.path.getmtime(filepath)
                            if dir_age > max_age:
                                shutil.rmtree(filepath, ignore_errors=True)
                                cleaned_count += 1
                                logger_sys.debug(f"清理旧临时目录: {dir_name}/{filename}")
                    except Exception as e:
                        logger_sys.warning(f"清理文件失败 {dir_name}/{filename}: {e}")
            except Exception as e:
                logger_sys.warning(f"扫描目录失败 {dir_name}: {e}")

            if cleaned_count > 0:
                logger_sys.info(f"临时文件清理完成 [{dir_name}]，删除了 {cleaned_count} 个文件")
                total_cleaned += cleaned_count

        if total_cleaned > 0:
            logger_sys.info(f"本轮临时文件清理总计: {total_cleaned} 个文件/目录")

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

_speaker_db_file_mtime = 0.0  # 声纹库文件上次加载时的 mtime

def load_speaker_db_if_changed():
    """读接口专用：仅当声纹库文件 mtime 变化时才重新读盘。
    写路径（注册/删除/确认样本）会直接更新内存 speaker_db，不走此函数。"""
    global _speaker_db_file_mtime
    try:
        mtime = os.path.getmtime(Config.SPEAKER_DB_FILE)
    except OSError:
        return  # 文件不存在时不动作，保留内存中的现有数据
    if mtime != _speaker_db_file_mtime:
        load_speaker_db()  # 内部持有 db_lock，与写路径互斥
        _speaker_db_file_mtime = mtime

# =================== 音频预处理 ===================
# ---- Silero VAD v5 切分引擎（2026-10-03：替代 fsmn-vad 内嵌切分）----
# 优点：婴儿声泛化更好；段间静音阈值放宽到 800ms（宝宝停顿不再切碎，切片更适合声纹注册）；
#       ONNX CPU 推理，不占 GPU。段级转写仍由 Paraformer 完成。
# 兼容：返回结构与 asr_pipeline.generate 完全一致（text + sentence_info），下游零改动；
#       异常时调用点自动回退 fsmn-vad 全链路。
_silero_vad_model = None


def _get_silero_vad():
    global _silero_vad_model
    if _silero_vad_model is None:
        from silero_vad import load_silero_vad
        _silero_vad_model = load_silero_vad()
    return _silero_vad_model


def _generate_with_silero(pipeline, wav_path, gen_kwargs):
    """Silero 切段 + 逐段 Paraformer 转写。输入为 16kHz mono wav（preprocess_audio 输出）。
    返回 [{text, sentence_info: [{text, start(ms), end(ms)}]}]，与 fsmn 链路同构。"""
    import numpy as np
    import soundfile as sf
    import torch
    from silero_vad import get_speech_timestamps

    wav, sr = sf.read(wav_path, dtype='float32', always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import torchaudio
        wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, 16000).numpy()

    ts = get_speech_timestamps(
        # silero-vad 6.x 参数名是 sampling_rate（旧版 4.x 才叫 fs）——用错会 TypeError 静默回退 fsmn
        torch.from_numpy(np.ascontiguousarray(wav)), _get_silero_vad(), sampling_rate=16000,
        threshold=Config.VAD_SPEECH_THRESHOLD,
        min_speech_duration_ms=Config.VAD_MIN_SPEECH_MS,
        min_silence_duration_ms=Config.VAD_MIN_SILENCE_MS,
        max_speech_duration_s=Config.VAD_MAX_SPEECH_S,
        speech_pad_ms=Config.VAD_SPEECH_PAD_MS,
    )
    if not ts:
        return []

    sentence_info = []
    text_parts = []
    for i, seg in enumerate(ts):
        a_ms, b_ms = int(seg['start'] / 16), int(seg['end'] / 16)  # 16k: sample/16 = ms
        piece_path = f"{wav_path}.s{i}.wav"
        sf.write(piece_path, wav[seg['start']:seg['end']], 16000)
        try:
            out = pipeline.generate(input=piece_path, **gen_kwargs)
        finally:
            try:
                os.remove(piece_path)
            except Exception:
                pass
        text = (out[0].get("text", "") if out else "") or ""
        sentence_info.append({"text": text, "start": a_ms, "end": b_ms})
        text_parts.append(text)

    return [{"text": "".join(text_parts), "sentence_info": sentence_info}]


def preprocess_audio(input_path, output_path, normalize=None):
    # normalize=None 时跟随全局配置；显式传 False 可在并发下安全跳过归一化
    use_normalize = Config.NORMALIZE_AUDIO if normalize is None else normalize
    # 如果启用了高级降噪，先进行降噪处理
    if Config.DENOISE_AUDIO:
        denoised_path = input_path + ".denoised.wav"
        if advanced_denoise(input_path, denoised_path):
            input_path = denoised_path
        else:
            logger_b.warning("高级降噪处理失败，使用原始音频")

    cmd = ["ffmpeg", "-v", "error", "-y", "-i", input_path]
    filters = ["loudnorm=I=-14:TP=-1.5:LRA=11"] if use_normalize else []
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



def transcribe_with_nano(audio_path):
    """用本地 Fun-ASR-Nano (audiocpp_server 8123, Metal) 识别切片; 失败返回None"""
    url = os.getenv('NANO_ASR_URL', 'http://127.0.0.1:8123/v1/audio/transcriptions')
    try:
        with open(audio_path, 'rb') as f:
            resp = requests.post(
                url,
                files={'file': (os.path.basename(audio_path), f, 'audio/wav')},
                data={'model': 'fun-asr-nano', 'language': 'auto'},
                timeout=60,
            )
        if resp.status_code == 200:
            return (resp.json().get('text') or '').strip() or None
    except Exception as e:
        logger_b.warning(f"      [Nano] 识别失败: {e}")
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
    neg_scores = {}   # 各模型与黑名单负样本的最大相似度（拒绝器候选分）

    logger_sys.info(f"🎯 开始声纹识别: 音频段路径={segment_path}")
    logger_sys.info(f"📋 声纹数据库包含 {len(speaker_db)} 个说话人")

    # 并行提取所有模型的 embedding
    embeddings = {}

    def _extract_emb(model_name, sv_pipe):
        emb = extract_embedding_from_file(sv_pipe, segment_path)
        return model_name, emb

    with ThreadPoolExecutor(max_workers=len(sv_pipelines)) as executor:
        futures = {executor.submit(_extract_emb, mn, sp): mn for mn, sp in sv_pipelines.items()}
        for future in as_completed(futures):
            model_name, emb_a = future.result()
            if emb_a is None:
                logger_sys.error(f"❌ 模型 {model_name} 特征提取失败")
                model_votes[model_name] = "Failed"
            else:
                embeddings[model_name] = emb_a

    # 逐模型打分
    for model_name, emb_a in embeddings.items():
        scores = []
        conf = Config.SV_MODELS[model_name]
        threshold = conf['threshold']
        gap = conf['gap']

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

        # 负样本黑名单比对: 该模型与所有负样本逐一算相似度, 取最大（向量已就绪, 仅加几次点积, 微秒级）
        if negative_samples:
            neg_best = 0.0
            for ne in negative_samples:
                ne_vec = (ne.get('embeddings') or {}).get(model_name)
                if ne_vec:
                    try:
                        s = 1 - cosine(emb_a.flatten(), np.array(ne_vec, dtype=np.float32).flatten())
                        if s > neg_best:
                            neg_best = s
                    except Exception:
                        continue
            if neg_best > 0:
                neg_scores[model_name] = neg_best

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
        # 【负样本拒绝器】在≥2个可比模型上"更像黑名单" → 拒绝归属（宁可漏放, 不误伤家人真声）
        cmp_models = [mn for mn in neg_scores if mn in model_scores]
        neg_hits = [mn for mn in cmp_models if neg_scores[mn] > model_scores[mn]]
        if len(cmp_models) >= 2 and len(neg_hits) >= 2:
            logger_b.info(f"🚫 [负样本拒绝] 黑名单更近的模型: {neg_hits} | 负样本分: "
                          + ", ".join(f"{mn}={neg_scores[mn]:.3f}" for mn in cmp_models)
                          + f" | 家人分: {model_scores} → 拒绝归属")
            return None, 0.0, [f"负样本拒绝: 黑名单在 {len(neg_hits)}/{len(cmp_models)} 个模型上更近"]

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

def detect_cry_from_full_audio(audio_path, source_filename=None):
    """
    【轨道A: 独立哭声检测】
    直接对 60s 原始音频进行声纹匹配，使用 CryDetectionConfig 参数。
    与 identify_speaker_fusion (轨道B) 完全独立，互不影响。

    Returns:
        (is_cry: bool, confidence: float, details: list[str])
    """
    if not CryDetectionConfig.ENABLED or not speaker_db:
        logger_b.info("🔍 [哭声检测] 跳过 (功能未启用或声纹库为空)")
        return False, 0.0, []

    _cry_detect_start = time.time()
    _display_name = source_filename or os.path.basename(audio_path)
    logger_b.info(f"🔍 [哭声检测] ═══════════════════════════════════════")
    logger_b.info(f"🔍 [哭声检测] 开始对完整音轨进行独立声纹分析")
    logger_b.info(f"   📄 文件: {_display_name}")
    logger_b.info(f"   🎯 目标声纹: {CryDetectionConfig.TARGET_SPEAKERS}")
    logger_b.info(f"   ⚙️ 参数: 分模型阈值={CryDetectionConfig.MODEL_THRESHOLDS} "
        f"(兜底={CryDetectionConfig.VOICEPRINT_THRESHOLD}), "
        f"间隔={CryDetectionConfig.VOICEPRINT_GAP}, 最少票数={CryDetectionConfig.MIN_VOTES}, "
        f"最低均值置信度={CryDetectionConfig.MIN_AVG_CONFIDENCE}, "
        f"强命中分={CryDetectionConfig.STRONG_MODEL_SCORE}, "
        f"最少强命中数={CryDetectionConfig.MIN_STRONG_MODELS}")
    logger_b.info(f"   🔧 模型数: {len(sv_pipelines)} ({', '.join(sv_pipelines.keys())})")

    target_speakers = CryDetectionConfig.TARGET_SPEAKERS
    cry_threshold = CryDetectionConfig.VOICEPRINT_THRESHOLD
    cry_gap = CryDetectionConfig.VOICEPRINT_GAP
    min_votes = CryDetectionConfig.MIN_VOTES
    min_avg_conf = CryDetectionConfig.MIN_AVG_CONFIDENCE
    strong_model_score = CryDetectionConfig.STRONG_MODEL_SCORE
    min_strong_models = CryDetectionConfig.MIN_STRONG_MODELS

    model_results = {}  # {model_name: (top_target_name, top_target_score, gap_to_others)}
    all_details = []

    # 并行提取所有模型的 embedding（串行约4.5s → 并行约1.5s）
    embeddings = {}

    def _extract_emb(model_name, sv_pipe):
        emb = extract_embedding_from_file(sv_pipe, audio_path)
        return model_name, emb

    with ThreadPoolExecutor(max_workers=len(sv_pipelines)) as executor:
        futures = {executor.submit(_extract_emb, mn, sp): mn for mn, sp in sv_pipelines.items()}
        for future in as_completed(futures):
            model_name, emb_a = future.result()
            if emb_a is None:
                all_details.append(f"  {model_name}: 特征提取失败")
                logger_b.warning(f"   ⚠️ {model_name}: 声纹特征提取失败")
            else:
                embeddings[model_name] = emb_a

    _emb_elapsed = time.time() - _cry_detect_start
    logger_b.info(f"   🧬 声纹特征提取完成: 成功={len(embeddings)}/{len(sv_pipelines)}, 耗时={_emb_elapsed:.2f}s")

    # 逐模型打分（embedding 已并行提取完毕）
    for model_name, emb_a in embeddings.items():
        # 对所有已注册说话人打分（排除语音识别专用的说话人）
        all_scores = []
        for name, speaker_data in speaker_db.items():
            if "avg_embeddings" not in speaker_data or model_name not in speaker_data["avg_embeddings"]:
                continue
            # 跳过语音识别用的说话人（如"大可"），避免干扰哭声检测
            if name in CryDetectionConfig.VOICE_RECOGNITION_SPEAKERS:
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
        logger_b.info(f"   {model_name}: [{scores_str}]")

        if target_hits:
            best_target_name, best_target_score = target_hits[0]
            best_other_name, best_other_score = non_target_scores[0] if non_target_scores else ("无", 0.0)
            gap = best_target_score - best_other_score

            logger_b.info(
                f"   {model_name}: target={best_target_name}={best_target_score:.3f}, "
                f"other={best_other_name}={best_other_score:.3f}, gap={gap:.3f}"
            )

            model_th = CryDetectionConfig.MODEL_THRESHOLDS.get(model_name, cry_threshold)
            passed_threshold = best_target_score >= model_th
            passed_gap = gap >= cry_gap
            passed = passed_threshold and passed_gap
            status = "✅ PASS" if passed else "❌ FAIL"

            # 详细输出判定过程
            fail_reasons = []
            if not passed_threshold:
                fail_reasons.append(f"分数{best_target_score:.3f}<阈值{model_th}")
            if not passed_gap:
                fail_reasons.append(f"间隔{gap:.3f}<要求{cry_gap}")
            fail_info = f" 原因: {', '.join(fail_reasons)}" if fail_reasons else ""
            logger_b.info(f"   {status} {model_name}: {best_target_name}={best_target_score:.3f} vs {best_other_name}={best_other_score:.3f} (gap={gap:.3f}){fail_info}")

            all_details.append(
                f"  {model_name}: {best_target_name}={best_target_score:.3f}, "
                f"other={best_other_name}={best_other_score:.3f} (gap={gap:.3f}) {status}"
            )

            if passed:
                model_results[model_name] = (best_target_name, best_target_score, gap)
        else:
            all_details.append(f"  {model_name}: 未找到目标说话人")
            logger_b.info(f"   ⚪ {model_name}: 未找到目标说话人 ({', '.join(target_speakers)})")

    # 投票判定
    vote_count = len(model_results)
    passed_scores = [v[1] for v in model_results.values()]
    avg_conf = float(np.mean(passed_scores)) if passed_scores else 0.0
    strong_model_count = sum(1 for score in passed_scores if score >= strong_model_score)
    winner_name = list(model_results.values())[0][0] if model_results else None

    _cry_detect_elapsed = time.time() - _cry_detect_start

    # 输出投票汇总
    logger_b.info(f"   ───── 投票汇总 ─────")
    logger_b.info(f"   📊 通过模型: {vote_count}/{len(sv_pipelines)} (需≥{min_votes})")
    if passed_scores:
        scores_detail = ", ".join([f"{m}={v[1]:.3f}" for m, v in model_results.items()])
        logger_b.info(f"   📊 通过分数: [{scores_detail}]")
        logger_b.info(f"   📊 平均置信度: {avg_conf:.3f} (需≥{min_avg_conf})")
        logger_b.info(f"   📊 强命中模型(≥{strong_model_score}): {strong_model_count} (需≥{min_strong_models})")
    else:
        logger_b.info(f"   📊 无模型通过阈值检查")

    if vote_count < min_votes:
        logger_b.info(f"   ❄️ [哭声检测 结论] 未检出哭声 — 票数不足 ({vote_count}<{min_votes}), 耗时={_cry_detect_elapsed:.2f}s")
        logger_b.info(f"🔍 [哭声检测] ═══════════════════════════════════════")
        all_details.append(f"结论: 未检出哭声 (票数不足: {vote_count}<{min_votes})")
        return False, 0.0, all_details

    if avg_conf < min_avg_conf:
        logger_b.info(
            f"   ❄️ [哭声检测 结论] 未检出哭声 — 平均置信度不足 ({avg_conf:.3f}<{min_avg_conf}), "
            f"票数={vote_count}, 耗时={_cry_detect_elapsed:.2f}s"
        )
        logger_b.info(f"🔍 [哭声检测] ═══════════════════════════════════════")
        all_details.append(
            f"结论: 未检出哭声 (平均置信度不足: {avg_conf:.3f}<{min_avg_conf:.3f}, 票数={vote_count})"
        )
        return False, 0.0, all_details

    if strong_model_count < min_strong_models:
        logger_b.info(
            f"   ❄️ [哭声检测 结论] 未检出哭声 — 强命中模型不足 ({strong_model_count}<{min_strong_models}), "
            f"平均置信度={avg_conf:.3f}, 耗时={_cry_detect_elapsed:.2f}s"
        )
        logger_b.info(f"🔍 [哭声检测] ═══════════════════════════════════════")
        all_details.append(
            f"结论: 未检出哭声 (强命中模型不足: {strong_model_count}<{min_strong_models}, 平均置信度={avg_conf:.3f})"
        )
        return False, 0.0, all_details

    logger_b.info(
        f"   🍼 [哭声检测 结论] ★ 检出哭声! 说话人={winner_name}, 票数={vote_count}/{len(sv_pipelines)}, "
        f"平均置信度={avg_conf:.3f}, 强命中模型数={strong_model_count}, 耗时={_cry_detect_elapsed:.2f}s"
    )
    logger_b.info(f"🔍 [哭声检测] ═══════════════════════════════════════")
    all_details.append(
        f"结论: 哭声检出 ({winner_name}, {vote_count}票, conf={avg_conf:.3f}, strong={strong_model_count})"
    )
    return True, avg_conf, all_details

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

@app.route("/api/illustration/<filename>")
def serve_illustration(filename):
    """提供宝宝哭声事件插图（静态文件，支持浏览器缓存）"""
    import os
    # 安全检查：防止路径穿越
    if filename != os.path.basename(filename) or '..' in filename:
        return jsonify({"error": "Invalid filename"}), 400
    illustration_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "illustrations")
    file_path = os.path.join(illustration_dir, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "Image not found"}), 404
    from flask import send_file
    return send_file(file_path, mimetype='image/jpeg', max_age=86400*7)  # 缓存7天

@app.route("/api/cry_event/<int:event_id>", methods=["GET"])
def api_get_cry_event(event_id):
    """获取单个事件的完整详情（弹窗展示用）"""
    from db_manager import get_baby_cry_event_by_id
    try:
        event = get_baby_cry_event_by_id(event_id)
        if not event:
            return jsonify({"error": "Event not found"}), 404
        # 构建音频URL列表
        event_files = event.get("event_files_json", [])
        audio_urls = []
        for f in event_files:
            if any(f.startswith(_r) for _r in RECORDS_ROOTS):
                f = _rel_to_records(f)  # 绝对路径 → 设备级相对路径（双源）
            elif f.startswith('/'):
                f = f.lstrip('/')
            audio_urls.append(f"/api/audio/{f}")

        # 检查插图文件是否存在
        illustration_url = event.get("illustration_url")
        if illustration_url:
            img_filename = illustration_url.replace("/api/illustration/", "")
            illustration_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "illustrations")
            img_path = os.path.join(illustration_dir, img_filename)
            if not os.path.exists(img_path):
                illustration_url = None  # 文件已丢失，不返回 URL

        return jsonify({
            "id": event["id"],
            "filename": event.get("filename"),
            "recording_time": event.get("recording_time"),
            "created_at": event.get("created_at"),
            "start_time": event.get("start_time", 0),
            "end_time": event.get("end_time", 0),
            "reason": event.get("reason"),
            "advice": event.get("advice"),
            "reason_category": event.get("reason_category"),
            "illustration_url": illustration_url,
            "file_count": len(event_files),
            "audio_urls": audio_urls,
            "has_illustration": illustration_url is not None,
            "sample_confirmed": bool(event.get("sample_confirmed")),
            "false_positive": bool(event.get("false_positive")),
            # 预切状态：候选片段清单已存在 → 试听/切换秒开，否则首次需 GPU 定位
            "presliced": os.path.exists(os.path.join(Config.TEMP_DIR, "preview_segments", f"{event_id}.json")),
        })
    except Exception as e:
        logger_a.error(f"获取事件详情失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/cry_events", methods=["GET"])
def api_get_cry_events():
    from db_manager import get_baby_cry_events
    try:
        limit = int(request.args.get('limit', 100))
        offset = int(request.args.get('offset', 0))
        date_filter = request.args.get('date')
        start_time_filter = request.args.get('start_time')
        end_time_filter = request.args.get('end_time')

        events, total = get_baby_cry_events(
            offset=offset,
            limit=limit,
            date_filter=date_filter,
            start_time_filter=start_time_filter,
            end_time_filter=end_time_filter
        )
        # 预切状态标注（一次 listdir；manifest 存在 → 前端显示"秒开"，否则提示需 GPU 定位）
        try:
            seg_dir = os.path.join(Config.TEMP_DIR, "preview_segments")
            presliced_ids = {f[:-5] for f in os.listdir(seg_dir) if f.endswith(".json")}
            for ev in events:
                ev["presliced"] = str(ev.get("id")) in presliced_ids
        except OSError:
            pass
        return jsonify({"events": events, "total": total})
    except Exception as e:
        logger_a.error(f"获取宝宝哭声记录失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/preview_progress", methods=["GET"])
def api_preview_progress():
    """预切总进度：已预切事件数 / 应预切总数（排除已删除与误报事件）"""
    try:
        seg_dir = os.path.join(Config.TEMP_DIR, "preview_segments")
        presliced = 0
        if os.path.isdir(seg_dir):
            presliced = sum(1 for f in os.listdir(seg_dir) if f.endswith(".json"))
        from db_manager import get_baby_cry_events
        _, total = get_baby_cry_events(offset=0, limit=1)
        pct = min(100.0, round(presliced * 100.0 / total, 1)) if total else 100.0
        return jsonify({"presliced": presliced, "total": total, "pct": pct})
    except Exception as e:
        logger_a.error(f"获取预切进度失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/analyze_cry", methods=["POST"])
@admin_required
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
        temp_dir = None
        try:
            logger_a.info(f"👶 [analyze_cry API] 开始运行声纹检测...")
            # 预处理音频
            temp_dir = tempfile.mkdtemp()
            proc_temp = os.path.join(temp_dir, "processed.wav")
            if preprocess_audio(audio_path, proc_temp):
                _, cry_confidence, cry_details = detect_cry_from_full_audio(proc_temp, source_filename=filename)
                logger_a.info(f"👶 [analyze_cry API] 声纹检测完成 - 置信度: {cry_confidence}, 详情数: {len(cry_details)}")
        except Exception as e:
            logger_a.warning(f"👶 [analyze_cry API] 声纹检测异常: {e}")
        finally:
            # 确保临时目录被清理
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

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

                # 移除控制字符（换行、制表符等），只保留可打印字符
                cleaned_text = ''.join(char for char in cleaned_text if char.isprintable() or char in '\n\r\t')

                # 尝试解析 JSON（可能不完整）
                category = "未知"
                reason = "未知"
                advice = "无"
                parse_success = False

                # 方法1：直接尝试解析整个清理后的文本（允许控制字符）
                try:
                    result = _json.loads(cleaned_text, strict=False)
                    category = result.get("category", "未知")
                    reason = result.get("reason", "未知")
                    advice = result.get("advice", "无")
                    parse_success = True
                    logger_a.info(f"👶 [analyze_cry API] 直接解析成功")
                except Exception as e1:
                    # 方法2：尝试找到 JSON 对象（允许控制字符）
                    try:
                        # 找到第一个 { 和最后一个 }
                        start_idx = cleaned_text.find('{')
                        end_idx = cleaned_text.rfind('}') + 1
                        if start_idx != -1 and end_idx > start_idx:
                            json_str = cleaned_text[start_idx:end_idx]
                            result = _json.loads(json_str, strict=False)
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
                    save_result = None
                    try:
                        save_result = save_cry_analysis(
                            filename, start_ms / 1000.0, end_ms / 1000.0,
                            reason, advice,
                            reason_category=category,
                            event_files=valid_paths,
                            audio_path=audio_path,
                            confidence=cry_confidence,
                            details=cry_details
                        )
                        if save_result:
                            logger_a.info(f"👶 [analyze_cry API] ✅ 分析结果已保存到数据库，ID={save_result}")
                        else:
                            logger_a.error(f"👶 [analyze_cry API] ❌ 保存分析结果失败，返回 None")
                            return jsonify({"error": "保存分析结果到数据库失败"}), 500
                    except Exception as save_err:
                        logger_a.error(f"👶 [analyze_cry API] ❌ 保存分析结果异常: {save_err}")
                        import traceback
                        logger_a.error(f"👶 [analyze_cry API] 异常堆栈: {traceback.format_exc()}")
                        return jsonify({"error": f"保存分析结果异常: {str(save_err)}"}), 500

                    # 异步生成插图（不阻塞返回）
                    _saved_id = save_result  # 闭包捕获
                    def generate_illustration():
                        try:
                            logger_a.info(f"🎨 [插图生成] ================ 开始生成场景插图 ================")
                            logger_a.info(f"🎨 [插图生成] ID={_saved_id}, 文件名: {filename}")
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
                                from db_manager import update_cry_event_image_by_id
                                update_result = update_cry_event_image_by_id(_saved_id, image_url)
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
                    return jsonify({"category": category, "reason": reason, "advice": advice, "saved_id": save_result})
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

# =================== 危险接口可选管理鉴权 ===================
# admin_required 已上移至模块顶部（所有路由之前），此处不再重复定义。

@app.route("/api/trigger_reprocess", methods=["POST"])
@admin_required
def trigger_reprocess():
    """触发重新处理历史音频任务"""
    global _history_reprocess_running, _history_reprocess_proc, _track_b_paused
    import subprocess

    # 安全检查：如果标记为运行中，但子进程已死，则自动重置状态
    if _history_reprocess_running and _history_reprocess_proc is not None:
        if _history_reprocess_proc.poll() is not None:
            # 子进程已退出，重置残留状态
            logger_sys.warning("⚠️ 检测到上次A轨进程已退出但状态未清理，自动重置。")
            _history_reprocess_running = False
            _history_reprocess_proc = None
            try:
                _history_reprocess_lock.release()
            except RuntimeError:
                pass  # 锁未持有，忽略
            _track_b_paused = False

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
        _track_b_paused = True
        logger_sys.warning("⚠️ 由于 A 轨历史分析启动，B 轨实时分析已自动暂停。")

        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reprocess_history_cries.py")
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
        os.makedirs(log_dir, exist_ok=True)
        asr_a_log_file = os.path.join(log_dir, "asr-a.log")

        # 写入启动信息到 asr-a.log
        is_targeted = bool(date_param or start_time or end_time)
        display_replace = "True (Targeted)" if is_targeted else force_replace
        start_msg = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🚀 开始执行历史音频重分析(过滤={date_param} {start_time}~{end_time}, 替换={display_replace})...\n"
        # 清空并写入 asr-a.log（前端读取这个文件）
        with open(asr_a_log_file, "w", encoding="utf-8") as f:
            f.write(start_msg)

        args = [sys.executable, "-u", script_path]
        args.append(date_param if date_param else "")
        args.append(start_time if start_time else "")
        args.append(end_time if end_time else "")
        if force_replace:
            args.append("--replace")

        def run_and_cleanup():
            global _history_reprocess_running, _history_reprocess_proc
            try:
                _history_reprocess_proc = subprocess.Popen(
                    args,
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
                end_msg = f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✅ 历史音频重分析任务已结束。\n"
                with open(asr_a_log_file, "a", encoding="utf-8") as f:
                    f.write(end_msg)

        threading.Thread(target=run_and_cleanup, daemon=True).start()

        logger_a.info(f"✅ 历史音频重分析线程已启动，输出至: {asr_a_log_file}")
        return jsonify({"message": "任务已在后台启动，请查看下方实时日志。"})
    except Exception as e:
        _history_reprocess_running = False
        _history_reprocess_lock.release()
        logger_a.error(f"❌ 运行历史分析脚本失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/stop_reprocess", methods=["POST"])
@admin_required
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
    "message": "",
    "logs": ""
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

def get_recent_process_logs(limit=100):
    """读取最近的 process 日志，供前端展示刷盘进度"""
    try:
        log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log", "history_process.log")
        if not os.path.exists(log_file):
            return ""

        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return ''.join(lines[-limit:])
    except Exception as e:
        logger_a.error(f"[刷盘] 读取process log失败: {e}")
        return f"读取刷盘日志失败: {e}"

def run_refresh_file_cache(target_dirs):
    """后台执行刷盘任务（双源架构：逐设备目录刷）"""
    global refresh_cache_task
    if isinstance(target_dirs, str):
        target_dirs = [target_dirs]
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
        f.write(f"[{time_module.strftime('%Y-%m-%d %H:%M:%S')}] 🔄 开始刷盘任务，扫描目录: {target_dirs}\n")
        f.flush()

    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nas-audio-notes-client'))
        from db_manager import refresh_file_cache as do_refresh, init_pool

        init_pool()
        # 使用回调函数实时更新进度和日志（双源：逐设备目录刷，计数累加）
        count = 0
        for target_dir in target_dirs:
            logger_a.info(f"[刷盘] 后台任务开始，扫描目录: {target_dir}")
            log_to_process(f"🔄 刷盘任务开始，目标目录: {target_dir}")
            count += do_refresh(target_dir, progress_callback=update_refresh_progress, log_callback=log_to_process)

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
@admin_required
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

    # 启动后台线程（双源：逐设备目录刷盘）
    target_dirs = list(FileMonitorConfig.SOURCES)
    thread = threading.Thread(target=run_refresh_file_cache, args=(target_dirs,))
    thread.daemon = True
    thread.start()

    logger_a.info(f"[刷盘] 启动后台任务，扫描目录: {target_dirs}")
    return jsonify({
        "message": "刷盘任务已启动",
        "status": "started",
        "task": refresh_cache_task
    })

@app.route("/api/refresh_file_cache/status", methods=["GET"])
def refresh_file_cache_status():
    """获取刷盘任务状态"""
    # 读取最近的刷盘日志
    logs = ""
    try:
        log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log", "history_process.log")
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
                logs = "".join(lines[-50:])  # 最后50行
    except Exception as e:
        logger_a.error(f"读取刷盘日志失败: {e}")

    return jsonify({
        "status": "success",
        "task": refresh_cache_task,
        "logs": logs
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

@app.route("/api/live_status", methods=["GET"])
def get_live_status():
    """获取 A/B 轨状态"""
    try:
        global _track_b_running, _track_b_paused, _history_reprocess_proc
        from db_manager import get_baby_cry_count
        # 检查 A 轨进程是否还在运行
        a_running = _history_reprocess_proc is not None and _history_reprocess_proc.poll() is None
        # 【2026-09-26】兼容外部启动的补跑（驱动脚本拉起，非 5008 子进程）：
        # 进度文件心跳新鲜(10分钟内)且状态为 running 时视为运行中，
        # 否则手机端在整个外部补跑期间会一直错误显示"暂停中"
        _progress_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log", "a_track_progress.json")
        if not a_running and os.path.exists(_progress_file):
            try:
                if time.time() - os.path.getmtime(_progress_file) < 600:
                    with open(_progress_file, "r", encoding="utf-8") as _pf:
                        if json.load(_pf).get("status") == "running":
                            a_running = True
            except Exception:
                pass
        today_cry_count = get_baby_cry_count()

        # 读取 A 轨日志（无论进程是否在运行，都返回日志内容）
        log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log", "asr-a.log")
        logs_a = ""
        if os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding='utf-8', errors='replace') as f:
                    # 使用 seek 从末尾读取最后100行，避免读取整个大文件
                    lines = f.readlines()
                    logs_a = ''.join(lines[-100:])  # 最后100行
            except Exception as e:
                logs_a = f"读取日志失败: {str(e)}"

        # 读取 A 轨结构化进度（由 reprocess_history_cries.py 写入）
        a_progress = None
        progress_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log", "a_track_progress.json")
        if os.path.exists(progress_file):
            try:
                with open(progress_file, "r", encoding="utf-8") as f:
                    a_progress = json.load(f)
                # 如果进程已退出但进度状态还是 running，修正为 stopped
                if not a_running and a_progress.get("status") == "running":
                    a_progress["status"] = "stopped"
            except Exception:
                a_progress = None

        # 获取 B 轨结构化统计
        with _b_stats_lock:
            b_stats = _b_stats.copy()

        # 计算 B 轨运行时长
        b_runtime = None
        if b_stats.get("started_at"):
            try:
                started = datetime.fromisoformat(b_stats["started_at"])
                diff = (datetime.now() - started).total_seconds()
                hours = int(diff // 3600)
                mins = int((diff % 3600) // 60)
                b_runtime = f"{hours}小时{mins}分" if hours > 0 else f"{mins}分钟"
            except Exception:
                b_runtime = None

        # 获取 Termux 上传停滞检测状态
        stall_status = audio_processor.get_stall_status()

        # 获取多设备恢复上传监控状态
        recovery_stall_status = recovery_monitor.get_all_stall_status()

        return jsonify({
            "a_running": a_running,
            "b_running": _track_b_running,
            "b_paused": _track_b_paused if _track_b_running else True,
            "logs_a": logs_a,
            "a_progress": a_progress,
            "today_cry_count": today_cry_count,
            "pid": _history_reprocess_proc.pid if _history_reprocess_proc else None,
            "a_reason": "哭声检测补跑中（转写让出 GPU）" if (a_running and _track_b_running and _track_b_paused) else None,
            "b_reason": "哭声检测补跑中（转写让出 GPU）" if (a_running and _track_b_running) else None,
            "message": "实时转写已暂停" if (_track_b_running and _track_b_paused) else ("实时转写运行中" if _track_b_running else "实时转写未启动"),
            "b_stats": {
                "started_at": b_stats.get("started_at"),
                "runtime": b_runtime,
                "today_record_count": b_stats.get("today_record_count", 0),
                "today_cry_count": b_stats.get("today_cry_count", 0),
                "last_event_time": b_stats.get("last_event_time"),
                "last_cry_time": b_stats.get("last_cry_time")
            },
            "upload_stall": stall_status,
            "recovery_devices_stall": recovery_stall_status
        })
    except Exception as e:
        return jsonify({"error": str(e), "a_running": False, "logs_a": ""}), 500

@app.route("/api/overview", methods=["GET"])
def get_overview():
    """系统总览：录音设备健康 + B 轨转写 + A 轨补跑 + 基础设施，一次请求拿全。
    设备/基础设施部分来自后台线程 ≤5 分钟的缓存快照（ffmpeg 响度检测较重，不随轮询实时跑）；
    B/A 轨部分每次请求实时读取（内存/本地文件，开销可忽略）。"""
    try:
        global _track_b_running, _track_b_paused, _history_reprocess_proc
        with _overview_cache_lock:
            devices = [d.copy() for d in _overview_cache["devices"]]
            infra = json.loads(json.dumps(_overview_cache["infra"]))
            scan_age = (int(time.time() - _overview_cache["updated_at"])
                        if _overview_cache["updated_at"] else None)
            total_now = _overview_cache.get("total_today")
            total_prev = _overview_cache.get("prev_total_today")

        # ---- A 轨（与 live_status 同一套判定：进程存活 或 进度文件心跳 10 分钟内 running）----
        a_running = _history_reprocess_proc is not None and _history_reprocess_proc.poll() is None
        progress_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log", "a_track_progress.json")
        if not a_running and os.path.exists(progress_file):
            try:
                if time.time() - os.path.getmtime(progress_file) < 600:
                    with open(progress_file, "r", encoding="utf-8") as f:
                        if json.load(f).get("status") == "running":
                            a_running = True
            except Exception:
                pass
        a_progress = None
        if os.path.exists(progress_file):
            try:
                with open(progress_file, "r", encoding="utf-8") as f:
                    a_progress = json.load(f)
                # 进程已退出但进度状态还是 running，修正为 stopped
                if not a_running and a_progress.get("status") == "running":
                    a_progress["status"] = "stopped"
            except Exception:
                a_progress = None

        # ---- B 轨：统计 + 心跳 + 积压 + 效率 ----
        with _b_stats_lock:
            b_stats = _b_stats.copy()
        now_ts = time.time()
        hour_rate = sum(1 for t in b_stats.get("done_window", []) if now_ts - t <= 3600)
        # DB 真实入库统计（缓存线程 5 分钟刷新；进程计数会因重启清零，仅作降级回退）
        db_stats = _overview_cache.get("db_stats") or {}
        ingest_1h = db_stats.get("ingest_1h")
        ingest_today = db_stats.get("ingest_today")
        cry_today = db_stats.get("cry_today")
        b_runtime = None
        if b_stats.get("started_at"):
            try:
                diff = (datetime.now() - datetime.fromisoformat(b_stats["started_at"])).total_seconds()
                hours, mins = int(diff // 3600), int((diff % 3600) // 60)
                b_runtime = f"{hours}小时{mins}分" if hours > 0 else f"{mins}分钟"
            except Exception:
                b_runtime = None
        # 心跳：audio_processor 每个文件处理前/每轮监听都会 touch（smb_watchdog 240s 判死同源）
        hb_age = None
        try:
            hb_file = getattr(audio_processor, "_B_TRACK_HEARTBEAT",
                              os.path.join(os.path.dirname(os.path.abspath(__file__)), "log", "b_track_heartbeat"))
            hb_age = int(time.time() - os.path.getmtime(hb_file))
        except Exception:
            pass
        # 积压 ≈ 各设备今日目录现存 m4a 总数（B 轨转写完会删除源文件，现存即待转写存量；
        # 来自 ≤5 分钟缓存，做的是趋势参考而非精确值）
        backlog = sum(d.get("today_count", 0) for d in devices) if devices else None
        # 积压趋势：最近两轮扫描(间隔 300s)的现存总数差 → 折算每小时净变化
        # 正=录制快于转写(积压在涨)，负=转写在消化积压，0=稳态；重启后首两轮 prev 为空则不出数
        backlog_trend = None
        if total_now is not None and total_prev is not None:
            backlog_trend = round((total_now - total_prev) / 300 * 3600)

        # ---- GPU 占用（推算，不做探测）：补跑 > 转写 > 空闲 ----
        if a_running:
            gpu_state = "补跑占用"
        elif _track_b_running:
            gpu_state = "转写占用"
        else:
            gpu_state = "空闲"

        return jsonify({
            "devices": devices,
            "devices_scan_age_sec": scan_age,
            "b_track": {
                "running": _track_b_running,
                "paused": _track_b_paused if _track_b_running else True,
                "heartbeat_age_sec": hb_age,
                "backlog": backlog,
                "backlog_trend_per_hour": backlog_trend,
                "hour_rate": hour_rate,
                "ingest_rate_per_hour": ingest_1h,
                "today_transcribed": ingest_today if ingest_today is not None else b_stats.get("today_record_count", 0),
                "today_cry_count": cry_today if cry_today is not None else b_stats.get("today_cry_count", 0),
                "runtime": b_runtime,
                "last_event_time": b_stats.get("last_event_time"),
            },
            "a_track": {
                "running": a_running,
                "progress": a_progress,
                "window": "01:00-06:30",
                "next_run": "01:05",
            },
            "infra": dict(infra, gpu_state=gpu_state),
        })
    except Exception as e:
        logger_sys.error(f"❌ /api/overview 失败: {e}")
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

@app.route("/api/recovery_devices/status", methods=["GET"])
def get_recovery_devices_status():
    """获取所有恢复监控设备的停滞状态"""
    try:
        status = recovery_monitor.get_all_stall_status()
        return jsonify({"devices": status})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/recovery_devices/trigger", methods=["POST"])
@admin_required
def trigger_device_recovery():
    """手动触发指定设备的恢复命令"""
    try:
        data = request.get_json(silent=True) or {}
        device_name = data.get("device_name") or request.args.get("device_name")
        if not device_name:
            return jsonify({"error": "缺少 device_name 参数"}), 400
        result = recovery_monitor.trigger_recovery(device_name)
        return jsonify(result)
    except Exception as e:
        logger.error(f"手动触发恢复失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/start_live", methods=["POST"])
@admin_required
def start_live():
    """启动实时监听（B 轨）"""
    try:
        import threading
        global _monitor_thread, _track_b_running, _track_b_paused
        if _track_b_running and _monitor_thread and _monitor_thread.is_alive():
            return jsonify({"message": "实时转写已在运行中", "status": "already_running"})
        _track_b_running = True
        _track_b_paused = False
        with _b_stats_lock:
            _b_stats["started_at"] = datetime.now().isoformat()
            _b_stats["today_record_count"] = 0
            _b_stats["today_cry_count"] = 0
            _b_stats["last_event_time"] = None
            _b_stats["last_cry_time"] = None
        _monitor_thread = threading.Thread(target=audio_processor.start_monitor, daemon=True)
        _monitor_thread.start()
        return jsonify({"message": "实时语音转写已启动", "status": "started"})
    except Exception as e:
        logger.error(f"启动 B 轨失败: {e}")
        return jsonify({"message": f"启动失败: {e}", "status": "error"}), 500

@app.route("/api/pause_live", methods=["POST"])
@admin_required
def pause_live():
    """暂停实时监听（B 轨）"""
    try:
        global _track_b_paused
        if not _track_b_running:
            return jsonify({"message": "实时转写未启动", "status": "not_running"})
        _track_b_paused = True
        return jsonify({"message": "实时转写已暂停", "status": "paused"})
    except Exception as e:
        return jsonify({"message": f"暂停失败: {e}", "status": "error"}), 500

@app.route("/api/stop_live", methods=["POST"])
@admin_required
def stop_live():
    """停止实时监听（B 轨）"""
    try:
        global _track_b_running, _track_b_paused, _monitor_thread
        _track_b_running = False
        _track_b_paused = True
        _monitor_thread = None
        return jsonify({"message": "实时转写已停止", "status": "stopped"})
    except Exception as e:
        return jsonify({"message": f"停止失败: {e}", "status": "error"}), 500

@app.route("/api/quick_cry_detect", methods=["POST"])
@admin_required
def quick_cry_detect():
    """
    【A轨快速哭声检测】
    仅进行声纹匹配检测哭声，跳过语音识别，速度提升10倍以上
    适用于历史文件批量处理场景
    """
    try:
        if 'audio_file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['audio_file']
        filename = file.filename

        # 保存临时文件
        temp_path = os.path.join(Config.TEMP_DIR, f"quick_cry_{int(time.time())}_{filename}")
        os.makedirs(Config.TEMP_DIR, exist_ok=True)
        file.save(temp_path)

        # 音频预处理（快速模式：跳过响度归一化，SV模型对响度不敏感，省约0.5-1s）
        # 通过参数传入而非改全局 Config，避免 Flask threaded=True 并发竞态
        proc_temp = os.path.join(Config.TEMP_DIR, f"quick_cry_proc_{int(time.time())}.wav")
        if not preprocess_audio(temp_path, proc_temp, normalize=False):
            # 清理临时文件
            for f in [temp_path, proc_temp]:
                if os.path.exists(f):
                    os.remove(f)
            return jsonify({"error": "Audio preprocessing failed"}), 500

        # 快速哭声检测（仅声纹匹配，无ASR）
        start_time = time.time()
        cry_detected, confidence, details = detect_cry_from_full_audio(proc_temp, source_filename=filename)
        detect_time = time.time() - start_time

        # 清理临时文件
        for f in [temp_path, proc_temp]:
            if os.path.exists(f):
                os.remove(f)

        return jsonify({
            "filename": filename,
            "is_baby_cry": cry_detected,
            "confidence": round(confidence, 4),
            "detect_time_ms": round(detect_time * 1000, 2),
            "details": details
        })

    except Exception as e:
        logger_a.error(f"快速哭声检测失败: {e}")
        return jsonify({"error": str(e)}), 500


# =================== 一键确认哭声事件为声纹样本 (自动喂样本库) ===================
SAMPLE_TARGET_SPEAKER = "Baby"       # 确认样本写入的说话人（与 speaker_db_multi.json 实际 key 一致）
SAMPLE_NAS_COPY_TIMEOUT = 60         # NAS 文件复制超时（秒）
SAMPLE_WINDOW_SECONDS = 8.0          # 自动定位的纯哭声窗长（与库中手工样本 2~9s 同量级）
SAMPLE_WINDOW_STEP = 2.0             # 滑窗步长
SAMPLE_SEED_MODEL = "eres2net_large" # 滑窗粗筛用模型（三模型中精度最高）

def _safe_copy_file(src, dst, timeout=SAMPLE_NAS_COPY_TIMEOUT):
    """带超时的文件复制：NAS(SMB) 读取可能挂死，放子线程执行，超时返回失败。
    【2026-09-20】改用 mkstemp 原子创建可写文件 + copyfileobj（不用 copy2 的
    copystat——NAS 源文件的怪权限位曾被同步到本地副本，导致后续 EPERM）。
    dst 若已给出则只取其扩展名，实际写入 mkstemp 生成的可写路径。"""
    result = {"ok": False, "error": None}

    def _copy():
        try:
            fd, real_dst = tempfile.mkstemp(
                prefix="copysrc_", suffix=os.path.splitext(dst)[1] or ".tmp", dir=os.path.dirname(os.path.abspath(dst)))
            os.close(fd)
            try:
                with open(src, "rb") as fsrc, open(real_dst, "wb") as fdst:
                    shutil.copyfileobj(fsrc, fdst)
                os.chmod(real_dst, 0o644)
                if os.path.exists(dst):
                    try:
                        os.remove(dst)
                    except Exception:
                        pass
                os.replace(real_dst, dst)
                result["ok"] = True
            except Exception:
                try:
                    os.remove(real_dst)
                except Exception:
                    pass
                raise
        except Exception as e:
            result["error"] = str(e)

    t = threading.Thread(target=_copy, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        result["error"] = f"文件复制超时({timeout}s)"
    return result["ok"], result["error"]


def _get_audio_duration(path, timeout=10):
    """ffprobe 获取音频时长（本地文件），失败返回 None"""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=timeout
        )
        return float(r.stdout.strip())
    except Exception:
        return None


def _find_top_cry_windows(src_local, top_n=5, step_scale=1):
    """在事件音频中自动定位最像 Baby 的纯哭声段（滑窗打分），返回 top_n 个候选窗。
    用声纹库 Baby 平均声纹做种子：单模型全窗粗筛 → top 窗 3 模型精排。
    step_scale>1 时放大滑窗步长（长录音降采样粗扫，推理量按倍数下降，精度略降）。
    返回 [(score, start_sec, end_sec), ...]（分数降序）；库中无声纹种子时返回 []。"""
    avg_embeddings = None
    if SAMPLE_TARGET_SPEAKER in speaker_db and "avg_embeddings" in speaker_db[SAMPLE_TARGET_SPEAKER]:
        avg_embeddings = speaker_db[SAMPLE_TARGET_SPEAKER]["avg_embeddings"]
    seed_avg = (avg_embeddings or {}).get(SAMPLE_SEED_MODEL)
    seed_pipe = sv_pipelines.get(SAMPLE_SEED_MODEL)
    if seed_avg is None or seed_pipe is None:
        return []
    seed_avg = np.array(seed_avg).flatten()

    duration = _get_audio_duration(src_local)
    if not duration or duration <= SAMPLE_WINDOW_SECONDS:
        return [(0.0, 0.0, float(duration or 0.0))]  # 整段不足一个窗，直接整段
    win_ms = int(SAMPLE_WINDOW_SECONDS * 1000)
    step_ms = int(SAMPLE_WINDOW_STEP * 1000 * max(1, int(step_scale)))
    total_ms = int(duration * 1000)

    windows = []
    start_ms = 0
    while True:
        end_ms = min(start_ms + win_ms, total_ms)
        windows.append((start_ms, end_ms))
        if end_ms >= total_ms:
            break
        start_ms += step_ms

    os.makedirs(Config.TEMP_DIR, exist_ok=True)
    # 粗筛：全窗单模型打分（只做相对排序，跳过响度归一化）
    scored = []
    for start_ms, end_ms in windows:
        wpath = os.path.join(Config.TEMP_DIR, f"crywin_{start_ms}_{int(time.time() * 1000)}.wav")
        try:
            if not extract_segment(src_local, start_ms, end_ms, wpath):
                continue
            emb = extract_embedding_from_file(seed_pipe, wpath)
            if emb is None:
                continue
            scored.append((1 - cosine(emb.flatten(), seed_avg), start_ms, end_ms))
        finally:
            try:
                os.remove(wpath)
            except Exception:
                pass

    if not scored:
        return []
    scored.sort(reverse=True)

    # 精排：top 8 窗走正式预处理 + 3 模型平均分，输出 top_n 个候选窗
    refined = []
    for _, s_ms, e_ms in scored[:8]:
        wpath = os.path.join(Config.TEMP_DIR, f"crywinp_{s_ms}_{int(time.time() * 1000)}.wav")
        ppath = os.path.join(Config.TEMP_DIR, f"crywinpp_{s_ms}_{int(time.time() * 1000)}.wav")
        try:
            if not extract_segment(src_local, s_ms, e_ms, wpath):
                continue
            if not preprocess_audio(wpath, ppath):
                continue
            sims = []
            for m, pipe in sv_pipelines.items():
                m_avg = avg_embeddings.get(m)
                if m_avg is None:
                    continue
                emb = extract_embedding_from_file(pipe, ppath)
                if emb is not None:
                    sims.append(1 - cosine(emb.flatten(), np.array(m_avg).flatten()))
            if sims:
                refined.append((sum(sims) / len(sims), s_ms / 1000, e_ms / 1000))
        finally:
            for p in (wpath, ppath):
                try:
                    os.remove(p)
                except Exception:
                    pass

    refined.sort(reverse=True)
    # 候选去重叠：相邻滑窗(步长2s)内容高度重复，"换一段"必须听到不同的音频。
    # 规则：与已选窗口重叠 <4s（即至少 4s 新内容）才入选
    selected = []
    for cand in refined:
        if all(cand[1] >= b - 4.0 for _, _, b in selected):
            selected.append(cand)
        if len(selected) >= top_n:
            break
    if not selected and refined:
        selected = [refined[0]]
    return selected


def _find_best_cry_window(src_local):
    """返回最像 Baby 哭声的单个窗口 (start_sec, end_sec, score)；无种子返回 (None, None, 0)。"""
    wins = _find_top_cry_windows(src_local, top_n=1)
    if not wins:
        return None, None, 0.0
    score, w_start, w_end = wins[0]
    return w_start, w_end, score


def preset_cry_segments(event_id, src_local, top_n=5, fast_windows=None, step_scale=1):
    """【2026-09-20 用户建议落地】分析时顺手预切哭声候选片段并存盘。
    之后预览/确认直接读文件——零 GPU 等待，彻底避开补跑抢 GPU 导致的预览卡顿。
    fast_windows: 调用方已知的 [(score, start_sec, end_sec), ...]，直接按位切片，
    跳过全量滑窗打分（历史事件补切用，秒级完成）；None 则走滑窗扫描。
    step_scale: 滑窗步长放大倍数（长录音降采样粗扫）。
    失败只降级到旧的"用时定位"路径，不影响主流程。"""
    try:
        seg_dir = os.path.join(Config.TEMP_DIR, "preview_segments")
        os.makedirs(seg_dir, exist_ok=True)
        manifest_path = os.path.join(seg_dir, f"{event_id}.json")
        if os.path.exists(manifest_path):
            return True
        if fast_windows:
            windows = fast_windows
        else:
            with gpu_lock:
                windows = _find_top_cry_windows(src_local, top_n=top_n, step_scale=step_scale)
        if not windows:
            return False
        manifest = []
        for i, (score, ws, we) in enumerate(windows):
            if we - ws < 1.0:
                continue
            seg_path = os.path.join(seg_dir, f"{event_id}_v{i}.wav")
            if extract_segment(src_local, int(ws * 1000), int(we * 1000), seg_path):
                manifest.append({"variant": i, "start": round(ws, 2), "end": round(we, 2),
                                 "score": round(float(score), 4)})
        if not manifest:
            return False
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, ensure_ascii=False)
        logger_b.info(f"🎵 [预切片段] 事件 {event_id} 已预存 {len(manifest)} 个候选段")
        return True
    except Exception as seg_err:
        logger_a.warning(f"预切片段失败 (event_id={event_id}): {seg_err}")
        return False


def _locate_and_copy_event_audio(event, copy=True):
    """定位事件音频源文件并带超时复制到本地 temp。
    优先级: audio_path(持久音频,分析完成后可能已清理) → event_files_json 中与
    filename 匹配的文件 → event_files_json 其余文件。复制动作即存在性检查。
    copy=False 时免拷贝直接返回第一个非空候选路径（ffmpeg input-seek 可直接
    远程读段，历史事件快速预切省去整段大文件拷贝）。
    返回 (本地路径 or None, 错误信息)"""
    candidates = []
    audio_path = event.get("audio_path")
    if audio_path:
        candidates.append(audio_path)
        if not os.path.isabs(audio_path):
            _abs = _resolve_under_records(audio_path)
            if _abs and _abs not in candidates:
                candidates.append(_abs)
    event_files = event.get("event_files_json") or []
    fname = event.get("filename")
    for f in event_files:
        if fname and os.path.basename(f) == fname and f not in candidates:
            candidates.append(f)
    for f in event_files:
        if f not in candidates:
            candidates.append(f)

    # 【2026-09-21】NAS 回退：老事件（尤其回合合并前 created 的实时事件，event_files 为空）
    # 持久音频被清理后，按录音日期到 NAS 各设备源目录找原文件；
    # 覆盖统一归档 processed/（B 轨处理后的文件都会移到各设备 processed/<日期>/，
    # 【2026-10-03 双源】设备清单含 Pixel-6/Pixel-5 及 Sony 兼容源）。
    # 复制动作自带存在性检查，多候选逐个尝试即可。
    try:
        # 注意：recording_time 从 DB 读出是 ISO 字符串而非 datetime，直接用文件名解析最稳
        _rec = parse_recording_time(fname or "")
        if _rec and fname:
            _date_str = _rec.strftime("%Y-%m-%d")
            for _dev in SOURCE_DEVICES + _LEGACY_DEVICES:
                for _root in RECORDS_ROOTS:
                    for _cand in (
                        os.path.join(_root, _dev, _date_str, fname),
                        os.path.join(_root, _dev, "processed", _date_str, fname),
                    ):
                        if _cand not in candidates:
                            candidates.append(_cand)
    except Exception:
        pass

    os.makedirs(Config.TEMP_DIR, exist_ok=True)
    if not copy:
        # 免拷贝模式: 第一个存在且非空的候选直接返回 (ffmpeg input-seek 远程读段)
        last_err = "事件无可定位的音频文件"
        for src in candidates:
            src = str(src).strip()
            if not src:
                continue
            try:
                if os.path.exists(src) and os.path.getsize(src) > 0:
                    return src, None
                last_err = f"{src}: 不存在或空文件"
            except OSError as _e:
                last_err = f"{src}: {_e}"
        return None, f"音频不可达(免拷贝模式) → {last_err}"
    src_ext = (os.path.splitext(candidates[0])[1] if candidates else "") or ".wav"
    local_path = os.path.join(
        Config.TEMP_DIR, f"confirm_src_{event.get('id')}_{int(time.time())}{src_ext}"
    )
    last_err = "事件无可定位的音频文件"
    for src in candidates:
        src = str(src).strip()
        if not src:
            continue
        ok, err = _safe_copy_file(src, local_path)
        if ok:
            return local_path, None
        last_err = f"{src}: {err}"
    if os.path.exists(local_path):
        try:
            os.remove(local_path)
        except Exception:
            pass
    return None, f"持久音频已清理且源文件不可达 → {last_err}"


@app.route("/api/preset_cry_segments/<int:event_id>", methods=["POST"])
@admin_required
def api_preset_cry_segments(event_id):
    """【2026-09-20】为历史事件预切哭声候选片段（批量补历史预览用，同步执行）。
    已有清单幂等返回；滑窗+精排约 10-60s（GPU 经 gpu_lock 与补跑排队共存）。"""
    try:
        from db_manager import get_baby_cry_event_by_id
        event = get_baby_cry_event_by_id(event_id)
        if not event:
            return jsonify({"error": "事件不存在"}), 404
        if event.get("is_deleted"):
            return jsonify({"skipped": "deleted"}), 200
        if event.get("false_positive"):
            return jsonify({"skipped": "false_positive"}), 200
        seg_dir = os.path.join(Config.TEMP_DIR, "preview_segments")
        if os.path.exists(os.path.join(seg_dir, f"{event_id}.json")):
            return jsonify({"already": True}), 200
        # 【2026-10-02 快速路径】历史事件 DB 里已有精确切片位置 (start_time/end_time, 秒)，
        # 直接按位切 3 个候选变体，跳过全量滑窗打分 (几小时录音的滑窗推理要 100-400s)。
        # 仅切片 ≤600s 时启用；超长粗切片 (整段录音级) 仍走旧滑窗保证精度。
        fast = None
        st, et = event.get("start_time"), event.get("end_time")
        try:
            st, et = float(st), float(et)
        except (TypeError, ValueError):
            st = et = None
        if st is not None and et is not None and 1.0 <= et - st <= 600:
            fast = [
                (1.0, st, et),
                (0.9, max(0.0, st - 2.0), et + 2.0),
                (0.8, max(0.0, st - 5.0), et + 5.0),
            ]
        src_local, locate_err = _locate_and_copy_event_audio(event, copy=not fast)
        if not src_local:
            return jsonify({"error": f"音频不可达: {locate_err}"}), 404
        try:
            # 超长粗切片(>600s, 整段录音级): 降采样滑窗(步长×12), 推理量减 12 倍
            scale = 12 if (st is None or et is None or et - st > 600) else 1
            ok = preset_cry_segments(event_id, src_local, fast_windows=fast, step_scale=scale)
            return jsonify({"ok": bool(ok)}), 200
        finally:
            # 清理批量定位产生的临时副本（持久音频 cry_*.wav 不在此列，绝不删）
            if os.path.basename(src_local).startswith("confirm_src_"):
                try:
                    os.remove(src_local)
                except Exception:
                    pass
    except Exception as e:
        logger_a.error(f"预切失败 (event_id={event_id}): {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/confirm_cry_sample/<int:event_id>", methods=["POST"])
@admin_required
def confirm_cry_sample(event_id):
    """
    【一键确认样本】将哭声事件自动截取的纯哭声段注册进声纹库 (自动喂样本库)。
    - 自动定位事件音频（持久音频或 NAS 原文件，NAS 复制带超时保护）
    - 用现有 Baby 声纹做种子，滑窗打分自动截取最像哭声的 ~8s 纯段
      （与库中手工采集的纯哭声样本同形态，避免整条 60s 稀释平均声纹）
    - 各声纹模型提取 embedding，追加样本并重算平均声纹
    - 幂等：已确认的事件直接返回，不重复入库
    """
    from db_manager import get_baby_cry_event_by_id, mark_sample_confirmed

    event = get_baby_cry_event_by_id(event_id)
    if not event:
        return jsonify({"error": "Event not found"}), 404
    if event.get("is_deleted"):
        return jsonify({"error": "事件已删除，无法确认为样本"}), 400
    if event.get("sample_confirmed"):
        return jsonify({
            "message": "该事件此前已确认为样本",
            "already_confirmed": True,
            "sample_id": event.get("sample_id"),
        })

    temp_files = []
    try:
        # 1. 定位并复制源音频（NAS 读取带超时保护）
        src_local, locate_err = _locate_and_copy_event_audio(event)
        if not src_local:
            return jsonify({"error": locate_err}), 500
        temp_files.append(src_local)

        duration = _get_audio_duration(src_local)
        if duration and duration < 2.0:
            return jsonify({"error": f"音频过短 ({duration:.1f}s)，无法作为样本"}), 400

        # 2~4. 定位哭声段 + 预处理 + 提取 embedding（GPU 密集，统一持 gpu_lock）
        sample_embeddings = {}
        emb_arrays = {}
        selected_window = None
        window_score = None
        with gpu_lock:
            # 2. 滑窗自动定位最像 Baby 的纯哭声段（与库中手工样本同形态）
            # 【2026-09-20】支持 variant：与试听端点一致的多候选序号，用户"换一段"后
            # 确认入库的就是他正在听的那段（试听门槛保证听过的才允许确认）
            try:
                confirm_variant = max(0, int((request.get_json(silent=True) or {}).get("variant") or 0))
            except (TypeError, ValueError):
                confirm_variant = 0
            wins = None
            # 优先读预存清单（零 GPU）；无预存才用时定位
            try:
                manifest_path = os.path.join(Config.TEMP_DIR, "preview_segments", f"{event_id}.json")
                if os.path.exists(manifest_path):
                    with open(manifest_path, encoding="utf-8") as mf:
                        manifest = json.load(mf)
                    if manifest:
                        if confirm_variant >= len(manifest):
                            confirm_variant %= len(manifest)
                        _m = next(m for m in manifest if m["variant"] == confirm_variant)
                        wins = [(_m.get("score", 0.0), _m["start"], _m["end"])]
            except Exception as man_err:
                logger_a.warning(f"确认读预存清单失败 (event_id={event_id})，回退用时定位: {man_err}")
                wins = None
            if wins is None:
                wins = _find_top_cry_windows(src_local, top_n=5)
            if wins:
                if confirm_variant >= len(wins):
                    confirm_variant = confirm_variant % len(wins)
                w_score, w_start, w_end = wins[confirm_variant]
            else:
                w_start, w_end, w_score = None, None, 0.0
            if w_start is None:
                w_start, w_end = 0.0, float(duration or 0.0)  # 库中无种子：回退整段
            if w_end - w_start < 1.0:
                return jsonify({"error": "未能定位出有效的哭声段，请检查该事件音频"}), 400
            selected_window = [round(w_start, 2), round(w_end, 2)]
            window_score = round(float(w_score), 4) if w_score else None

            if duration and (w_end - w_start) >= duration * 0.95:
                seg_path = src_local  # 选中的段几乎覆盖全长，无需剪辑
            else:
                seg_path = os.path.join(Config.TEMP_DIR, f"confirm_seg_{event_id}_{int(time.time())}.wav")
                if not extract_segment(src_local, int(w_start * 1000), int(w_end * 1000), seg_path):
                    return jsonify({"error": "剪出哭声段失败 (ffmpeg)"}), 500
                temp_files.append(seg_path)

            # 3. 预处理（与打分管线一致：响度归一化 + 16k 单声道）
            proc_path = os.path.join(Config.TEMP_DIR, f"confirm_proc_{event_id}_{int(time.time())}.wav")
            if not preprocess_audio(seg_path, proc_path):
                return jsonify({"error": "音频预处理失败"}), 500
            temp_files.append(proc_path)

            # 4. 提取各模型 embedding
            for model_name, sv_pipe in sv_pipelines.items():
                emb = extract_embedding_from_file(sv_pipe, proc_path)
                if emb is not None:
                    emb_arrays[model_name] = emb
                    sample_embeddings[model_name] = emb.tolist()

        if not sample_embeddings:
            return jsonify({"error": "声纹特征提取失败"}), 500

        # 5. 合并入声纹库（db_lock 保护，与注册/识别流程互斥）
        sample_id = f"event_{event_id}_{int(time.time())}"
        speaker_dir = os.path.join("speaker_samples", SAMPLE_TARGET_SPEAKER)
        os.makedirs(speaker_dir, exist_ok=True)
        sample_audio_path = os.path.join(speaker_dir, f"{sample_id}.wav")
        shutil.copy2(proc_path, sample_audio_path)

        sample_info = {
            "id": sample_id,
            "filename": event.get("filename"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "audio_path": sample_audio_path,
            "embeddings": sample_embeddings,
            "source": "confirmed_cry_event",
            "event_id": event_id,
        }
        similarities = {}
        low_similarity_warning = []

        with db_lock:
            existing = speaker_db.get(SAMPLE_TARGET_SPEAKER)
            if existing and existing.get("avg_embeddings"):
                # 合并前计算新样本与现有平均声纹的逐模型相似度（用于反馈）
                for model_name, emb in emb_arrays.items():
                    avg_emb = existing["avg_embeddings"].get(model_name)
                    if avg_emb is None:
                        continue
                    sim = 1 - cosine(emb.flatten(), np.array(avg_emb).flatten())
                    similarities[model_name] = round(float(sim), 4)
                    ref_th = CryDetectionConfig.MODEL_THRESHOLDS.get(
                        model_name, CryDetectionConfig.VOICEPRINT_THRESHOLD)
                    if sim < ref_th:
                        low_similarity_warning.append(
                            f"{model_name}: {sim:.3f} (低于参考阈值 {ref_th})")

                existing.setdefault("samples", []).append(sample_info)
                # 重算所有样本的平均 embedding（与注册增强逻辑一致）
                all_model_embeddings = {m: [] for m in sv_pipelines.keys()}
                for sample in existing["samples"]:
                    for m, e in sample.get("embeddings", {}).items():
                        all_model_embeddings[m].append(np.array(e))
                existing["avg_embeddings"] = {
                    m: np.mean(lst, axis=0).tolist()
                    for m, lst in all_model_embeddings.items() if lst
                }
                total_samples = len(existing["samples"])
            else:
                # 声纹库中尚无该说话人（兜底新建）
                speaker_db[SAMPLE_TARGET_SPEAKER] = {
                    "samples": [sample_info],
                    "avg_embeddings": {m: e.tolist() for m, e in emb_arrays.items()},
                }
                total_samples = 1

            with open(Config.SPEAKER_DB_FILE, "w", encoding="utf-8") as f:
                json.dump(speaker_db, f, indent=2, ensure_ascii=False)

        logger_a.info(
            f"🧬 [确认样本] 事件 {event_id} ({event.get('filename')}) 已注册为 "
            f"[{SAMPLE_TARGET_SPEAKER}] 样本 sample_id={sample_id}, "
            f"选中段 {selected_window}, 当前共 {total_samples} 个样本, 相似度={similarities}")

        # 6. 标记 DB（在声纹库写成功之后，保证幂等标记与实际入库一致）
        if not mark_sample_confirmed(event_id, sample_id):
            logger_a.warning(f"🧬 [确认样本] 事件 {event_id} 声纹已入库但标记 DB 失败")

        return jsonify({
            "message": (f"已自动截取 {selected_window[0]}s~{selected_window[1]}s 哭声段，"
                        f"加入 [{SAMPLE_TARGET_SPEAKER}] 声纹库"),
            "already_confirmed": False,
            "sample_id": sample_id,
            "sample_count": total_samples,
            "selected_window": selected_window,
            "window_score": window_score,
            "similarities": similarities,
            "low_similarity_warning": low_similarity_warning,
        })

    except Exception as e:
        logger_a.error(f"确认哭声样本失败 (event_id={event_id}): {e}")
        import traceback
        logger_a.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    finally:
        for f in temp_files:
            try:
                if f and os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass


@app.route("/api/cry_segment_preview/<int:event_id>", methods=["GET"])
@admin_required
def cry_segment_preview(event_id):
    """
    【哭声片段试听】返回事件最像哭声的纯段音频（与"确认为样本"入库的是同一段）。
    - 优先滑窗自动定位（与 confirm_cry_sample 同款逻辑），失败回退事件的 start/end_time
    - 结果缓存到 TEMP（事件音频不变，切片可复用，避免每次都跑 GPU 滑窗打分）
    - 前端要求：用户须先试听本片段，才允许点击"确认为样本"
    """
    from db_manager import get_baby_cry_event_by_id

    event = get_baby_cry_event_by_id(event_id)
    if not event:
        return jsonify({"error": "Event not found"}), 404

    # 【2026-09-20】variant：候选片段序号（0=最佳，1..N=换一段）。自动定位可能选偏，
    # 提供多候选让用户耳朵裁决；确认样本时按所选 variant 入库对应片段
    try:
        variant = max(0, int(request.args.get("variant", 0) or 0))
    except (TypeError, ValueError):
        variant = 0
    if variant > 9:
        variant = 9
    cached = os.path.join(
        Config.TEMP_DIR,
        f"preview_cryseg_{event_id}.wav" if variant == 0 else f"preview_cryseg_{event_id}_v{variant}.wav",
    )
    if os.path.exists(cached) and os.path.getsize(cached) > 1000:
        resp = send_file(cached, mimetype="audio/wav")
        resp.headers["X-Cry-Variant"] = str(variant)
        return resp
    # 优先读分析时预存的候选片段（零 GPU，秒开）；无预存才走"用时定位"
    seg_dir = os.path.join(Config.TEMP_DIR, "preview_segments")
    manifest_path = os.path.join(seg_dir, f"{event_id}.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, encoding="utf-8") as mf:
                manifest = json.load(mf)
            if manifest:
                if variant >= len(manifest):
                    variant %= len(manifest)
                seg_path = os.path.join(seg_dir, f"{event_id}_v{variant}.wav")
                if os.path.isfile(seg_path) and os.path.getsize(seg_path) > 1000:
                    resp = send_file(seg_path, mimetype="audio/wav")
                    resp.headers["X-Cry-Variant"] = str(variant)
                    resp.headers["X-Cry-Windows"] = str(len(manifest))
                    return resp
                # 【2026-10-02】清单在但切片 wav 缺失(批量预切中断遗留 110 个事件)：
                # 按清单 start/end 免拷贝直切(ffmpeg 远程读段, 秒级)，省掉 GPU 滑窗
                # 打分(100-400s)；结果写入长期缓存，下次直接秒回
                _cand = manifest[variant] if isinstance(manifest[variant], dict) else {}
                _st = float(_cand.get("start") or 0.0)
                _et = float(_cand.get("end") or 0.0)
                if _et - _st >= 1.0:
                    _src, _loc_err = _locate_and_copy_event_audio(event, copy=False)
                    if _src:
                        try:
                            if extract_segment(_src, int(_st * 1000), int(_et * 1000), cached):
                                logger_b.info(f"🎵 [试听快速路径] 事件 {event_id} 清单直切 v{variant} ({_st:.1f}-{_et:.1f}s)")
                                resp = send_file(cached, mimetype="audio/wav")
                                resp.headers["X-Cry-Variant"] = str(variant)
                                resp.headers["X-Cry-Windows"] = str(len(manifest))
                                resp.headers["X-Cry-Fastpath"] = "manifest"
                                return resp
                        except Exception as fast_err:
                            logger_a.warning(f"清单直切失败 (event_id={event_id})，落穿用时定位: {fast_err}")
                    else:
                        logger_a.warning(f"清单直切无源 (event_id={event_id}): {_loc_err}")
                logger_a.warning(f"预切 wav 缺失，落穿用时定位 (event_id={event_id}, v{variant})")
        except Exception as man_err:
            logger_a.warning(f"读取预切片段失败 (event_id={event_id})，回退用时定位: {man_err}")

    temp_files = []
    try:
        src_local, locate_err = _locate_and_copy_event_audio(event)
        if not src_local:
            return jsonify({"error": locate_err}), 500
        temp_files.append(src_local)

        duration = _get_audio_duration(src_local) or 0.0

        # 滑窗定位最像哭声的 ~8s 段（GPU 密集，持 gpu_lock）；失败回退事件时间段
        windows = []
        try:
            with gpu_lock:
                windows = _find_top_cry_windows(src_local, top_n=5)
        except Exception as win_err:
            logger_a.warning(f"滑窗定位失败 (event_id={event_id})，回退事件时间段: {win_err}")
        used_slide_window = bool(windows)
        if used_slide_window:
            if variant >= len(windows):
                variant = variant % len(windows)
            _score, w_start, w_end = windows[variant]
            windows_total = len(windows)
        else:
            # 回退场景（声纹管线未就绪/无种子/滑窗异常）不缓存，待服务完全就绪后可重新定位
            windows_total = 0
            w_start = float(event.get("start_time") or 0.0)
            w_end = float(event.get("end_time") or min(8.0, duration))
            logger_a.info(f"哭声试听回退事件时间段 (event_id={event_id}): {w_start:.1f}-{w_end:.1f}s")
        if w_end - w_start < 1.0:
            w_start, w_end = 0.0, min(8.0, max(duration, 1.0))
        w_start = max(0.0, min(w_start, max(duration - 1.0, 0.0)))
        w_end = max(w_start + 1.0, min(w_end, duration if duration > 0 else w_end))
        logger_a.info(
            f"哭声试听窗口 (event_id={event_id}): {w_start:.2f}-{w_end:.2f}s "
            f"滑窗定位={used_slide_window} variant={variant}/{windows_total}"
        )

        os.makedirs(Config.TEMP_DIR, exist_ok=True)
        if duration > 0 and (w_end - w_start) >= duration * 0.95:
            send_path = src_local  # 段几乎覆盖全长，直接发原文件
        else:
            # 只有滑窗定位结果才写长期缓存；回退结果写临时文件用完即弃
            out_path = cached if used_slide_window else os.path.join(Config.TEMP_DIR, f"preview_cryseg_{event_id}_tmp.wav")
            if not extract_segment(src_local, int(w_start * 1000), int(w_end * 1000), out_path):
                return jsonify({"error": "切片失败 (ffmpeg)"}), 500
            send_path = out_path

        resp = send_file(send_path, mimetype="audio/wav")
        resp.headers["X-Cry-Variant"] = str(variant)
        resp.headers["X-Cry-Windows"] = str(windows_total)
        return resp
    except Exception as e:
        logger_a.error(f"哭声片段试听失败 (event_id={event_id}): {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        for f in temp_files:
            try:
                if f and os.path.exists(f) and f != send_path:
                    os.remove(f)
            except Exception:
                pass


@app.route("/api/cry_event_feedback/<int:event_id>", methods=["POST"])
@admin_required
def cry_event_feedback(event_id):
    """
    【人工反馈】哭声事件标注：
    - verdict=false_positive：误报（假阳性）。DB 标记 false_positive=TRUE，
      供后续训练/阈值校准使用；与 sample_confirmed 互斥（已喂样本的事件不允许标误报）。
    """
    from db_manager import get_baby_cry_event_by_id, mark_event_false_positive

    data = request.get_json(silent=True) or {}
    verdict = data.get("verdict")
    if verdict != "false_positive":
        return jsonify({"error": "verdict 仅支持 false_positive（确认哭声请走 confirm_cry_sample）"}), 400

    event = get_baby_cry_event_by_id(event_id)
    if not event:
        return jsonify({"error": "Event not found"}), 404
    if event.get("sample_confirmed"):
        return jsonify({"error": "该事件已确认为样本（真哭声），不能同时标记为误报"}), 400

    if not mark_event_false_positive(event_id):
        return jsonify({"error": "标记失败（事件不存在或数据库异常）"}), 500

    logger_a.info(f"人工反馈：事件 {event_id} 标记为误报 "
                  f"(confidence={event.get('confidence')}, category={event.get('reason_category')})")
    return jsonify({
        "message": "已标记为误报",
        "event_id": event_id,
        "confidence": event.get("confidence"),
        "reason_category": event.get("reason_category"),
    })


@app.route("/speakers", methods=["GET"])
def get_speakers():
    """获取所有说话人列表"""
    try:
        # 读内存声纹库（mtime 变化时才重读盘，避免每次请求 JSON 解析）
        load_speaker_db_if_changed()
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
        # 读内存声纹库（mtime 变化时才重读盘，避免每次请求 JSON 解析）
        load_speaker_db_if_changed()
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
@admin_required
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
@admin_required
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
            _ap = _resolve_sample_audio_path(sample_to_remove)
            if _ap:
                try:
                    os.remove(_ap)
                    logger_b.info(f"🗑️ 删除了音频文件: {_ap}")
                except Exception as e:
                    logger_b.warning(f"⚠️ 删除音频文件失败: {_ap}, 错误: {str(e)}")

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
        # 读内存声纹库（mtime 变化时才重读盘，避免每次请求 JSON 解析）
        load_speaker_db_if_changed()
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

# =================== 声纹注册防重复（内容指纹） ===================
def _sample_fingerprint(wav_path):
    """预处理后标准 wav 的 MD5 —— 同源音频字节一致，作为防重复注册指纹"""
    try:
        with open(wav_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None

def _find_voiceprint_duplicate(fp):
    """跨说话人全局查重。返回 (说话人, 样本) 或 (None, None)。
    顺带懒补存量样本缺失的指纹（从已存档的样本 wav 计算）。"""
    if not fp:
        return None, None
    fp_added = False
    for pd_ in speaker_db.values():
        for s_ in pd_.get('samples', []):
            if not s_.get('fingerprint'):
                s_['fingerprint'] = _sample_fingerprint(s_.get('audio_path', ''))
                if s_['fingerprint']:
                    fp_added = True
    hit_spk, hit_s = None, None
    for spk, pd_ in speaker_db.items():
        for s_ in pd_.get('samples', []):
            if s_.get('fingerprint') == fp:
                hit_spk, hit_s = spk, s_
                break
        if hit_spk:
            break
    if fp_added:
        try:
            with open(Config.SPEAKER_DB_FILE, 'w', encoding='utf-8') as f:
                json.dump(speaker_db, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger_b.warning(f"声纹库指纹懒补落盘失败（不影响本次注册）: {e}")
    return hit_spk, hit_s


# =================== 声纹注册质量守门 ===================
# 铁律: 单人清晰说话短音频入库; 太短嵌入不稳, 太长/错人样本会稀释声纹
VP_MIN_DURATION_S = 1.5    # 低于此长度嵌入不稳定
VP_MAX_DURATION_S = 30.0   # 过长可能混入多人
VP_MIN_SIMILARITY = 0.5    # 增强模式: 新样本与现有平均的最低余弦相似度（同域同人实测 0.7+）

def _vp_wav_duration(path):
    """读 wav 时长（秒），异常返回 0"""
    import wave as _wave
    try:
        with _wave.open(path, 'rb') as w:
            return w.getnframes() / max(1, w.getframerate())
    except Exception:
        return 0.0

def _vp_cosine(a, b):
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def _voiceprint_quality_gate(proc_temp, speaker_name, sample_embeddings, enhance_mode):
    """注册质量守门: ①时长门槛 ②增强模式相似度防稀释。
    返回 (ok, similarity, error_dict, http_status)；ok=True 时 error_dict 为 None。"""
    dur = _vp_wav_duration(proc_temp)
    if dur < VP_MIN_DURATION_S:
        return False, None, {"error": f"切片太短（{dur:.1f}s），嵌入不稳定。请选 3 秒以上的清晰单人说话片段"}, 400
    if dur > VP_MAX_DURATION_S:
        return False, None, {"error": f"切片过长（{dur:.1f}s），可能混入多人。请选 5~10 秒的单人清晰片段"}, 400
    if enhance_mode and speaker_name in speaker_db:
        avg_embs = speaker_db[speaker_name].get("avg_embeddings") or {}
        sims = [_vp_cosine(sample_embeddings[m], avg_embs[m]) for m in sample_embeddings if m in avg_embs]
        if sims:
            best = max(sims)
            if best < VP_MIN_SIMILARITY:
                logger_b.warning(f"🚫 声纹质量守门拒绝: 「{speaker_name}」新样本相似度 {best:.2f} < {VP_MIN_SIMILARITY}")
                return False, round(best, 3), {
                    "error": f"该样本与「{speaker_name}」现有声纹相似度仅 {best:.2f}，疑似不是同一人。为防稀释已拒绝注册；若确属本人请换清晰片段重试",
                    "similarity": round(best, 3), "below_threshold": True
                }, 409
            return True, round(best, 3), None, 0
    return True, None, None, 0

@app.route("/speaker/register", methods=["POST"])
@admin_required
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

            # 内容指纹防重复注册（跨说话人全局查重）
            fp = _sample_fingerprint(proc_temp)
            dup_spk, dup_s = _find_voiceprint_duplicate(fp)
            if dup_spk:
                logger_b.info(f"🚫 拒绝重复注册: 该音频已归属 [{dup_spk}] (样本 {dup_s.get('id')}, {dup_s.get('timestamp')})")
                return jsonify({
                    "error": f"该音频已注册到「{dup_spk}」名下（{dup_s.get('timestamp', '')}），无需重复注册",
                    "duplicate": True, "existing_speaker": dup_spk
                }), 409

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

            # 质量守门: 时长 + 增强模式相似度（防稀释）
            ok, sim, qerr, qstatus = _voiceprint_quality_gate(proc_temp, speaker_name, sample_embeddings, enhance_mode)
            if not ok:
                return jsonify(qerr), qstatus

            # 生成唯一的样本ID
            sample_id = f"{int(time.time())}_{hash(audio_file.filename) % 10000}"

            # 保存处理后的音频文件
            sample_audio_path = os.path.join(speaker_dir, f"{sample_id}.wav")
            shutil.copy2(proc_temp, sample_audio_path)

            sample_info = {
                "id": sample_id,
                "filename": audio_file.filename,
                "source_key": request.form.get('source_key', ''),
                "source": "web",
                "fingerprint": fp,
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

            resp = {
                "message": f"Speaker '{speaker_name}' {'enhanced' if enhance_mode else 'registered'} successfully.",
                "sample_count": len(speaker_db[speaker_name]["samples"])
            }
            if sim is not None:
                resp["similarity"] = sim  # 与现有平均嵌入的余弦相似度, 供前端建立手感
            return jsonify(resp)

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

# =================== 负样本黑名单（拒绝器） ===================
# 用户在移动端把"电视/动画等非家人声音"标为负样本 → 向量以个体形式存档
# 归属时只做"拒绝"（更像黑名单就不标任何人），绝不参与正向归属
NEGATIVE_DB_FILE = "negative_samples.json"
negative_samples = []
neg_lock = threading.Lock()
NEG_PENDING_DIR = os.path.join("negative_samples", "pending")

def _process_negative_job(job_file):
    """后台处理单个负样本建模任务: 预处理 + GPU提取三模型向量 + 入黑名单"""
    temp_files = []
    try:
        with open(job_file, 'r', encoding='utf-8') as f:
            job = json.load(f)
        wav_path = job.get('wav') or ''
        src_name = job.get('source') or 'seg.wav'
        if not wav_path or not os.path.exists(wav_path):
            raise RuntimeError(f"待处理音频缺失: {wav_path}")
        proc_temp = os.path.join(Config.TEMP_DIR, f"neg_proc_{int(time.time())}.wav")
        temp_files.append(proc_temp)
        if not preprocess_audio(wav_path, proc_temp):
            raise RuntimeError("音频预处理失败")
        embeddings = {}
        with gpu_lock:
            for model_name, sv_pipe in sv_pipelines.items():
                emb = extract_embedding_from_file(sv_pipe, proc_temp)
                if emb is not None:
                    embeddings[model_name] = emb.tolist()
        if not embeddings:
            raise RuntimeError("声纹提取失败")
        entry_id = job.get('id') or f"neg_{time.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        entry = {
            "id": entry_id,
            "ts": time.strftime('%Y-%m-%d %H:%M:%S'),
            "source": src_name[:200],
            "embeddings": embeddings,
        }
        neg_dir = "negative_samples"
        os.makedirs(neg_dir, exist_ok=True)
        entry["audio"] = os.path.join(neg_dir, f"{entry_id}.wav")
        try:
            shutil.copy2(proc_temp, entry["audio"])
        except Exception as e:
            logger_sys.warning(f"负样本音频副本保存失败(不影响黑名单): {e}")
            entry.pop("audio", None)
        with neg_lock:
            negative_samples.append(entry)
            with open(NEGATIVE_DB_FILE, 'w', encoding='utf-8') as f:
                json.dump(negative_samples, f, ensure_ascii=False)
        logger_b.info(f"🚫 [负样本] 已建模入黑名单: {entry['id']} | 共 {len(negative_samples)} 条")
        for p in (job_file, wav_path):
            try:
                os.remove(p)
            except Exception:
                pass
        return True
    except Exception as e:
        logger_b.error(f"负样本建模失败({os.path.basename(job_file)}): {e}")
        try:
            os.rename(job_file, job_file + ".failed")  # 保留现场, 不阻塞后续任务
        except Exception:
            pass
        return False
    finally:
        for tmp in temp_files:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

def _negative_worker_loop():
    os.makedirs(NEG_PENDING_DIR, exist_ok=True)
    while True:
        try:
            if not sv_pipelines:
                time.sleep(5)   # 模型未就绪(启动中/加载失败)时等待
                continue
            jobs = sorted(fn for fn in os.listdir(NEG_PENDING_DIR) if fn.endswith('.json'))
            if not jobs:
                time.sleep(2)
                continue
            _process_negative_job(os.path.join(NEG_PENDING_DIR, jobs[0]))
        except Exception as e:
            logger_sys.error(f"负样本工作线程异常: {e}")
            time.sleep(5)

def start_negative_worker():
    threading.Thread(target=_negative_worker_loop, daemon=True, name="negative-worker").start()
    logger_sys.info("🚫 负样本后台建模线程已启动")

def load_negative_samples():
    global negative_samples
    try:
        if os.path.exists(NEGATIVE_DB_FILE):
            with open(NEGATIVE_DB_FILE, 'r', encoding='utf-8') as f:
                negative_samples = json.load(f)
        # ID 唯一性清理: 历史数据可能出现重复 id(旧版时间戳+hash 生成), 仅保留首条
        seen, uniq = set(), []
        for n in negative_samples:
            nid = n.get('id')
            if nid and nid in seen:
                continue
            seen.add(nid)
            uniq.append(n)
        if len(uniq) != len(negative_samples):
            logger_sys.warning(f"负样本黑名单发现重复 id: {len(negative_samples)} → {len(uniq)} 条, 已去重")
            negative_samples = uniq
            try:
                with open(NEGATIVE_DB_FILE, 'w', encoding='utf-8') as f:
                    json.dump(negative_samples, f, ensure_ascii=False)
            except Exception as e:
                logger_sys.error(f"负样本黑名单去重回写失败: {e}")
        logger_sys.info(f"📦 负样本黑名单已加载: {len(negative_samples)} 条")
    except Exception as e:
        logger_sys.error(f"负样本黑名单加载失败: {e}")
        negative_samples = []

def _negative_source_exists(src):
    """同源去重: 该切片路径已在黑名单或待处理队列中则返回 True"""
    if not src:
        return False
    with neg_lock:
        if any(n.get('source') == src for n in negative_samples):
            return True
    try:
        if os.path.isdir(NEG_PENDING_DIR):
            for fn in glob.glob(os.path.join(NEG_PENDING_DIR, '*.json')):
                try:
                    with open(fn, 'r', encoding='utf-8') as f:
                        if json.load(f).get('source') == src:
                            return True
                except Exception:
                    continue
    except Exception:
        pass
    return False

@app.route('/speaker/negative', methods=['POST'])
@admin_required
def register_negative_sample():
    """登记负样本(异步): 音频入队即返回202, 后台线程提取三模型向量进黑名单(仅拒绝用)"""
    try:
        if 'audio_file' not in request.files:
            return jsonify({"error": "audio_file is required"}), 400
        file = request.files['audio_file']
        src_name = request.form.get('source_path', '').strip() or (file.filename or 'seg.wav')
        if _negative_source_exists(src_name):
            logger_b.info(f"🚫 [负样本] 重复标记忽略: {src_name[:80]}")
            return jsonify({"ok": True, "duplicate": True})
        os.makedirs(NEG_PENDING_DIR, exist_ok=True)
        job_id = f"neg_{time.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        wav_path = os.path.join(NEG_PENDING_DIR, f"{job_id}.wav")
        file.save(wav_path)
        with open(os.path.join(NEG_PENDING_DIR, f"{job_id}.json"), 'w', encoding='utf-8') as f:
            json.dump({"id": job_id, "source": src_name[:200], "wav": wav_path,
                       "ts": time.strftime('%Y-%m-%d %H:%M:%S')}, f, ensure_ascii=False)
        logger_b.info(f"🚫 [负样本] 已入队: {job_id} | {src_name[:80]}")
        return jsonify({"ok": True, "queued": True, "id": job_id}), 202
    except Exception as e:
        logger_b.error(f"负样本入队失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/speaker/negative/list', methods=['GET'])
def list_negative_samples():
    """负样本黑名单列表(不含向量, 供样本管理页核对; 接口低频, 直接读盘保证与其它端写入一致)"""
    with neg_lock:
        load_negative_samples()
        items = [{"id": n.get("id"), "ts": n.get("ts"), "source": n.get("source"),
                  "has_audio": bool(n.get("audio") and os.path.exists(n.get("audio")))}
                 for n in reversed(negative_samples)]
    return jsonify({"total": len(items), "samples": items})

@app.route('/speaker/negative/<neg_id>/audio', methods=['GET'])
def get_negative_sample_audio(neg_id):
    """试听负样本音频副本"""
    with neg_lock:
        entry = next((n for n in negative_samples if n.get("id") == neg_id), None)
    if not entry:
        return jsonify({"error": "negative sample not found"}), 404
    ap = entry.get("audio")
    if not ap or not os.path.exists(ap):
        return jsonify({"error": "audio file missing"}), 404
    return send_file(ap, mimetype="audio/wav", as_attachment=True, download_name=f"{neg_id}.wav")

@app.route('/speaker/negative/<neg_id>', methods=['DELETE'])
@admin_required
def delete_negative_sample(neg_id):
    """从黑名单移除单条负样本(回滚)"""
    global negative_samples
    with neg_lock:
        before = len(negative_samples)
        entry = next((n for n in negative_samples if n.get("id") == neg_id), None)
        negative_samples = [n for n in negative_samples if n.get("id") != neg_id]
        if len(negative_samples) == before:
            return jsonify({"error": "not found"}), 404
        try:
            with open(NEGATIVE_DB_FILE, 'w', encoding='utf-8') as f:
                json.dump(negative_samples, f, ensure_ascii=False)
        except Exception as e:
            logger_sys.error(f"负样本黑名单写盘失败: {e}")
    if entry and entry.get("audio") and os.path.exists(entry["audio"]):
        try:
            os.remove(entry["audio"])
        except Exception:
            pass
    logger_b.info(f"🗑️ [负样本] 已移除: {neg_id} | 剩余 {len(negative_samples)} 条")
    return jsonify({"ok": True, "total": len(negative_samples)})

@app.route("/register", methods=["POST"])
@admin_required
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
            skipped_dup = []
            skipped_quality = []   # 质量守门不合格（太短/太长/相似度过低）
            batch_sims = {}        # filename -> 与现有声纹的相似度（增强模式）
            batch_fps = {}
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

                # 内容指纹防重复注册（跨说话人全局查重 + 本批次互查）
                fp_i = _sample_fingerprint(proc_temp)
                if fp_i and (fp_i in batch_fps or _find_voiceprint_duplicate(fp_i)[0]):
                    dup_owner = batch_fps.get(fp_i) or _find_voiceprint_duplicate(fp_i)[0]
                    logger_b.info(f"🚫 跳过重复文件 {file.filename}: 已归属 [{dup_owner}]")
                    skipped_dup.append(file.filename)
                    continue
                if fp_i:
                    batch_fps[fp_i] = speaker_name

                # 为每个模型提取嵌入
                sample_embeddings = {}
                for model_name, sv_pipe in sv_pipelines.items():
                    emb = extract_embedding_from_file(sv_pipe, proc_temp)
                    if emb is not None:
                        sample_embeddings[model_name] = emb.tolist()
                        model_embeddings[model_name].append(emb)
                    else:
                        logger_b.warning(f"⚠️ 从 {file.filename} 提取 {model_name} embedding 失败。")

                # 质量守门: 时长门槛 + 增强模式相似度防稀释（不合格跳过该文件, 不拖累其余）
                if sample_embeddings:
                    ok, sim, qerr, qstatus = _voiceprint_quality_gate(proc_temp, speaker_name, sample_embeddings, enhance_mode)
                    if not ok:
                        logger_b.info(f"🚫 质量守门跳过 {file.filename}: {qerr.get('error', '')[:80]}")
                        skipped_quality.append(f"{file.filename}（{qerr.get('error', '').split('。')[0]}）")
                        continue
                    if sim is not None:
                        batch_sims[file.filename] = sim

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
                        "source_key": request.form.get('source_key', ''),
                        "source": "upload",
                        "fingerprint": fp_i,
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
                if skipped_dup and not new_samples:
                    return jsonify({
                        "error": f"全部 {len(skipped_dup)} 个文件都是重复样本（已存在于声纹库），未做任何修改",
                        "duplicate": True, "duplicates_skipped": skipped_dup
                    }), 409
                if skipped_quality and not new_samples:
                    return jsonify({
                        "error": f"全部 {len(skipped_quality)} 个文件未通过质量守门: {'; '.join(skipped_quality[:3])}",
                        "quality_skipped": skipped_quality
                    }), 409
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
            resp = {
                "message": f"Speaker '{speaker_name}' {action} successfully.",
                "samples_added": len(new_samples)
            }
            if skipped_dup:
                resp["duplicates_skipped"] = skipped_dup
                resp["message"] += f"（跳过 {len(skipped_dup)} 个重复文件: {', '.join(skipped_dup[:3])}{'...' if len(skipped_dup) > 3 else ''}）"
            if skipped_quality:
                resp["quality_skipped"] = skipped_quality
                resp["message"] += f"（质量守门跳过 {len(skipped_quality)} 个: {'; '.join(skipped_quality[:2])}{'...' if len(skipped_quality) > 2 else ''}）"
            if batch_sims:
                resp["similarities"] = batch_sims
                avg_sim = sum(batch_sims.values()) / len(batch_sims)
                resp["similarity"] = round(avg_sim, 3)
            return jsonify(resp)

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


@app.route("/api/voiceprint_registered", methods=["GET"])
@admin_required
def get_voiceprint_registered():
    """已注册声纹的样本清单 —— 前端「记录」页据此标注已入库的段，防止重复注册"""
    registered = []
    for spk, pd in speaker_db.items():
        for s in pd.get('samples', []):
            registered.append({
                "speaker": spk,
                "filename": s.get('filename', ''),
                "source_key": s.get('source_key', ''),
                "timestamp": s.get('timestamp', ''),
            })
    source_keys = sorted({r['source_key'] for r in registered if r['source_key']})
    return jsonify({
        "registered": registered,
        "source_keys": source_keys,
        "total": len(registered)
    })

@app.route("/transcribe", methods=["POST"])
@app.route("/transcribes", methods=["POST"])
@admin_required
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

            # 更新 B 轨统计（同步端点：收到即处理，接收时刻窗口即实时吞吐速率）
            with _b_stats_lock:
                _b_stats["today_record_count"] += 1
                _b_stats["last_event_time"] = datetime.now().strftime('%H:%M:%S')
                now_ts = time.time()
                w = _b_stats["done_window"]
                w.append(now_ts)
                while w and now_ts - w[0] > 3600:
                    w.pop(0)

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
            # 【2026-10-04 夜间降级】audio_processor 对凌晨(1-6点)录音携带 skip_asr=true：
            # 跳过 FunASR 转写（GPU 让给凌晨哭声补跑），轨道A哭声检测照跑——
            # 半夜哭声告警不再盲区；转写由历史补跑链路兜底，凌晨对话本就稀少。
            # backfill_pixels / reprocess_history_cries 的提交不带此参数，不受影响。
            if request.form.get('skip_asr', 'false').lower() == 'true':
                logger_b.info("  🌙 [夜间降级] 跳过 VAD & ASR 转写，仅执行轨道A哭声检测")
                res = None
            else:
                _gen_kwargs = dict(language="auto", use_itn=True, use_punc=True)
                if Config.ASR_HOTWORD:
                    _gen_kwargs["hotword"] = Config.ASR_HOTWORD  # SeACo 热词偏置
                if Config.VAD_ENGINE == "silero":
                    try:
                        res = _generate_with_silero(asr_pipeline, proc_temp, _gen_kwargs)
                    except Exception as _vad_ex:
                        logger_b.warning(f"  ⚠️ Silero VAD 异常（{_vad_ex.__class__.__name__}: {_vad_ex}），本次回退 fsmn-vad 全链路")
                        res = asr_pipeline.generate(input=proc_temp, **_gen_kwargs)
                else:
                    res = asr_pipeline.generate(input=proc_temp, **_gen_kwargs)

            # 【轨道A: 独立哭声检测】直接对完整 60s 原始音频做声纹匹配
            # 使用 CryDetectionConfig 独立参数，与轨道B (VAD+语音识别) 完全隔离
            cry_detected = False
            cry_detection_completed = False
            skip_cry_flag = request.form.get('skip_cry', 'false').lower() == 'true'
            # 【2026-09-21】历史文件防线：录音时间早于 6 小时前的一律按历史文件处理——
            # 不发即时报警邮件、不发 Webhook、不写实时事件（历史检测由补跑脚本负责，
            # 那条链路有自己的入库方式）。防止手机端积压补传的旧录音在凌晨触发
            # 轰炸式"重复告警"。与 audio_processor 的提交侧防线互为兜底。
            _rec_t = None
            try:
                from db_manager import parse_recording_time as _prt
                _rec_t = _prt(file.filename)
                if _rec_t and (datetime.now() - _rec_t).total_seconds() > CryDetectionConfig.HISTORY_DROP_AGE_HOURS * 3600:
                    skip_cry_flag = True
            except Exception:
                pass
            # 上传来源设备名（audio_processor 从 NAS 路径推导，如 Sony-2），用于哭声报警 Webhook
            source_device = (request.form.get('source_device') or '').strip()
            try:
                cry_detected, cry_confidence, cry_details = detect_cry_from_full_audio(proc_temp, source_filename=file.filename)
                cry_detection_completed = True

                if cry_detected:
                    logger_b.info(f"  🍼 [轨道A] 哭声确认! 文件={file.filename}, 置信度={cry_confidence:.3f}, 启动报警流程...")
                    logger_b.info(f"      检测详情: {' | '.join(cry_details)}")

                    # 更新 B 轨统计
                    with _b_stats_lock:
                        _b_stats["today_cry_count"] = _b_stats.get("today_cry_count", 0) + 1
                        _b_stats["last_cry_time"] = datetime.now().strftime('%H:%M:%S')

                    if skip_cry_flag:
                        logger_b.info(f"      [skip_cry] 哭声已标记，历史模式不发送即时邮件")
                    else:
                        # 【2026-09-21 哭声回合合并】先看能否并入进行中的回合：
                        # 10 分钟内连续检出的哭声录音属于同一场哭闹，并入最近一个
                        # 仍处于分析占位状态的事件（追加 event_files），不新建事件、
                        # 不重复报警。深度分析由该事件的延迟线程在合并窗口关闭后
                        # 统一执行（其上下文收集会覆盖全部相邻录音）。
                        _merge_target = None
                        try:
                            from db_manager import find_mergeable_cry_event, append_file_to_cry_event
                            _merge_target = find_mergeable_cry_event(_rec_t, window_minutes=10)
                        except Exception as _merge_err:
                            logger_b.warning(f"      ⚠️ [回合合并] 合并检查失败: {_merge_err}")

                        if _merge_target:
                            _merged = append_file_to_cry_event(_merge_target['id'], file.filename)
                            if _merged:
                                logger_b.info(
                                    f"      🍼 [轨道A] 哭声并入回合 #{_merge_target['id']}"
                                    f"（距上一段 {abs(_merge_target['diff_sec'])/60:.1f} 分钟，同一回合），不重复报警"
                                )
                            else:
                                logger_b.warning(f"      ⚠️ [回合合并] 文件追加失败，按新回合处理")
                                _merge_target = None  # 追加失败走新回合，避免丢事件

                        # 【2026-10-06 根因修复】原代码此处 `else:` 缩进为 24 空格，误绑定到
                        # `if not _merge_target:`（而非 `if in_cooldown:`）。后果：下面整段
                        # "正式报警"（写占位事件 + 发即时邮件 + Webhook + 延迟深度分析）
                        # 只在"找到可并入回合"时执行；而全新哭闹回合（无 analyzing 事件可并入）
                        # 是最常见情形 → 永远不入库、不报警。这正是 10 月"一次没提醒"的根因。
                        # 正确语义：无回合可并入 且 不在冷却期 → 走正式报警。
                        _do_cry_alert = False
                        if not _merge_target:
                            # 冷却机制
                            global _last_cry_trigger_time
                            now = time.time()
                            with _cry_cooldown_lock:
                                in_cooldown = (now - _last_cry_trigger_time) < CryDetectionConfig.COOLDOWN_SEC
                                if not in_cooldown:
                                    _last_cry_trigger_time = now

                            if in_cooldown:
                                elapsed = int(now - _last_cry_trigger_time)
                                logger_b.info(f"      [冷却中] 距上次哭声分析 {elapsed}s，冷却期 {CryDetectionConfig.COOLDOWN_SEC}s 内跳过")
                            else:
                                _do_cry_alert = True

                        if _do_cry_alert:
                            # ── 正式报警：先保存占位，后续在分析和插图生成后发送邮件 ──

                            # 注意：proc_temp 会在请求结束后被 finally 清理，需要先复制到持久位置
                            import shutil as _shutil
                            _cry_persist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_cry")
                            os.makedirs(_cry_persist_dir, exist_ok=True)
                            _persist_ts = int(time.time())
                            _persist_audio_path = os.path.join(_cry_persist_dir, f"cry_{_persist_ts}.wav")
                            _shutil.copy2(proc_temp, _persist_audio_path)
                            logger_b.info(f"💾 [BabyCry] 已复制音频到持久路径: {_persist_audio_path}")

                            # [即时占位] 立即写入数据库（包含声纹投票详情）
                            from db_manager import save_cry_analysis
                            placeholder_id = save_cry_analysis(
                                file.filename, 0, audio_duration,
                                "深度分析中 (等待合并窗口关闭)...", "请稍候内容更新",
                                reason_category="analyzing", event_files=[],
                                audio_path=_persist_audio_path,
                                confidence=cry_confidence,
                                details=cry_details,
                                device=source_device or None
                            )

                            # 更新持久文件名包含 placeholder_id
                            if placeholder_id:
                                _new_persist_path = os.path.join(_cry_persist_dir, f"cry_{placeholder_id}_{_persist_ts}.wav")
                                os.rename(_persist_audio_path, _new_persist_path)
                                _persist_audio_path = _new_persist_path
                                try:
                                    from db_manager import update_cry_event_audio_path
                                    update_cry_event_audio_path(placeholder_id, _persist_audio_path)
                                except Exception as path_update_err:
                                    logger_b.warning(f"⚠️ [BabyCry] 更新持久音频路径失败: {path_update_err}")

                                # 【2026-09-20】后台预切哭声候选片段：报警不被阻塞，GPU 由 gpu_lock 排队
                                if placeholder_id:
                                    import threading as _threading
                                    _t = _threading.Thread(
                                        target=preset_cry_segments,
                                        args=(placeholder_id, _persist_audio_path),
                                        name=f"preset-seg-{placeholder_id}", daemon=True)
                                    _t.start()

                            def start_delayed_analysis(fname, a_path, dur, p_id, cry_conf, cry_det, src_device=""):
                                # 从文件名提取时间作为第一封邮件的时间范围
                                from db_manager import parse_recording_time
                                rec_time = parse_recording_time(fname)
                                if rec_time:
                                    time_range_str = rec_time.strftime('%Y-%m-%d %H:%M:%S')
                                else:
                                    time_range_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                                # ── 第1封邮件: 即时报警 (立即发送) ──
                                logger_a.info(f"📧 [邮件] 发送即时报警邮件...")
                                send_cry_alert_email(
                                    fname, cry_conf, cry_det,
                                    reason="正在深度分析中，请稍候...",
                                    advice="系统正在收集完整上下文音频，稍后将发送详细分析报告",
                                    category="analyzing",
                                    image_data=None,
                                    time_range=time_range_str
                                )

                                # 【2026-09-20】同步向外部 Webhook 推送哭声报警
                                # （.env: CRY_WEBHOOK_URL/CRY_WEBHOOK_TOKEN，异步不阻塞，仅即时报警发一次）
                                send_cry_webhook(
                                    cry_conf,
                                    event_id=p_id,
                                    filename=fname,
                                    recording_time=rec_time.strftime('%Y-%m-%d %H:%M:%S') if rec_time else None,
                                    time_range=time_range_str,
                                    audio_duration=dur,
                                    device=src_device or None,
                                    models=cry_det,
                                )

                                # 【2026-09-21 回合合并】延迟 720s = 合并窗口 10min + 2min 缓冲，
                                # 确保窗口内后续并入的录音都被收齐后，一次性做深度分析
                                # （process_baby_cry_async 的上下文收集会覆盖全部相邻录音）
                                logger_b.info(f"⏳ [BabyCry] 已启动延迟分析线程，等待 720s (合并窗口关闭) 后更新占位 (ID={p_id})...")
                                time.sleep(720)

                                # 执行分析并获取结果
                                logger_b.info(f"🔍 [BabyCry] 开始执行深度分析...")
                                reason, advice, category = process_baby_cry_async(fname, a_path, 0, dur * 1000, placeholder_id=p_id, cry_conf=cry_conf, cry_det=cry_det)

                                # 验证数据库更新是否成功
                                analysis_ok = False
                                event_time_range = time_range_str
                                try:
                                    from db_manager import get_baby_cry_event_by_id
                                    record = get_baby_cry_event_by_id(p_id)
                                    if record:
                                        reason = record.get('reason')
                                        advice = record.get('advice')
                                        category = record.get('reason_category')
                                        # 判断分析是否真正完成（不是占位文本）
                                        if category and category != 'analyzing':
                                            analysis_ok = True

                                        # 获取事件的时间范围
                                        rec_time = record.get('recording_time')
                                        start_time = record.get('start_time', 0)
                                        end_time = record.get('end_time', 0)
                                        if rec_time:
                                            from datetime import timedelta
                                            start_dt = datetime.fromisoformat(rec_time) + timedelta(milliseconds=start_time)
                                            end_dt = datetime.fromisoformat(rec_time) + timedelta(milliseconds=end_time)
                                            event_time_range = f"{start_dt.strftime('%Y-%m-%d %H:%M:%S')} ~ {end_dt.strftime('%H:%M:%S')}"

                                        logger_b.info(f"✅ [BabyCry] 数据库记录: category={category}, reason={reason[:50] if reason else 'None'}..., 分析成功={analysis_ok}")
                                    else:
                                        logger_b.error(f"❌ [BabyCry] 数据库记录未找到 (ID={p_id})")
                                except Exception as e:
                                    logger_b.error(f"❌ [BabyCry] 验证数据库更新失败: {e}")

                                # 生成插图（仅分析成功时）
                                image_data_for_email = None
                                image_url = None  # 5008 侧相对路径（/api/illustration/xxx），生成失败保持 None
                                try:
                                    if analysis_ok and reason and reason != "未知" and "深度分析中" not in (reason or ""):
                                        logger_a.info(f"🎨 [邮件插图] 开始生成邮件插图...")
                                        image_prompt = (
                                            f"创作一幅温暖治愈的儿童绘本风格卡通插图。"
                                            f"主角：一个可爱的2岁半中国宝宝（2023年8月出生）。"
                                            f"场景描述：{reason}。"
                                            f"展现这个宝宝在此情境下的真实日常生活场景。"
                                            f"风格：柔和的粉彩色调、柔和的灯光、可爱的卡通形象、情感丰富、表情生动，"
                                            f"绘本插画风格、温馨氛围、细节丰富的背景。"
                                            f"重要：请在画面中添加中文文字（如对话框、场景标注等），使用中文。"
                                        )
                                        image_url = call_gemini_image_api(image_prompt)
                                        if image_url:
                                            logger_a.info(f"🎨 [邮件插图] ✅ 插图生成成功: {image_url}")
                                            # 更新数据库中的 illustration_url
                                            try:
                                                from db_manager import update_cry_event_image_by_id
                                                update_cry_event_image_by_id(p_id, image_url)
                                                logger_a.info(f"🎨 [邮件插图] 已更新数据库 illustration_url (ID={p_id})")
                                            except Exception as db_img_err:
                                                logger_a.error(f"🎨 [邮件插图] 更新数据库失败: {db_img_err}")
                                            # 将文件路径转为 base64 以便邮件嵌入
                                            try:
                                                illustration_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "illustrations")
                                                img_filename = image_url.replace("/api/illustration/", "")
                                                img_full_path = os.path.join(illustration_dir, img_filename)
                                                if os.path.exists(img_full_path):
                                                    import base64 as _b64
                                                    with open(img_full_path, 'rb') as _f:
                                                        img_bytes = _f.read()
                                                    image_data_for_email = f"data:image/jpeg;base64,{_b64.b64encode(img_bytes).decode()}"
                                                    logger_a.info(f"🎨 [邮件插图] 已转为 base64 ({len(img_bytes)} bytes) 用于邮件嵌入")
                                                else:
                                                    logger_a.warning(f"🎨 [邮件插图] 插图文件不存在: {img_full_path}")
                                            except Exception as img_err:
                                                logger_a.error(f"🎨 [邮件插图] base64 转换失败: {img_err}")
                                        else:
                                            logger_a.warning(f"🎨 [邮件插图] 生成返回 None")
                                    else:
                                        logger_a.warning(f"🎨 [邮件插图] 跳过生成：分析未完成或 reason 无效 (category={category})")
                                except Exception as e:
                                    logger_a.error(f"🎨 [邮件插图] 生成失败: {e}")
                                    import traceback
                                    logger_a.error(f"🎨 [邮件插图] 异常堆栈: {traceback.format_exc()}")

                                # ── 第2封邮件: 完整深度分析报告 ──
                                if analysis_ok:
                                    logger_a.info(f"📧 [邮件] 发送深度分析完整报告邮件...")
                                else:
                                    logger_a.warning(f"📧 [邮件] 分析未完成，发送失败通知邮件...")
                                    reason = reason or "深度分析未能完成"
                                    advice = advice or "系统自动分析失败，请稍后查看页面获取最新状态"
                                    category = category or "分析失败"

                                send_cry_alert_email(
                                    fname, cry_conf, cry_det,
                                    reason=reason, advice=advice, category=category,
                                    image_data=image_data_for_email,
                                    time_range=event_time_range
                                )

                                # 【2026-09-20】深度分析完成后，向外部 Webhook 推送"分析报告"（第二推）
                                # 无论插图是否生成成功，均如实上报分析报告（有图带图，无图亦推送原因与安抚建议）
                                if analysis_ok or image_url:
                                    send_cry_analysis_webhook(
                                        event_id=p_id,
                                        status="ok" if analysis_ok else "failed",
                                        category=category,
                                        reason=reason,
                                        advice=advice,
                                        confidence=cry_conf,
                                        filename=fname,
                                        recording_time=rec_time.strftime('%Y-%m-%d %H:%M:%S') if rec_time else None,
                                        time_range=event_time_range,
                                        device=src_device or None,
                                        illustration_path=image_url,
                                    )

                                # 分析失败时保留音频，供未完成事件自动重试使用。
                                if analysis_ok:
                                    try:
                                        if os.path.exists(a_path):
                                            os.remove(a_path)
                                            logger_b.info(f"🗑️ [BabyCry] 已清理持久音频: {a_path}")
                                    except:
                                        pass
                                else:
                                    logger_b.info(f"💾 [BabyCry] 分析未完成，保留持久音频等待自动重试: {a_path}")

                            threading.Thread(
                                target=start_delayed_analysis,
                                args=(file.filename, _persist_audio_path, audio_duration, placeholder_id, cry_confidence, cry_details, source_device),
                                daemon=True
                            ).start()
            except Exception as ex:
                logger_b.warning(f"    ⚠️ [哭声检测] 检测过程异常: {ex}")
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
                        sensevoice_text = None  # 只在长段分支赋值，短段/切片失败时保持 None（否则 UnboundLocalError，Silero 短段一多就暴露）

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

                            # 按日期分类的目录结构: <设备>/audio_segments/YYYY-MM-DD/filename/（双源按设备隔离）
                            _seg_dev = source_device if source_device else (SOURCE_DEVICES[0] if SOURCE_DEVICES else "unknown")
                            segments_dir = os.path.join(RECORDS_ROOT, _seg_dev, "audio_segments", date_str, base_filename)
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

                                # 3. 补全其他信息 (Whisper/Emotion/Nano)
                                whisper_text = None
                                nano_text = None
                                emotion = sensevoice_emotion

                                if identity is not None:
                                    if emotion is None:
                                        emotion = detect_emotion_for_segment(seg_wav_temp)
                                    # Whisper 对比转写已移除（效果不佳）, whisper_text 恒为 null
                                    nano_text = transcribe_with_nano(seg_wav_temp)
                                    logger_b.info(f"      [性能] 已识别说话人 {identity}")
                                else:
                                    logger_b.info(f"      [性能] 未识别说话人，跳过后续处理")
                                    whisper_text = None
                                    nano_text = None

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
                            nano_text = None


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
                            "nano_text": nano_text,
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
                    logger_b.info("  🍼 [轨道A] 补偿机制启动: VAD未命中，手动添加全局哭声片段。")
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

                    # 保存到数据库（device: 上传来源设备, 供前端标注）
                    success = save_to_db(file.filename, full_text, processed_segments, recording_time, summary,
                                         device=source_device or None)

                    if success:
                        # 归属统计: 让仪表盘结论能区分「切片总数」与「已归属注册人」
                        _spk_cnt = {}
                        for _s in processed_segments:
                            _sp = (_s.get("spk") or "Unknown") if isinstance(_s, dict) else "Unknown"
                            _spk_cnt[_sp] = _spk_cnt.get(_sp, 0) + 1
                        _spk_detail = " · ".join(f"{k}{v}" for k, v in sorted(_spk_cnt.items(), key=lambda x: -x[1]))
                        logger_b.info(f"📊 归属统计: {len(processed_segments)}段 · {_spk_detail}")
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
            else:
                # 0 有效语音段（纯静音/未过 VAD）或夜间降级文件不入库，但必须打带 recording_time
                # 的闭合标记——否则 web_viewer 仪表盘的日志配对永远无法闭合，文件会一直
                # 假性显示"处理中"（2026-10-04）
                if request.form.get('skip_asr', 'false').lower() == 'true':
                    logger_b.info(f"⭕ 夜间降级仅哭声检测, 跳过转写入库 (recording_time: {parse_recording_time(file.filename)})")
                else:
                    logger_b.info(f"⭕ 无有效语音段, 跳过入库 (recording_time: {parse_recording_time(file.filename)})")
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

def _resolve_sample_audio_path(sample):
    """解析样本音频绝对路径: 兼容旧库中的 Windows 反斜杠路径与相对路径"""
    ap = (sample.get("audio_path") or "").replace("\\", "/")
    if not ap:
        return None
    if not os.path.isabs(ap):
        ap = os.path.join(os.path.dirname(os.path.abspath(__file__)), ap)
    return ap if os.path.exists(ap) else None


@app.route("/speaker/<speaker_name>/sample/<sample_id>/audio")
def get_sample_audio(speaker_name, sample_id):
    """获取指定说话人样本的音频文件"""
    try:
        # 读内存声纹库（mtime 变化时才重读盘，避免每次请求 JSON 解析）
        load_speaker_db_if_changed()
        if speaker_name not in speaker_db:
            return jsonify({"error": f"Speaker '{speaker_name}' not found."}), 404

        speaker_data = speaker_db[speaker_name]
        if "samples" not in speaker_data:
            return jsonify({"error": f"No samples found for speaker '{speaker_name}'."}), 404

        # 查找指定样本
        for sample in speaker_data["samples"]:
            if sample["id"] == sample_id:
                ap = _resolve_sample_audio_path(sample)
                if ap:
                    return send_file(ap, as_attachment=True, download_name=sample["filename"])
                else:
                    return jsonify({"error": f"Audio file for sample '{sample_id}' not found."}), 404

        return jsonify({"error": f"Sample '{sample_id}' not found for speaker '{speaker_name}'."}), 404
    except Exception as e:
        logger_sys.error(f"获取样本音频文件失败: {str(e)}")
        return jsonify({"error": "Failed to retrieve sample audio"}), 500


@app.route('/audio_segments/<path:filename>')
def serve_audio_segment(filename):
    """提供音频片段静态文件服务（多根：本地镜像优先，NAS 历史回退）"""
    from werkzeug.exceptions import NotFound
    try:
        rel = filename.lstrip('/')
        for _root in RECORDS_ROOTS:
            for dev in SOURCE_DEVICES + _LEGACY_DEVICES:
                audio_segments_dir = os.path.join(_root, dev, 'audio_segments')
                try:
                    return send_from_directory(audio_segments_dir, rel)
                except NotFound:
                    continue
        # 兼容单源直挂（--source-path 覆盖等场景）
        return send_from_directory(os.path.join(FileMonitorConfig.SOURCE_DIR, 'audio_segments'), rel)
    except Exception as e:
        logger_sys.error(f"获取音频片段失败: {str(e)}")
        return jsonify({"error": "Audio segment not found"}), 404


@app.route('/api/audio/<filename>')
def serve_audio_file(filename):
    """提供原始音频文件服务（如 TermuxAudioRecording_xxx.m4a）"""
    try:
        # 清理路径（移除开头的 /）
        clean_filename = filename.lstrip('/')
        # 双源：设备级路径直接定位；旧式相对路径（无设备前缀）逐设备探测
        _abs = _resolve_under_records(clean_filename)
        if _abs:
            _dir, _base = os.path.split(_abs)
            return send_from_directory(_dir, _base)
        return send_from_directory(RECORDS_ROOT, clean_filename)
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
            # 转换为相对 records 根的路径（含设备级；旧式路径保持原样由 serve 端探测）
            # 【2026-10-04 多根】兼容本地镜像与 NAS 两种绝对路径前缀
            if any(f.startswith(_r) for _r in RECORDS_ROOTS):
                f = _rel_to_records(f)
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
        # 创建一个带真实写入检测的客户端对象
        class SSEClient:
            def __init__(self):
                self._closed = False
            def write(self, msg):
                if self._closed:
                    raise ConnectionError("Client disconnected")
                return msg

        client = SSEClient()

        # 添加客户端到SSE处理器
        sse_handler.add_client(client)
        try:
            # 保持连接打开，每30秒发送心跳以检测连接状态
            while True:
                time.sleep(30)
                yield ": heartbeat\n\n"
        except GeneratorExit:
            pass
        finally:
            client._closed = True
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
        FileMonitorConfig.SOURCES = [args.source_path]
        SOURCE_DEVICES[:] = [os.path.basename(args.source_path.rstrip("/"))]
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
        logger_sys.warning("⚠️ 数据库连接池初始化失败！服务将继续运行，数据库功能暂不可用，连接恢复后自动生效")
    else:
        print("初始化数据库表结构...")
        if not init_db():
            logger_sys.warning("⚠️ 数据库表结构初始化失败，但服务将继续运行")

    load_models()

    # 打印 Gemini Key 轮换配置
    print(f"Gemini API Key 轮询: {len(LLMConfig.GEMINI_API_KEYS)} 个 Key 已加载")
    for i, k in enumerate(LLMConfig.GEMINI_API_KEYS):
        print(f"  Key {i+1}: ...{k[-6:]}")

    # 启动临时文件清理定时任务
    cleanup_temp_dir()
    logger_sys.info("临时文件清理定时任务已启动")

    # 启动未完成哭声分析自动重试
    threading.Timer(120, retry_incomplete_cry_analyses).start()  # 启动2分钟后首次执行，给系统初始化时间
    logger_sys.info(f"未完成哭声分析自动重试已启动 (间隔{RETRY_INCOMPLETE_INTERVAL}秒)")

    # 启动文件监控模块 (已解耦)
    # 【2026-10-04 修复】原逻辑在 app.run() 之前直接启动, catch-up 请求会打在
    # 尚未 listen 的端口上 → Connection refused → 文件被误归档 failed/（历史已积累 169 个）。
    # 改为延迟启动: 等 Flask 就绪后 catch-up 才能成功提交。
    def _deferred_start_monitor():
        time.sleep(50)  # 实测 5008 从启动到可接收请求约 45s（模型加载+listen）
        global _track_b_running
        _track_b_running = True
        audio_processor.start_monitor()
        logger_sys.info("B 轨文件监控已延迟启动（等待 Flask 就绪后）")

    threading.Thread(target=_deferred_start_monitor, daemon=True).start()

    # 启动多设备恢复上传监控（仅监控停滞和自动恢复，不参与识别）
    # 内部会扫描 NAS 目录，可能挂起，用线程+超时保护，避免阻塞启动
    _rec_thread = threading.Thread(target=recovery_monitor.start_recovery_monitors, daemon=True)
    _rec_thread.start()
    _rec_thread.join(10)
    if _rec_thread.is_alive():
        logger_sys.warning("⚠️ 恢复监控初始化扫描超时(10s)，将在后台继续，不阻塞服务启动")

    # 启动系统总览设备健康扫描（后台 5 分钟缓存，/api/overview 只读快照）
    threading.Thread(target=_overview_cache_loop, daemon=True, name="overview-scan").start()
    logger_sys.info("系统总览设备健康扫描线程已启动 (间隔300s)")

    print("🎉 服务启动成功！")
    print("📌 声纹注册页面: http://127.0.0.1:5008/register_page")
    print("📌 语音转录API: http://127.0.0.1:5008/transcribes (本地监控专用)")
    print("📌 外部调用API: http://127.0.0.1:5008/transcribe (保留给NAS使用)")
    print(f"📂 文件监控目录: {FileMonitorConfig.SOURCE_DIR}")
    print(f"⏱️  扫描间隔: {FileMonitorConfig.SCAN_INTERVAL}秒")
    print("🔧 API使用方法: POST请求，参数名 'audio_file'，上传音频文件")
    print("🔍 示例命令: curl -X POST -F \"audio_file=@your_audio.wav\" http://127.0.0.1:5008/transcribes")
    app.run(host=Config.HOST, port=Config.PORT, debug=False, threaded=True)
