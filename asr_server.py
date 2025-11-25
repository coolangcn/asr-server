#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, logging, json, threading, subprocess, time, traceback, tempfile
import numpy as np
from scipy.spatial.distance import cosine
from flask import Flask, request, jsonify, render_template, send_file, Response
from funasr import AutoModel  # ASR 用 FunASR
from modelscope.pipelines import pipeline  # SV 用 ModelScope
from modelscope.utils.constant import Tasks
import torch
import torchaudio
import shutil
import re
from collections import Counter
from db_manager import save_to_db
from logging.handlers import TimedRotatingFileHandler
import whisper

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
        return None, "neutral"
    
    try:
        result = sensevoice_pipeline.generate(
            input=audio_path,
            language="auto",
            use_itn=True
        )
        
        if not result or len(result) == 0:
            return None, "neutral"
        
        raw_text = result[0].get("text", "")
        
        # 提取情感
        emotion = "neutral"
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
        return None, "neutral"

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
        emotion = "neutral"
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
        logger.info(f"🔍 开始使用模型: {model_name}")
        
        emb_a = extract_embedding_from_file(sv_pipe, segment_path)
        if emb_a is None:
            logger.error(f"❌ 模型 {model_name} 特征提取失败")
            model_votes[model_name] = "Failed"
            continue

        scores = []
        conf = Config.SV_MODELS[model_name]
        threshold = conf['threshold']
        gap = conf['gap']
        logger.info(f"📌 模型 {model_name} 阈值: {threshold}, 置信度间隔: {gap}")

        for name, speaker_data in speaker_db.items():
            # 使用平均嵌入进行比较
            if "avg_embeddings" not in speaker_data or model_name not in speaker_data["avg_embeddings"]: 
                continue
            emb_b = np.array(speaker_data["avg_embeddings"][model_name]).flatten()
            score = 1 - cosine(emb_a.flatten(), emb_b)
            scores.append((name, score))
            logger.info(f"💯 模型 {model_name} 与说话人 {name} 的相似度: {score:.6f}")

        if not scores:
            logger.warning(f"⚠️ 模型 {model_name} 未找到匹配的说话人数据")
            model_votes[model_name] = "NoDB"
            continue

        scores.sort(key=lambda x: x[1], reverse=True)
        top1_name, top1_score = scores[0]
        top2_name, top2_score = scores[1] if len(scores) > 1 else (None, 0.0)
        score_gap = top1_score - top2_score
        
        logger.info(f"🏆 模型 {model_name} 识别结果: 第一名 {top1_name} (得分: {top1_score:.6f}), 第二名 {top2_name} (得分: {top2_score:.6f}), 差距: {score_gap:.6f}")

        if top1_score >= threshold and score_gap >= gap:
            model_votes[model_name] = top1_name
            model_scores[model_name] = top1_score
            logger.info(f"✅ 模型 {model_name} 验证通过: {top1_name} (得分: {top1_score:.6f} ≥ 阈值 {threshold})")
        else:
            model_votes[model_name] = "Unknown"
            model_scores[model_name] = top1_score
            reason = []
            if top1_score < threshold:
                reason.append(f"得分 {top1_score:.6f} < 阈值 {threshold}")
            if score_gap < gap:
                reason.append(f"差距 {score_gap:.6f} < 置信度间隔 {gap}")
            logger.info(f"❌ 模型 {model_name} 验证失败: {', '.join(reason)}")

    logger.info(f"📊 多模型投票结果: {model_votes}")
    
    # 2/3投票逻辑
    votes = [v for v in model_votes.values() if v not in ["Unknown", "Failed", "NoDB"]]
    if not votes:
        logger.info("❌ 交叉验证失败: 所有模型均未识别出有效候选人")
        return None, 0.0, []

    vote_counts = Counter(votes)
    most_common_vote = vote_counts.most_common(1)[0]
    winner, count = most_common_vote
    
    # 至少需要2票
    if count >= 2:
        # 计算获胜者的平均置信度
        winning_scores = [model_scores[model] for model, vote in model_votes.items() if vote == winner]
        avg_confidence = np.mean(winning_scores)
        
        logger.info(f"🎉 交叉验证成功 (多数票): [{winner}] 获得 {count} 票 | 平均置信度: {avg_confidence:.3f}")
        
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
        
        logger.info(f"❌ 交叉验证失败: 没有候选人获得足够票数 (多数票 ≥ 2)")
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
            for f in temp_files:
                if os.path.exists(f):
                    try: os.remove(f)
                    except: pass

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    # 确保临时目录存在
    os.makedirs(Config.TEMP_DIR, exist_ok=True)
    request_start = time.time()
    temp_files = []

    with gpu_lock:
        try:
            if 'audio_file' not in request.files: return jsonify({"error": "No file uploaded"}), 400
            
            file = request.files['audio_file']
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
                        emotion = "neutral"
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
                        if (end - start) > Config.MIN_SPEAKER_DURATION_MS:
                            seg_wav = os.path.join(Config.TEMP_DIR, f"seg_{start}_{i}_{int(time.time())}.wav")
                            if extract_segment(proc_temp, start, end, seg_wav):
                                temp_files.append(seg_wav)
                                identity, confidence, recognition_details = identify_speaker_fusion(seg_wav)
                                # 情感检测
                                emotion = detect_emotion_for_segment(seg_wav)
                                # Whisper对比识别
                                whisper_text = transcribe_with_whisper(seg_wav)
                                
                                # SenseVoice识别和情感检测
                                sensevoice_text, sensevoice_emotion = transcribe_with_sensevoice(seg_wav)
                                
                                # 使用SenseVoice的情感结果(如果检测到非neutral)
                                if sensevoice_emotion != "neutral":
                                    emotion = sensevoice_emotion
                                
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


                                if Config.SAVE_LONG_SENTENCES and len(clean_text) >= Config.MIN_TEXT_LENGTH_TO_SAVE and not is_noise(clean_text):
                                    try:
                                        os.makedirs(Config.LONG_SENTENCES_DIR, exist_ok=True)
                                        timestamp = int(time.time())
                                        speaker_name = identity or "Unknown"
                                        saved_filename = f"{timestamp}_{speaker_name}_{len(clean_text)}chars.wav"
                                        saved_path = os.path.join(Config.LONG_SENTENCES_DIR, saved_filename)
                                        shutil.copy2(seg_wav, saved_path)
                                        
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
                                            else:
                                                f.write(f"\n=== Whisper 识别结果 ===\n(未启用或识别失败)\n")
                                        
                                        logger.info(f"      [长句保存] 已保存 {len(clean_text)} 字音频: {saved_filename}")
                                    except Exception as e:
                                        logger.warning(f"      [长句保存] 保存失败: {e}")
                        else:
                            logger.info(f"      [3.{i+1}] 分段时长过短({end-start}ms)，跳过声纹识别。")
                            # 即使跳过声纹识别，也要初始化这些变量
                            emotion = "neutral"
                            whisper_text = None


                        if Config.ONLY_REGISTERED_SPEAKERS and identity is None: continue
                        
                        processed_segments.append({
                            "text": clean_text, "start": start, "end": end,
                            "spk": identity or "Unknown", "emotion": emotion,
                            "whisper_text": whisper_text,
                            "sensevoice_text": sensevoice_text,
                            "confidence": float(f"{confidence:.3f}"),
                            "recognition_details": recognition_details
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
            
            # 保存到数据库
            if segments:  # 只有在有分段时才保存
                try:
                    save_to_db(file.filename, full_text, segments)
                    logger.info(f"💾 [数据库保存] 已保存到数据库: {file.filename}")
                except Exception as save_err:
                    logger.error(f"❌ [数据库保存] 保存失败: {save_err}")
            
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
if __name__ == "__main__":
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        logger.critical("❌ 系统未安装 FFmpeg！")
        sys.exit(1)

    load_models()
    print("🎉 服务启动成功！")
    print("📌 声纹注册页面: http://127.0.0.1:5008/register_page")
    print("📌 语音转录API: http://127.0.0.1:5008/transcribe")
    print("🔧 API使用方法: POST请求，参数名 'audio_file'，上传音频文件")
    print("🔍 示例命令: curl -X POST -F \"audio_file=@your_audio.wav\" http://127.0.0.1:5008/transcribe")
    app.run(host=Config.HOST, port=Config.PORT, debug=False, threaded=True)