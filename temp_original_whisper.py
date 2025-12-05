#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, logging, json, threading, subprocess, time, traceback, tempfile
import numpy as np
from scipy.spatial.distance import cosine
from flask import Flask, request, jsonify, render_template, send_file, Response
from funasr import AutoModel  # ASR 鐢?FunASR
from modelscope.pipelines import pipeline  # SV 鐢?ModelScope
from modelscope.utils.constant import Tasks
import torch
import torchaudio
import shutil
import re
from collections import Counter
from db_manager import save_to_db
from logging.handlers import TimedRotatingFileHandler
import whisper

# =================銆?閰嶇疆 銆?================
class Config:
    DEVICE = "cuda:0"
    HOST = '0.0.0.0'
    PORT = 5008
    SPEAKER_DB_FILE = "speaker_db_multi.json"
    
    ONLY_REGISTERED_SPEAKERS = False
    # ASR妯″瀷閰嶇疆 - Paraformer (鏀寔VAD鍒嗘鍜岃璇濅汉鍒嗙)
    ASR_MODEL = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"  # 浠?SenseVoiceSmall 鍒囨崲鍒?Paraformer
    VAD_MODEL = "fsmn-vad"       # VAD妯″瀷
    SPK_MODEL = "cam++"          # 璇磋瘽浜哄垎绂绘ā鍨? 
    PUNC_MODEL = "ct-punc"       # 鏍囩偣鎭㈠妯″瀷
    
    # VAD鍙傛暟閰嶇疆(涓篜araformer浼樺寲)
    VAD_MAX_SINGLE_SEGMENT = 15000  # ms - 鍗曟鏈€闀挎椂闂?    VAD_MAX_END_SILENCE = 300       # ms - 娈靛熬闈欓煶闃堝€?    VAD_SIL_TO_SPEECH = 50          # ms - 闈欓煶鍒拌闊抽槇鍊?    VAD_SPEECH_TO_SIL = 80          # ms - 璇煶鍒伴潤闊抽槇鍊?    
    SV_MODELS = {
        "eres2net_large": {
            "id": "iic/speech_eres2net_large_200k_sv_zh-cn_16k-common",
            "rev": "v1.0.0",
            "threshold": 0.40,  # 闄嶄綆闃堝€间互鎻愰珮璇嗗埆鐜?            "gap": 0.08         # 闄嶄綆缃俊搴﹂棿闅旇姹?        },
        "rdino_ecapa": {
            "id": "iic/speech_rdino_ecapa_tdnn_sv_zh-cn_cnceleb_16k",
            "rev": "v1.0.0",
            "threshold": 0.40,  # 闄嶄綆闃堝€间互鎻愰珮璇嗗埆鐜?            "gap": 0.08         # 闄嶄綆缃俊搴﹂棿闅旇姹?        },
        "camplusplus": {
            "id": "iic/speech_campplus_sv_zh-cn_16k-common",
            "rev": "v1.0.0",
            "threshold": 0.40,  # 闄嶄綆闃堝€间互鎻愰珮璇嗗埆鐜?            "gap": 0.08         # 闄嶄綆缃俊搴﹂棿闅旇姹?        }
    }
    
    MIN_SPEAKER_DURATION_MS = 800
    NORMALIZE_AUDIO = True
    DENOISE_AUDIO = False  # 鍚敤楂樼骇闄嶅櫔
# ==========================================

EMOTION_TAGS = {
    "<|happy|>": "happy", "<|sad|>": "sad", "<|angry|>": "angry",
    "<|neutral|>": "neutral", "<|laughter|>": "laughter", "<|fearful|>": "fearful",
    "<|disgusted|>": "disgusted", "<|surprised|>": "surprised", "<|EMO_UNKNOWN|>": "neutral"
}
INVALID_TAGS = {"<|nospeech|>", "<|BGM|>", "<|Event_UNK|>", "<|music|>"}

# 鏂板锛氬畾涔夎璇濅汉鏁版嵁缁撴瀯
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

# 鍒涘缓鏃ュ織闃熷垪鐢ㄤ簬SSE
export_logger = logging.getLogger('export_logger')
export_logger.setLevel(logging.INFO)

# 鑷畾涔夋棩蹇楀鐞嗗櫒锛屽皢鏃ュ織娑堟伅鍙戦€佸埌SSE杩炴帴
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

# 鍒涘缓鏃ュ織澶勭悊鍣?log_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

# 鍒涘缓骞堕厤缃甋SE澶勭悊鍣?sse_handler = SSEHandler()
sse_handler.setFormatter(log_formatter)
sse_handler.setLevel(logging.INFO)

# 閰嶇疆鏍规棩蹇楄褰曞櫒
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 鍒涘缓鏂囦欢澶勭悊鍣紝鐢ㄤ簬灏嗘棩蹇楀啓鍏ユ枃浠讹紙姣?0鍒嗛挓杞浆涓€娆★級
file_handler = TimedRotatingFileHandler(
    'asr-server.log', 
    when='M',           # 鎸夊垎閽熻疆杞?    interval=10,        # 姣?0鍒嗛挓
    backupCount=144,    # 淇濈暀144涓枃浠讹紙24灏忔椂锛?    encoding='utf-8'
)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler) # 娣诲姞鏂囦欢澶勭悊鍣?
logger.addHandler(console_handler)
logger.addHandler(sse_handler)

app = Flask(__name__)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

asr_pipeline = None
sv_pipelines = {}
speaker_db = {}
gpu_lock = threading.Lock()
db_lock = threading.Lock()

# =================== 妯″瀷鍔犺浇 ===================
def load_models():
    global asr_pipeline, sv_pipelines
    print("\n====== 馃殌 鍚姩 SOTA 铻嶅悎鏈嶅姟 ======")
    
    load_speaker_db()

    # 2. 鍔犺浇 ASR (FunASR)
    print(f"馃 鍔犺浇 ASR: {Config.ASR_MODEL} ...")
    # 2. 鍔犺浇 ASR (FunASR Paraformer + VAD + 璇磋瘽浜哄垎绂?
    print(f"馃 鍔犺浇 ASR: {Config.ASR_MODEL} (鏀寔VAD鍒嗘鍜岃璇濅汉鍒嗙) ...")
    asr_pipeline = AutoModel(
        model=Config.ASR_MODEL,       # paraformer-zh
        vad_model=Config.VAD_MODEL,   # fsmn-vad
        punc_model=Config.PUNC_MODEL, # ct-punc (鏍囩偣鎭㈠)
        spk_model=Config.SPK_MODEL,   # cam++ (璇磋瘽浜哄垎绂?
        vad_kwargs={
            "max_single_segment_time": Config.VAD_MAX_SINGLE_SEGMENT,
            "max_end_silence_time": Config.VAD_MAX_END_SILENCE,
            "sil_to_speech_time_thres": Config.VAD_SIL_TO_SPEECH,
            "speech_to_sil_time_thres": Config.VAD_SPEECH_TO_SIL
        },
        device=Config.DEVICE, 
        disable_update=True
    )
    print("鉁?Paraformer妯″瀷鍔犺浇瀹屾垚锛屽凡鍚敤VAD鍒嗘鍜岃璇濅汉鍒嗙鍔熻兘")

    # 3. 鍔犺浇 SV 妯″瀷
    for name, conf in Config.SV_MODELS.items():
        print(f"馃攳 鍔犺浇 SV [{name}] : {conf['id']} ...")
        sv_pipelines[name] = pipeline(
            task=Tasks.speaker_verification,
            model=conf['id'], 
            model_revision=conf['rev'], 
            device=Config.DEVICE.split(':')[0]
        )
    print(f"鉁?鏈嶅姟灏辩华 | ASR: SenseVoice | SV: {list(sv_pipelines.keys())}\n")

def load_speaker_db():
    global speaker_db
    with db_lock:
        if os.path.exists(Config.SPEAKER_DB_FILE):
            try:
                with open(Config.SPEAKER_DB_FILE, 'r', encoding='utf-8') as f:
                    loaded_db = json.load(f)
                
                # 鍏煎鏃ф暟鎹粨鏋?                converted_db = {}
                for name, data in loaded_db.items():
                    if "samples" in data and "avg_embeddings" in data:
                        # 鏂版暟鎹粨鏋勶紝鐩存帴浣跨敤
                        converted_db[name] = data
                    else:
                        # 鏃ф暟鎹粨鏋勶紝杞崲涓烘柊缁撴瀯
                        logger.info(f"馃攧 杞崲鏃ф暟鎹粨鏋?for speaker: {name}")
                        converted_db[name] = {
                            "samples": [],  # 鏃ф暟鎹粨鏋勬病鏈夋牱鏈俊鎭?                            "avg_embeddings": data  # 鏃ф暟鎹粨鏋勭洿鎺ユ槸宓屽叆瀛楀吀
                        }
                
                speaker_db = converted_db
                logger.info(f"馃摎 澹扮汗搴撳凡鎸傝浇: {len(speaker_db)} 浜?)
            except Exception as e:
                logger.error(f"澹扮汗搴撴崯鍧? {e}")
                speaker_db = {}
        else:
            logger.warning(f"鈿狅笍 鏈壘鍒?{Config.SPEAKER_DB_FILE}锛屽皢鍒涘缓鏂扮殑鏁版嵁搴撱€?)
            speaker_db = {}

# =================== 闊抽棰勫鐞?===================
def preprocess_audio(input_path, output_path):
    # 濡傛灉鍚敤浜嗛珮绾ч檷鍣紝鍏堣繘琛岄檷鍣鐞?    if Config.DENOISE_AUDIO:
        denoised_path = input_path + ".denoised.wav"
        if advanced_denoise(input_path, denoised_path):
            input_path = denoised_path
        else:
            logger.warning("楂樼骇闄嶅櫔澶勭悊澶辫触锛屼娇鐢ㄥ師濮嬮煶棰?)
    
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", input_path]
    filters = ["loudnorm=I=-14:TP=-1.5:LRA=11"] if Config.NORMALIZE_AUDIO else []
    if filters: cmd.extend(["-af", ",".join(filters)])
    cmd.extend(["-ac", "1", "-ar", "16000", output_path])
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        # 娓呯悊涓存椂闄嶅櫔鏂囦欢
        if Config.DENOISE_AUDIO and input_path.endswith(".denoised.wav"):
            try:
                os.remove(input_path)
            except:
                pass
        return True
    except Exception as e:
        logger.error(f"FFmpeg 棰勫鐞嗗け璐? {e}")
        return False

def advanced_denoise(input_path, output_path):
    """浣跨敤璋卞噺娉曡繘琛岄珮绾ч檷鍣?""
    try:
        # 鍔犺浇闊抽
        waveform, sample_rate = torchaudio.load(input_path)
        
        # 濡傛灉閲囨牱鐜囦笉鏄?6kHz锛屽厛杩涜閲嶉噰鏍?        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        # 杞崲涓哄崟澹伴亾
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 绠€鍖栫殑璋卞噺娉曢檷鍣?        # 杩欓噷鎴戜滑浣跨敤涓€涓畝鍖栫殑瀹炵幇锛屽疄闄呭簲鐢ㄤ腑鍙互浣跨敤鏇村鏉傜殑绠楁硶
        audio_np = waveform.numpy()[0]
        
        # 璁＄畻鐭椂鍌呴噷鍙跺彉鎹?        from scipy import signal
        frequencies, times, Zxx = signal.stft(audio_np, fs=sample_rate, nperseg=512)
        
        # 浼拌鍣０璋憋紙鍋囪鍓?00ms涓哄櫔澹帮級
        noise_seg_len = min(int(0.1 * sample_rate), len(audio_np))
        noise_segment = audio_np[:noise_seg_len]
        _, _, noise_stft = signal.stft(noise_segment, fs=sample_rate, nperseg=512)
        noise_spectrum = np.mean(np.abs(noise_stft), axis=1)
        
        # 搴旂敤璋卞噺娉?        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)
        
        # 鍑忓幓鍣０璋辩殑浼拌鍊?        noise_factor = 1.5
        magnitude_denoised = np.maximum(magnitude - noise_factor * noise_spectrum[:, np.newaxis], 0)
        
        # 閲嶆瀯淇″彿
        Zxx_denoised = magnitude_denoised * np.exp(1j * phase)
        _, audio_denoised = signal.istft(Zxx_denoised, fs=sample_rate)
        
        # 瑁佸壀鍒板師濮嬮暱搴?        audio_denoised = audio_denoised[:len(audio_np)]
        
        # 淇濆瓨闄嶅櫔鍚庣殑闊抽
        waveform_denoised = torch.tensor(audio_denoised).unsqueeze(0)
        torchaudio.save(output_path, waveform_denoised, sample_rate)
        
        return True
    except Exception as e:
        logger.error(f"楂樼骇闄嶅櫔澶勭悊澶辫触: {e}")
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
    浣跨敤Whisper璇嗗埆闊抽鐗囨锛堜綔涓篎unASR鐨勫姣斿弬鑰冿級
    
    Args:
        audio_path: 闊抽鐗囨璺緞
        
    Returns:
        str: Whisper璇嗗埆鐨勬枃鏈紝濡傛灉澶辫触杩斿洖None
    """
    if not Config.ENABLE_WHISPER_COMPARISON or whisper_model is None:
        return None
    
    try:
        result = whisper_model.transcribe(
            audio_path,
            language='zh',
            fp16=True,  # GPU鍔犻€?            verbose=False
        )
        whisper_text = result['text'].strip()
        logger.info(f"      [Whisper瀵规瘮] {whisper_text}")
        return whisper_text
    except Exception as e:
        logger.warning(f"      [Whisper瀵规瘮] 璇嗗埆澶辫触: {e}")
        return None

def detect_emotion_for_segment(audio_path):
    """浣跨敤SenseVoice妫€娴嬮煶棰戞鐨勬儏鎰?""
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
        logger.info(f"      [SenseVoice鎯呮劅] 鍘熷杈撳嚭: {raw_text}")
        
        # 鎻愬彇鎯呮劅鏍囩
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
                logger.info(f"      [SenseVoice鎯呮劅] 妫€娴嬪埌鎯呮劅: {emotion}")
                break
        
        return emotion
    except Exception as e:
        logger.warning(f"      [SenseVoice鎯呮劅] 妫€娴嬪け璐? {e}")
        return "neutral"


# =================== 鎻愬彇 embedding ===================
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
        logger.error(f"鉂?extract_embedding 澶辫触: {e}")
        return None

# =================== 澶氭ā鍨嬩氦鍙夐獙璇?===================
def identify_speaker_fusion(segment_path):
    if not speaker_db: 
        logger.info("馃し鈥嶁檪锔?澹扮汗鏁版嵁搴撲负绌猴紝鏃犳硶杩涜璇嗗埆")
        return None, 0.0, []

    model_votes = {}
    model_scores = {}

    logger.info(f"馃幆 寮€濮嬪０绾硅瘑鍒? 闊抽娈佃矾寰?{segment_path}")
    logger.info(f"馃搵 澹扮汗鏁版嵁搴撳寘鍚?{len(speaker_db)} 涓璇濅汉")

    for model_name, sv_pipe in sv_pipelines.items():
        logger.info(f"馃攳 寮€濮嬩娇鐢ㄦā鍨? {model_name}")
        
        emb_a = extract_embedding_from_file(sv_pipe, segment_path)
        if emb_a is None:
            logger.error(f"鉂?妯″瀷 {model_name} 鐗瑰緛鎻愬彇澶辫触")
            model_votes[model_name] = "Failed"
            continue

        scores = []
        conf = Config.SV_MODELS[model_name]
        threshold = conf['threshold']
        gap = conf['gap']
        logger.info(f"馃搶 妯″瀷 {model_name} 闃堝€? {threshold}, 缃俊搴﹂棿闅? {gap}")

        for name, speaker_data in speaker_db.items():
            # 浣跨敤骞冲潎宓屽叆杩涜姣旇緝
            if "avg_embeddings" not in speaker_data or model_name not in speaker_data["avg_embeddings"]: 
                continue
            emb_b = np.array(speaker_data["avg_embeddings"][model_name]).flatten()
            score = 1 - cosine(emb_a.flatten(), emb_b)
            scores.append((name, score))
            logger.info(f"馃挴 妯″瀷 {model_name} 涓庤璇濅汉 {name} 鐨勭浉浼煎害: {score:.6f}")

        if not scores:
            logger.warning(f"鈿狅笍 妯″瀷 {model_name} 鏈壘鍒板尮閰嶇殑璇磋瘽浜烘暟鎹?)
            model_votes[model_name] = "NoDB"
            continue

        scores.sort(key=lambda x: x[1], reverse=True)
        top1_name, top1_score = scores[0]
        top2_name, top2_score = scores[1] if len(scores) > 1 else (None, 0.0)
        score_gap = top1_score - top2_score
        
        logger.info(f"馃弳 妯″瀷 {model_name} 璇嗗埆缁撴灉: 绗竴鍚?{top1_name} (寰楀垎: {top1_score:.6f}), 绗簩鍚?{top2_name} (寰楀垎: {top2_score:.6f}), 宸窛: {score_gap:.6f}")

        if top1_score >= threshold and score_gap >= gap:
            model_votes[model_name] = top1_name
            model_scores[model_name] = top1_score
            logger.info(f"鉁?妯″瀷 {model_name} 楠岃瘉閫氳繃: {top1_name} (寰楀垎: {top1_score:.6f} 鈮?闃堝€?{threshold})")
        else:
            model_votes[model_name] = "Unknown"
            model_scores[model_name] = top1_score
            reason = []
            if top1_score < threshold:
                reason.append(f"寰楀垎 {top1_score:.6f} < 闃堝€?{threshold}")
            if score_gap < gap:
                reason.append(f"宸窛 {score_gap:.6f} < 缃俊搴﹂棿闅?{gap}")
            logger.info(f"鉂?妯″瀷 {model_name} 楠岃瘉澶辫触: {', '.join(reason)}")

    logger.info(f"馃搳 澶氭ā鍨嬫姇绁ㄧ粨鏋? {model_votes}")
    
    # 2/3鎶曠エ閫昏緫
    votes = [v for v in model_votes.values() if v not in ["Unknown", "Failed", "NoDB"]]
    if not votes:
        logger.info("鉂?浜ゅ弶楠岃瘉澶辫触: 鎵€鏈夋ā鍨嬪潎鏈瘑鍒嚭鏈夋晥鍊欓€変汉")
        return None, 0.0, []

    vote_counts = Counter(votes)
    most_common_vote = vote_counts.most_common(1)[0]
    winner, count = most_common_vote
    
    # 鑷冲皯闇€瑕?绁?    if count >= 2:
        # 璁＄畻鑾疯儨鑰呯殑骞冲潎缃俊搴?        winning_scores = [model_scores[model] for model, vote in model_votes.items() if vote == winner]
        avg_confidence = np.mean(winning_scores)
        
        logger.info(f"馃帀 浜ゅ弶楠岃瘉鎴愬姛 (澶氭暟绁?: [{winner}] 鑾峰緱 {count} 绁?| 骞冲潎缃俊搴? {avg_confidence:.3f}")
        
        # 鐢熸垚璇︾粏淇℃伅
        recognition_details = []
        for model_name, result in model_votes.items():
            if result in ["Unknown", "Failed", "NoDB"]:
                recognition_details.append(f"妯″瀷 {model_name}: {result}")
            else:
                recognition_details.append(f"妯″瀷 {model_name}: 璇嗗埆涓?{result} (鐩镐技搴? {model_scores.get(model_name, 0):.6f})")
        recognition_details.append(f"鏈€缁堣瘑鍒粨鏋? {winner} (澶氭暟绁? {count} 绁? 骞冲潎缃俊搴? {avg_confidence:.3f})")
        
        return winner, avg_confidence, recognition_details
    else:
        # 鐢熸垚璇嗗埆澶辫触鐨勮缁嗕俊鎭?        recognition_details = []
        for model_name, result in model_votes.items():
            recognition_details.append(f"妯″瀷 {model_name}: {result} (鐩镐技搴? {model_scores.get(model_name, 0):.6f})")
        recognition_details.append("鏈€缁堣瘑鍒粨鏋? 璇嗗埆澶辫触锛屾病鏈夊€欓€変汉鑾峰緱瓒冲绁ㄦ暟 (澶氭暟绁?鈮?2)")
        
        logger.info(f"鉂?浜ゅ弶楠岃瘉澶辫触: 娌℃湁鍊欓€変汉鑾峰緱瓒冲绁ㄦ暟 (澶氭暟绁?鈮?2)")
        return None, 0.0, []

# =================== Flask 鎺ュ彛 ===================
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
    """鑾峰彇鎵€鏈夎璇濅汉鍒楄〃"""
    try:
        # 閲嶆柊鍔犺浇澹扮汗鏁版嵁搴撲互纭繚鏁版嵁鏄渶鏂扮殑
        load_speaker_db()
        # 杩斿洖璇磋瘽浜哄垪琛紙涓嶅寘鍚叿浣撶殑embedding鏁版嵁锛?        speakers_summary = {}
        for name, data in speaker_db.items():
            sample_count = len(data.get("samples", []))
            model_names = list(data.get("avg_embeddings", {}).keys())
            speakers_summary[name] = {
                "sample_count": sample_count,
                "models": model_names
            }
        return jsonify({"speakers": speakers_summary})
    except Exception as e:
        logger.error(f"鑾峰彇璇磋瘽浜哄垪琛ㄥけ璐? {str(e)}")
        return jsonify({"error": "Failed to retrieve speakers"}), 500

@app.route("/speaker/<speaker_name>", methods=["GET"])
def get_speaker_samples(speaker_name):
    """鑾峰彇鎸囧畾璇磋瘽浜虹殑鏍锋湰鍒楄〃"""
    try:
        # 閲嶆柊鍔犺浇澹扮汗鏁版嵁搴撲互纭繚鏁版嵁鏄渶鏂扮殑
        load_speaker_db()
        if speaker_name not in speaker_db:
            return jsonify({"error": f"Speaker '{speaker_name}' not found."}), 404
            
        speaker_data = speaker_db[speaker_name]
        # 杩斿洖鏍锋湰淇℃伅锛堜笉鍖呭惈鍏蜂綋鐨別mbedding鏁版嵁锛?        samples_info = []
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
        logger.error(f"鑾峰彇璇磋瘽浜烘牱鏈垪琛ㄥけ璐? {str(e)}")
        return jsonify({"error": "Failed to retrieve speaker samples"}), 500

@app.route("/speaker/<speaker_name>", methods=["DELETE"])
def delete_speaker(speaker_name):
    """鍒犻櫎鎸囧畾璇磋瘽浜?""
    try:
        with db_lock:
            if speaker_name in speaker_db:
                del speaker_db[speaker_name]
                # 淇濆瓨鏇存柊鍚庣殑鏁版嵁搴?                with open(Config.SPEAKER_DB_FILE, 'w', encoding='utf-8') as f:
                    json.dump(speaker_db, f, indent=2, ensure_ascii=False)
                logger.info(f"鉁?鎴愬姛鍒犻櫎璇磋瘽浜? {speaker_name}")
                return jsonify({"message": f"Speaker '{speaker_name}' deleted successfully."})
            else:
                return jsonify({"error": f"Speaker '{speaker_name}' not found."}), 404
    except Exception as e:
        logger.error(f"鍒犻櫎璇磋瘽浜哄け璐? {str(e)}")
        return jsonify({"error": "Failed to delete speaker"}), 500

@app.route("/speaker/<speaker_name>/sample/<sample_id>", methods=["DELETE"])
def delete_speaker_sample(speaker_name, sample_id):
    """鍒犻櫎鎸囧畾璇磋瘽浜虹殑鐗瑰畾鏍锋湰"""
    try:
        with db_lock:
            if speaker_name not in speaker_db:
                return jsonify({"error": f"Speaker '{speaker_name}' not found."}), 404
                
            speaker_data = speaker_db[speaker_name]
            if "samples" not in speaker_data:
                return jsonify({"error": f"No samples found for speaker '{speaker_name}'."}), 404
                
            # 鏌ユ壘骞跺垹闄ゆ寚瀹氭牱鏈?            samples = speaker_data["samples"]
            sample_to_remove = None
            sample_index = -1
            for i, sample in enumerate(samples):
                if sample["id"] == sample_id:
                    sample_to_remove = sample
                    sample_index = i
                    break
                    
            if sample_to_remove is None:
                return jsonify({"error": f"Sample '{sample_id}' not found for speaker '{speaker_name}'."}), 404
                
            # 鍒犻櫎鏍锋湰鐨勯煶棰戞枃浠?            if "audio_path" in sample_to_remove and os.path.exists(sample_to_remove["audio_path"]):
                try:
                    os.remove(sample_to_remove["audio_path"])
                    logger.info(f"馃棏锔?鍒犻櫎浜嗛煶棰戞枃浠? {sample_to_remove['audio_path']}")
                except Exception as e:
                    logger.warning(f"鈿狅笍 鍒犻櫎闊抽鏂囦欢澶辫触: {sample_to_remove['audio_path']}, 閿欒: {str(e)}")
            
            # 浠庢暟鎹簱涓Щ闄ゆ牱鏈褰?            del samples[sample_index]
                
            # 濡傛灉鍒犻櫎鏍锋湰鍚庢病鏈夊墿浣欐牱鏈紝鍒欏垹闄ゆ暣涓璇濅汉
            if len(samples) == 0:
                del speaker_db[speaker_name]
                # 鍒犻櫎璇磋瘽浜虹殑鐩綍
                speaker_dir = os.path.join("speaker_samples", speaker_name)
                if os.path.exists(speaker_dir):
                    try:
                        shutil.rmtree(speaker_dir)
                        logger.info(f"馃棏锔?鍒犻櫎浜嗚璇濅汉鐩綍: {speaker_dir}")
                    except Exception as e:
                        logger.warning(f"鈿狅笍 鍒犻櫎璇磋瘽浜虹洰褰曞け璐? {speaker_dir}, 閿欒: {str(e)}")
                
                with open(Config.SPEAKER_DB_FILE, 'w', encoding='utf-8') as f:
                    json.dump(speaker_db, f, indent=2, ensure_ascii=False)
                logger.info(f"馃棏锔?鍒犻櫎浜嗚璇濅汉 {speaker_name}锛堟渶鍚庝竴涓牱鏈凡鍒犻櫎锛?)
                return jsonify({"message": f"Speaker '{speaker_name}' deleted (last sample removed)."})

            # 閲嶆柊璁＄畻骞冲潎宓屽叆
            all_model_embeddings = {model_name: [] for model_name in sv_pipelines.keys()}
            for sample in samples:
                for model_name, emb in sample["embeddings"].items():
                    all_model_embeddings[model_name].append(np.array(emb))
            
            # 璁＄畻鏂扮殑骞冲潎宓屽叆
            new_avg_embeddings = {}
            for model_name, emb_list in all_model_embeddings.items():
                if emb_list:
                    avg_emb = np.mean(emb_list, axis=0)
                    new_avg_embeddings[model_name] = avg_emb.tolist()
            
            speaker_db[speaker_name]["avg_embeddings"] = new_avg_embeddings
            
            # 淇濆瓨鏇存柊鍚庣殑鏁版嵁搴?            with open(Config.SPEAKER_DB_FILE, 'w', encoding='utf-8') as f:
                json.dump(speaker_db, f, indent=2, ensure_ascii=False)
                
            logger.info(f"馃棏锔?鍒犻櫎浜嗚璇濅汉 {speaker_name} 鐨勬牱鏈?{sample_id}")
            return jsonify({
                "message": f"Sample '{sample_id}' deleted from speaker '{speaker_name}'.",
                "remaining_samples": len(samples)
            })
    except Exception as e:
        logger.error(f"鍒犻櫎璇磋瘽浜烘牱鏈け璐? {str(e)}")
        return jsonify({"error": "Failed to delete speaker sample"}), 500

@app.route("/register", methods=["POST"])
def register_speaker():
    temp_files = []
    with gpu_lock:
        try:
            if 'speaker_name' not in request.form or not request.form['speaker_name']:
                return jsonify({"error": "Speaker name is required"}), 400
            
            speaker_name = request.form['speaker_name']
            audio_files = request.files.getlist('audio_files')
            
            # 鑷姩妫€娴嬫槸鍚﹂渶瑕佸寮烘ā寮?            enhance_mode = speaker_name in speaker_db

            if not audio_files:
                return jsonify({"error": "At least one audio file is required"}), 400

            action = "澧炲己" if enhance_mode else "娉ㄥ唽"
            logger.info(f"馃摜 寮€濮媨action}鏂板０绾? {speaker_name} | 鏂囦欢鏁? {len(audio_files)}")
            
            # 鍒涘缓璇磋瘽浜烘牱鏈洰褰?            speaker_dir = os.path.join("speaker_samples", speaker_name)
            if not os.path.exists(speaker_dir):
                os.makedirs(speaker_dir)
            
            # 鏀堕泦鏂版牱鏈暟鎹?            new_samples = []
            model_embeddings = {model_name: [] for model_name in sv_pipelines.keys()}

            for file in audio_files:
                raw_temp = os.path.join(tempfile.gettempdir(), f"reg_raw_{int(time.time())}_{file.filename}")
                file.save(raw_temp)
                temp_files.append(raw_temp)
                
                proc_temp = os.path.join(tempfile.gettempdir(), f"reg_proc_{int(time.time())}.wav")
                temp_files.append(proc_temp)

                if not preprocess_audio(raw_temp, proc_temp):
                    logger.warning(f"鈿狅笍 鏂囦欢 {file.filename} 棰勫鐞嗗け璐ワ紝宸茶烦杩囥€?)
                    continue

                # 涓烘瘡涓ā鍨嬫彁鍙栧祵鍏?                sample_embeddings = {}
                for model_name, sv_pipe in sv_pipelines.items():
                    emb = extract_embedding_from_file(sv_pipe, proc_temp)
                    if emb is not None:
                        sample_embeddings[model_name] = emb.tolist()
                        model_embeddings[model_name].append(emb)
                    else:
                        logger.warning(f"鈿狅笍 浠?{file.filename} 鎻愬彇 {model_name} embedding 澶辫触銆?)

                # 淇濆瓨鏍锋湰淇℃伅鍜岄煶棰戞枃浠?                if sample_embeddings:  # 鍙湁褰撹嚦灏戞湁涓€涓ā鍨嬫垚鍔熸彁鍙栧祵鍏ユ椂鎵嶄繚瀛樻牱鏈?                    # 鐢熸垚鍞竴鐨勬牱鏈琁D
                    sample_id = f"{int(time.time())}_{hash(file.filename) % 10000}"
                    
                    # 淇濆瓨澶勭悊鍚庣殑闊抽鏂囦欢
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

            # 璁＄畻姣忎釜妯″瀷鐨勫钩鍧囧祵鍏?            avg_embeddings = {}
            for model_name, emb_list in model_embeddings.items():
                if not emb_list:
                    continue
                avg_emb = np.mean(emb_list, axis=0)
                avg_embeddings[model_name] = avg_emb.tolist()
                logger.info(f"  - 妯″瀷 [{model_name}] 澶勭悊浜?{len(emb_list)} 涓牱鏈?)

            if not avg_embeddings:
                return jsonify({"error": "Failed to extract embeddings from any samples"}), 500

            with db_lock:
                # 濡傛灉璇磋瘽浜哄凡瀛樺湪锛屽垯娣诲姞鏂版牱鏈苟鏇存柊骞冲潎宓屽叆
                if enhance_mode and speaker_name in speaker_db:
                    # 娣诲姞鏂版牱鏈埌鐜版湁鏍锋湰鍒楄〃
                    if "samples" not in speaker_db[speaker_name]:
                        speaker_db[speaker_name]["samples"] = []
                    speaker_db[speaker_name]["samples"].extend(new_samples)
                    
                    # 閲嶆柊璁＄畻鎵€鏈夋牱鏈殑骞冲潎宓屽叆
                    all_model_embeddings = {model_name: [] for model_name in sv_pipelines.keys()}
                    
                    # 娣诲姞鐜版湁鏍锋湰鐨勫祵鍏?                    for sample in speaker_db[speaker_name]["samples"]:
                        for model_name, emb in sample["embeddings"].items():
                            all_model_embeddings[model_name].append(np.array(emb))
                    
                    # 閲嶆柊璁＄畻骞冲潎宓屽叆
                    new_avg_embeddings = {}
                    for model_name, emb_list in all_model_embeddings.items():
                        if emb_list:
                            avg_emb = np.mean(emb_list, axis=0)
                            new_avg_embeddings[model_name] = avg_emb.tolist()
                    
                    speaker_db[speaker_name]["avg_embeddings"] = new_avg_embeddings
                    logger.info(f"馃攧 澧炲己浜嗚璇濅汉 {speaker_name} 鐨勫０绾癸紝鏂板 {len(new_samples)} 涓牱鏈?)
                else:
                    # 鍒涘缓鏂扮殑璇磋瘽浜烘潯鐩?                    speaker_db[speaker_name] = {
                        "samples": new_samples,
                        "avg_embeddings": avg_embeddings
                    }
                    logger.info(f"馃啎 鍒涘缓浜嗘柊璇磋瘽浜?{speaker_name} 鐨勫０绾癸紝鍖呭惈 {len(new_samples)} 涓牱鏈?)
                    
                # 淇濆瓨鏇存柊鍚庣殑鏁版嵁搴?                with open(Config.SPEAKER_DB_FILE, 'w', encoding='utf-8') as f:
                    json.dump(speaker_db, f, indent=2, ensure_ascii=False)
            
            logger.info(f"鉁?澹扮汗{action}鎴愬姛: {speaker_name}")
            return jsonify({
                "message": f"Speaker '{speaker_name}' {action} successfully.",
                "samples_added": len(new_samples)
            })

        except Exception as e:
            logger.error(f"鉂?娉ㄥ唽寮傚父: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"error": "An internal error occurred during registration."} ), 500
        finally:
            for f in temp_files:
                if os.path.exists(f):
                    try: os.remove(f)
                    except: pass

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    request_start = time.time()
    temp_files = []

    with gpu_lock:
        try:
            if 'audio_file' not in request.files: return jsonify({"error": "No file uploaded"}), 400
            
            file = request.files['audio_file']
            raw_temp = os.path.join(tempfile.gettempdir(), f"raw_{int(time.time())}_{file.filename}")
            file.save(raw_temp)
            temp_files.append(raw_temp)
            proc_temp = os.path.join(tempfile.gettempdir(), f"proc_{int(time.time())}.wav")
            temp_files.append(proc_temp)
            
            logger.info(f"馃摜 鏀跺埌杞綍浠诲姟: {file.filename}")
            
            logger.info("  [鐢熷懡鍛ㄦ湡: 1. 闊抽棰勫鐞哴 寮€濮?(FFmpeg闄嶅櫔銆侀噸閲囨牱銆佸綊涓€鍖?...")
            if not preprocess_audio(raw_temp, proc_temp):
                return jsonify({"error": "Audio preprocessing failed"}), 500
            logger.info("  [鐢熷懡鍛ㄦ湡: 1. 闊抽棰勫鐞哴 瀹屾垚銆?)

            audio_duration = 0
            try:
                probe = subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', proc_temp])
                audio_duration = float(probe)
            except: pass

            logger.info("  [鐢熷懡鍛ㄦ湡: 2. VAD & ASR] 寮€濮?(FunASR璇煶妫€娴嬩笌鏂囧瓧杞綍)...")
            res = asr_pipeline.generate(input=proc_temp, language="auto", use_itn=True, use_punc=True)
            # logger.info(f"  [VAD 璋冭瘯] FunASR generate() 鍘熷杩斿洖: {json.dumps(res, ensure_ascii=False, indent=2)}")
            full_text = ""
            segments = []

            if res and isinstance(res, list) and len(res) > 0:
                item = res[0]
                full_text = item.get("text", "")
                
                raw_segments = item.get("sentence_info", [])
                logger.info(f"  [鐢熷懡鍛ㄦ湡: 2. VAD & ASR] 瀹屾垚, VAD妫€鍑?{len(raw_segments)} 涓垎娈点€?)

                if not raw_segments and full_text:
                    raw_segments = [{"text": full_text, "start": 0, "end": int(audio_duration * 1000)}]

                processed_segments = []
                
                if raw_segments:
                    logger.info("  [鐢熷懡鍛ㄦ湡: 3. 閫愭澹扮汗璇嗗埆] 寮€濮?..")
                    for i, seg in enumerate(raw_segments):
                        raw_text = seg.get("text", "")
                        start, end = seg.get("start", 0), seg.get("end", 0)
                        logger.info(f"    [3.{i+1}] 澶勭悊鍒嗘 {start}ms - {end}ms...")
                        
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
                            logger.info(f"      [3.{i+1}] 鍒嗘鏂囨湰鍦ㄦ竻娲楀悗涓虹┖锛屽凡璺宠繃銆?)
                            continue

                        identity, confidence = None, 0.0
                        recognition_details = []
                        if (end - start) > Config.MIN_SPEAKER_DURATION_MS:
                            seg_wav = os.path.join(tempfile.gettempdir(), f"seg_{start}_{int(time.time())}.wav")
                            if extract_segment(proc_temp, start, end, seg_wav):
                                temp_files.append(seg_wav)
                                identity, confidence, recognition_details = identify_speaker_fusion(seg_wav)
                                # 鎯呮劅妫€娴?                                emotion = detect_emotion_for_segment(seg_wav)
                                # Whisper瀵规瘮璇嗗埆
                                whisper_text = transcribe_with_whisper(seg_wav)
                        else:
                            logger.info(f"      [3.{i+1}] 鍒嗘鏃堕暱杩囩煭({end-start}ms)锛岃烦杩囧０绾硅瘑鍒€?)
                            # 鍗充娇璺宠繃澹扮汗璇嗗埆锛屼篃瑕佸垵濮嬪寲杩欎簺鍙橀噺
                            emotion = "neutral"
                            whisper_text = None


                        if Config.ONLY_REGISTERED_SPEAKERS and identity is None: continue
                        
                        processed_segments.append({
                            "text": clean_text, "start": start, "end": end,
                            "spk": identity or "Unknown", "emotion": emotion,
                            "whisper_text": whisper_text,
                            "confidence": float(f"{confidence:.3f}"),
                            "recognition_details": recognition_details
                        })
                    logger.info("  [鐢熷懡鍛ㄦ湡: 3. 閫愭澹扮汗璇嗗埆] 瀹屾垚銆?)

                segments = processed_segments
                full_text = "".join([s["text"] for s in segments]) # Reconstruct from clean segments

            process_time = time.time() - request_start
            rtf = process_time / audio_duration if audio_duration > 0 else 0
            # RTF(Real-Time Factor)鏄疄鏃跺洜瀛愶紝璇勪及绯荤粺澶勭悊閫熷害涓庨煶棰戞椂闀跨殑姣旂巼
            # RTF < 1琛ㄧず鍙互瀹炴椂澶勭悊锛孯TF瓒婁綆绯荤粺鎬ц兘瓒婂ソ
            logger.info(f"鉁?瀹屾垚! 闊抽:{audio_duration:.1f}s | 鑰楁椂:{process_time:.2f}s | RTF:{rtf:.3f} (RTF < 1琛ㄧず鍙疄鏃跺鐞嗭紝鍊艰秺浣庢€ц兘瓒婂ソ)")

            logger.info("  [鐢熷懡鍛ㄦ湡: 4. 缁勮鍝嶅簲] 寮€濮?..")
            response_data = {
                "full_text": full_text,
                "segments": segments,
                "meta": {
                    "process_time": process_time,
                    "audio_duration": audio_duration,
                    "rtf": rtf,
                    "rtf_description": "Real-Time Factor(瀹炴椂鍥犲瓙)锛屽鐞嗘椂闂?闊抽鏃堕暱锛孯TF < 1琛ㄧず鍙疄鏃跺鐞嗭紝鍊艰秺浣庢€ц兘瓒婂ソ"
                }
            }
            logger.info(f"馃摛  [鐢熷懡鍛ㄦ湡: 4. 缁勮鍝嶅簲] 瀹屾垚, 杩斿洖 /transcribe 缁撴灉: {json.dumps(response_data, ensure_ascii=False, indent=2)}")
            
            # 淇濆瓨鍒版暟鎹簱
            if segments:  # 鍙湁鍦ㄦ湁鍒嗘鏃舵墠淇濆瓨
                try:
                    save_to_db(file.filename, full_text, segments)
                    logger.info(f"馃捑 [鏁版嵁搴撲繚瀛榏 宸蹭繚瀛樺埌鏁版嵁搴? {file.filename}")
                except Exception as save_err:
                    logger.error(f"鉂?[鏁版嵁搴撲繚瀛榏 淇濆瓨澶辫触: {save_err}")
            
            return jsonify(response_data)

        except Exception as e:
            logger.error(f"鉂?澶勭悊寮傚父: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"error": str(e)}), 500
        finally:
            for f in temp_files:
                if os.path.exists(f):
                    try: os.remove(f)
                    except: pass

@app.route("/speaker/<speaker_name>/sample/<sample_id>/audio")
def get_sample_audio(speaker_name, sample_id):
    """鑾峰彇鎸囧畾璇磋瘽浜烘牱鏈殑闊抽鏂囦欢"""
    try:
        # 閲嶆柊鍔犺浇澹扮汗鏁版嵁搴撲互纭繚鏁版嵁鏄渶鏂扮殑
        load_speaker_db()
        if speaker_name not in speaker_db:
            return jsonify({"error": f"Speaker '{speaker_name}' not found."}), 404
            
        speaker_data = speaker_db[speaker_name]
        if "samples" not in speaker_data:
            return jsonify({"error": f"No samples found for speaker '{speaker_name}'."}), 404
            
        # 鏌ユ壘鎸囧畾鏍锋湰
        for sample in speaker_data["samples"]:
            if sample["id"] == sample_id:
                if "audio_path" in sample and os.path.exists(sample["audio_path"]):
                    return send_file(sample["audio_path"], as_attachment=True, download_name=sample["filename"])
                else:
                    return jsonify({"error": f"Audio file for sample '{sample_id}' not found."}), 404
        
        return jsonify({"error": f"Sample '{sample_id}' not found for speaker '{speaker_name}'."}), 404
    except Exception as e:
        logger.error(f"鑾峰彇鏍锋湰闊抽鏂囦欢澶辫触: {str(e)}")
        return jsonify({"error": "Failed to retrieve sample audio"}), 500


@app.route("/logs/stream")
def stream_logs():
    """SSE endpoint for real-time log streaming"""
    def generate_logs():
        # 鍒涘缓涓€涓柊鐨勫鎴风杩炴帴
        client = type('Client', (), {'write': lambda self, msg: print(msg, end='', flush=True) or msg})
        
        # 娣诲姞瀹㈡埛绔埌SSE澶勭悊鍣?        sse_handler.add_client(client)
        try:
            # 淇濇寔杩炴帴鎵撳紑
            while True:
                time.sleep(1)
        except GeneratorExit:
            # 瀹㈡埛绔柇寮€杩炴帴鏃剁Щ闄ゅ鎴风
            sse_handler.remove_client(client)
    
    return Response(generate_logs(), mimetype='text/event-stream')

# =================== 鍚姩 ===================
if __name__ == "__main__":
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        logger.critical("鉂?绯荤粺鏈畨瑁?FFmpeg锛?)
        sys.exit(1)

    load_models()
    print("馃帀 鏈嶅姟鍚姩鎴愬姛锛?)
    print("馃搶 澹扮汗娉ㄥ唽椤甸潰: http://127.0.0.1:5008/register_page")
    print("馃搶 璇煶杞綍API: http://127.0.0.1:5008/transcribe")
    print("馃敡 API浣跨敤鏂规硶: POST璇锋眰锛屽弬鏁板悕 'audio_file'锛屼笂浼犻煶棰戞枃浠?)
    print("馃攳 绀轰緥鍛戒护: curl -X POST -F \"audio_file=@your_audio.wav\" http://127.0.0.1:5008/transcribe")
    app.run(host=Config.HOST, port=Config.PORT, debug=False, threaded=True)
