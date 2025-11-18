#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, logging, json, threading, subprocess, time, traceback, tempfile
import numpy as np
from scipy.spatial.distance import cosine
from flask import Flask, request, jsonify
from funasr import AutoModel  # ASR 用 FunASR
from modelscope.pipelines import pipeline  # SV 用 ModelScope
from modelscope.utils.constant import Tasks

# =================【 配置 】=================
class Config:
    DEVICE = "cuda:0"
    HOST = '0.0.0.0'
    PORT = 5000
    SPEAKER_DB_FILE = "speaker_db_multi.json"  # 使用新的数据库
    
    ONLY_REGISTERED_SPEAKERS = True
    ASR_MODEL = "iic/SenseVoiceSmall"
    
    # SV 模型
    SV_MODELS = {
        "eres2net_large": {
            "id": "iic/speech_eres2net_large_200k_sv_zh-cn_16k-common",
            "rev": "v1.0.0",
            "threshold": 0.50,
            "gap": 0.10
        },
        "rdino_ecapa": {
            "id": "iic/speech_rdino_ecapa_tdnn_sv_zh-cn_cnceleb_16k",
            "rev": "v1.0.0",
            "threshold": 0.50,
            "gap": 0.10
        }
    }
    
    MIN_SPEAKER_DURATION_MS = 800
    NORMALIZE_AUDIO = True
# ==========================================

EMOTION_TAGS = {
    "<|happy|>": "happy", "<|sad|>": "sad", "<|angry|>": "angry",
    "<|neutral|>": "neutral", "<|laughter|>": "laughter", "<|fearful|>": "fearful",
    "<|disgusted|>": "disgusted", "<|surprised|>": "surprised", "<|EMO_UNKNOWN|>": "neutral"
}
INVALID_TAGS = {"<|nospeech|>", "<|BGM|>", "<|Event_UNK|>", "<|music|>"}

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger()
app = Flask(__name__)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

asr_pipeline = None
sv_pipelines = {}
speaker_db = {}
gpu_lock = threading.Lock() 

# =================== 模型加载 ===================
def load_models():
    global asr_pipeline, sv_pipelines, speaker_db
    print("\n====== 🚀 启动 SOTA 融合服务 ======")
    
    # 1. 加载声纹库
    if os.path.exists(Config.SPEAKER_DB_FILE):
        try:
            with open(Config.SPEAKER_DB_FILE, 'r', encoding='utf-8') as f:
                speaker_db = json.load(f)
            print(f"📚 声纹库已挂载: {len(speaker_db)} 人")
        except Exception as e:
            logger.error(f"声纹库损坏: {e}")
    else:
        logger.warning(f"⚠️ 未找到 {Config.SPEAKER_DB_FILE}")

    # 2. 加载 ASR (FunASR)
    print(f"🧠 加载 ASR: {Config.ASR_MODEL} ...")
    asr_pipeline = AutoModel(model=Config.ASR_MODEL, trust_remote_code=True, device=Config.DEVICE, disable_update=True)

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

# =================== 音频预处理 ===================
def preprocess_audio(input_path, output_path):
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", input_path]
    filters = ["loudnorm=I=-14:TP=-1.5:LRA=11"] if Config.NORMALIZE_AUDIO else []
    if filters: cmd.extend(["-af", ",".join(filters)])
    cmd.extend(["-ac", "1", "-ar", "16000", output_path])
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        return True
    except Exception as e:
        logger.error(f"FFmpeg 预处理失败: {e}")
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

# =================== 提取 embedding ===================
def extract_embedding_from_file(sv_pipe, wav_path):
    import torchaudio
    import torch

    try:
        model = sv_pipe.model
        audio, sr = torchaudio.load(wav_path)
        if sr != 16000:
            resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            audio = resample(audio)
        audio = audio.mean(dim=0)      # 保证 shape [T]
        audio = audio.unsqueeze(0)     # shape [1, T]

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
        return None, 0.0

    model_votes = {}
    model_scores = {}

    for model_name, sv_pipe in sv_pipelines.items():
        emb_a = extract_embedding_from_file(sv_pipe, segment_path)
        if emb_a is None:
            model_votes[model_name] = "Failed"
            continue

        scores = []
        conf = Config.SV_MODELS[model_name]
        threshold = conf['threshold']
        gap = conf['gap']

        for name, db_embs in speaker_db.items():
            if model_name not in db_embs: continue
            emb_b = np.array(db_embs[model_name]).flatten()
            score = 1 - cosine(emb_a.flatten(), emb_b)
            scores.append((name, score))

        if not scores:
            model_votes[model_name] = "NoDB"
            continue

        scores.sort(key=lambda x: x[1], reverse=True)
        top1_name, top1_score = scores[0]
        top2_name, top2_score = scores[1] if len(scores) > 1 else (None, 0.0)
        score_gap = top1_score - top2_score

        if top1_score >= threshold and score_gap >= gap:
            model_votes[model_name] = top1_name
            model_scores[model_name] = top1_score
        else:
            model_votes[model_name] = "Unknown"
            model_scores[model_name] = top1_score

    votes = list(model_votes.values())
    first_vote = votes[0]
    if first_vote not in ["Unknown", "Failed", "NoDB"] and all(v == first_vote for v in votes):
        avg_confidence = np.mean(list(model_scores.values()))
        logger.info(f"🔍 交叉验证成功: [{first_vote}] | Avg_Score: {avg_confidence:.3f} | Votes: {votes}")
        return first_vote, avg_confidence
    else:
        return None, 0.0

# =================== Flask 接口 ===================
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
            
            logger.info(f"📥 收到任务: {file.filename}")
            if not preprocess_audio(raw_temp, proc_temp):
                return jsonify({"error": "Audio preprocessing failed"}), 500
            
            audio_duration = 0
            try:
                probe = subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', proc_temp])
                audio_duration = float(probe)
            except: pass

            res = asr_pipeline.generate(input=proc_temp, language="auto", use_itn=True)
            full_text = ""
            segments = []

            if res and isinstance(res, list) and len(res) > 0:
                item = res[0]
                full_text = item.get("text", "")
                if "sentence_info" in item:
                    raw_segments = item["sentence_info"]
                else:
                    raw_segments = [{"text": full_text, "start": 0, "end": int(audio_duration * 1000)}]

                processed_segments = []
                for seg in raw_segments:
                    raw_text = seg.get("text", "")
                    start = seg.get("start", 0)
                    end = seg.get("end", 0)
                    if any(tag in raw_text for tag in INVALID_TAGS): continue

                    emotion = "neutral"
                    for tag, emo_code in EMOTION_TAGS.items():
                        if tag in raw_text:
                            emotion = emo_code
                            break

                    clean_text = raw_text
                    for tag in (list(EMOTION_TAGS.keys()) + list(INVALID_TAGS) + ["<|zh|>", "<|en|>", "<|yue|>", "<|withitn|>", "<|speech|>"]):
                        clean_text = clean_text.replace(tag, "")
                    clean_text = clean_text.strip()
                    if not clean_text: continue

                    identity = None
                    confidence = 0.0
                    duration_ms = end - start
                    if duration_ms > Config.MIN_SPEAKER_DURATION_MS:
                        seg_wav = os.path.join(tempfile.gettempdir(), f"seg_{start}_{int(time.time())}.wav")
                        if extract_segment(proc_temp, start, end, seg_wav):
                            temp_files.append(seg_wav)
                            identity, confidence = identify_speaker_fusion(seg_wav)

                    if Config.ONLY_REGISTERED_SPEAKERS and identity is None: continue
                    if identity is None: identity = "Unknown"

                    processed_segments.append({
                        "text": clean_text,
                        "start": start,
                        "end": end,
                        "spk": identity,
                        "emotion": emotion,
                        "confidence": float(f"{confidence:.3f}")
                    })

                segments = processed_segments
                if Config.ONLY_REGISTERED_SPEAKERS:
                    full_text = "".join([s["text"] for s in segments])

            process_time = time.time() - request_start
            rtf = process_time / audio_duration if audio_duration > 0 else 0
            logger.info(f"✅ 完成! 音频:{audio_duration:.1f}s | 耗时:{process_time:.2f}s | RTF:{rtf:.3f}")

            return jsonify({
                "full_text": full_text,
                "segments": segments,
                "meta": {
                    "process_time": process_time,
                    "audio_duration": audio_duration,
                    "rtf": rtf
                }
            })

        except Exception as e:
            logger.error(f"❌ 处理异常: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"error": str(e)}), 500

        finally:
            for f in temp_files:
                if os.path.exists(f):
                    try: os.remove(f)
                    except: pass

# =================== 启动 ===================
if __name__ == "__main__":
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        logger.critical("❌ 系统未安装 FFmpeg！")
        sys.exit(1)

    load_models()
    app.run(host=Config.HOST, port=Config.PORT, debug=False, threaded=True)
