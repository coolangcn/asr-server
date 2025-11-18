#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, tempfile, subprocess
import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import torch

# 和 asr_server.py 保持一致
SPEAKER_DB_FILE = "speaker_db_multi.json"

SV_MODELS = {
    "eres2net_large": {
        "id": "iic/speech_eres2net_large_200k_sv_zh-cn_16k-common",
        "rev": "v1.0.0",
    },
    "rdino_ecapa": {
        "id": "iic/speech_rdino_ecapa_tdnn_sv_zh-cn_cnceleb_16k",
        "rev": "v1.0.0",
    }
}

def preprocess_audio(input_path, output_path):
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-i", input_path,
        "-ac", "1", "-ar", "16000",
        output_path
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except:
        return False

def extract_embedding(sv_pipe, wav_path):
    """使用底层模型提取 embedding（避免两输入报错）"""
    try:
        model = sv_pipe.model

        import torchaudio
        audio, sr = torchaudio.load(wav_path)

        if sr != 16000:
            resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            audio = resample(audio)

        # [C, T] -> [1, T]
        audio = audio.mean(dim=0, keepdim=True)

        # 不要 unsqueeze，不要变成 [1,1,T]
        # 模型要求 [1, T]

        with torch.no_grad():
            out = model(audio)
            if isinstance(out, dict):
                emb = out.get("spk_embedding")
            else:
                emb = out

        return emb.squeeze().cpu().numpy().tolist()

    except Exception as e:
        print("❌ Extract embedding failed:", e)
        return None


def register_speaker(name, audio_file):
    print(f"📥 处理音频: {audio_file}")

    # 1. 预处理
    tmp = os.path.join(tempfile.gettempdir(), f"reg_{os.path.basename(audio_file)}")
    if not preprocess_audio(audio_file, tmp):
        print("❌ FFmpeg 转换失败")
        return

    # 2. 加载数据库
    if os.path.exists(SPEAKER_DB_FILE):
        with open(SPEAKER_DB_FILE, "r", encoding="utf-8") as f:
            db = json.load(f)
    else:
        db = {}

    if name not in db:
        db[name] = {}

    # 3. 遍历所有模型提取 embedding
    for model_name, conf in SV_MODELS.items():
        print(f"  🔍 使用模型 {model_name} 提取 embedding ...")

        sv = pipeline(
            task=Tasks.speaker_verification,
            model=conf["id"],
            model_revision=conf["rev"],
            device="cuda"
        )

        emb = extract_embedding(sv, tmp)
        if emb is None:
            print(f"  ❌ {model_name} 提取失败")
            continue

        db[name][model_name] = emb
        print(f"  ✅ {model_name} 提取成功, 维度: {len(emb)}")

    # 4. 保存
    with open(SPEAKER_DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

    print(f"\n🎉 声纹注册完成: {name}")
    print(f"📦 已保存到: {SPEAKER_DB_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="说话人名称")
    parser.add_argument("--audio", required=True, help="单段音频（3-10 秒）")
    args = parser.parse_args()

    register_speaker(args.name, args.audio)
