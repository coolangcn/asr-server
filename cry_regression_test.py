#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
哭声检测回归测试
=================
对已验证的真哭声源录音（DB 事件 486/487/488，6-30）和 no_cry 负样本，
分别测试两种预处理路径的三模型 Baby 打分：
  A) 无 loudnorm（quick_cry_detect 快速模式，用户之前测试用的路径）
  B) loudnorm 归一化（正式实时路径 /transcribes 用的路径）

用于定位 7-1 之后漏检的真实环节，并为重定阈值提供数据。
"""
import os
import sys
import json
import subprocess
import numpy as np
from scipy.spatial.distance import cosine

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import torch
import torchaudio

DB_FILE = "speaker_db_multi.json"
TMP = "/tmp/cry_reg_test"
os.makedirs(TMP, exist_ok=True)

SV_MODELS = {
    "eres2net_large": "iic/speech_eres2net_large_200k_sv_zh-cn_16k-common",
    "rdino_ecapa": "iic/speech_rdino_ecapa_tdnn_sv_zh-cn_cnceleb_16k",
    "camplusplus": "iic/speech_campplus_sv_zh-cn_16k-common",
}

# 真哭声正样本（DB 已确认事件）
POS_SAMPLES = [
    ("event486", "/Volumes/download/records/Sony-2/processed/2026-06-30/TermuxAudioRecording_2026-06-30_20-56-04.m4a"),
    ("event487", "/Volumes/download/records/Sony-2/processed/2026-06-30/TermuxAudioRecording_2026-06-30_21-19-18.m4a"),
    ("event488", "/Volumes/download/records/Sony-2/processed/2026-06-30/TermuxAudioRecording_2026-06-30_22-01-44.m4a"),
]

TARGET = ["baby", "宝宝"]
SKIP_SPEAKERS = ["大可", "妈妈", "婆婆"]  # VOICE_RECOGNITION_SPEAKERS，与线上打分口径一致


def load_pipelines():
    pipes = {}
    for name, mid in SV_MODELS.items():
        print(f"🔍 加载 SV [{name}] ...")
        pipes[name] = pipeline(task=Tasks.speaker_verification, model=mid, model_revision="v1.0.0", device="cpu")
    return pipes


def preprocess(input_path, output_path, normalize):
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", input_path]
    if normalize:
        cmd.extend(["-af", "loudnorm=I=-14:TP=-1.5:LRA=11"])
    cmd.extend(["-ac", "1", "-ar", "16000", output_path])
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return output_path


def extract_embedding(sv_pipe, wav_path):
    model = sv_pipe.model
    audio, sr = torchaudio.load(wav_path)
    if sr != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(audio)
    audio = audio.mean(dim=0, keepdim=True)
    with torch.no_grad():
        out = model(audio)
        emb = out.get("spk_embedding") if isinstance(out, dict) else out
    return emb.squeeze().cpu().numpy()


def cos(a, b):
    return 1 - cosine(np.asarray(a).flatten(), np.asarray(b).flatten())


def score_against_db(emb, db, model_name):
    """复现线上 detect_cry_from_full_audio 打分口径"""
    scores = {}
    for name, sdata in db.items():
        if name in SKIP_SPEAKERS:
            continue
        avg = sdata.get("avg_embeddings", {}).get(model_name)
        if avg is None:
            continue
        scores[name] = cos(emb, avg)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    target_hits = [(n, s) for n, s in ranked if n.lower() in TARGET]
    others = [(n, s) for n, s in ranked if n.lower() not in TARGET]
    best_target = target_hits[0] if target_hits else ("无", 0.0)
    best_other = others[0] if others else ("无", 0.0)
    gap = best_target[1] - best_other[1]
    return ranked, best_target, best_other, gap


def main():
    db = json.load(open(DB_FILE, encoding="utf-8"))
    pipes = load_pipelines()

    inputs = []
    for label, src in POS_SAMPLES:
        if not os.path.exists(src):
            print(f"⚠️ 源文件不存在，跳过: {src}")
            continue
        inputs.append((label, src, True))

    # 负样本
    import psycopg2
    conn = psycopg2.connect(host="192.168.1.188", port=5433, user="postgres", password="cncncncn", dbname="postgres")
    cur = conn.cursor()
    cur.execute("""
        SELECT filename, processed_at FROM processed_files_a
        WHERE filename LIKE 'TermuxAudioRecording_2026-06-30%'
          AND filename NOT IN (SELECT filename FROM baby_cry_events)
        ORDER BY processed_at DESC LIMIT 8
    """)
    neg_rows = cur.fetchall()
    conn.close()

    base = "/Volumes/download/records/Sony-2/processed"
    neg_files = []
    for fname, _ in neg_rows:
        date_part = fname.split("_")[1] if "_" in fname else None
        if not date_part:
            continue
        p = os.path.join(base, date_part, fname)
        if os.path.exists(p):
            neg_files.append(p)
    neg_files = neg_files[:4]
    print(f"负样本(no_cry 文件): {len(neg_files)} 个")
    for i, p in enumerate(neg_files):
        inputs.append((f"neg{i+1}", p, False))

    results = []
    for label, src, is_pos in inputs:
        for normalize in [False, True]:
            tag = "loudnorm" if normalize else "raw"
            out = os.path.join(TMP, f"{label}_{tag}.wav")
            preprocess(src, out, normalize)
            row = {"label": label, "mode": tag, "is_pos": is_pos, "per_model": {}}
            verdict_votes = 0
            for m_name, pipe in pipes.items():
                emb = extract_embedding(pipe, out)
                ranked, best_t, best_o, gap = score_against_db(emb, db, m_name)
                passed = bool(best_t[1] >= 0.65 and gap >= 0.15)
                if passed:
                    verdict_votes += 1
                row["per_model"][m_name] = {
                    "target": f"{best_t[0]}={best_t[1]:.3f}",
                    "other": f"{best_o[0]}={best_o[1]:.3f}",
                    "gap": round(gap, 3),
                    "pass": passed,
                }
                t_str = ", ".join([f"{n}={s:.3f}" for n, s in ranked])
                print(f"  [{label}|{tag}] {m_name}: {t_str}  {'✅' if passed else '❌'}")
            row["votes"] = verdict_votes
            row["result"] = "DETECT" if verdict_votes >= 3 else "MISS"
            print(f"  ▶ {label} [{tag}]: {row['result']} (votes={verdict_votes}/3)\n")
            results.append(row)

    with open("cry_regression_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "═" * 60)
    print("📊 汇总: raw(无归一化,quick_cry_detect口径) vs loudnorm(正式路径口径)")
    for label, _, _ in inputs:
        raw_row = next((r for r in results if r["label"] == label and r["mode"] == "raw"), None)
        ln_row = next((r for r in results if r["label"] == label and r["mode"] == "loudnorm"), None)
        if raw_row and ln_row:
            print(f"  {label}: raw={raw_row['result']}{raw_row['votes']}票  loudnorm={ln_row['result']}{ln_row['votes']}票")


if __name__ == "__main__":
    main()
