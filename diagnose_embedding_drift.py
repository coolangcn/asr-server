#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
声纹漂移诊断脚本
=================
用 speaker_db 中已存储的样本 wav，在当前运行时重新提取 embedding，
与数据库中存储的 embedding（3月计算）做余弦相似度对比，量化运行时漂移。

对比维度：
1. new_sample_emb vs stored_sample_emb  —— 模型前向/运行时漂移（同一wav同模型）
2. new_sample_emb vs stored_avg_emb     —— 线上实际比对路径
3. 模拟线上打分：new Baby emb vs 库内所有人 avg —— 验证 Baby 是否还能最高分命中

输出：drift_report.json + 控制台报告
"""
import os
import sys
import json
import numpy as np
from scipy.spatial.distance import cosine

os.environ.setdefault("HF_HUB_OFFLINE", "0")

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import torch
import torchaudio

DB_FILE = "speaker_db_multi.json"
REPORT_FILE = "drift_report.json"

SV_MODELS = {
    "eres2net_large": {
        "id": "iic/speech_eres2net_large_200k_sv_zh-cn_16k-common",
        "rev": "v1.0.0",
    },
    "rdino_ecapa": {
        "id": "iic/speech_rdino_ecapa_tdnn_sv_zh-cn_cnceleb_16k",
        "rev": "v1.0.0",
    },
    "camplusplus": {
        "id": "iic/speech_campplus_sv_zh-cn_16k-common",
        "rev": "v1.0.0",
    },
}


def load_pipelines():
    pipes = {}
    for name, conf in SV_MODELS.items():
        print(f"🔍 加载 SV [{name}]: {conf['id']} ...")
        pipes[name] = pipeline(
            task=Tasks.speaker_verification,
            model=conf["id"],
            model_revision=conf["rev"],
            device="cpu",
        )
    return pipes


def extract_embedding(sv_pipe, wav_path):
    """与 asr_server.extract_embedding_from_file 完全一致的提取逻辑"""
    try:
        model = sv_pipe.model
        audio, sr = torchaudio.load(wav_path)
        if sr != 16000:
            resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            audio = resample(audio)
        audio = audio.mean(dim=0, keepdim=True)  # [C, T] -> [1, T]
        with torch.no_grad():
            out = model(audio)
            if isinstance(out, dict):
                emb = out.get("spk_embedding")
            else:
                emb = out
        return emb.squeeze().cpu().numpy()
    except Exception as e:
        print(f"   ❌ extract_embedding 失败 ({wav_path}): {e}")
        return None


def cos(a, b):
    return 1 - cosine(np.asarray(a).flatten(), np.asarray(b).flatten())


def main():
    db = json.load(open(DB_FILE, encoding="utf-8"))
    pipes = load_pipelines()

    report = {"models": list(SV_MODELS.keys()), "speakers": {}}

    for spk_name, spk_data in db.items():
        samples = spk_data.get("samples", [])
        usable = [s for s in samples if s.get("audio_path") and os.path.exists(s["audio_path"])]
        if not usable:
            print(f"\n⏭️  {spk_name}: 无可用样本 wav（{len(samples)} 条全部缺失），跳过重算")
            report["speakers"][spk_name] = {"usable_samples": 0, "total_samples": len(samples)}
            continue

        print(f"\n══ {spk_name}: {len(usable)}/{len(samples)} 条样本可用 ══")
        spk_report = {"usable_samples": len(usable), "total_samples": len(samples), "samples": [],
                      "sim_vs_db_avgs": {}}

        new_emb_cache = {}  # (wav_path, m_name) -> emb

        for s in usable:
            wav_path = s["audio_path"]
            print(f"  📄 {wav_path}")
            row = {"wav": wav_path, "per_model": {}}

            for m_name, pipe in pipes.items():
                stored = s.get("embeddings", {}).get(m_name)
                if stored is None:
                    continue
                cache_key = (wav_path, m_name)
                if cache_key not in new_emb_cache:
                    new_emb_cache[cache_key] = extract_embedding(pipe, wav_path)
                new_emb = new_emb_cache[cache_key]
                if new_emb is None:
                    continue

                sim_vs_sample = cos(new_emb, stored)
                avg = spk_data.get("avg_embeddings", {}).get(m_name)
                sim_vs_avg = cos(new_emb, avg) if avg is not None else None

                row["per_model"][m_name] = {
                    "sim_vs_stored_sample": round(sim_vs_sample, 4),
                    "sim_vs_stored_avg": round(sim_vs_avg, 4) if sim_vs_avg is not None else None,
                }
                print(f"     {m_name}: vs样本={sim_vs_sample:.4f}, vs均值={sim_vs_avg:.4f}" if sim_vs_avg is not None
                      else f"     {m_name}: vs样本={sim_vs_sample:.4f}")

            row["per_model"] and spk_report["samples"].append(row)

        # 模拟线上打分：该说话人所有可用样本的新 embedding vs 库内所有人 avg
        for m_name, pipe in pipes.items():
            new_embs = []
            for s in usable:
                e = new_emb_cache.get((s["audio_path"], m_name))
                if e is not None:
                    new_embs.append(e)
            if not new_embs:
                continue
            scores = {}
            for other, odata in db.items():
                avg = odata.get("avg_embeddings", {}).get(m_name)
                if avg is None:
                    continue
                sims = [cos(e, avg) for e in new_embs]
                scores[other] = round(float(np.mean(sims)), 4)
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            spk_report["sim_vs_db_avgs"][m_name] = ranked
            top_str = ", ".join([f"{n}={v}" for n, v in ranked])
            print(f"  🎯 [{m_name}] 新embedding对库内均值打分: {top_str}")

        report["speakers"][spk_name] = spk_report

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 汇总
    print("\n" + "═" * 60)
    print("📊 漂移汇总（new vs stored，余弦相似度；1.0=无漂移）")
    for m_name in SV_MODELS:
        sims = []
        for spk, spk_r in report["speakers"].items():
            for row in spk_r.get("samples", []):
                v = row["per_model"].get(m_name)
                if v and v.get("sim_vs_stored_sample") is not None:
                    sims.append(v["sim_vs_stored_sample"])
        if sims:
            print(f"  {m_name}: 样本级 sim 分布 min={min(sims):.4f} mean={np.mean(sims):.4f} max={max(sims):.4f} (n={len(sims)})")

    print(f"\n✅ 报告已保存: {REPORT_FILE}")


if __name__ == "__main__":
    main()
