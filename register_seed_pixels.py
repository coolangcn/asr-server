# -*- coding: utf-8 -*-
"""声纹种子注册脚本（Pixel 新域）
从 NAS 上的 Pixel 分钟录音中切出指定片段，注册为指定说话人的声纹样本。
服务端 /speaker/register 自带 MD5 指纹查重 + source_key 防重复注册。

单条用法:
  python3 register_seed_pixels.py --file /Volumes/download/records/Pixel-5/2026-10-03/TermuxAudioRecording_2026-10-03_21-00-00.m4a --speaker 大可 --start 5 --end 35
批量用法（manifest 每行: 文件路径<TAB>说话人<TAB>起始秒<TAB>结束秒）:
  python3 register_seed_pixels.py --manifest seeds.txt
"""
import os, sys, argparse, subprocess, tempfile

import requests
from dotenv import load_dotenv

load_dotenv('/Users/mac/asr-server/.env')
ASR = "http://localhost:5008"
TOKEN = (os.getenv("ADMIN_TOKEN") or "").strip()
HEADERS = {"X-Admin-Token": TOKEN} if TOKEN else {}


def register_one(path, speaker, start, end):
    if not os.path.exists(path):
        print(f"  ✗ 文件不存在: {path}")
        return False
    base = os.path.basename(path)
    src_key = f"{base}#{int(start)}-{int(end)}"
    tmp = tempfile.mkdtemp(prefix="seed_")
    wav = os.path.join(tmp, f"seed_{int(start)}.wav")
    # 切片 → 16k mono wav（服务端还会做响度归一/降噪预处理）
    r = subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-ss", str(start), "-to", str(end), "-i", path,
                        "-ar", "16000", "-ac", "1", wav], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(wav) or os.path.getsize(wav) < 10000:
        print(f"  ✗ 切片失败 {src_key}: {r.stderr.strip()[:120]}")
        return False
    with open(wav, 'rb') as f:
        resp = requests.post(f"{ASR}/speaker/register", headers=HEADERS,
                             files={'audio_file': (f"seed_{speaker}_{int(start)}.wav", f, 'audio/wav')},
                             data={'speaker_name': speaker, 'source_key': src_key}, timeout=300)
    if resp.status_code == 200:
        print(f"  ✓ [{speaker}] {base} #{int(start)}-{int(end)}s 注册成功")
        return True
    elif resp.status_code == 409:
        print(f"  ⏭ [{speaker}] {src_key} 已注册过（指纹查重命中），跳过")
        return True
    else:
        print(f"  ✗ [{speaker}] {src_key} HTTP {resp.status_code}: {resp.text[:150]}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="NAS 分钟录音 m4a 路径")
    ap.add_argument("--speaker", help="说话人名（须与声纹库一致）")
    ap.add_argument("--start", type=float, default=0, help="起始秒")
    ap.add_argument("--end", type=float, default=30, help="结束秒")
    ap.add_argument("--manifest", help="批量清单：每行 文件<TAB>说话人<TAB>起始秒<TAB>结束秒")
    args = ap.parse_args()

    jobs = []
    if args.manifest:
        for line in open(args.manifest, encoding='utf-8'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) != 4:
                print(f"  ✗ manifest 格式错误（需4列TAB分隔）: {line[:80]}")
                continue
            jobs.append((parts[0], parts[1], float(parts[2]), float(parts[3])))
    elif args.file and args.speaker:
        jobs.append((args.file, args.speaker, args.start, args.end))
    else:
        ap.error("需要 --file+--speaker 或 --manifest")

    ok = sum(1 for j in jobs if register_one(*j))
    print(f"\n完成: {ok}/{len(jobs)} 成功")


if __name__ == "__main__":
    main()
