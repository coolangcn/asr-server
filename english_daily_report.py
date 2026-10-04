#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
英语启蒙日报 —— 英语暴露量统计 + 宝宝新词捕捉器

数据源: transcriptions 表 (segments_json 含说话人 spk + 文本 + 时长)
输出:
  - english_enlightenment/reports/YYYY-MM-DD.json  日报数据
  - english_enlightenment/vocab_history.json       宝宝词汇史 (first_seen/count)
  - english_enlightenment/rejected_words.json      LLM 判定为 ASR 误识别的词(不再当新词)
  - 邮件推送 (--push)

用法:
  python3 english_daily_report.py                 # 统计今天, 不发邮件
  python3 english_daily_report.py --push          # 统计今天 + 邮件推送
  python3 english_daily_report.py --date 2026-09-28
  python3 english_daily_report.py --backfill 60   # 从60天前补到昨天, 只建词汇史不推送
"""

import os
import re
import json
import argparse
import logging
from datetime import datetime, timedelta, date
from collections import defaultdict

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

import requests
import psycopg2

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger('EnEnglishDaily')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'english_enlightenment')
REPORTS_DIR = os.path.join(OUT_DIR, 'reports')
VOCAB_FILE = os.path.join(OUT_DIR, 'vocab_history.json')
REJECT_FILE = os.path.join(OUT_DIR, 'rejected_words.json')

DATABASE_URL = os.getenv('DATABASE_URL', '')
CHILD_SPEAKERS = {s.strip() for s in os.getenv('ENLIGHTENMENT_CHILD_SPEAKERS', '大可,Baby').split(',') if s.strip()}

GEMINI_BASE = os.getenv('GEMINI_API_BASE_URL', 'https://generativelanguage.googleapis.com')
GEMINI_KEYS = [k.strip() for k in os.getenv('GEMINI_API_KEYS', '').split(',') if k.strip()]
GEMINI_MODEL = os.getenv('GEMINI_MODEL_NAME', 'gemini-2.0-flash')
GEMINI_FALLBACK = os.getenv('GEMINI_FALLBACK_MODEL_NAME', '')

IMG_BASE = os.getenv('IMAGE_NEWAPI_BASE_URL', '').strip('"')
IMG_KEY = os.getenv('IMAGE_NEWAPI_API_KEY', '').strip('"')
IMG_MODEL = os.getenv('IMAGE_NEWAPI_MODEL', '').strip('"')

# 英文判定: 一段话里 ASCII 单词占比 ≥ 该阈值且至少 2 个英文词, 才计入"英文暴露时长"
EN_RATIO_THRESHOLD = 0.5
MIN_EN_WORDS = 2


# ---------------- 数据获取 ----------------

def fetch_day_segments(day: date):
    """拉取某天的所有语音段: [{spk, text, duration, recording_time}]"""
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute(
        "SELECT recording_time, segments_json FROM transcriptions "
        "WHERE recording_time >= %s AND recording_time < %s AND segments_json IS NOT NULL",
        (datetime.combine(day, datetime.min.time()),
         datetime.combine(day + timedelta(days=1), datetime.min.time())),
    )
    segs_out = []
    for rt, sj in cur.fetchall():
        try:
            segs = json.loads(sj)
        except Exception:
            continue
        for s in segs:
            tq = s.get('text_quality') or {}
            if tq.get('is_noise') or (tq.get('noise_score') or 0) > 0.6:
                continue
            text = (s.get('text') or s.get('sensevoice_text') or '').strip()
            if not text:
                continue
            metrics = s.get('speech_metrics') or {}
            dur = metrics.get('duration_seconds')
            if not dur:
                dur = ((s.get('end') or 0) - (s.get('start') or 0)) / 1000.0
            segs_out.append({
                'spk': '大可' if (s.get('spk') or 'Unknown') in CHILD_SPEAKERS else (s.get('spk') or 'Unknown'),
                'text': text,
                'duration': float(dur or 0),
                'recording_time': rt.strftime('%H:%M') if rt else '',
            })
    cur.close()
    conn.close()
    return segs_out


# ---------------- 英文识别 ----------------

WORD_RE = re.compile(r"[A-Za-z][A-Za-z']+")

def normalize_word(w: str):
    w = w.lower().strip("'")
    return w if len(w) >= 2 else None


def split_english(seg):
    """返回 (english_tokens, 预估英文词数, is_english_seg)
    SenseVoice 常把英文粘连成一个长串(secretmissionistoget...)，
    ≥8 字符的串按每 5 字符 ≈ 1 词估算真实词数。"""
    tokens = [normalize_word(w) for w in WORD_RE.findall(seg['text'])]
    tokens = [t for t in tokens if t]
    cn_chars = len(re.findall(r'[\u4e00-\u9fff]', seg['text']))
    word_estimate = sum(max(1, len(t) // 5) if len(t) >= 8 else 1 for t in tokens)
    if tokens:
        ratio = word_estimate / (word_estimate + cn_chars)
    else:
        ratio = 0.0
    is_en = ratio >= EN_RATIO_THRESHOLD and word_estimate >= MIN_EN_WORDS
    return tokens, word_estimate, is_en


# ---------------- LLM (Gemini, 多 key 轮询) ----------------

def gemini_generate(prompt, max_output=1024, temperature=0.2):
    if not GEMINI_KEYS:
        return None
    models = [GEMINI_MODEL] + ([GEMINI_FALLBACK] if GEMINI_FALLBACK else [])
    for model in models:
        url = f"{GEMINI_BASE}/v1beta/models/{model}:generateContent"
        for key in GEMINI_KEYS:
            try:
                resp = requests.post(
                    url,
                    params={'key': key},
                    json={
                        'contents': [{'parts': [{'text': prompt}]}],
                        'generationConfig': {
                            'temperature': temperature,
                            'maxOutputTokens': max_output,
                            # Gemini 3.x 是思考模型: 低档位控制思考开销, 防止吃光 maxOutputTokens
                            'thinkingConfig': {'thinkingLevel': 'low'},
                        },
                    },
                    timeout=90,
                )
                if resp.status_code == 429:
                    continue  # 换下一个 key
                if resp.status_code == 404:
                    break      # 模型不存在, 换备用模型
                resp.raise_for_status()
                parts = resp.json()['candidates'][0]['content']['parts']
                return ''.join(p.get('text', '') for p in parts)
            except Exception as e:
                logger.warning(f"Gemini 请求失败 ({model}): {e}")
                continue
    return None


def validate_new_words(new_words):
    """LLM 过滤 ASR 误识别词。返回 (确认的新词列表, 拒绝的词列表)"""
    if not new_words:
        return [], []
    prompt = (
        "下面是从家庭录音 ASR 转写中提取的、一个三岁孩子(及家人对他说的话)可能说出的英文单词候选。\n"
        "请过滤掉明显的语音识别乱码/误识别(不是真实英文单词、或不可能出自幼儿及日常家庭对话)。\n"
        "只输出 JSON: {\"keep\": [\"word1\",...], \"drop\": [{\"word\":\"x\",\"reason\":\"简短原因\"}]}\n"
        f"候选: {json.dumps(new_words, ensure_ascii=False)}"
    )
    text = gemini_generate(prompt)
    if not text:
        logger.warning("LLM 校验不可用, 本次新词全部保留为待定(计入 keep)")
        return list(new_words), []
    m = re.search(r'\{.*\}', text, re.S)
    if not m:
        return list(new_words), []
    try:
        out = json.loads(m.group(0))
        keep = [w for w in out.get('keep', []) if w in new_words]
        drop = [d['word'] for d in out.get('drop', []) if d.get('word')]
        # 兜底: keep/drop 之外的候选默认保留
        keep += [w for w in new_words if w not in keep and w not in drop]
        return keep, drop
    except Exception:
        logger.warning("LLM 返回 JSON 解析失败, 新词全部保留为待定")
        return list(new_words), []


# ---------------- 插画 ----------------

def gen_illustration(word: str):
    """为新词生成庆祝插画, 返回 dataURI 或 None"""
    if not (IMG_BASE and IMG_KEY and IMG_MODEL):
        return None
    prompt = (
        f"Warm children's picture book illustration, soft watercolor style, "
        f"a happy toddler celebrating learning a new English word '{word}', "
        f"the word '{word}' floating in playful letters around the child, cozy home, pastel colors, no text other than the word"
    )
    try:
        resp = requests.post(
            f"{IMG_BASE}/images/generations",
            headers={'Authorization': f'Bearer {IMG_KEY}'},
            json={'model': IMG_MODEL, 'prompt': prompt, 'n': 1, 'size': '1024x1024'},
            timeout=120,
        )
        resp.raise_for_status()
        d = resp.json()['data'][0]
        if d.get('b64_json'):
            return f"data:image/png;base64,{d['b64_json']}"
        if d.get('url'):
            return d['url']
    except Exception as e:
        logger.warning(f"插画生成失败: {e}")
    return None


# ---------------- 词汇史 ----------------

def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return default


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


# ---------------- 单日统计 ----------------

def analyze_day(day: date):
    segs = fetch_day_segments(day)
    stats = {
        'date': day.isoformat(),
        'segments_total': len(segs),
        'by_speaker': {},          # spk -> {speech_min, english_min, english_words}
        'child_english_tokens': defaultdict(int),
        'child_english_segments': [],  # 宝宝含英文的原句(供 LLM 拆词)
    }
    for seg in segs:
        tokens, word_estimate, is_en = split_english(seg)
        spk = seg['spk']
        b = stats['by_speaker'].setdefault(spk, {'speech_min': 0.0, 'english_min': 0.0, 'english_words': 0})
        b['speech_min'] += seg['duration'] / 60.0
        if is_en:
            b['english_min'] += seg['duration'] / 60.0
        b['english_words'] += word_estimate

        if spk == '大可' and tokens:
            for t in tokens:
                if len(t) <= 8:  # 长粘连串不计数, 交给 LLM 拆词
                    stats['child_english_tokens'][t] += 1
            stats['child_english_segments'].append(f"[{seg['recording_time']}] {seg['text'][:60]}")
    return stats


def extract_child_words_llm(segments):
    """LLM 从宝宝原句中拆词(处理英文粘连)并提取真实说出的英文单词"""
    if not segments:
        return []
    prompt = (
        "以下是一个三岁孩子说话的家庭录音转写(语音识别输出，英文单词常粘连无空格，也常中英混杂)。\n"
        "请提取孩子真实说出的英文单词；粘连的英文请拆开(如 secretmissionistoget → secret mission)。\n"
        "只输出 JSON 数组，如 [\"hello\",\"mission\"]；没有英文则输出 []。\n"
        "转写:\n" + "\n".join(segments[:40])
    )
    text = gemini_generate(prompt)
    if not text:
        return []
    m = re.search(r'\[.*\]', text, re.S)
    if not m:
        return []
    try:
        words = json.loads(m.group(0))
        return sorted({normalize_word(w) for w in words if normalize_word(w)})
    except Exception:
        return []


def update_vocab(stats, day: date):
    """用当天宝宝英文词更新词汇史, 返回新词 dict {word: contexts}"""
    vocab = load_json(VOCAB_FILE, {})
    rejected = set(load_json(REJECT_FILE, []))

    tokens = stats['child_english_tokens']
    # 候选1: 正则分出的干净短词; 候选2: LLM 从粘连原句里拆出的词
    short_candidates = sorted(
        (w for w in tokens if w not in vocab and w not in rejected),
        key=lambda w: -tokens[w],
    )[:20]
    llm_words = extract_child_words_llm(stats['child_english_segments'])
    candidates = sorted(set(short_candidates) | (set(llm_words) - set(rejected) - set(vocab)))[:30]
    if not candidates:
        return {}

    kept, dropped = validate_new_words(candidates)

    for w in dropped:
        rejected.add(w)
    if dropped:
        save_json(REJECT_FILE, sorted(rejected))

    new_words = {}
    for w in kept:
        vocab[w] = {'first_seen': day.isoformat(), 'count': 0}
        ctx = [s for s in stats['child_english_segments'] if w in s.lower()][:2]
        new_words[w] = ctx
    # 词频从历史日报重算(幂等): 同一天重复运行不会重复累加
    totals = defaultdict(int)
    for fn in os.listdir(REPORTS_DIR):
        if not fn.endswith('.json'):
            continue
        try:
            rep = json.load(open(os.path.join(REPORTS_DIR, fn)))
        except Exception:
            continue
        for w, c in (rep.get('child_english_tokens') or {}).items():
            totals[w] += c
    for w in vocab:
        vocab[w]['count'] = totals.get(w, 0)
    if totals:
        today_str = day.isoformat()
        for w in vocab:
            if totals.get(w) and vocab[w]['first_seen'] <= today_str:
                vocab[w]['last_seen'] = today_str
    save_json(VOCAB_FILE, vocab)
    return new_words


# ---------------- 报告与推送 ----------------

def render_report_text(stats, new_words):
    lines = [f"📖 大可英语启蒙日报 · {stats['date']}", '']
    lines.append('—— 今日英语暴露量 ——')
    rows = sorted(stats['by_speaker'].items(), key=lambda kv: -kv[1]['english_min'])
    for spk, b in rows:
        who = '👶 大可' if spk in CHILD_SPEAKERS else spk
        lines.append(f"  {who}: 开口 {b['speech_min']:.0f} 分钟 | 英文环境 {b['english_min']:.1f} 分钟 | 英文词 {b['english_words']}")
    total_en = sum(b['english_min'] for b in stats['by_speaker'].values())
    lines.append(f"  ➜ 全家英文暴露合计: {total_en:.1f} 分钟")

    child_tokens = stats['child_english_tokens']
    if child_tokens:
        top = sorted(child_tokens.items(), key=lambda kv: -kv[1])[:12]
        lines.append('')
        lines.append(f"—— 宝宝开口的英文 ({sum(child_tokens.values())} 词次) ——")
        lines.append('  ' + ' · '.join(f"{w}×{c}" for w, c in top))

    if new_words:
        lines.append('')
        lines.append(f"🎉 今日新词 ({len(new_words)}): " + '、'.join(new_words.keys()))
        for w, ctx in list(new_words.items())[:5]:
            if ctx:
                lines.append(f"  · {w}: {ctx[0]}")
    else:
        lines.append('')
        lines.append('今日无新词 (或无宝宝开口的英文)')
    return '\n'.join(lines)


def push_email(day_str, stats, new_words):
    from email_utils import send_email_sync  # 需在 load_dotenv 之后导入
    image = None
    if new_words:
        first = sorted(new_words.keys(), key=lambda w: -stats['child_english_tokens'].get(w, 0))[0]
        image = gen_illustration(first)
    subject = f"🎉 英语启蒙日报 {day_str}" + (f" · 新词: {'/'.join(list(new_words)[:5])}" if new_words else "")
    ok = send_email_sync(subject, render_report_text(stats, new_words), image_data=image)
    logger.info("邮件推送: %s", "成功" if ok else "失败/跳过")


# ---------------- 主流程 ----------------

def process_day(day: date, push: bool):
    stats = analyze_day(day)
    stats['by_speaker'] = {k: v for k, v in stats['by_speaker'].items()}
    new_words = update_vocab(stats, day)
    stats['new_words'] = {w: {'contexts': c} for w, c in new_words.items()}
    stats['child_english_tokens'] = dict(stats['child_english_tokens'])

    save_json(os.path.join(REPORTS_DIR, f"{day.isoformat()}.json"), stats)
    logger.info("[%s] 段数=%d 说话人=%d 宝宝英文词次=%d 新词=%d",
                day, stats['segments_total'], len(stats['by_speaker']),
                sum(stats['child_english_tokens'].values()), len(new_words))

    text = render_report_text(stats, new_words)
    print(text)
    if push:
        push_email(day.isoformat(), stats, new_words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', help='统计某天 YYYY-MM-DD, 默认今天')
    ap.add_argument('--backfill', type=int, metavar='N', help='从N天前补到昨天, 只建词汇史')
    ap.add_argument('--push', action='store_true', help='邮件推送')
    args = ap.parse_args()

    os.makedirs(REPORTS_DIR, exist_ok=True)

    if args.backfill:
        today = date.today()
        days = [today - timedelta(days=i) for i in range(args.backfill, 0, -1)]
        logger.info("补跑 %d 天 (建立词汇史基线, 不推送)...", len(days))
        for d in days:
            try:
                process_day(d, push=False)
            except Exception as e:
                logger.error("[%s] 补跑失败: %s", d, e)
        vocab = load_json(VOCAB_FILE, {})
        logger.info("补跑完成, 词汇史共 %d 词", len(vocab))
    else:
        day = datetime.strptime(args.date, '%Y-%m-%d').date() if args.date else date.today()
        process_day(day, push=args.push)


if __name__ == '__main__':
    main()
