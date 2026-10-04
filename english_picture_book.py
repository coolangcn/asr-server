#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘本日记 v2 —— 把宝宝每天的真实生活"发散"成一册多页英文绘本 + 逐页插画 + TTS 讲述

主角: 大可, 小女孩, 生日 2023-08-09 (年龄按故事日期自动计算)
流程: 当日转写精华片段 → Gemini 发散成 4-6 页连续小故事
      → 逐页生图(固定角色设定保证形象一致) → edge-tts 每页音频(英文朗读+中文讲述)
存档:
  english_enlightenment/picturebook/YYYY-MM-DD.json      v2 结构 {pages: [...]}
  english_enlightenment/picturebook/images/<date>_pN.png 逐页插画
  english_enlightenment/picturebook/audio/<date>_pN.mp3  逐页语音
  v1 单页旧条目保持兼容(服务端自动归一化为单页书)

用法:
  python3 english_picture_book.py                    # 今天
  python3 english_picture_book.py --push             # 今天 + 邮件推送
  python3 english_picture_book.py --date 2026-09-30 --force --push
  python3 english_picture_book.py --backfill 30      # 补写历史(不推送)
"""

import os
import re
import json
import time
import shutil
import asyncio
import argparse
import tempfile
import subprocess
import threading
from datetime import datetime, date, timedelta

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

import requests

from english_daily_report import (
    fetch_day_segments, gemini_generate, load_json, save_json, logger,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BOOK_DIR = os.path.join(BASE_DIR, 'english_enlightenment', 'picturebook')
IMG_DIR = os.path.join(BOOK_DIR, 'images')
AUDIO_DIR = os.path.join(BOOK_DIR, 'audio')

BIRTHDAY = date(2023, 8, 9)
PAGE_MIN, PAGE_MAX = 4, 20  # 页数由素材量动态决定(2026-10-02: 素材 28→86 条/天, 上限 14→20 配合放宽)

# 固定角色设定: 每页插画 prompt 都带上, 保证全书形象一致
CHAR_SHEET = (
    "the main character Dake: a cute 3-year-old Chinese girl with short black bob hair "
    "and straight bangs, big sparkling brown eyes, rosy round cheeks, wearing a "
    "mint-green long-sleeve dress and white sneakers"
)

# TTS 声线 (可用环境变量覆盖)
TTS_EN_VOICE = os.getenv('PB_TTS_EN_VOICE', 'en-US-AnaNeural')       # Ana: 美音童声
TTS_ZH_VOICE = os.getenv('PB_TTS_ZH_VOICE', 'zh-CN-XiaoxiaoNeural')  # 晓晓: 温暖女声讲述

STORY_PROMPT = """你是一位儿童英语启蒙绘本作家。下面是一个{age}岁小女孩"大可"一天的家庭录音转写片段
(可能有识别噪声、中英混杂、无意义碎语，说话人在方括号里)。

请根据这些真实素材，把大可的今天温柔地改编成一册连续绘本小故事。
页数由素材决定：今日共 {n_moments} 条有效片段。若片段较多（{n_moments} ≥ 40 条），
   页数至少要 {prich} 页（上限 {pmax} 页），把当天不同时段的多个事件都编进故事，不要只提炼一条主线；
   片段少就紧凑一些（最少 {pmin} 页）。
1. 不是逐字复述，而是把当天真实发生的小事串联成有起承转合的小故事，允许适当加入一点点想象力
2. 每页 1-2 句极简英文（{age}岁孩子能听懂的词汇，现在简单时为主），每页配中文翻译
3. 页与页之间要连贯，像翻页讲故事；最后一页给一个温暖收尾
4. 每页一个 "scene"：一句英文画面描述（用于插画生成），只描述这一页的画面内容，
   不要写大可的外貌服饰（外貌由系统统一提供，保持全书一致）
5. title: 英文书名，不超过 6 个词，充满童趣
6. 每页还要输出 "lines"：把这一页拆成分角色台词，用于多声线朗读——
   - spk 只能取: "narrator"(旁白) / "大可"(小女孩) / "妈妈" / "爸爸" / "婆婆"(外婆)
   - narrator 负责讲述，大可/妈妈/爸爸/婆婆只说自己的台词（语气要有童趣和故事感）
   - 素材里出现婆婆的对话时，故事里要自然地让婆婆出场；没出现就不要硬加
   - 每个元素 {{"spk": "...", "en": "英文", "zh": "中文"}}，en/zh 内容对应一致，lines 合起来就是这一页的全部内容
   - text_en = 所有 en 空格连接；text_zh = 所有 zh 连接

只输出 JSON:
{{"title": "...", "pages": [{{"text_en": "...", "text_zh": "...", "scene": "...", "lines": [{{"spk": "...", "en": "...", "zh": "..."}}, ...]}}, ...]}}

今日片段:
{moments}"""


def age_of(day: date) -> int:
    return day.year - BIRTHDAY.year - ((day.month, day.day) < (BIRTHDAY.month, BIRTHDAY.day))


# ---------------- 故事主笔 (NAS CPA gemini-3.8-flash-high, 失败回退 Google API) ----------------

def story_llm_generate(prompt):
    """用 CPA 网关的 gemini-3.8-flash-high(high 思考档)写故事; 复用 IMAGE_CPA 连接配置"""
    base = os.getenv('IMAGE_CPA_BASE_URL', '').strip('"').rstrip('/')
    key = os.getenv('IMAGE_CPA_API_KEY', '').strip('"')
    model = os.getenv('PB_STORY_CPA_MODEL', 'gemini-3.8-flash-high').strip('"')
    if not (base and key):
        return None
    try:
        resp = requests.post(
            f"{base}/chat/completions",
            headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
            json={
                'model': model,
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.7,
                'max_tokens': 12000,  # 思考模型: 预算需覆盖思考+正文
            },
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content']
    except Exception as e:
        logger.warning("CPA 故事生成失败(%s): %s", model, e)
        return None


def _oa_story_llm(env_prefix, default_model, prompt):
    """OpenAI 兼容网关通用故事主笔 (PB_*_BASE_URL/_API_KEY/_MODEL), 思考模型需足额 max_tokens"""
    base = os.getenv(f'{env_prefix}_BASE_URL', '').strip('"').rstrip('/')
    key = os.getenv(f'{env_prefix}_API_KEY', '').strip('"')
    model = os.getenv(f'{env_prefix}_MODEL', default_model).strip('"')
    if not (base and key):
        logger.warning("%s 主笔未配置(%s_*), 跳过", env_prefix, env_prefix)
        return None
    try:
        resp = requests.post(
            f"{base}/chat/completions",
            headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
            json={
                'model': model,
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.7,
                'max_tokens': 12000,
            },
            timeout=240,
        )
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content']
    except Exception as e:
        logger.warning("%s(%s) 故事生成失败: %s", env_prefix, model, e)
        return None


# 变体对照册 → 主笔函数 (失败不回退: 两册同故事无对照意义)
VARIANT_LLMS = {
    'grok': lambda p: _oa_story_llm('PB_GROK', 'grok-chat-fast', p),
    'agnes': lambda p: _oa_story_llm('PB_AGNES', 'agnes-2.5-pro-alpha', p),
}


# ---------------- 片段挑选 ----------------

def pick_moments(segs, max_lines=90):
    """从当天语音段里挑有信息量的片段: 每小时自适应(稀疏时段全取, 密集时段取最长15条), 按时间排序。
    2026-10-02: 上限 28→90 条、截断 80→120 字、过滤 <8 字碎语 (9-30 实测覆盖率 23%→~75%)"""
    by_hour = {}
    for s in segs:
        if len(s['text']) < 8:
            continue
        hour = (s['recording_time'] or '00:00')[:2]
        by_hour.setdefault(hour, []).append(s)
    moments = []
    for hour in sorted(by_hour):
        lines = sorted(by_hour[hour], key=lambda s: -len(s['text']))[:15]
        for s in lines:
            moments.append(f"[{s['spk']} {s['recording_time']}] {s['text'][:120]}")
    return moments[:max_lines]


# ---------------- 生图 (三通道: new-api → CPA gemini → grok2api) ----------------

def _img_newapi(prompt):
    """NAS new-api 网关 (agnes-image-2.5-flash), 返回图片 bytes 或 None"""
    base_url = os.getenv('IMAGE_NEWAPI_BASE_URL', '').strip('"').rstrip('/')
    api_key = os.getenv('IMAGE_NEWAPI_API_KEY', '').strip('"')
    model = os.getenv('IMAGE_NEWAPI_MODEL', '').strip('"') or 'agnes-image-2.5-flash'
    if not (base_url and api_key):
        return None
    try:
        resp = requests.post(
            f"{base_url}/images/generations",
            headers={'Authorization': f'Bearer {api_key}'},
            json={'model': model, 'prompt': prompt, 'n': 1, 'size': '1024x1024'},
            timeout=120,
        )
        resp.raise_for_status()
        item = ((resp.json() or {}).get('data') or [{}])[0]
        if item.get('b64_json'):
            import base64
            return base64.b64decode(item['b64_json'])
        if item.get('url'):
            img = requests.get(item['url'], timeout=60)
            img.raise_for_status()
            return img.content
    except Exception as e:
        logger.warning(f"🎨 [new-api:{model}] {e}")
    return None


def _img_cpa(prompt):
    """NAS CPA 的 gemini-3.1-flash-image, chat 接口返回 data URI"""
    base_url = os.getenv('IMAGE_CPA_BASE_URL', '').strip('"').rstrip('/')
    api_key = os.getenv('IMAGE_CPA_API_KEY', '').strip('"')
    model = os.getenv('IMAGE_CPA_MODEL', '').strip('"') or 'gemini-3.1-flash-image'
    if not (base_url and api_key):
        return None
    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers={'Authorization': f'Bearer {api_key}'},
            json={'model': model, 'messages': [{'role': 'user', 'content': prompt}]},
            timeout=120,
        )
        resp.raise_for_status()
        message = ((resp.json() or {}).get('choices') or [{}])[0].get('message', {})
        images = message.get('images') or []
        url = (images[0].get('image_url', {}) or {}).get('url', '') if images else ''
        if url.startswith('data:'):
            import base64
            return base64.b64decode(url.split(',', 1)[1])
    except Exception as e:
        logger.warning(f"🎨 [CPA:{model}] {e}")
    return None


def _img_grok(prompt):
    """NAS grok2api 的 grok-imagine-image-2.0, 容器内 url 需重写为外部地址"""
    base_url = os.getenv('IMAGE_GROK2API_BASE_URL', '').strip('"').rstrip('/')
    api_key = os.getenv('IMAGE_GROK2API_API_KEY', '').strip('"')
    model = os.getenv('IMAGE_GROK2API_MODEL', '').strip('"') or 'grok-imagine-image-2.0'
    if not (base_url and api_key):
        return None
    try:
        resp = requests.post(
            f"{base_url}/images/generations",
            headers={'Authorization': f'Bearer {api_key}'},
            json={'model': model, 'prompt': prompt, 'n': 1},
            timeout=120,
        )
        resp.raise_for_status()
        item = ((resp.json() or {}).get('data') or [{}])[0]
        if item.get('b64_json'):
            import base64
            return base64.b64decode(item['b64_json'])
        url = item.get('url', '')
        if not url:
            return None
        from urllib.parse import urlparse
        parsed = urlparse(base_url)
        fixed = url.replace('http://127.0.0.1:8000', f"{parsed.scheme}://{parsed.netloc}")
        img = requests.get(fixed, headers={'Authorization': f'Bearer {api_key}'}, timeout=60)
        img.raise_for_status()
        return img.content
    except Exception as e:
        logger.warning(f"🎨 [grok2api:{model}] {e}")
    return None


# 每册专属生图通道 (对照实验: 三个模型家族各自包办一册的插画); 首选失败按序回退
_IMG_CPA = ('CPA', _img_cpa)
_IMG_NEWAPI = ('new-api', _img_newapi)
_IMG_GROK = ('Grok', _img_grok)
IMG_CHANNELS = {
    'cpa':   (_IMG_CPA, _IMG_NEWAPI, _IMG_GROK),
    'grok':  (_IMG_GROK, _IMG_CPA, _IMG_NEWAPI),
    'agnes': (_IMG_NEWAPI, _IMG_CPA, _IMG_GROK),  # agnes-image-2.5-flash 在 new-api 网关
}


def gen_page_image(scene: str, char_sheet: str, engine: str = 'cpa'):
    """生成一页插画: engine 对应的专属通道优先, 失败按序回退。返回 (bytes, 通道名) 或 (None, '')"""
    prompt = (
        f"Children's picture book page, warm watercolor style, consistent character design. "
        f"{char_sheet}. Scene: {scene}. Cozy family home, soft pastel colors, no text."
    )
    for ch, fn in IMG_CHANNELS.get(engine, IMG_CHANNELS['cpa']):
        data = fn(prompt)
        if data:
            return data, ch
    return None, ''


# ---------------- TTS (Fish Audio 多角色优先, edge-tts 兜底; ffmpeg 拼接) ----------------

FISH_API_BASE = os.getenv('FISH_AUDIO_BASE_URL', 'https://api.fish.audio').rstrip('/')
FISH_API_KEY = os.getenv('FISH_AUDIO_API_KEY', '').strip('"')
FISH_TTS_MODEL = os.getenv('FISH_TTS_MODEL', 's2.1-pro-free')  # 免费模型, 走请求头
# 多 key 轮换: FISH_AUDIO_API_KEYS 优先, 现有单 key 永远排第一(兜底可用)
FISH_KEYS = [k.strip().strip('"') for k in os.getenv('FISH_AUDIO_API_KEYS', '').split(',') if k.strip()]
if FISH_API_KEY:
    FISH_KEYS = [FISH_API_KEY] + [k for k in FISH_KEYS if k != FISH_API_KEY]
elif not FISH_KEYS:
    FISH_KEYS = []
_fish_key_idx = 0
_fish_bad_keys = set()  # 401/402 的 key 本进程内永久剔除

# 角色 → 声线映射 (reference_id 可在 .env 覆盖)
FISH_ROLE_VOICES = {
    'narrator': os.getenv('FISH_VOICE_NARRATOR', '').strip('"'),
    '大可': os.getenv('FISH_VOICE_DAKE', '').strip('"'),
    '妈妈': os.getenv('FISH_VOICE_MOM', '').strip('"'),
    '爸爸': os.getenv('FISH_VOICE_DAD', '').strip('"'),
    '婆婆': os.getenv('FISH_VOICE_GRANDMA', '').strip('"'),
}
_fish_disabled = False  # 遇到 402/401 等硬错误后, 本进程不再重试 Fish


def fish_tts(text: str, voice_id: str, out_path: str):
    """Fish Audio 单段 TTS: 多 key 轮换, 401/402 剔除坏 key, 429 换下一个; 全失败返回 False"""
    global _fish_disabled, _fish_key_idx, _fish_bad_keys
    if _fish_disabled or not (FISH_KEYS and voice_id):
        return False
    live = [k for k in FISH_KEYS if k not in _fish_bad_keys]
    if not live:
        logger.warning("🔊 [Fish] 全部 %d 个 key 均无效, 本次改用 edge-tts 兜底", len(FISH_KEYS))
        _fish_disabled = True
        return False
    for attempt in range(len(live)):
        key = live[(_fish_key_idx + attempt) % len(live)]
        try:
            resp = requests.post(
                f'{FISH_API_BASE}/v1/tts',
                headers={
                    'Authorization': f'Bearer {key}',
                    'Content-Type': 'application/json',
                    'model': FISH_TTS_MODEL,
                },
                json={'text': text, 'reference_id': voice_id, 'format': 'mp3'},
                timeout=90,
            )
            if resp.status_code in (401, 402):
                bad_idx = FISH_KEYS.index(key) + 1
                _fish_bad_keys.add(key)
                logger.warning("🔊 [Fish] key#%d 无效(401/402), 剔除后剩 %d 个可用", bad_idx, len(live) - attempt - 1)
                continue
            if resp.status_code == 429:
                logger.warning("🔊 [Fish] key#%d 限流(429), 换下一个 key", FISH_KEYS.index(key) + 1)
                continue
            resp.raise_for_status()
            if len(resp.content) < 1000:
                logger.warning("🔊 [Fish] 返回音频过小(%dB)", len(resp.content))
                return False
            _fish_key_idx = (_fish_key_idx + attempt + 1) % len(live)  # 下次换 key 均摊负载
            with open(out_path, 'wb') as f:
                f.write(resp.content)
            return True
        except Exception as e:
            logger.warning("🔊 [Fish] TTS 失败(key#%d): %s", FISH_KEYS.index(key) + 1, e)
    logger.warning("🔊 [Fish] 本轮全部 %d 个 key 不可用, 本次改用 edge-tts 兜底", len(live))
    _fish_disabled = True
    return False


def _make_silence(out_path: str, dur: float = 0.35):
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=mono',
             '-t', str(dur), '-c:a', 'libmp3lame', '-b:a', '128k', out_path],
            capture_output=True, timeout=30, check=True)
        return True
    except Exception:
        return False


def _normalize_loudness(p: str):
    """单段响度归一(EBU R128, I=-16), 让角色/旁白音量一致。失败则保留原样。"""
    target = os.getenv('PB_LOUDNORM_I', '-16').strip()
    out = p + '.norm.mp3'
    try:
        r = subprocess.run(
            ['ffmpeg', '-y', '-i', p, '-filter:a',
             f'loudnorm=I={target}:TP=-1.5:LRA=11', '-ar', '44100',
             '-c:a', 'libmp3lame', '-b:a', '128k', out],
            capture_output=True, timeout=60)
        if r.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 500:
            os.replace(out, p)
    except Exception:
        pass


def fish_tts_page(lines, out_path: str):
    """多角色朗读一页: 每句台词按角色声线, 先英文后中文, 句间 0.35s。
    每段合成后做响度归一, 避免旁白响角色轻。返回 out_path 或 None(失败, 由调用方回退 edge-tts)"""
    global _fish_disabled
    if _fish_disabled or not FISH_API_KEY:
        return None
    tmp = tempfile.mkdtemp(prefix='pb_fish_')
    parts = []
    try:
        for i, line in enumerate(lines):
            spk = (line.get('spk') or 'narrator').strip()
            vid = FISH_ROLE_VOICES.get(spk) or FISH_ROLE_VOICES['narrator'] or FISH_ROLE_VOICES['大可']
            for lang in ('en', 'zh'):
                txt = (line.get(lang) or '').strip()
                if not txt:
                    continue
                p = os.path.join(tmp, f'{i:02d}_{lang}.mp3')
                if fish_tts(txt, vid, p):
                    if os.getenv('PB_LOUDNORM', '1') != '0':
                        _normalize_loudness(p)
                    parts.append(p)
                    if _fish_disabled:
                        return None
            if parts:
                sil = os.path.join(tmp, f'{i:02d}_sil.mp3')
                if _make_silence(sil):
                    parts.append(sil)
        if not parts:
            return None
        # 重新编码拼接, 规避不同片段采样率不一致
        lst = os.path.join(tmp, 'list.txt')
        with open(lst, 'w') as f:
            for p in parts:
                f.write(f"file '{p}'\n")
        subprocess.run(
            ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', lst,
             '-c:a', 'libmp3lame', '-b:a', '128k', out_path],
            capture_output=True, timeout=120, check=True)
        return out_path
    except Exception as e:
        logger.warning("🔊 [Fish] 页面拼接失败: %s", e)
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---- edge-tts 单声线兜底 ----
TTS_EN_VOICE = os.getenv('PB_TTS_EN_VOICE', 'en-US-AnaNeural')       # Ana: 美音童声
TTS_ZH_VOICE = os.getenv('PB_TTS_ZH_VOICE', 'zh-CN-XiaoxiaoNeural')  # 晓晓: 温暖女声讲述


def _tts_segment(text, voice, out_path, rate=None):
    import edge_tts

    async def _run():
        comm = edge_tts.Communicate(text, voice, rate=rate) if rate else edge_tts.Communicate(text, voice)
        await comm.save(out_path)

    asyncio.run(_run())


def tts_page(text_en: str, text_zh: str, out_path: str):
    """edge-tts 兜底: 先英文朗读再中文讲述, 拼接为单个 mp3。返回 out_path 或 None"""
    tmp = tempfile.mkdtemp(prefix='pb_tts_')
    parts = []
    try:
        if text_en.strip():
            p = os.path.join(tmp, 'en.mp3')
            _tts_segment(text_en, TTS_EN_VOICE, p, rate='-10%')
            parts.append(p)
        if text_zh.strip():
            p = os.path.join(tmp, 'zh.mp3')
            _tts_segment(text_zh, TTS_ZH_VOICE, p)
            parts.append(p)
        if not parts:
            return None
        if len(parts) == 1:
            shutil.copyfile(parts[0], out_path)
        else:
            lst = os.path.join(tmp, 'list.txt')
            with open(lst, 'w') as f:
                for p in parts:
                    f.write(f"file '{p}'\n")
            subprocess.run(
                ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', lst, '-c', 'copy', out_path],
                capture_output=True, timeout=60, check=True,
            )
        return out_path
    except Exception as e:
        logger.warning(f"🔊 edge-tts 生成失败: {e}")
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------- 生成一册书 ----------------

def write_book(day: date, engine: str = 'cpa', tag: str = ''):
    """生成 v2 多页绘本, 返回 book dict 或 None。
    engine: 故事主笔 ('cpa'=gemini-3.8-flash-high / 'grok'=grok-4.7 / 'agnes'=agnes-2.5-flash)
    tag:    文件名变体后缀 (''=主册, 'grok'/'agnes'=对照版) → {date}_grok_pN.png 等"""
    segs = fetch_day_segments(day)
    moments = pick_moments(segs)
    if len(moments) < 3:
        logger.info("[%s%s] 有效片段不足(%d), 跳过绘本", day, f"/{tag}" if tag else "", len(moments))
        return None

    age = age_of(day)
    prich = max(PAGE_MIN + 1, PAGE_MAX * 2 // 3)  # 素材丰富时的最低页数(20 上限 → 13)
    full_prompt = STORY_PROMPT.format(age=age, pmin=PAGE_MIN, pmax=PAGE_MAX, prich=prich,
                                      n_moments=len(moments), moments="\n".join(moments))
    # 主笔选择: 变体册(grok/agnes)失败不回退(两册同故事无对照意义), 直接跳过等下次重试
    text = None
    if engine in VARIANT_LLMS:
        text = VARIANT_LLMS[engine](full_prompt)
        if text:
            logger.info("[%s%s] 故事主笔: %s", day, f"/{tag}" if tag else "", engine)
        else:
            logger.warning("[%s%s] %s 主笔不可用, 跳过本次对照册", day, f"/{tag}" if tag else "", engine)
            return None
    if not text:
        text = story_llm_generate(full_prompt)
        if text:
            logger.info("[%s%s] 故事主笔: CPA gemini-3.8-flash-high", day, f"/{tag}" if tag else "")
    if not text:
        text = gemini_generate(full_prompt, max_output=6000, temperature=0.7)
        if text:
            logger.info("[%s%s] 故事主笔: 回退 Google API", day, f"/{tag}" if tag else "")
    if not text:
        logger.warning("[%s%s] 故事生成失败(LLM 无响应)", day, f"/{tag}" if tag else "")
        return None
    m = re.search(r'\{.*\}', text, re.S)
    if not m:
        logger.warning("[%s] 故事 JSON 解析失败", day)
        return None
    try:
        story = json.loads(m.group(0))
    except Exception:
        logger.warning("[%s] 故事 JSON 解析异常", day)
        return None

    raw_pages = story.get('pages') or []
    if not isinstance(raw_pages, list) or not raw_pages:
        logger.warning("[%s] 故事页数为空", day)
        return None
    raw_pages = raw_pages[:PAGE_MAX]

    sfx = f"_{tag}" if tag else ""
    char_sheet = CHAR_SHEET.replace('3-year-old', f'{age}-year-old')
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)

    pages = []
    for i, raw in enumerate(raw_pages, 1):
        if not isinstance(raw, dict):
            continue
        lines = raw.get('lines') if isinstance(raw.get('lines'), list) else []
        lines = [{'spk': (l.get('spk') or 'narrator'), 'en': (l.get('en') or '').strip(),
                  'zh': (l.get('zh') or '').strip()} for l in lines if isinstance(l, dict)]
        text_en = (raw.get('text_en') or '').strip() or ' '.join(l['en'] for l in lines if l['en'])
        text_zh = (raw.get('text_zh') or '').strip() or ''.join(l['zh'] for l in lines if l['zh'])
        scene = (raw.get('scene') or '').strip()
        if not (text_en or text_zh):
            continue
        page = {'seq': len(pages) + 1, 'text_en': text_en, 'text_zh': text_zh, 'scene': scene}
        if lines:
            page['lines'] = lines

        # 逐页插画 → 落盘, JSON 里存相对路径
        if scene:
            img_path = os.path.join(IMG_DIR, f"{day.isoformat()}{sfx}_p{page['seq']}.png")
            try:
                data, ch = gen_page_image(scene, char_sheet, engine)
                if data:
                    with open(img_path, 'wb') as f:
                        f.write(data)
                    page['image'] = f"images/{day.isoformat()}{sfx}_p{page['seq']}.png"
                    page['image_engine'] = ch
                    logger.info("[%s%s] p%d 插画完成 (%.0fKB, %s)", day, sfx, page['seq'], len(data) / 1024, ch)
            except Exception as e:
                logger.warning("[%s%s] p%d 插画失败: %s", day, sfx, page['seq'], e)

        # 逐页 TTS: Fish Audio 多角色优先, edge-tts 单声线兜底
        audio_rel = f"audio/{day.isoformat()}{sfx}_p{page['seq']}.mp3"
        audio_path = os.path.join(AUDIO_DIR, f"{day.isoformat()}{sfx}_p{page['seq']}.mp3")
        try:
            done = fish_tts_page(lines, audio_path) if lines else None
            if done:
                page['audio'] = audio_rel
                page['audio_engine'] = 'fish-multi'
            else:
                done = tts_page(text_en, text_zh, audio_path)
                if done:
                    page['audio'] = audio_rel
                    page['audio_engine'] = 'edge-tts'
            if done:
                logger.info("[%s%s] p%d 语音完成 (%s)", day, sfx, page['seq'], page['audio_engine'])
        except Exception as e:
            logger.warning("[%s%s] p%d 语音失败: %s", day, sfx, page['seq'], e)

        pages.append(page)

    book = {
        'version': 2,
        'date': day.isoformat(),
        'title': story.get('title') or 'A Day with Dake',
        'protagonist': '大可',
        'age': age,
        'pages': pages,
        'moments_used': len(moments),
    }
    if tag:
        book['engine'] = engine  # 变体册标记主笔模型
    logger.info("[%s%s] 绘本 v2 生成完成: 《%s》 %d 页", day, sfx, book['title'], len(pages))
    return book


def render_email_text(book):
    lines = [f"📖 《{book['title']}》 · {book['date']} · {len(book.get('pages', []))}页", '']
    for p in book.get('pages', []):
        lines.append(f"P{p['seq']}  {p.get('text_en', '')}")
        lines.append(f"     {p.get('text_zh', '')}")
        lines.append('')
    return '\n'.join(lines)


def process_day(day: date, push: bool = False, force: bool = False, make_video: bool = True,
                variant: str = ''):
    """生成一天绘本并落盘。variant='' 主册(CPA主笔+邮件+视频); variant='grok'/'agnes' 对照册(对应主笔, 无视频/邮件)"""
    os.makedirs(BOOK_DIR, exist_ok=True)
    out = os.path.join(BOOK_DIR, f"{day.isoformat()}{('.' + variant) if variant else ''}.json")
    if os.path.exists(out) and not force:
        page = load_json(out, None)
        if page:
            logger.info("[%s%s] 已有绘本页, 跳过生成 (--force 可重生成)", day, f"/{variant}" if variant else "")
            book = page if page.get('pages') else {
                'version': 1, 'date': page.get('date'), 'title': page.get('title'),
                'pages': [{'seq': 1, 'text_en': page.get('story_en', ''),
                           'text_zh': page.get('story_zh', ''), 'image': page.get('image')}],
            }
        else:
            return
    else:
        book = write_book(day, engine=variant or 'cpa', tag=variant)
        if not book:
            return
        save_json(out, book)
        if variant:
            return book  # 变体册: 不做视频/邮件
        if make_video:
            try:
                final = make_book_video(day.isoformat(), book)
                if final:
                    save_json(out, book)  # 回写 video 路径
            except Exception as e:
                logger.warning("[%s] 视频生成失败(不影响绘本): %s", day, e)
        return book

    print(render_email_text(book))
    if push:
        from email_utils import send_email_sync
        # 封面: 第 1 页插画转 dataURI (兼容 v1 dataURI/URL/相对路径三种存法)
        image = None
        p1 = (book.get('pages') or [{}])[0]
        img = p1.get('image') or ''
        try:
            if img.startswith('data:image'):
                image = img
            elif img.startswith('images/'):
                import base64
                with open(os.path.join(BOOK_DIR, img), 'rb') as f:
                    image = 'data:image/png;base64,' + base64.b64encode(f.read()).decode()
            elif img.startswith('http'):
                r = requests.get(img, timeout=60)
                import base64
                image = 'data:image/png;base64,' + base64.b64encode(r.content).decode()
        except Exception as e:
            logger.warning("封面获取失败: %s", e)
        ok = send_email_sync(
            f"📖 大可绘本日记 {day.isoformat()} · 《{book.get('title', '')}》",
            render_email_text(book), image_data=image)
        logger.info("邮件推送: %s", "成功" if ok else "失败/跳过")


# ---------------- 绘本视频 (agnes-video-2.5-flash keyframe 模式) ----------------

VIDEO_DIR = os.path.join(BOOK_DIR, 'video')
AGNES_BASE = os.getenv('AGNES_VIDEO_BASE_URL', 'https://apihub.agnes-ai.com/v1').strip('"').rstrip('/')
AGNES_MODEL = os.getenv('AGNES_VIDEO_MODEL', 'agnes-video-2.5-flash').strip('"')
AGNES_KEYS = [k.strip().strip('"') for k in os.getenv('AGNES_API_KEYS', '').split(',') if k.strip()]
PB_PUBLIC_BASE = os.getenv('PB_PUBLIC_BASE_URL', '').strip('"').rstrip('/')
PB_PREVIEW_TOKEN = os.getenv('CRY_PREVIEW_TOKEN', '').strip('"')
PB_VIDEO_ENABLED = os.getenv('PB_VIDEO_ENABLED', '1').strip() == '1'


def _media_duration(path):
    try:
        out = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', path],
            capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip())
    except Exception:
        return None


def _agnes_create(prompt, first_frame_url, seconds):
    """创建视频任务: 8 key 轮换, 队列满(503)等待重试。返回 video_id 或 None"""
    body = {'model': AGNES_MODEL, 'prompt': prompt, 'mode': 'keyframe',
            'first_frame': first_frame_url, 'seconds': str(int(seconds)),
            'size': '720P', 'aspect_ratio': '16:9', 'n': 1}
    queue_waits = 0
    for attempt in range(60):  # 队列满时最长约 30 分钟
        key = AGNES_KEYS[attempt % len(AGNES_KEYS)]
        try:
            resp = requests.post(f'{AGNES_BASE}/videos',
                                 headers={'Authorization': f'Bearer {key}'}, json=body, timeout=60)
            if resp.status_code == 503:
                queue_waits += 1
                if queue_waits % 4 == 1:
                    logger.info("🎬 [agnes] 队列满, 继续等待 (%d)", queue_waits)
                time.sleep(30)
                continue
            if resp.status_code in (401, 403):
                continue  # 换下一个 key
            if resp.status_code == 400:
                logger.warning("🎬 [agnes] 请求被拒(400): %s", resp.text[:200])
                return None
            resp.raise_for_status()
            d = resp.json() or {}
            vid = d.get('video_id') or d.get('id') or d.get('task_id')
            if vid:
                logger.info("🎬 [agnes] 任务已创建: %s (使用 key: %s...%s)", vid, key[:10], key[-6:])
                return vid, key
            logger.warning("🎬 [agnes] 创建响应无 id: %s", str(d)[:200])
            return None, None
        except Exception as e:
            logger.warning("🎬 [agnes] 创建异常: %s", e)
            time.sleep(10)
    logger.warning("🎬 [agnes] 队列持续满, 放弃本页视频")
    return None, None


def _agnes_poll(video_id, auth_key=None, timeout_s=1200):
    """轮询任务直到 completed/failed, 返回视频 url 或 None"""
    start = time.time()
    last_log = 0
    while time.time() - start < timeout_s:
        key = auth_key or AGNES_KEYS[int(time.time()) % len(AGNES_KEYS)]
        try:
            r = requests.get('https://apihub.agnes-ai.com/agnesapi',
                             params={'video_id': video_id, 'model_name': AGNES_MODEL},
                             headers={'Authorization': f'Bearer {key}'}, timeout=30)
            if r.ok:
                d = r.json() or {}
                st = d.get('status')
                if st == 'completed':
                    return d.get('url')
                if st == 'failed':
                    logger.warning("🎬 [agnes] 任务失败: %s", str(d.get('error'))[:200])
                    return None
                if time.time() - last_log > 60:
                    last_log = time.time()
                    logger.info("🎬 [agnes] 生成中 progress=%s%%", d.get('progress'))
        except Exception as e:
            logger.warning("🎬 [agnes] 轮询异常: %s", e)
        time.sleep(5)
    logger.warning("🎬 [agnes] 轮询超时(%ds)", timeout_s)
    return None


def agnes_video_page(page, day_iso, out_path):
    """keyframe 模式: 本页插画为首帧生成动态片段, 时长对齐本页配音。返回 out_path 或 None"""
    scene = page.get('scene') or 'a child playing at home'
    if PB_PREVIEW_TOKEN:
        first_frame = f"{PB_PUBLIC_BASE}/pbpub/{day_iso}/{page['seq']}.png?t={PB_PREVIEW_TOKEN}"
    else:
        first_frame = f"{PB_PUBLIC_BASE}/pbpub/{day_iso}/{page['seq']}.png"
    audio_dur = _media_duration(os.path.join(BOOK_DIR, page['audio'])) if page.get('audio') else None
    seconds = min(12, max(4, int(round((audio_dur or 8) + 1))))
    prompt = (
        f"Gently bring this children's picture book illustration to life: {scene}. "
        "Warm watercolor style, subtle natural motion, soft slow camera drift, "
        "the character's face and clothing must stay identical to the first frame, "
        "cozy dreamy bedtime-story mood, no text, no scene change."
    )
    vid, auth_key = _agnes_create(prompt, first_frame, seconds)
    if not vid:
        return None
    url = _agnes_poll(vid, auth_key=auth_key)
    if not url:
        return None
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        if len(r.content) < 50_000:
            logger.warning("🎬 [agnes] 视频文件过小(%dB)", len(r.content))
            return None
        with open(out_path, 'wb') as f:
            f.write(r.content)
        logger.info("🎬 [%s] p%d 视频片段完成 (%.1fs 目标/ %.0fKB)",
                    day_iso, page['seq'], seconds, len(r.content) / 1024)
        return out_path
    except Exception as e:
        logger.warning("🎬 [agnes] 下载失败: %s", e)
        return None


def compose_book_video(day_iso, book, clips):
    """逐页片段与配音对齐 (画面不足补末帧/超出截断), 合成整本视频。返回最终路径或 None"""
    tmp = tempfile.mkdtemp(prefix='pb_vid_')
    segs = []
    try:
        for page in book.get('pages', []):
            clip = clips.get(page['seq'])
            audio_rel = page.get('audio')
            if not clip or not audio_rel:
                continue
            audio_path = os.path.join(BOOK_DIR, audio_rel)
            adur = _media_duration(audio_path) or 0
            if adur <= 0:
                continue
            seg = os.path.join(tmp, f"seg_{page['seq']:02d}.mp4")
            subprocess.run(
                ['ffmpeg', '-y', '-i', clip, '-i', audio_path, '-filter_complex',
                 '[0:v]scale=720:720:force_original_aspect_ratio=decrease,'
                 'pad=720:720:(ow-iw)/2:(oh-ih)/2,fps=25,'
                 f'tpad=stop_mode=clone:stop_duration=20[v]',
                 '-map', '[v]', '-map', '1:a', '-t', f'{adur:.2f}',
                 '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
                 '-c:a', 'aac', '-b:a', '160k', '-movflags', '+faststart', seg],
                capture_output=True, timeout=300, check=True)
            segs.append(seg)
        if not segs:
            return None
        lst = os.path.join(tmp, 'list.txt')
        with open(lst, 'w') as f:
            for s in segs:
                f.write(f"file '{s}'\n")
        final = os.path.join(VIDEO_DIR, f'{day_iso}.mp4')
        subprocess.run(
            ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', lst,
             '-c', 'copy', '-movflags', '+faststart', final],
            capture_output=True, timeout=300, check=True)
        logger.info("🎬 [%s] 整本视频合成完成: %s (%.1fMB)",
                    day_iso, final, os.path.getsize(final) / 1024 / 1024)
        return final
    except Exception as e:
        logger.warning("🎬 合成失败: %s", e)
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def make_book_video(day_iso, book):
    """逐页生成视频片段并合成整本; 结果写回 book JSON。返回最终路径或 None"""
    if not (PB_VIDEO_ENABLED and AGNES_KEYS and PB_PUBLIC_BASE):
        logger.info("🎬 视频未启用或缺少配置, 跳过")
        return None
    os.makedirs(VIDEO_DIR, exist_ok=True)
    clips = {}
    for page in book.get('pages', []):
        if not (page.get('image') and page.get('audio')):
            continue
        out = os.path.join(VIDEO_DIR, f"{day_iso}_p{page['seq']}.mp4")
        if os.path.isfile(out):  # 幂等: 断点续跑
            clips[page['seq']] = out
            continue
        logger.info("🎬 [%s] p%d 开始生成视频片段...", day_iso, page['seq'])
        r = agnes_video_page(page, day_iso, out)
        if r:
            clips[page['seq']] = r
            page['video'] = f"video/{day_iso}_p{page['seq']}.mp4"
    if not clips:
        logger.warning("🎬 [%s] 无可用片段, 不合成", day_iso)
        return None
    final = compose_book_video(day_iso, book, clips)
    if final:
        book['video'] = f"video/{day_iso}.mp4"
    return final


def revoice_day(day_iso: str) -> bool:
    """用当前 .env 音色重新为已有绘本配音(故事/插画/视频全部不动, 幂等)。
    注意: 已合成的整本视频保留旧音频版本, 不重做(避免浪费)。"""
    out = os.path.join(BOOK_DIR, f"{day_iso}.json")
    book = load_json(out, None)
    if not book or not book.get('pages'):
        logger.warning("[%s] 无绘本 JSON, 跳过 revoice", day_iso)
        return False
    for page in book['pages']:
        seq = page.get('seq', 1)
        lines = page.get('lines') or []
        audio_rel = f"audio/{day_iso}_p{seq}.mp3"
        audio_path = os.path.join(AUDIO_DIR, f"{day_iso}_p{seq}.mp3")
        try:
            done = fish_tts_page(lines, audio_path) if lines else None
            if done:
                page['audio'] = audio_rel
                page['audio_engine'] = 'fish-multi'
            else:
                done = tts_page(page.get('text_en', ''), page.get('text_zh', ''), audio_path)
                if done:
                    page['audio'] = audio_rel
                    page['audio_engine'] = 'edge-tts'
            logger.info("[%s] p%d revoice %s", day_iso, seq, 'OK' if done else 'FAIL')
        except Exception as e:
            logger.warning("[%s] p%d revoice 失败: %s", day_iso, seq, e)
        time.sleep(0.3)
    save_json(out, book)
    logger.info("[%s] revoice 完成 (%d 页, 视频保留不动)", day_iso, len(book['pages']))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', help='生成某天 YYYY-MM-DD, 默认今天')
    ap.add_argument('--backfill', type=int, metavar='N', help='从N天前补到昨天')
    ap.add_argument('--push', action='store_true', help='邮件推送')
    ap.add_argument('--force', action='store_true', help='已存在也重新生成')
    ap.add_argument('--video', action='store_true', help='只为已有绘本补做视频(不重新生成故事)')
    ap.add_argument('--revoice', action='store_true', help='用当前音色为已有绘本重新配音(不改故事/插画)')
    ap.add_argument('--all-history', action='store_true', help='按 DB 全部有转写日期补跑(跳过已有, 不做视频)')
    ap.add_argument('--grok', action='store_true', help='只生成 Grok 对照册(与 --date/--force 配合)')
    ap.add_argument('--agnes', action='store_true', help='只生成 Agnes 对照册(与 --date/--force 配合)')
    args = ap.parse_args()

    if args.revoice:
        if args.all_history:
            books = sorted((f[:-5] for f in os.listdir(BOOK_DIR)
                            if f.endswith('.json') and not re.search(r'\.[a-z0-9_-]{1,20}\.json$', f)),
                           reverse=True)  # 最新日期优先, 旧的慢慢补 (排除 .grok.json/.agnes.json 等变体册)
            logger.info("revoice 全部 %d 册 (最新优先)...", len(books))
            for i, d in enumerate(books, 1):
                try:
                    revoice_day(d)
                    logger.info("进度 %d/%d", i, len(books))
                except Exception as e:
                    logger.warning("[%s] revoice 失败: %s", d, e)
        elif args.date:
            revoice_day(args.date)
        else:
            revoice_day(date.today().isoformat())
        return

    if args.all_history:
        import psycopg2
        conn = psycopg2.connect(os.getenv('DATABASE_URL', ''))
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT recording_time::date FROM transcriptions "
                    "WHERE segments_json IS NOT NULL ORDER BY 1")
        days = [r[0] for r in cur.fetchall() if r[0]]
        cur.close()
        conn.close()
        logger.info("DB 共 %d 个有转写日期, 开始补跑", len(days))
        done = skip = 0
        for d in days:
            try:
                if os.path.exists(os.path.join(BOOK_DIR, f'{d.isoformat()}.json')):
                    skip += 1
                    continue
                book = process_day(d, push=False, make_video=False)
                done += 1 if book else 0
            except Exception as e:
                logger.warning("[%s] 补跑失败: %s", d, e)
        logger.info("全历史补跑完成: 新生成 %d 册, 跳过已有 %d 天", done, skip)
        return

    if args.video:
        day = datetime.strptime(args.date, '%Y-%m-%d').date() if args.date else date.today()
        out = os.path.join(BOOK_DIR, f"{day.isoformat()}.json")
        book = load_json(out, None)
        if not book or not book.get('pages'):
            logger.warning("[%s] 无绘本 JSON, 先生成绘本再补视频", day)
            return
        final = make_book_video(day.isoformat(), book)
        if final:
            save_json(out, book)
        return

    if args.backfill:
        today = date.today()
        days = [today - timedelta(days=i) for i in range(args.backfill, 0, -1)]
        logger.info("补写绘本 %d 天...", len(days))
        for d in days:
            try:
                process_day(d, push=False, make_video=False)
            except Exception as e:
                logger.error("[%s] 补写失败: %s", d, e)
        logger.info("补写完成, 书架共 %d 本", len([f for f in os.listdir(BOOK_DIR) if f.endswith('.json')]))
    else:
        day = datetime.strptime(args.date, '%Y-%m-%d').date() if args.date else date.today()
        single = [('grok', args.grok), ('agnes', args.agnes)]
        for v, flag in single:
            if flag:
                # 只生成该变体对照册 (可 --force 重生成)
                process_day(day, push=False, force=args.force, variant=v)
                return
        # 每日主流程: 主册(CPA) 与 Grok/Agnes 对照册并行生成, 全部完成后进程才退出
        variants = ['grok', 'agnes']
        variant_errs = {}

        def _variant_worker(v):
            try:
                process_day(day, push=False, force=args.force, variant=v)
            except Exception as e:
                variant_errs[v] = e
                logger.warning("[%s] %s 对照册失败(不影响主册): %s", day, v, e)

        threads = []
        for v in variants:
            t = threading.Thread(target=_variant_worker, args=(v,), name=f'{v}-book', daemon=True)
            t.start()
            threads.append(t)
        process_day(day, push=args.push, force=args.force)
        for t in threads:
            t.join(timeout=3600)  # 等并行册收尾(launchd one-shot 进程不能提前退出)
        for v in variant_errs:
            logger.warning("[%s] %s 对照册本次未完成, 明天自动重试", day, v)


if __name__ == '__main__':
    main()
