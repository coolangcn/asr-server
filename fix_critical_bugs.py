#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修复关键Bug: 恢复detect_emotion_for_segment函数并隐藏VAD调试日志"""

with open('asr_server.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 1. 先找到 transcribe_with_whisper 函数的位置，在其后添加 detect_emotion_for_segment
insert_pos = None
for i, line in enumerate(lines):
    if 'def transcribe_with_whisper(' in line:
        # 找到函数结束位置
        for j in range(i+1, len(lines)):
            if lines[j].strip() and not lines[j].startswith(' ') and not lines[j].startswith('\t'):
                insert_pos = j
                break
            elif j < len(lines) - 1 and lines[j].strip() == '' and lines[j+1].strip() and lines[j+1].startswith('# ==='):
                insert_pos = j
                break
        break

if insert_pos:
    emotion_function = '''
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

'''
    lines.insert(insert_pos, emotion_function)

# 2. 注释VAD调试日志
for i, line in enumerate(lines):
    if '[VAD 调试]' in line and 'logger.info' in line:
        # 注释掉这行
        indent = len(line) - len(line.lstrip())
        lines[i] = ' ' * indent + '# ' + line.lstrip()

with open('asr_server.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("✅ 已修复:")
print("   1. 恢复 detect_emotion_for_segment() 函数")
print("   2. 隐藏 [VAD 调试] 日志输出")
