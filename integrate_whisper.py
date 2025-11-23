#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在segment处理中添加Whisper识别调用"""

with open('asr_server.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 找到声纹识别后的位置（在append之前）
for i, line in enumerate(lines):
    if 'identity, confidence, recognition_details = identify_speaker_fusion(seg_wav)' in line:
        # 在这行之后添加情感检测和Whisper识别
        indent = '                                '
        new_lines = [
            indent + '# 情感检测\n',
            indent + 'emotion = detect_emotion_for_segment(seg_wav)\n',
            indent + '# Whisper对比识别\n',
            indent + 'whisper_text = transcribe_with_whisper(seg_wav)\n'
        ]
        lines[i+1:i+1] = new_lines
        break

# 找到processed_segments.append并添加whisper_text字段
for i, line in enumerate(lines):
    if '"emotion": emotion,' in line:
        # 在emotion之后添加whisper_text
        indent = ' ' * (len(line) - len(line.lstrip()))
        new_line = indent + '"whisper_text": whisper_text,\n'
        lines[i+1:i+1] = [new_line]
        break

with open('asr_server.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("✅ 成功集成Whisper到segment处理流程!")
print("  - 在声纹识别后调用detect_emotion_for_segment()")
print("  - 在情感检测后调用transcribe_with_whisper()")
print("  - 在返回结果中添加whisper_text字段")
