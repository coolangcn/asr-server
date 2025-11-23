#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""添加缺失的配置项"""

with open('asr_server.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 检查是否已有ENABLE_EMOTION_DETECTION
if 'ENABLE_EMOTION_DETECTION' not in content:
    # 在Config类中的WHISPER配置后添加EMOTION配置
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        if 'ENABLE_WHISPER_COMPARISON' in line:
            # 找到这行后，在下一行插入EMOTION配置
            indent = '    '
            emotion_config = f'''
{indent}# SenseVoice情感检测配置
{indent}EMOTION_MODEL = "iic/SenseVoiceSmall"
{indent}ENABLE_EMOTION_DETECTION = True
'''
            lines.insert(i+1, emotion_config)
            break
    
    content = '\n'.join(lines)
    
    with open('asr_server.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ 已添加 ENABLE_EMOTION_DETECTION 配置")
else:
    print("⚠️ 配置已存在，无需添加")
