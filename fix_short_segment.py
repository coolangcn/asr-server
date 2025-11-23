#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修复短segment whisper_text未初始化问题"""

with open('asr_server.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 找到跳过声纹识别的地方并添加初始化
for i, line in enumerate(lines):
    if '分段时长过短' in line and '跳过声纹识别' in line:
        # 在下一行添加变量初始化
        indent = ' ' * 28  # 保持缩进一致
        lines.insert(i+1, f"{indent}# 即使跳过声纹识别，也要初始化这些变量\n")
        lines.insert(i+2, f"{indent}emotion = \"neutral\"\n")
        lines.insert(i+3, f"{indent}whisper_text = None\n")
        break

with open('asr_server.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("✅ 成功修复短segment whisper_text未初始化问题!")
print("   - 在跳过声纹识别时初始化 emotion 和 whisper_text")
