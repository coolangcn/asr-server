#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修改代码也显示neutral情感，用于测试"""

# 读取文件
with open('web_viewer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 找到过滤neutral的代码并注释掉过滤
old_code = '''                // 生成情感标签 (只显示非neutral的情感)
                const emotionTags = Object.entries(emotionStats)
                    .filter(([emo]) => emo !== 'neutral')'''

new_code = '''                // 生成情感标签 (显示所有情感包括neutral，用于测试)
                const emotionTags = Object.entries(emotionStats)
                    // .filter(([emo]) => emo !== 'neutral')  // 临时注释以测试'''

content = content.replace(old_code, new_code)

# 写回文件
with open('web_viewer.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("已修改为显示所有情感(包括neutral)")
print("请重启web_viewer.py以查看效果")
