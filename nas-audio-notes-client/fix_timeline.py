#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单修补:改进时光轴情感显示和头像样式
"""

with open('web_viewer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 修改1: 头像样式 - 使用渐变色和单个大写字母
old_avatar = '''<div class="w-10 h-10 rounded-full bg-gray-800 border border-gray-700 flex items-center justify-center shadow-lg">
                                    <span class="text-xs font-bold text-gray-300">${seg.spk.substring(0,2)}</span>
                                </div>'''

new_avatar = '''<div class="w-10 h-10 rounded-full bg-gradient-to-br from-${textColor}-500 to-${textColor}-700 border-2 border-${textColor}-400/30 flex items-center justify-center shadow-lg shadow-${textColor}-500/20">
                                    <span class="text-sm font-bold text-white">${seg.spk.substring(0,1).toUpperCase()}</span>
                                </div>'''

content = content.replace(old_avatar, new_avatar)

# 修改2: 情感显示 - 添加emoji并显示所有情感
old_emotion = '''${seg.emotion && seg.emotion !== 'neutral' ? 
                                        `<span class="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-gray-800 text-gray-400 ml-2 border border-gray-700">
                                            ${seg.emotion}
                                        </span>` : ''}'''

new_emotion = '''${seg.emotion ? 
                                        `<span class="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium bg-purple-500/10 text-purple-300 ml-2 border border-purple-500/30">
                                            <span>${getEmotionIcon(seg.emotion)}</span>
                                            <span>${seg.emotion}</span>
                                        </span>` : ''}'''

content = content.replace(old_emotion, new_emotion)

with open('web_viewer.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ 修补成功!")
print("- 头像改为彩色渐变 + 单字母大写")
print("- 情感显示改为emoji + 名称,紫色主题")
