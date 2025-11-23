#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""安全地添加Whisper对比显示到web_viewer.py"""

with open('web_viewer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 找到segment文本显示区域，在emotion之后添加whisper对比
old_display = """                                    ${seg.emotion ? 
                                        `<span class="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium bg-purple-500/10 text-purple-300 ml-2 border border-purple-500/30">
                                            <span>${getEmotionIcon(seg.emotion)}</span>
                                            <span>${seg.emotion}</span>
                                        </span>` : ''}
                                </div>
                            </div>"""

new_display = """                                    ${seg.emotion ? 
                                        `<span class="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium bg-purple-500/10 text-purple-300 ml-2 border border-purple-500/30">
                                            <span>${getEmotionIcon(seg.emotion)}</span>
                                            <span>${seg.emotion}</span>
                                        </span>` : ''}
                                </div>
                                ${seg.whisper_text ? 
                                    `<div class="text-gray-500 text-xs mt-1 pl-4 border-l-2 border-gray-700/50">
                                        <span class="text-gray-600">↪️ Whisper: </span>${seg.whisper_text}
                                    </div>` : ''}
                            </div>"""

content = content.replace(old_display, new_display)

with open('web_viewer.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ 成功添加Whisper对比显示到Web界面!")
print("   - 在segment文本下方显示Whisper识别结果")
print("   - 使用↪️ 符号和灰色样式区分")
