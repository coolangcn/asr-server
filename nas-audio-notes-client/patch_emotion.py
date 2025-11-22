#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修补web_viewer.py文件,添加情感展示功能到仪表盘视图
"""

# 读取文件
with open('web_viewer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 找到renderDashboard函数中的代码位置
# 在说话人标签之后添加情感统计代码

# 第一步:在sp eakerTags定义之后添加情感统计代码
emotion_stats_code = '''
                // 统计情感分布
                const emotionStats = {};
                item.segments.forEach(s => {
                    const emo = s.emotion || 'neutral';
                    emotionStats[emo] = (emotionStats[emo] || 0) + 1;
                });
                
                // 生成情感标签 (只显示非neutral的情感)
                const emotionTags = Object.entries(emotionStats)
                    .filter(([emo]) => emo !== 'neutral')
                    .map(([emo, count]) => {
                        const icon = getEmotionIcon(emo);
                        return `<span class="text-xs px-2 py-1 rounded-md bg-purple-500/10 text-purple-400 border border-purple-500/30 flex items-center gap-1">
                            <span>${icon}</span>
                            <span>${emo}</span>
                            <span class="text-[10px] opacity-70">×${count}</span>
                        </span>`;
                    }).join('');
'''

# 替换点1: 在speakerTags后添加情感统计
old_pattern_1 = '''                }).join('');

                html += `'''

new_pattern_1 = '''                }).join('');
''' +  emotion_stats_code + '''
                html += `'''

content = content.replace(old_pattern_1, new_pattern_1, 1)

# 替换点2: 修改卡片底部显示结构
old_pattern_2 = '''                    <div class="flex flex-wrap gap-2 mt-auto pt-4 border-t border-gray-800/50">
                        ${speakerTags || '<span class="text-xs text-gray-600">无说话人</span>'}
                    </div>
'''

new_pattern_2 = '''                    <div class="space-y-2 mt-auto pt-4 border-t border-gray-800/50">
                        <div class="flex flex-wrap gap-2">
                            ${speakerTags || '<span class="text-xs text-gray-600">无说话人</span>'}
                        </div>
                        ${emotionTags ? `<div class="flex flex-wrap gap-2 pt-2 border-t border-gray-800/30">
                            ${emotionTags}
                        </div>` : ''}
                    </div>
'''

content = content.replace(old_pattern_2, new_pattern_2, 1)

# 写回文件
with open('web_viewer.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("修补完成!")
print("已在仪表盘视图中添加情感展示功能")
