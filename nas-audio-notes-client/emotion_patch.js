// Patch for adding emotion display to dashboard
// Insert this after line 539 (after speakerTags definition)

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

// Then modify the card HTML structure around line 560 to include emotionTags section:
<div class="space-y-2 mt-auto pt-4 border-t border-gray-800/50">
    <div class="flex flex-wrap gap-2">
        ${speakerTags || '<span class="text-xs text-gray-600">无说话人</span>'}
    </div>
    ${emotionTags ? `<div class="flex flex-wrap gap-2 pt-2 border-t border-gray-800/30">
                            ${emotionTags}
                        </div>` : ''}
</div>
