// =================== ECharts 可视化图表 ===================

// 初始化情感曲线图
function initEmotionChart() {
    const chartDom = document.getElementById('emotion-chart');
    if (!chartDom) return;

    const myChart = echarts.init(chartDom);

    // 显示加载动画
    myChart.showLoading({
        text: '加载中...',
        color: '#6366f1',
        textColor: '#fff',
        maskColor: 'rgba(0, 0, 0, 0.3)'
    });

    // 获取数据
    fetch('/api/emotion-timeline')
        .then(res => res.json())
        .then(data => {
            myChart.hideLoading();

            if (!data.timeline || data.timeline.length === 0) {
                myChart.setOption({
                    title: {
                        text: '暂无数据',
                        left: 'center',
                        top: 'center',
                        textStyle: { color: '#9ca3af', fontSize: 16 }
                    }
                });
                return;
            }

            const dates = data.timeline.map(d => d.date);
            const scores = data.timeline.map(d => d.score);

            const option = {
                backgroundColor: 'transparent',
                tooltip: {
                    trigger: 'axis',
                    backgroundColor: 'rgba(17, 24, 39, 0.95)',
                    borderColor: 'rgba(99, 102, 241, 0.3)',
                    textStyle: { color: '#fff' },
                    formatter: function (params) {
                        const item = data.timeline[params[0].dataIndex];
                        let html = `<div style="padding: 8px;">
                            <div style="font-weight: bold; margin-bottom: 8px;">${item.date}</div>
                            <div style="margin-bottom: 4px;">情感分数: <span style="color: ${params[0].value >= 0 ? '#10b981' : '#ef4444'}; font-weight: bold;">${params[0].value.toFixed(3)}</span></div>
                            <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.1);">`;

                        for (const [emotion, count] of Object.entries(item.emotions)) {
                            const icon = getEmotionIcon(emotion);
                            html += `<div style="display: flex; justify-content: space-between; margin: 4px 0;">
                                <span>${icon} ${emotion}</span>
                                <span style="color: #9ca3af;">×${count}</span>
                            </div>`;
                        }
                        html += `</div></div>`;
                        return html;
                    }
                },
                grid: {
                    left: '3%',
                    right: '4%',
                    bottom: '3%',
                    top: '10%',
                    containLabel: true
                },
                xAxis: {
                    type: 'category',
                    data: dates,
                    axisLine: { lineStyle: { color: '#374151' } },
                    axisLabel: {
                        color: '#9ca3af',
                        rotate: 45,
                        fontSize: 11
                    }
                },
                yAxis: {
                    type: 'value',
                    min: -1,
                    max: 1,
                    axisLine: { lineStyle: { color: '#374151' } },
                    axisLabel: { color: '#9ca3af' },
                    splitLine: {
                        lineStyle: {
                            color: '#374151',
                            type: 'dashed'
                        }
                    }
                },
                series: [{
                    name: '情感分数',
                    type: 'line',
                    data: scores,
                    smooth: true,
                    symbol: 'circle',
                    symbolSize: 8,
                    lineStyle: {
                        width: 3,
                        color: {
                            type: 'linear',
                            x: 0, y: 0, x2: 1, y2: 0,
                            colorStops: [
                                { offset: 0, color: '#ec4899' },
                                { offset: 0.5, color: '#8b5cf6' },
                                { offset: 1, color: '#6366f1' }
                            ]
                        }
                    },
                    itemStyle: {
                        color: function (params) {
                            return params.value >= 0 ? '#10b981' : '#ef4444';
                        },
                        borderWidth: 2,
                        borderColor: '#111827'
                    },
                    areaStyle: {
                        color: {
                            type: 'linear',
                            x: 0, y: 0, x2: 0, y2: 1,
                            colorStops: [
                                { offset: 0, color: 'rgba(139, 92, 246, 0.3)' },
                                { offset: 1, color: 'rgba(139, 92, 246, 0.05)' }
                            ]
                        }
                    },
                    markLine: {
                        silent: true,
                        symbol: 'none',
                        lineStyle: {
                            color: '#6b7280',
                            type: 'dashed',
                            width: 2
                        },
                        label: {
                            color: '#9ca3af',
                            fontSize: 11
                        },
                        data: [
                            { yAxis: 0, label: { formatter: '中性线' } }
                        ]
                    }
                }]
            };

            myChart.setOption(option);
        })
        .catch(err => {
            myChart.hideLoading();
            console.error('加载情感曲线图失败:', err);
            myChart.setOption({
                title: {
                    text: '加载失败',
                    subtext: err.message,
                    left: 'center',
                    top: 'center',
                    textStyle: { color: '#ef4444', fontSize: 16 },
                    subtextStyle: { color: '#9ca3af', fontSize: 12 }
                }
            });
        });

    // 响应式调整
    window.addEventListener('resize', () => myChart.resize());
}

// 初始化对话热力图
function initHeatmapChart() {
    const chartDom = document.getElementById('heatmap-chart');
    if (!chartDom) return;

    const myChart = echarts.init(chartDom);

    myChart.showLoading({
        text: '加载中...',
        color: '#f97316',
        textColor: '#fff',
        maskColor: 'rgba(0, 0, 0, 0.3)'
    });

    fetch('/api/heatmap')
        .then(res => res.json())
        .then(data => {
            myChart.hideLoading();

            if (!data.speakers || data.speakers.length === 0) {
                myChart.setOption({
                    title: {
                        text: '暂无数据',
                        left: 'center',
                        top: 'center',
                        textStyle: { color: '#9ca3af', fontSize: 16 }
                    }
                });
                return;
            }

            const option = {
                backgroundColor: 'transparent',
                tooltip: {
                    position: 'top',
                    backgroundColor: 'rgba(17, 24, 39, 0.95)',
                    borderColor: 'rgba(249, 115, 22, 0.3)',
                    textStyle: { color: '#fff' },
                    formatter: function (params) {
                        return `<div style="padding: 8px;">
                            <div style="font-weight: bold; margin-bottom: 4px;">${data.speakers[params.value[1]]}</div>
                            <div style="color: #9ca3af;">时间: ${data.hours[params.value[0]]}</div>
                            <div style="margin-top: 4px; color: #f97316; font-weight: bold;">对话次数: ${params.value[2]}</div>
                        </div>`;
                    }
                },
                grid: {
                    left: '10%',
                    right: '5%',
                    bottom: '3%',
                    top: '5%',
                    containLabel: true
                },
                xAxis: {
                    type: 'category',
                    data: data.hours,
                    axisLine: { lineStyle: { color: '#374151' } },
                    axisLabel: {
                        color: '#9ca3af',
                        fontSize: 10,
                        interval: 2
                    },
                    splitArea: {
                        show: true,
                        areaStyle: {
                            color: ['rgba(31, 41, 55, 0.1)', 'rgba(31, 41, 55, 0.2)']
                        }
                    }
                },
                yAxis: {
                    type: 'category',
                    data: data.speakers,
                    axisLine: { lineStyle: { color: '#374151' } },
                    axisLabel: {
                        color: '#9ca3af',
                        fontSize: 11
                    },
                    splitArea: {
                        show: true,
                        areaStyle: {
                            color: ['rgba(31, 41, 55, 0.1)', 'rgba(31, 41, 55, 0.2)']
                        }
                    }
                },
                visualMap: {
                    min: 0,
                    max: data.max_value,
                    calculable: true,
                    orient: 'horizontal',
                    left: 'center',
                    bottom: '0%',
                    textStyle: { color: '#9ca3af' },
                    inRange: {
                        color: ['#1f2937', '#f97316', '#dc2626']
                    }
                },
                series: [{
                    name: '对话次数',
                    type: 'heatmap',
                    data: data.data,
                    label: {
                        show: true,
                        color: '#fff',
                        fontSize: 10
                    },
                    emphasis: {
                        itemStyle: {
                            shadowBlur: 10,
                            shadowColor: 'rgba(249, 115, 22, 0.5)'
                        }
                    }
                }]
            };

            myChart.setOption(option);
        })
        .catch(err => {
            myChart.hideLoading();
            console.error('加载对话热力图失败:', err);
            myChart.setOption({
                title: {
                    text: '加载失败',
                    subtext: err.message,
                    left: 'center',
                    top: 'center',
                    textStyle: { color: '#ef4444', fontSize: 16 },
                    subtextStyle: { color: '#9ca3af', fontSize: 12 }
                }
            });
        });

    window.addEventListener('resize', () => myChart.resize());
}

// 情感图标辅助函数
function getEmotionIcon(emotion) {
    const icons = {
        'happy': '😊',
        'neutral': '😐',
        'sad': '😢',
        'angry': '😠'
    };
    return icons[emotion] || '❓';
}

// 当切换到统计分析页面时初始化图表
const originalSwitchTab = window.switchTab;
window.switchTab = function (tabName) {
    if (originalSwitchTab) {
        originalSwitchTab(tabName);
    }

    if (tabName === 'analysis') {
        // 延迟初始化，确保 DOM 已渲染
        setTimeout(() => {
            initEmotionChart();
            initHeatmapChart();
        }, 100);
    }
};
