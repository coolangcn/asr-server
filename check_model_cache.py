#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查当前模型缓存位置和大小
"""

import os
import shutil
from pathlib import Path

def get_dir_size(path):
    """计算目录大小"""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    except:
        pass
    return total

def format_size(bytes):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} PB"

print("=" * 60)
print("检查模型缓存位置")
print("=" * 60)
print()

# 检查常见的缓存目录
cache_dirs = {
    "HuggingFace": [
        Path.home() / ".cache" / "huggingface",
        Path("D:/AI/model_cache/huggingface")
    ],
    "ModelScope": [
        Path.home() / ".cache" / "modelscope",
        Path("D:/AI/model_cache/modelscope")
    ],
    "Whisper": [
        Path.home() / ".cache" / "whisper",
        Path("D:/AI/model_cache/whisper")
    ],
    "Torch": [
        Path.home() / ".cache" / "torch",
        Path("D:/AI/model_cache/torch")
    ]
}

total_c_size = 0
total_d_size = 0

for name, paths in cache_dirs.items():
    print(f"📦 {name} 模型:")
    for path in paths:
        if path.exists():
            size = get_dir_size(str(path))
            formatted_size = format_size(size)
            drive = str(path)[0].upper()
            if drive == 'C':
                total_c_size += size
            elif drive == 'D':
                total_d_size += size
            print(f"   ✅ {path}")
            print(f"      大小: {formatted_size}")
        else:
            print(f"   ❌ {path} (不存在)")
    print()

print("=" * 60)
print("汇总:")
print("=" * 60)
print(f"C盘缓存总大小: {format_size(total_c_size)}")
print(f"D盘缓存总大小: {format_size(total_d_size)}")
print()

if total_c_size > 0:
    print("💡 建议:")
    print("   1. 运行 setup_model_cache.bat 设置环境变量")
    print("   2. 手动移动C盘模型到D盘对应目录")
    print("   3. 删除C盘旧文件释放空间")
    print()
    print("移动命令示例:")
    print(f"   xcopy /E /I /Y \"%USERPROFILE%\\.cache\\huggingface\" \"D:\\AI\\model_cache\\huggingface\"")
    print(f"   xcopy /E /I /Y \"%USERPROFILE%\\.cache\\modelscope\" \"D:\\AI\\model_cache\\modelscope\"")
    print(f"   xcopy /E /I /Y \"%USERPROFILE%\\.cache\\whisper\" \"D:\\AI\\model_cache\\whisper\"")
    print(f"   xcopy /E /I /Y \"%USERPROFILE%\\.cache\\torch\" \"D:\\AI\\model_cache\\torch\"")
