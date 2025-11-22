#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查数据库中的emotion字段"""
import json
from db_manager import init_pool, get_transcripts

# 初始化数据库连接
if init_pool():
    # 获取最新的一条记录
    items = get_transcripts(offset=0, limit=1)
    
    if items:
        item = items[0]
        print(f"文件名: {item['filename']}")
        print(f"分段数: {len(item.get('segments', []))}")
        print("\n前3个分段的情感信息:")
        for i, seg in enumerate(item.get('segments', [])[:3]):
            print(f"  分段 {i+1}:")
            print(f"    文本: {seg.get('text', '')[:50]}...")
            print(f"    说话人: {seg.get('spk', 'N/A')}")
            print(f"    情感: {seg.get('emotion', 'N/A')}")
            print()
    else:
        print("数据库中没有记录")
else:
    print("无法连接数据库")
