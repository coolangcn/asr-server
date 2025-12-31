#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import psycopg2
from psycopg2 import pool
import json
import os
import re
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

# PostgreSQL 连接配置
# 优先使用环境变量，如果未设置则使用默认值
DATABASE_URL = os.getenv(
    'DATABASE_URL',
    'postgresql://cnncn:74123698cN@cncn.postgres.database.azure.com:5432/postgres?sslmode=require'
)

# 连接池
connection_pool = None

# 东八区时区 (UTC+8)
UTC_PLUS_8 = timezone(timedelta(hours=8))

def init_pool():
    """初始化数据库连接池"""
    global connection_pool
    try:
        connection_pool = psycopg2.pool.SimpleConnectionPool(
            1, 10,  # 最小和最大连接数
            DATABASE_URL
        )
        if connection_pool:
            print("[DB] PostgreSQL连接池创建成功")
            return True
    except Exception as e:
        print(f"[DB Error] 创建连接池失败: {e}")
        return False

def get_connection():
    """从连接池获取连接"""
    if connection_pool:
        return connection_pool.getconn()
    return None

def return_connection(conn):
    """归还连接到连接池"""
    if connection_pool and conn:
        connection_pool.putconn(conn)

def parse_recording_time(filename: str) -> Optional[datetime]:
    """
    从文件名中解析录音时间
    支持格式: TermuxAudioRecording_2025-11-23_12-56-54.m4a
    
    Args:
        filename: 文件名
        
    Returns:
        datetime对象，如果无法解析则返回None
    """
    # 匹配格式: YYYY-MM-DD_HH-MM-SS
    pattern = r'(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})'
    match = re.search(pattern, filename)
    
    if match:
        year, month, day, hour, minute, second = map(int, match.groups())
        try:
            return datetime(year, month, day, hour, minute, second)
        except ValueError:
            # 日期值无效（如月份13）
            return None
    
    # 如果无法解析，返回None（调用者应使用当前时间）
    return None

def init_db():
    """初始化数据库表结构"""
    conn = None
    try:
        conn = get_connection()
        if not conn:
            print("[DB Error] 无法获取数据库连接")
            return False
            
        cursor = conn.cursor()
        
        # 创建转录记录表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS transcriptions (
            id SERIAL PRIMARY KEY,
            filename TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            recording_time TIMESTAMP,
            full_text TEXT,
            segments_json TEXT,
            topics_json TEXT,
            summary_json TEXT
        );
        ''')
        
        # 创建索引以提高查询性能
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_created_at ON transcriptions(created_at DESC);
        ''')
        
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_filename ON transcriptions(filename);
        ''')
        
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_recording_time ON transcriptions(recording_time DESC NULLS LAST);
        ''')
        
        conn.commit()
        cursor.close()
        print("[DB] 数据库表结构初始化成功")
        return True
    except Exception as e:
        print(f"[DB Error] 数据库初始化失败: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            return_connection(conn)

def save_to_db(filename: str, full_text: str, segments_list: List[Dict], 
               recording_time: Optional[datetime] = None, summary: Optional[Dict] = None) -> bool:
    """
    保存转录记录到数据库（如果文件已存在则覆盖）
    
    Args:
        filename: 文件名
        full_text: 完整文本
        segments_list: 分段列表
        recording_time: 录音时间（可选，如果为None则尝试从文件名解析）
        summary: 智能摘要（可选）
    """
    conn = None
    try:
        conn = get_connection()
        if not conn:
            print("[DB Error] 无法获取数据库连接")
            return False
            
        cursor = conn.cursor()
        segments_json = json.dumps(segments_list, ensure_ascii=False)
        summary_json = json.dumps(summary, ensure_ascii=False) if summary else None
        
        # 如果没有提供recording_time,尝试从文件名解析
        if recording_time is None:
            recording_time = parse_recording_time(filename)
        
        # 先删除已存在的记录（如果有）
        cursor.execute(
            "DELETE FROM transcriptions WHERE filename = %s",
            (filename,)
        )
        deleted_count = cursor.rowcount
        
        # 获取东八区当前时间
        created_at = datetime.now(UTC_PLUS_8)
        
        # 插入新记录
        cursor.execute(
            "INSERT INTO transcriptions (filename, created_at, full_text, segments_json, recording_time, summary_json) VALUES (%s, %s, %s, %s, %s, %s)",
            (filename, created_at, full_text, segments_json, recording_time, summary_json)
        )
        
        conn.commit()
        cursor.close()
        time_str = recording_time.strftime('%Y-%m-%d %H:%M:%S') if recording_time else '当前时间'
        
        if deleted_count > 0:
            print(f"  [DB] 覆盖 {filename} (录音时间: {time_str}, 删除了 {deleted_count} 条旧记录)")
        else:
            print(f"  [DB] 新增 {filename} (录音时间: {time_str})")
        
        return True
    except Exception as e:
        print(f"  [DB Error] {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            return_connection(conn)

def update_topics(filename: str, topics_dict: dict) -> bool:
    """更新转录记录的 LLM 主题信息"""
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return False
        
        cursor = conn.cursor()
        topics_json = json.dumps(topics_dict, ensure_ascii=False)
        
        cursor.execute(
            "UPDATE transcriptions SET topics_json = %s WHERE filename = %s",
            (topics_json, filename)
        )
        
        conn.commit()
        updated = cursor.rowcount > 0
        cursor.close()
        
        if updated:
            print(f"  [DB] 更新 {filename} 的主题信息")
        return updated
            
    except Exception as e:
        print(f"  [DB Error] 更新主题失败: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            return_connection(conn)

def get_transcripts(offset: int = 0, limit: int = 100) -> List[Dict]:
    """获取最近的转录记录，支持分页"""
    conn = None
    try:
        conn = get_connection()
        if not conn:
            print("[DB Error] 无法获取数据库连接")
            return []
            
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, filename, created_at, full_text, segments_json, recording_time FROM transcriptions ORDER BY COALESCE(recording_time, created_at) DESC LIMIT %s OFFSET %s",
            (limit, offset)
        )
        
        rows = cursor.fetchall()
        cursor.close()
        
        results = []
        for row in rows:
            data = {
                'id': row[0],
                'filename': row[1],
                'created_at': row[2].isoformat() if row[2] else None,
                'full_text': row[3],
                'segments_json': row[4],
                'recording_time': row[5].isoformat() if row[5] else None
            }
            # 解析segments_json
            try:
                data['segments'] = json.loads(data['segments_json']) if data['segments_json'] else []
            except:
                data['segments'] = []
            
            results.append(data)
        
        return results
    except Exception as e:
        print(f"[DB Error] 查询失败: {e}")
        return []
    finally:
        if conn:
            return_connection(conn)

def test_connection() -> bool:
    """测试数据库连接"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"[DB] PostgreSQL连接成功: {version[0]}")
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"[DB Error] 连接测试失败: {e}")
        return False

def close_pool():
    """关闭连接池"""
    global connection_pool
    if connection_pool:
        connection_pool.closeall()
        print("[DB] 连接池已关闭")

if __name__ == "__main__":
    # 测试代码
    print("测试数据库连接...")
    if test_connection():
        print("初始化连接池...")
        if init_pool():
            print("初始化数据库表结构...")
            init_db()
            close_pool()
