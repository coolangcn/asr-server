#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import psycopg2
from psycopg2 import pool
import json
import re
from datetime import datetime
from typing import List, Dict, Optional

# PostgreSQL 连接配置
DATABASE_URL = "postgresql://cnncn:74123698cN@cncn.postgres.database.azure.com:5432/postgres?sslmode=require"

# 连接池
connection_pool = None

def init_pool(db_url: str = None):
    """初始化数据库连接池"""
    global connection_pool
    target_url = db_url or DATABASE_URL
    try:
        connection_pool = psycopg2.pool.SimpleConnectionPool(
            2, 20,  # 最小和最大连接数
            target_url,
            connect_timeout=10
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
    支持格式: 
        - TermuxAudioRecording_2025-11-23_12-56-54.m4a
        - 2026-03-30_09-29-10.m4a
        - recording-20251123-125654.m4a
    
    Args:
        filename: 文件名
        
    Returns:
        datetime对象，如果无法解析则返回None
    """
    # 尝试多种格式
    patterns = [
        # 格式1: YYYY-MM-DD_HH-MM-SS (如 2026-03-30_09-29-10)
        (r'(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})', '%Y-%m-%d_%H-%M-%S'),
        # 格式2: YYYYMMDD-HHMMSS (如 recording-20251123-125654)
        (r'(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})', '%Y%m%d-%H%M%S'),
    ]
    
    for pattern, _ in patterns:
        match = re.search(pattern, filename)
        if match:
            year, month, day, hour, minute, second = map(int, match.groups())
            try:
                return datetime(year, month, day, hour, minute, second)
            except ValueError:
                continue
    
    # 如果无法解析，返回None
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
            segments_json TEXT
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
               recording_time: Optional[datetime] = None) -> bool:
    """
    保存转录记录到数据库
    
    Args:
        filename: 文件名
        full_text: 完整文本
        segments_list: 分段列表
        recording_time: 录音时间（可选，如果为None则使用当前时间）
    """
    conn = None
    try:
        conn = get_connection()
        if not conn:
            print("[DB Error] 无法获取数据库连接")
            return False
            
        cursor = conn.cursor()
        segments_json = json.dumps(segments_list, ensure_ascii=False)
        
        # 如果没有提供recording_time,尝试从文件名解析
        if recording_time is None:
            recording_time = parse_recording_time(filename)
        
        cursor.execute(
            "INSERT INTO transcriptions (filename, full_text, segments_json, recording_time) VALUES (%s, %s, %s, %s)",
            (filename, full_text, segments_json, recording_time)
        )
        
        conn.commit()
        cursor.close()
        time_str = recording_time.strftime('%Y-%m-%d %H:%M:%S') if recording_time else '当前时间'
        print(f"  [DB] Saved {filename} (录音时间: {time_str})")
        return True
    except Exception as e:
        print(f"  [DB Error] {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            return_connection(conn)

def get_transcripts(offset: int = 0, limit: int = 100, db_url: str = None) -> List[Dict]:
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

def fix_recording_time() -> int:
    """
    修复现有数据中 recording_time 为 NULL 的记录
    从文件名中解析时间并更新数据库
    
    Returns:
        修复的记录数量
    """
    conn = None
    try:
        conn = get_connection()
        if not conn:
            print("[DB Error] 无法获取数据库连接")
            return 0
            
        cursor = conn.cursor()
        
        # 获取所有 recording_time 为 NULL 的记录
        cursor.execute(
            "SELECT id, filename FROM transcriptions WHERE recording_time IS NULL"
        )
        rows = cursor.fetchall()
        
        fixed_count = 0
        for row in rows:
            id, filename = row
            recording_time = parse_recording_time(filename)
            
            if recording_time:
                cursor.execute(
                    "UPDATE transcriptions SET recording_time = %s WHERE id = %s",
                    (recording_time, id)
                )
                fixed_count += 1
                print(f"[DB Fix] 修复记录 {id}: {filename} -> {recording_time}")
        
        conn.commit()
        cursor.close()
        
        print(f"[DB Fix] 共修复 {fixed_count} 条记录")
        return fixed_count
    except Exception as e:
        print(f"[DB Error] 修复 recording_time 失败: {e}")
        if conn:
            conn.rollback()
        return 0
    finally:
        if conn:
            return_connection(conn)

def save_analysis_progress(session_id: str, all_dates: list, loaded_count: int,
                           has_more: bool, current_date: str, dates_state: dict) -> bool:
    """
    保存宝宝哭声分析进度到数据库

    Args:
        session_id: 会话ID
        all_dates: 所有日期列表
        loaded_count: 已加载数量
        has_more: 是否还有更多
        current_date: 当前处理日期
        dates_state: 各日期状态字典
    """
    conn = None
    try:
        conn = get_connection()
        if not conn:
            print("[DB Error] 无法获取数据库连接")
            return False
            
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO babycry_analysis_progress 
            (session_id, all_dates, loaded_count, has_more, processing_date, dates_state, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (session_id)
            DO UPDATE SET
                all_dates = EXCLUDED.all_dates,
                loaded_count = EXCLUDED.loaded_count,
                has_more = EXCLUDED.has_more,
                processing_date = EXCLUDED.processing_date,
                dates_state = EXCLUDED.dates_state,
                updated_at = CURRENT_TIMESTAMP
        ''', (session_id, json.dumps(all_dates), loaded_count, has_more,
              current_date, json.dumps(dates_state)))
        
        conn.commit()
        cursor.close()
        return True
    except Exception as e:
        print(f"[DB Error] 保存分析进度失败: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            return_connection(conn)

def load_analysis_progress(session_id: str) -> dict:
    """
    从数据库加载宝宝哭声分析进度
    
    Args:
        session_id: 会话ID
        
    Returns:
        进度字典，如果没有找到则返回None
    """
    conn = None
    try:
        conn = get_connection()
        if not conn:
            print("[DB Error] 无法获取数据库连接")
            return None
            
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT all_dates, loaded_count, has_more, processing_date, dates_state
            FROM babycry_analysis_progress
            WHERE session_id = %s
        ''', (session_id,))

        row = cursor.fetchone()
        cursor.close()

        if row:
            return {
                'all_dates': row[0],
                'loaded_count': row[1],
                'has_more': row[2],
                'processing_date': row[3],
                'dates_state': row[4]
            }
        return None
    except Exception as e:
        print(f"[DB Error] 加载分析进度失败: {e}")
        return None
    finally:
        if conn:
            return_connection(conn)

def clear_analysis_progress(session_id: str) -> bool:
    """
    清除数据库中的分析进度
    
    Args:
        session_id: 会话ID
    """
    conn = None
    try:
        conn = get_connection()
        if not conn:
            print("[DB Error] 无法获取数据库连接")
            return False
            
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM babycry_analysis_progress 
            WHERE session_id = %s
        ''', (session_id,))
        
        conn.commit()
        cursor.close()
        return True
    except Exception as e:
        print(f"[DB Error] 清除分析进度失败: {e}")
        if conn:
            conn.rollback()
        return False
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

def save_date_stats_to_redis(date_info: dict) -> bool:
    """保存日期统计到 Valkey"""
    import os
    VALKEY_URI = os.environ.get('VALKEY_URI', '')
    if not VALKEY_URI:
        print(f"  [Valkey Error] VALKEY_URI 环境变量未设置")
        return False
    try:
        import valkey
        r = valkey.from_url(VALKEY_URI)
        pipe = r.pipeline()
        for date_str, info in date_info.items():
            pipe.hset('babycry:date_stats', date_str, info.get('fileCount', 0))
        pipe.execute()
        return True
    except Exception as e:
        print(f"  [Valkey Error] 保存日期统计失败: {e}")
        return False

def get_date_stats_from_redis() -> dict:
    """从 Valkey 获取所有日期统计"""
    import os
    VALKEY_URI = os.environ.get('VALKEY_URI', '')
    if not VALKEY_URI:
        print(f"  [Valkey Error] VALKEY_URI 环境变量未设置")
        return {}
    try:
        import valkey
        r = valkey.from_url(VALKEY_URI)
        stats = r.hgetall('babycry:date_stats')
        result = {}
        for date_str, file_count in stats.items():
            date_str = date_str.decode('utf-8') if isinstance(date_str, bytes) else date_str
            count = int(file_count) if file_count else 0
            result[date_str] = {
                'fileCount': count,
                'processedCount': 0,
                'status': 'pending'
            }
        return result
    except Exception as e:
        print(f"  [Valkey Error] 获取日期统计失败: {e}")
        return {}

def clear_date_stats_in_redis() -> bool:
    """清空 Valkey 中的日期统计"""
    import os
    VALKEY_URI = os.environ.get('VALKEY_URI', '')
    if not VALKEY_URI:
        print(f"  [Valkey Error] VALKEY_URI 环境变量未设置")
        return False
    try:
        import valkey
        r = valkey.from_url(VALKEY_URI)
        r.delete('babycry:date_stats')
        return True
    except Exception as e:
        print(f"  [Valkey Error] 清空日期统计失败: {e}")
        return False

if __name__ == "__main__":
    # 测试代码
    print("测试数据库连接...")
    if test_connection():
        print("初始化连接池...")
        if init_pool():
            print("初始化数据库表结构...")
            init_db()
            close_pool()
