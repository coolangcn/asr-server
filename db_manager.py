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

def init_pool(db_url: str = None):
    """初始化数据库连接池"""
    global connection_pool
    target_url = db_url or DATABASE_URL
    try:
        connection_pool = psycopg2.pool.SimpleConnectionPool(
            1, 10,  # 最小和最大连接数
            target_url
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
        
        # 创建宝宝哭声分析表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS baby_cry_events (
            id SERIAL PRIMARY KEY,
            filename TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            recording_time TIMESTAMP,
            start_time REAL,
            end_time REAL,
            reason TEXT,
            advice TEXT,
            reason_category TEXT,
            event_files_json TEXT,
            illustration_url TEXT
        );
        ''')
        
        # 兼容性升级逻辑
        try:
            cursor.execute("ALTER TABLE baby_cry_events ADD COLUMN IF NOT EXISTS reason_category TEXT;")
            cursor.execute("ALTER TABLE baby_cry_events ADD COLUMN IF NOT EXISTS event_files_json TEXT;")
            cursor.execute("ALTER TABLE baby_cry_events ADD COLUMN IF NOT EXISTS audio_path TEXT;")
            cursor.execute("ALTER TABLE baby_cry_events ADD COLUMN IF NOT EXISTS confidence REAL;")
            cursor.execute("ALTER TABLE baby_cry_events ADD COLUMN IF NOT EXISTS details_json TEXT;")
            cursor.execute("ALTER TABLE baby_cry_events ADD COLUMN IF NOT EXISTS illustration_url TEXT;")
        except Exception as e:
            print(f"[DB] 字段升级提示: {e}")

        # 创建已处理文件记录表 (A轨历史扫描专用)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS processed_files_a (
            id SERIAL PRIMARY KEY,
            filename TEXT UNIQUE NOT NULL,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT
        );
        ''')
        
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_processed_a_filename ON processed_files_a(filename);
        ''')

        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_cry_filename ON baby_cry_events(filename);
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

def get_baby_cry_events(offset: int = 0, limit: int = 100, 
                        date_filter: str = None, 
                        start_time_filter: str = None, 
                        end_time_filter: str = None) -> List[Dict]:
    """获取记录的宝宝哭声分析事件，支持分页和时间筛选"""
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return []
            
        cursor = conn.cursor()
        
        # 构建动态 SQL
        query = "SELECT id, filename, created_at, recording_time, start_time, end_time, reason, advice, reason_category, event_files_json, audio_path, confidence, details_json, illustration_url FROM baby_cry_events"
        where_clauses = []
        params = []
        
        # 1. 日期过滤 (YYYY-MM-DD)
        if date_filter:
            where_clauses.append("recording_time::date = %s")
            params.append(date_filter)
            
        # 2. 时间段过滤 (HH-MM) - 注意文件名和数据库中通常是这个格式
        # 这里我们假设过滤的是 recording_time 的时刻
        if start_time_filter:
            # 将 HH-MM 转换为 HH:MM 供 SQL 比较
            sql_start = start_time_filter.replace('-', ':')
            where_clauses.append("recording_time::time >= %s")
            params.append(sql_start)
            
        if end_time_filter:
            sql_end = end_time_filter.replace('-', ':')
            where_clauses.append("recording_time::time <= %s")
            params.append(sql_end)
            
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
            
        query += " ORDER BY COALESCE(recording_time, created_at) DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        
        cursor.execute(query, tuple(params))
        
        rows = cursor.fetchall()
        cursor.close()
        
        results = []
        for row in rows:
            results.append({
                'id': row[0],
                'filename': row[1],
                'created_at': row[2].isoformat() if row[2] else None,
                'recording_time': row[3].isoformat() if row[3] else None,
                'start_time': row[4],
                'end_time': row[5],
                'reason': row[6],
                'advice': row[7],
                'reason_category': row[8],
                'event_files_json': json.loads(row[9]) if row[9] else [],
                'audio_path': row[10] if len(row) > 10 else None,
                'confidence': row[11] if len(row) > 11 else 0.0,
                'details': json.loads(row[12]) if len(row) > 12 and row[12] else [],
                'illustration_url': row[13] if len(row) > 13 else None
            })
        
        return results
    except Exception as e:
        print(f"[DB Error] 查询哭声记录失败: {e}")
        return []
    finally:
        if conn:
            return_connection(conn)

def get_baby_cry_event_by_id(event_id: int) -> Dict:
    """根据 ID 获取单个宝宝哭声分析事件"""
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return None

        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, filename, created_at, recording_time, start_time, end_time, reason, advice, reason_category, event_files_json, audio_path, confidence, details_json, illustration_url FROM baby_cry_events WHERE id = %s",
            (event_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None

        return {
            'id': row[0],
            'filename': row[1],
            'created_at': row[2].isoformat() if row[2] else None,
            'recording_time': row[3].isoformat() if row[3] else None,
            'start_time': float(row[4]) if row[4] else 0,
            'end_time': float(row[5]) if row[5] else 0,
            'reason': row[6],
            'advice': row[7],
            'reason_category': row[8],
            'event_files_json': json.loads(row[9]) if row[9] else [],
            'audio_path': row[10],
            'confidence': float(row[11]) if row[11] else 0,
            'details_json': row[12],
            'illustration_url': row[13]
        }
    except Exception as e:
        print(f"[DB Error] 查询哭声记录失败: {e}")
        return None
    finally:
        if conn:
            return_connection(conn)

def save_cry_analysis(filename: str, start_time: float, end_time: float, reason: str, advice: str, 
                      reason_category: str = None, event_files: list = None, audio_path = None, confidence: float = 0.0, details: list = None, illustration_url: str = None) -> bool:
    """保存宝宝哭声分析结果"""
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return False
            
        cursor = conn.cursor()
        recording_time = parse_recording_time(filename)
        created_at = datetime.now(UTC_PLUS_8)
        event_files_json = json.dumps(event_files, ensure_ascii=False) if event_files else None
        details_json = json.dumps(details, ensure_ascii=False) if details else None
        
        cursor.execute(
            "INSERT INTO baby_cry_events (filename, created_at, recording_time, start_time, end_time, reason, advice, reason_category, event_files_json, audio_path, confidence, details_json, illustration_url) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (filename, created_at, recording_time, start_time, end_time, reason, advice, reason_category, event_files_json, audio_path, confidence, details_json, illustration_url)
        )
        
        conn.commit()
        # 获取刚插入的 ID
        cursor.execute("SELECT lastval()")
        new_id = cursor.fetchone()[0]
        cursor.close()
        print(f"  [DB] 已保存宝宝哭声分析占位 [{filename}] {start_time:.1f}s-{end_time:.1f}s: ID={new_id}")
        return new_id
    except Exception as e:
        print(f"  [DB Error] 保存哭声分析失败: {e}")
        if conn:
            conn.rollback()
        return None
    finally:
        if conn:
            return_connection(conn)

def update_cry_analysis(event_id: int, reason: str, advice: str, 
                       reason_category: str = None, event_files: list = None, confidence: float = None, details: list = None) -> bool:
    """更新已有的宝宝哭声分析结果 (主要用于异步回调)"""
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return False
            
        cursor = conn.cursor()
        event_files_json = json.dumps(event_files, ensure_ascii=False) if event_files else None
        details_json = json.dumps(details, ensure_ascii=False) if details else None
        
        # 构建更新动态 SQL 以支持可选参数
        updates = ["reason = %s", "advice = %s", "reason_category = %s", "event_files_json = %s"]
        params = [reason, advice, reason_category, event_files_json]
        
        if confidence is not None:
            updates.append("confidence = %s")
            params.append(confidence)
        if details_json is not None:
            updates.append("details_json = %s")
            params.append(details_json)
            
        params.append(event_id)
        query = f"UPDATE baby_cry_events SET {', '.join(updates)} WHERE id = %s"
        
        cursor.execute(query, tuple(params))
        
        conn.commit()
        updated = cursor.rowcount > 0
        cursor.close()
        if updated:
            print(f"  [DB] 已更新宝宝哭声深度分析详情: ID={event_id}")
        return updated
    except Exception as e:
        print(f"  [DB Error] 更新哭声分析失败: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            return_connection(conn)

def update_cry_event_image(filename: str, illustration_url: str) -> bool:
    """更新宝宝哭声事件的插图 URL"""
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return False
            
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE baby_cry_events SET illustration_url = %s WHERE filename = %s",
            (illustration_url, filename)
        )
        
        conn.commit()
        updated = cursor.rowcount > 0
        cursor.close()
        if updated:
            print(f"  [DB] 已更新插图：{filename}")
        return updated
    except Exception as e:
        print(f"  [DB Error] 更新插图失败：{e}")
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

def is_file_processed_a(filename: str) -> bool:
    """检查文件是否已由 A 轨历史扫描处理过"""
    conn = None
    try:
        conn = get_connection()
        if not conn: return False
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM processed_files_a WHERE filename = %s", (filename,))
        exists = cursor.fetchone() is not None
        cursor.close()
        return exists
    except Exception as e:
        print(f"  [DB Error] 检查处理进度失败: {e}")
        return False
    finally:
        if conn: return_connection(conn)

def mark_file_processed_a(filename: str, status: str = "success") -> bool:
    """标记文件为 A 轨已处理"""
    conn = None
    try:
        conn = get_connection()
        if not conn: return False
        cursor = conn.cursor()
        # 冲突时更新处理时间
        cursor.execute(
            "INSERT INTO processed_files_a (filename, processed_at, status) VALUES (%s, %s, %s) ON CONFLICT (filename) DO UPDATE SET processed_at = EXCLUDED.processed_at, status = EXCLUDED.status",
            (filename, datetime.now(UTC_PLUS_8), status)
        )
        conn.commit()
        cursor.close()
        return True
    except Exception as e:
        print(f"  [DB Error] 标记处理进度失败: {e}")
        if conn: conn.rollback()
        return False
    finally:
        if conn: return_connection(conn)

def get_date_processing_stats() -> dict:
    """获取每个日期的处理进度统计 {date_str: processed_count}"""
    conn = None
    try:
        conn = get_connection()
        if not conn: return {}
        cursor = conn.cursor()
        result = {}

        # 格式1: /path/YYYY-MM-DD/file.m4a (带斜杠)
        cursor.execute('''
            SELECT (regexp_matches(filename, '/(\d{4}-\d{2}-\d{2})/'))[1] as date_str, COUNT(*) as cnt
            FROM processed_files_a
            WHERE filename ~ '/\d{4}-\d{2}-\d{2}/'
            GROUP BY date_str
        ''')
        for row in cursor.fetchall():
            if row[0]:
                result[row[0]] = row[1]

        # 格式2: TermuxAudioRecording_YYYY-MM-DD_HH-MM-SS.acc 或 recording-YYYYMMDD-HHMMSS.m4a
        # 日期在文件名中，前面是下划线或短横线
        cursor.execute('''
            SELECT (regexp_matches(filename, '[_-](\d{4}-\d{2}-\d{2})[_-]'))[1] as date_str, COUNT(*) as cnt
            FROM processed_files_a
            WHERE filename ~ '[_-]\d{4}-\d{2}-\d{2}[_-]'
            GROUP BY date_str
        ''')
        for row in cursor.fetchall():
            if row[0]:
                result[row[0]] = result.get(row[0], 0) + row[1]

        # 格式3: recording-YYYYMMDD-HHMMSS.m4a (无横杠格式)
        cursor.execute('''
            SELECT SUBSTRING(filename FROM '\d{4}\d{2}\d{2}') as date_str, COUNT(*) as cnt
            FROM processed_files_a
            WHERE filename ~ '\d{4}\d{2}\d{2}' AND filename NOT LIKE '%-%-%'
            GROUP BY date_str
        ''')
        for row in cursor.fetchall():
            if row[0] and len(row[0]) == 8:
                date_str = f"{row[0][:4]}-{row[0][4:6]}-{row[0][6:]}"
                result[date_str] = result.get(date_str, 0) + row[1]

        cursor.close()
        return result
    except Exception as e:
        print(f"  [DB Error] 获取日期处理进度失败: {e}")
        import traceback
        traceback.print_exc()
        return {}
    finally:
        if conn: return_connection(conn)

def get_date_file_counts() -> dict:
    """获取每个日期的文件数量（从 babycry_date_cache 表）"""
    conn = None
    try:
        conn = get_connection()
        if not conn: return {}
        cursor = conn.cursor()
        cursor.execute('SELECT date_str, file_count FROM babycry_date_cache')
        result = {row[0]: row[1] for row in cursor.fetchall()}
        cursor.close()
        return result
    except Exception as e:
        print(f"  [DB Error] 获取日期文件数量失败: {e}")
        return {}
    finally:
        if conn: return_connection(conn)

def delete_cry_events_by_date(date_str: str) -> int:
    """删除指定日期（YYYY-MM-DD）的哭声分析事件和处理进度"""
    conn = None
    deleted_count = 0
    try:
        conn = get_connection()
        if not conn: return 0
        cursor = conn.cursor()
        
        # 1. 删除哭声事件
        cursor.execute(
            "DELETE FROM baby_cry_events WHERE recording_time::date = %s",
            (date_str,)
        )
        deleted_count = cursor.rowcount
        
        # 2. 删除对应的处理进度 (以便重新处理)
        # 注意：processed_files_a 中的文件名包含日期信息，如 Sony-2/2026-03-28/xxx.m4a
        # 我们使用模糊匹配来删除
        cursor.execute(
            "DELETE FROM processed_files_a WHERE filename LIKE %s",
            (f"%{date_str}%",)
        )
        
        conn.commit()
        cursor.close()
        print(f"  [DB] 已清除日期 {date_str} 的 {deleted_count} 条哭声事件记录及相关处理进度")
        return deleted_count
    except Exception as e:
        print(f"  [DB Error] 清除日期记录失败: {e}")
        if conn: conn.rollback()
        return 0
    finally:
        if conn: return_connection(conn)

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
