#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import psycopg2
from psycopg2 import pool
import json
import os
import re
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# PostgreSQL 连接配置
# 优先使用环境变量或 .env；未设置时数据库功能会在 init_pool 中降级失败
DATABASE_URL = os.getenv(
    'DATABASE_URL',
    ''
)

# 连接池
connection_pool = None

# 东八区时区 (UTC+8)
UTC_PLUS_8 = timezone(timedelta(hours=8))

def init_pool(db_url: str = None):
    """初始化数据库连接池，带重试机制"""
    global connection_pool
    target_url = db_url or DATABASE_URL
    if not target_url:
        print("[DB Error] DATABASE_URL 未设置")
        return False
    
    for attempt in range(3):
        try:
            connection_pool = psycopg2.pool.SimpleConnectionPool(
                3, 30,  # 最小3个，最大30个连接（B轨Catch-up需要更多并发连接）
                target_url
            )
            if connection_pool:
                print(f"[DB] PostgreSQL连接池创建成功 (尝试 {attempt + 1}/3, 最大30连接)")
                return True
        except Exception as e:
            print(f"[DB Error] 创建连接池失败 (尝试 {attempt + 1}/3): {e}")
            if attempt < 2:
                import time
                wait_time = 2 ** attempt  # 指数退避: 1s, 2s
                print(f"[DB] {wait_time}秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"[DB Error] 所有重试都失败了")
                return False
    return False

def get_connection(max_retries=3, retry_delay=0.5):
    """从连接池获取连接，带重试机制"""
    if connection_pool:
        for attempt in range(max_retries):
            try:
                return connection_pool.getconn()
            except psycopg2.pool.PoolError as e:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                print(f"[DB Error] 连接池耗尽，无法获取连接 (重试{max_retries}次后)")
                return None
            except Exception as e:
                print(f"[DB Error] 获取连接异常: {e}")
                return None
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
            cursor.execute("ALTER TABLE baby_cry_events ADD COLUMN IF NOT EXISTS is_deleted BOOLEAN DEFAULT FALSE;")
            cursor.execute("UPDATE baby_cry_events SET is_deleted = FALSE WHERE is_deleted IS NULL;")
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

        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_cry_recording_not_deleted
        ON baby_cry_events(recording_time DESC NULLS LAST)
        WHERE is_deleted = FALSE;
        ''')

        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_cry_created_not_deleted
        ON baby_cry_events(created_at DESC)
        WHERE is_deleted = FALSE;
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
                        end_time_filter: str = None) -> tuple:
    """获取记录的宝宝哭声分析事件，支持分页和时间筛选
    返回: (events_list, total_count)
    """
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return ([], 0)
            
        cursor = conn.cursor()
        
        # 构建 WHERE 子句
        where_clauses = ["is_deleted = FALSE"]
        params = []
        
        # 1. 日期过滤 (YYYY-MM-DD)
        if date_filter:
            where_clauses.append("recording_time::date = %s")
            params.append(date_filter)
            
        # 2. 时间段过滤 (HH-MM)
        if start_time_filter:
            sql_start = start_time_filter.replace('-', ':')
            where_clauses.append("recording_time::time >= %s")
            params.append(sql_start)
            
        if end_time_filter:
            sql_end = end_time_filter.replace('-', ':')
            where_clauses.append("recording_time::time <= %s")
            params.append(sql_end)
        
        where_sql = " WHERE " + " AND ".join(where_clauses)
            
        # 查询总数
        count_query = f"SELECT COUNT(*) FROM baby_cry_events{where_sql}"
        cursor.execute(count_query, tuple(params))
        total_count = cursor.fetchone()[0]
        
        # 查询分页数据
        query = "SELECT id, filename, created_at, recording_time, start_time, end_time, LEFT(reason, 80) as reason_preview, reason_category, LEFT(advice, 120) as suggestion_preview, jsonb_array_length(event_files_json::jsonb) as file_count, CASE WHEN illustration_url IS NOT NULL THEN true ELSE FALSE END as has_illustration FROM baby_cry_events"
        query += where_sql + " ORDER BY COALESCE(recording_time, created_at) DESC LIMIT %s OFFSET %s"
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
                'start_time': float(row[4]) if row[4] else 0.0,
                'end_time': float(row[5]) if row[5] else 0.0,
                'reason_preview': row[6] or '',
                'reason_category': row[7],
                'suggestion_preview': row[8] or '',
                'file_count': int(row[9]) if row[9] else 0,
                'has_illustration': bool(row[10]) if len(row) > 10 else False,
            })
        
        return (results, total_count)
    except Exception as e:
        print(f"[DB Error] 查询哭声记录失败: {e}")
        return ([], 0)
    finally:
        if conn:
            return_connection(conn)

def get_baby_cry_count(date_filter: str = None) -> int:
    """获取指定日期的宝宝哭声事件数量，默认按东八区今天统计"""
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return 0

        cursor = conn.cursor()
        target_date = date_filter or datetime.now(UTC_PLUS_8).date().isoformat()
        cursor.execute(
            "SELECT COUNT(*) FROM baby_cry_events WHERE COALESCE(recording_time, created_at)::date = %s AND is_deleted = FALSE",
            (target_date,)
        )
        row = cursor.fetchone()
        cursor.close()
        return int(row[0]) if row and row[0] is not None else 0
    except Exception as e:
        print(f"[DB Error] 查询哭声数量失败: {e}")
        return 0
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

def update_cry_event_audio_path(event_id: int, audio_path: str) -> bool:
    """更新宝宝哭声事件的持久音频路径"""
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return False

        cursor = conn.cursor()
        cursor.execute(
            "UPDATE baby_cry_events SET audio_path = %s WHERE id = %s",
            (audio_path, event_id)
        )
        conn.commit()
        updated = cursor.rowcount > 0
        cursor.close()
        if updated:
            print(f"  [DB] 已更新哭声事件音频路径: ID={event_id}")
        return updated
    except Exception as e:
        print(f"  [DB Error] 更新哭声事件音频路径失败(ID={event_id}): {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            return_connection(conn)

def update_cry_event_image(filename: str, illustration_url: str) -> bool:
    """更新宝宝哭声事件的插图 URL（按 filename 匹配）"""
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

def update_cry_event_image_by_id(event_id: int, illustration_url: str) -> bool:
    """更新宝宝哭声事件的插图 URL（按 ID 精确匹配，推荐使用）"""
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return False
            
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE baby_cry_events SET illustration_url = %s WHERE id = %s",
            (illustration_url, event_id)
        )
        
        conn.commit()
        updated = cursor.rowcount > 0
        cursor.close()
        if updated:
            print(f"  [DB] 已更新插图：ID={event_id}")
        return updated
    except Exception as e:
        print(f"  [DB Error] 更新插图失败(ID={event_id})：{e}")
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
    """标记文件为 A 轨已处理（统一用纯文件名，避免路径不一致导致重复记录）"""
    import os
    filename = os.path.basename(filename)  # 统一用纯文件名
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

def get_processed_files_for_date(date_str: str) -> set:
    """获取特定日期已处理的文件**名称**集合（用于精确断点续传）
    
    Args:
        date_str: 日期字符串，格式: YYYY-MM-DD
    
    Returns:
        已处理文件的文件名集合（统一使用 os.path.basename 提取）
    
    支持的文件名格式:
        1. /path/2025-11-15/file.m4a (日期文件夹)
        2. TermuxAudioRecording_2025-11-15_12-56-54.m4a (有横杠)
        3. recording-20251115-125944.m4a (无横杠 YYYYMMDD)
        4. recording-2025-11-15-125944.m4a (混合格式)
    """
    conn = None
    try:
        conn = get_connection()
        if not conn: return set()
        cursor = conn.cursor()
        result = set()

        # 格式1: 路径中包含 /YYYY-MM-DD/ (日期文件夹)
        cursor.execute(
            "SELECT filename FROM processed_files_a WHERE filename LIKE %s",
            (f'%/{date_str}/%',)
        )
        for row in cursor.fetchall():
            result.add(os.path.basename(row[0]))

        # 格式2: 文件名中包含 _YYYY-MM-DD_ 或 -YYYY-MM-DD- (有横杠格式)
        cursor.execute(
            "SELECT filename FROM processed_files_a WHERE filename ~ %s",
            (f'[_-]{date_str}[_-]',)
        )
        for row in cursor.fetchall():
            result.add(os.path.basename(row[0]))

        # 格式3: 文件名中包含 YYYYMMDD (无横杠格式)
        date_no_dash = date_str.replace('-', '')
        cursor.execute(
            "SELECT filename FROM processed_files_a WHERE filename ~ %s",
            (date_no_dash,)
        )
        for row in cursor.fetchall():
            result.add(os.path.basename(row[0]))

        # 格式4: 文件名中包含 YYYY-MM-DD 但前后分隔符不固定 (混合格式)
        cursor.execute(
            "SELECT filename FROM processed_files_a WHERE filename ~ %s",
            (date_str,)
        )
        for row in cursor.fetchall():
            result.add(os.path.basename(row[0]))

        cursor.close()
        return result
    except Exception as e:
        print(f"  [DB Error] 获取日期已处理文件列表失败: {e}")
        import traceback
        traceback.print_exc()
        return set()
    finally:
        if conn: return_connection(conn)

def get_cry_files_for_date(date_str: str) -> list:
    """获取指定日期已标记为 cry 的文件名列表
    
    Args:
        date_str: 日期字符串，格式: YYYY-MM-DD
    
    Returns:
        cry 文件名列表
    """
    conn = None
    try:
        conn = get_connection()
        if not conn: return []
        cursor = conn.cursor()
        result = []

        # 查找该日期所有 status='cry' 的文件
        cursor.execute(
            "SELECT filename FROM processed_files_a WHERE status='cry' AND filename ~ %s",
            (date_str,)
        )
        for row in cursor.fetchall():
            result.append(row[0])

        cursor.close()
        return sorted(result)
    except Exception as e:
        print(f"  [DB Error] 获取日期cry文件列表失败: {e}")
        return []
    finally:
        if conn: return_connection(conn)


def get_unanalyzed_cry_dates() -> list:
    """获取有 cry 记录但缺少有效 baby_cry_events 分析的日期列表
    
    包含两类情况：
    1. processed_files_a 中 status='cry' 但 baby_cry_events 中无对应记录
    2. baby_cry_events 中有记录但分析不完整（reason为空/category=analyzing/category=未分类）
    
    Returns:
        日期字符串列表，格式: ['2025-11-17', '2025-11-18', ...]
    """
    conn = None
    try:
        conn = get_connection()
        if not conn: return []
        cursor = conn.cursor()

        # 情况1：processed_files_a 有 cry 但 baby_cry_events 无记录
        cursor.execute("""
            SELECT DISTINCT cry_date FROM (
                SELECT substring(filename from '\d{4}-\d{2}-\d{2}') AS cry_date
                FROM processed_files_a
                WHERE status = 'cry' AND filename ~ '\d{4}-\d{2}-\d{2}'
            ) AS cry_dates
            WHERE cry_date NOT IN (
                SELECT DISTINCT recording_time::date::text
                FROM baby_cry_events
                WHERE recording_time IS NOT NULL
            )
            ORDER BY cry_date
        """)
        missing_dates = set(row[0] for row in cursor.fetchall() if row[0])

        # 情况2：baby_cry_events 有记录但分析不完整
        cursor.execute("""
            SELECT DISTINCT recording_time::date::text
            FROM baby_cry_events
            WHERE recording_time IS NOT NULL
              AND is_deleted = FALSE
              AND (reason IS NULL OR reason = '' OR reason = '未知'
                   OR reason_category = 'analyzing' 
                   OR reason_category = '未分类' OR reason_category = '未知'
                   OR reason_category IS NULL)
            ORDER BY recording_time::date::text
        """)
        incomplete_dates = set(row[0] for row in cursor.fetchall() if row[0])

        result = sorted(missing_dates | incomplete_dates)
        cursor.close()
        return result
    except Exception as e:
        print(f"  [DB Error] 获取未分析cry日期列表失败: {e}")
        return []
    finally:
        if conn: return_connection(conn)


def get_incomplete_cry_events(date_str: str = None) -> list:
    """获取分析不完整的哭声事件列表（需要重新分析）
    
    不完整的定义：reason为空/未知、reason_category=analyzing/未分类/未知/NULL
    
    Args:
        date_str: 可选，指定日期格式 YYYY-MM-DD。为空则返回所有不完整事件。
    
    Returns:
        事件字典列表，包含 id, filename, recording_time, reason_category, audio_path 等
    """
    conn = None
    try:
        conn = get_connection()
        if not conn: return []
        cursor = conn.cursor()

        query = """
            SELECT id, filename, recording_time, reason_category, reason, advice, 
                   event_files_json, audio_path, confidence
            FROM baby_cry_events
            WHERE is_deleted = FALSE
              AND (reason IS NULL OR reason = '' OR reason = '未知'
                   OR reason_category = 'analyzing' 
                   OR reason_category = '未分类' OR reason_category = '未知'
                   OR reason_category IS NULL)
        """
        params = []
        if date_str:
            query += " AND recording_time::date = %s"
            params.append(date_str)
        query += " ORDER BY recording_time"

        cursor.execute(query, tuple(params))
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'filename': row[1],
                'recording_time': row[2].isoformat() if row[2] else None,
                'reason_category': row[3],
                'reason': row[4],
                'advice': row[5],
                'event_files': json.loads(row[6]) if row[6] else [],
                'audio_path': row[7],
                'confidence': float(row[8]) if row[8] else 0.0,
            })
        cursor.close()
        return results
    except Exception as e:
        print(f"  [DB Error] 获取不完整分析事件列表失败: {e}")
        return []
    finally:
        if conn: return_connection(conn)


def get_completed_events_for_date(date_str: str) -> list:
    """获取指定日期已完成分析的事件列表（用于检查cry文件是否已被覆盖）

    Args:
        date_str: 日期格式 YYYY-MM-DD

    Returns:
        事件字典列表，包含 id, filename, event_files 等
    """
    conn = None
    try:
        conn = get_connection()
        if not conn: return []
        cursor = conn.cursor()

        query = """
            SELECT id, filename, recording_time, reason_category, reason, advice,
                   event_files_json, audio_path, confidence
            FROM baby_cry_events
            WHERE is_deleted = FALSE
              AND recording_time::date = %s
              AND reason IS NOT NULL AND reason != ''
              AND reason != '未知'
              AND reason_category IS NOT NULL
              AND reason_category != 'analyzing'
              AND reason_category != '未分类'
              AND reason_category != '未知'
            ORDER BY recording_time
        """

        cursor.execute(query, (date_str,))
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'filename': row[1],
                'recording_time': row[2].isoformat() if row[2] else None,
                'reason_category': row[3],
                'reason': row[4],
                'advice': row[5],
                'event_files': json.loads(row[6]) if row[6] else [],
                'audio_path': row[7],
                'confidence': float(row[8]) if row[8] else 0.0,
            })
        cursor.close()
        return results
    except Exception as e:
        print(f"  [DB Error] 获取已完成事件列表失败: {e}")
        return []
    finally:
        if conn: return_connection(conn)


def get_completed_cry_covered_files_for_date(date_str: str, include_event_files: bool = True) -> set:
    """获取指定日期已完成哭声事件覆盖的文件名集合。

    用于历史扫描跳过 B 轨实时分析已经完成的哭声事件，避免重复检测和重复
    Gemini 深度分析。include_event_files=False 时只返回事件主文件，避免把上下文
    音频误当成已检测哭声。
    """
    conn = None
    try:
        conn = get_connection()
        if not conn: return set()
        cursor = conn.cursor()

        date_no_dash = date_str.replace('-', '')
        cursor.execute("""
            SELECT filename, event_files_json
            FROM baby_cry_events
            WHERE is_deleted = FALSE
              AND (
                  recording_time::date = %s
                  OR filename LIKE %s
                  OR filename LIKE %s
                  OR event_files_json LIKE %s
                  OR event_files_json LIKE %s
              )
              AND reason IS NOT NULL AND reason != ''
              AND reason != '未知'
              AND reason_category IS NOT NULL
              AND reason_category != 'analyzing'
              AND reason_category != '未分类'
              AND reason_category != '未知'
        """, (date_str, f'%{date_str}%', f'%{date_no_dash}%', f'%{date_str}%', f'%{date_no_dash}%'))

        covered_files = set()
        for filename, event_files_json in cursor.fetchall():
            if filename:
                covered_files.add(os.path.basename(filename))
            if include_event_files and event_files_json:
                try:
                    for path in json.loads(event_files_json):
                        if path:
                            covered_files.add(os.path.basename(path))
                except Exception:
                    pass

        cursor.close()
        return covered_files
    except Exception as e:
        print(f"  [DB Error] 获取已完成哭声覆盖文件失败: {e}")
        return set()
    finally:
        if conn: return_connection(conn)


def soft_delete_incomplete_events_for_files(date_str: str, filenames: list) -> int:
    """软删除指定日期中，与给定文件名列表相关的未完成事件
    
    只软删除（标记 is_deleted = TRUE），不物理删除。
    只删除与 filenames 相关的事件，且事件必须是不完整的。
    
    Args:
        date_str: 日期字符串 YYYY-MM-DD
        filenames: 文件名列表（只删除这些文件涉及的事件）
    
    Returns:
        软删除的记录数
    """
    if not filenames:
        return 0
    
    conn = None
    try:
        conn = get_connection()
        if not conn: return 0
        cursor = conn.cursor()
        
        # 构建文件名占位符
        placeholders = ','.join(['%s'] * len(filenames))
        
        # 软删除条件：
        # 1. 日期匹配
        # 2. filename 在给定列表中，或 event_files_json 中包含这些文件名
        # 3. 事件不完整（reason/category 为空或未知）
        # 4. 尚未被软删除
        
        # 构建 LIKE 模式数组（避免 f-string 中使用反斜杠）
        like_patterns = []
        pattern_params = []
        for fn in filenames:
            like_patterns.append('%s')
            pattern_params.append('%' + fn + '%')
        
        patterns_sql = ','.join(like_patterns)
        
        cursor.execute("""
            UPDATE baby_cry_events 
            SET is_deleted = TRUE
            WHERE recording_time::date = %s
              AND is_deleted = FALSE
              AND (
                  filename IN ({placeholders})
                  OR event_files_json::text LIKE ANY(ARRAY[{patterns_sql}])
              )
              AND (reason IS NULL OR reason = '' OR reason = '未知'
                   OR reason_category = 'analyzing' 
                   OR reason_category = '未分类' OR reason_category = '未知'
                   OR reason_category IS NULL)
        """.format(placeholders=placeholders, patterns_sql=patterns_sql), 
        [date_str] + filenames + pattern_params)
        
        deleted = cursor.rowcount
        conn.commit()
        cursor.close()
        
        if deleted > 0:
            print(f"  [DB] 已软删除 {date_str} 的 {deleted} 条未完成事件（涉及 {len(filenames)} 个文件）")
        return deleted
    except Exception as e:
        print(f"  [DB Error] 软删除未完成事件失败: {e}")
        if conn: conn.rollback()
        return 0
    finally:
        if conn: return_connection(conn)


def delete_incomplete_cry_events(date_str: str) -> int:
    """软删除指定日期的分析不完整事件（重新分析前清理）
    
    改为软删除：标记 is_deleted = TRUE，不物理删除。
    
    Args:
        date_str: 日期字符串 YYYY-MM-DD
    
    Returns:
        软删除的记录数
    """
    conn = None
    try:
        conn = get_connection()
        if not conn: return 0
        cursor = conn.cursor()
        cursor.execute(
            """UPDATE baby_cry_events 
               SET is_deleted = TRUE
               WHERE recording_time::date = %s
                 AND is_deleted = FALSE
                 AND (reason IS NULL OR reason = '' OR reason = '未知'
                      OR reason_category = 'analyzing' 
                      OR reason_category = '未分类' OR reason_category = '未知'
                      OR reason_category IS NULL)""",
            (date_str,)
        )
        deleted = cursor.rowcount
        conn.commit()
        cursor.close()
        if deleted > 0:
            print(f"  [DB] 已软删除 {date_str} 的 {deleted} 条不完整事件")
        return deleted
    except Exception as e:
        print(f"  [DB Error] 软删除不完整分析事件失败: {e}")
        if conn: conn.rollback()
        return 0
    finally:
        if conn: return_connection(conn)


def get_uncovered_cry_count(date_str: str) -> int:
    """获取指定日期中，被标记为cry但未被baby_cry_events事件覆盖的文件数量
    
    即：在 processed_files_a 中 status='cry'，但该文件名不在 baby_cry_events 的 
    event_files_json 或 filename 字段中的文件数。
    
    Args:
        date_str: 日期字符串 YYYY-MM-DD
    
    Returns:
        未被覆盖的cry文件数量
    """
    conn = None
    try:
        conn = get_connection()
        if not conn: return 0
        cursor = conn.cursor()

        # 获取该日期所有 cry 标记文件（统一提取文件名，兼容路径和纯文件名两种格式）
        cursor.execute("""
            SELECT filename FROM processed_files_a 
            WHERE status = 'cry' AND filename LIKE %s
        """, (f'%{date_str}%',))
        cry_files = set(os.path.basename(row[0]) for row in cursor.fetchall() if row[0])

        if not cry_files:
            return 0

        # 获取该日期所有事件中包含的文件名（从 event_files_json 和 filename 提取）
        cursor.execute("""
            SELECT event_files_json, filename FROM baby_cry_events
            WHERE recording_time::date = %s AND is_deleted = FALSE
        """, (date_str,))
        
        covered_files = set()
        for row in cursor.fetchall():
            # 从 filename 字段（统一提取文件名）
            if row[1]:
                covered_files.add(os.path.basename(row[1]))
            # 从 event_files_json 字段
            if row[0]:
                try:
                    paths = json.loads(row[0])
                    for p in paths:
                        covered_files.add(os.path.basename(p))
                except:
                    pass

        # 未覆盖的文件数
        uncovered = cry_files - covered_files
        cursor.close()
        return len(uncovered)
    except Exception as e:
        print(f"  [DB Error] 获取未覆盖cry文件数量失败: {e}")
        return 0
    finally:
        if conn: return_connection(conn)


def get_all_cry_dates() -> list:
    """获取所有有 cry 记录的日期列表（从 processed_files_a 表中提取）
    
    Returns:
        日期字符串列表，格式: ['2025-11-17', '2025-11-18', ...]
    """
    conn = None
    try:
        conn = get_connection()
        if not conn: return []
        cursor = conn.cursor()

        # 从 filename 中提取日期，按日期分组
        cursor.execute("""
            SELECT DISTINCT substring(filename from '(\d{4}-\d{2}-\d{2})') as d
            FROM processed_files_a 
            WHERE status='cry' AND filename ~ '\d{4}-\d{2}-\d{2}'
            ORDER BY d
        """)
        result = [row[0] for row in cursor.fetchall() if row[0]]

        cursor.close()
        return result
    except Exception as e:
        print(f"  [DB Error] 获取cry日期列表失败: {e}")
        return []
    finally:
        if conn: return_connection(conn)


def refresh_file_cache(target_dir: str, audio_exts=('.m4a', '.mp3', '.wav', '.aac', '.flac', '.ogg', '.acc'), progress_callback=None, log_callback=None, ttl_seconds=86400, skip_completed_dates=None) -> int:
    """扫描目录并刷新文件缓存到Redis，返回扫描到的文件数量

    Args:
        target_dir: 目标目录
        audio_exts: 音频文件扩展名
        progress_callback: 进度回调函数，接收 (count, current_dir) 参数
        log_callback: 日志回调函数，接收 (message) 参数
        ttl_seconds: 缓存过期时间（秒），默认 86400（24小时）。过期后下次读取会自动触发重新刷盘
        skip_completed_dates: 需要跳过的已完成日期集合，如 {'2025-11-17', '2025-11-21'}
    """
    import os
    import re
    import time
    import valkey

    VALKEY_URI = os.environ.get('VALKEY_URI', '')

    def log(msg):
        print(msg)
        if log_callback:
            log_callback(msg)

    def log_progress(count, current_dir):
        if progress_callback:
            progress_callback(count, current_dir)

    log(f"  [刷盘] 开始扫描目录: {target_dir}")
    if skip_completed_dates:
        log(f"  [刷盘] 跳过已完成日期: {len(skip_completed_dates)} 个")
    start_time = time.time()

    if not VALKEY_URI:
        log("  [Valkey Error] VALKEY_URI 环境变量未设置")
        return -1

    try:
        # 连接 Valkey
        r = valkey.from_url(VALKEY_URI)

        # 清空旧缓存
        keys = r.keys('babycry:*')
        if keys:
            r.delete(*keys)
        log("  [刷盘] 已清空旧缓存")

        # 扫描目录
        count = 0
        batch_size = 500  # Redis pipeline 批量大小
        batch_data = []
        last_callback_time = time.time()
        date_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2})$')
        current_container = ""
        processed_in_dir = 0
        skipped_dates = 0

        for root, dirs, files in os.walk(target_dir):
            # 只扫描 processed 目录下的文件，跳过其他所有目录
            current_container = os.path.basename(root)

            # 检查路径中是否包含 processed 目录
            normalized_root = root.replace('\\', '/')
            path_parts = normalized_root.split('/')
            is_in_processed = 'processed' in path_parts
            is_processed_dir = current_container == 'processed'
            is_date_dir = bool(date_pattern.match(current_container))

            # 如果不在 processed 目录下，跳过（但允许进入 processed 子目录）
            if not is_in_processed and not is_processed_dir:
                dirs[:] = [d for d in dirs if d == 'processed']
                continue

            # 在 processed 目录内，只保留日期格式的子目录
            if is_processed_dir:
                # 过滤掉已完成的日期
                if skip_completed_dates:
                    dirs[:] = [d for d in dirs if date_pattern.match(d) and d not in skip_completed_dates]
                    skipped_dates = len([d for d in dirs if date_pattern.match(d) and d in skip_completed_dates])
                else:
                    dirs[:] = [d for d in dirs if date_pattern.match(d)]
            elif is_date_dir:
                # 在日期目录内，不再深入子目录
                dirs[:] = []
                processed_in_dir = 0

            # 跳过已完成的日期目录（如果进入时被误放进来）
            if is_date_dir and skip_completed_dates and current_container in skip_completed_dates:
                if skipped_dates % 50 == 0:
                    log(f"  [刷盘] 跳过已完成日期: {current_container}")
                skipped_dates += 1
                dirs[:] = []
                continue

            # 每秒回调一次进度
            current_time = time.time()
            if progress_callback and current_time - last_callback_time >= 1:
                log_progress(count, current_container)
                last_callback_time = current_time

            # 处理文件
            for file in files:
                if file.startswith('.'):
                    continue
                if not file.lower().endswith(audio_exts):
                    continue

                filepath = os.path.join(root, file)

                # 提取日期
                if is_date_dir:
                    date_str = current_container
                else:
                    m = re.search(r'/(\d{4}-\d{2}-\d{2})/', filepath)
                    date_str = m.group(1) if m else 'unknown'

                # 准备 Valkey 数据
                batch_data.append(('babycry:file:' + filepath, date_str + '|' + file))
                batch_data.append(('babycry:date:' + date_str, filepath))
                batch_data.append(('babycry:files', filepath))
                count += 1
                processed_in_dir += 1

                # 每5000个文件打印一次进度
                if processed_in_dir % 5000 == 0:
                    log(f"  [刷盘调试] 目录 {current_container} 已处理 {processed_in_dir} 个文件，总count: {count}")

                # 批量写入 Valkey (每3个元素为一条记录)，带 TTL 自动过期
                if len(batch_data) >= batch_size * 3:
                    pipe = r.pipeline()
                    for i in range(0, len(batch_data), 3):
                        key1, val1 = batch_data[i]
                        key2, val2 = batch_data[i+1]
                        key3, val3 = batch_data[i+2]
                        pipe.set(key1, val1, ex=ttl_seconds)
                        pipe.sadd(key2, val2)
                        pipe.expire(key2, ttl_seconds)
                        pipe.sadd(key3, val3)
                        pipe.expire(key3, ttl_seconds)
                    pipe.execute()
                    batch_data = []

        # 写入剩余数据，带 TTL
        if batch_data:
            pipe = r.pipeline()
            for i in range(0, len(batch_data), 3):
                key1, val1 = batch_data[i]
                key2, val2 = batch_data[i+1]
                key3, val3 = batch_data[i+2]
                pipe.set(key1, val1, ex=ttl_seconds)
                pipe.sadd(key2, val2)
                pipe.expire(key2, ttl_seconds)
                pipe.sadd(key3, val3)
                pipe.expire(key3, ttl_seconds)
            pipe.execute()

        # 最后回调一次
        log_progress(count, current_container)
        elapsed = time.time() - start_time
        log(f"  [刷盘完成] 共扫描 {count} 个音频文件，耗时: {elapsed:.1f}秒，平均速度: {count/elapsed:.1f}个/秒" if elapsed > 0 else f"  [刷盘完成] 共扫描 {count} 个音频文件")
        return count

    except Exception as e:
        log(f"  [Redis Error] 刷新文件缓存失败: {e}")
        import traceback
        traceback.print_exc()
        return -1

def check_cache_freshness() -> dict:
    """检查 Redis 文件缓存的新鲜度
    
    Returns:
        dict: {
            'fresh': bool,          # 缓存是否新鲜
            'total_keys': int,      # 总 key 数量
            'expired_keys': int,    # 已过期的 key 数量
            'ttl_min': int,         # 最小剩余 TTL（秒），-1=无TTL，-2=key不存在
            'ttl_avg': float,       # 平均剩余 TTL（秒）
        }
    """
    import os
    VALKEY_URI = os.environ.get('VALKEY_URI', '')
    if not VALKEY_URI:
        return {'fresh': False, 'total_keys': 0, 'expired_keys': 0, 'ttl_min': -2, 'ttl_avg': 0}
    try:
        import valkey
        r = valkey.from_url(VALKEY_URI)
        
        # 检查 babycry:files 这个集合的 TTL 作为代表
        main_key = 'babycry:files'
        ttl = r.ttl(main_key)
        
        # ttl 返回值: -2=key不存在, -1=key存在但无TTL, >0=剩余秒数
        if ttl == -2:
            # 缓存不存在
            return {'fresh': False, 'total_keys': 0, 'expired_keys': 0, 'ttl_min': -2, 'ttl_avg': 0}
        elif ttl == -1:
            # 缓存存在但没有 TTL（旧缓存），视为不新鲜
            return {'fresh': False, 'total_keys': -1, 'expired_keys': 0, 'ttl_min': -1, 'ttl_avg': 0}
        else:
            # 有 TTL，检查剩余时间是否充足（<1小时视为不新鲜）
            fresh = ttl > 3600
            total = r.scard(main_key)
            return {'fresh': fresh, 'total_keys': total, 'expired_keys': 0, 'ttl_min': ttl, 'ttl_avg': float(ttl)}
    except Exception as e:
        print(f"  [Valkey Error] 检查缓存新鲜度失败: {e}")
        return {'fresh': False, 'total_keys': 0, 'expired_keys': 0, 'ttl_min': -2, 'ttl_avg': 0}


def get_file_cache_from_redis(date_str: str = None) -> list:
    """从 Valkey 获取文件列表（使用 pipeline 优化）"""
    import os
    VALKEY_URI = os.environ.get('VALKEY_URI', '')
    if not VALKEY_URI:
        print(f"  [Valkey Error] VALKEY_URI 环境变量未设置")
        return []
    try:
        import valkey
        r = valkey.from_url(VALKEY_URI)

        if date_str:
            filepaths = r.smembers('babycry:date:' + date_str)
        else:
            filepaths = r.smembers('babycry:files')

        result = []
        # 使用 pipeline 批量获取（最多 10000 个一批）
        batch_size = 10000
        filepath_list = list(filepaths)
        total = len(filepath_list)

        for i in range(0, len(filepath_list), batch_size):
            batch = filepath_list[i:i+batch_size]
            pipe = r.pipeline()
            for fp in batch:
                fp_str = fp.decode('utf-8') if isinstance(fp, bytes) else fp
                pipe.get('babycry:file:' + fp_str)
            values = pipe.execute()

            for j, val in enumerate(values):
                fp = filepath_list[i + j]
                fp_str = fp.decode('utf-8') if isinstance(fp, bytes) else fp
                if val:
                    val_str = val.decode('utf-8') if isinstance(val, bytes) else val
                    parts = val_str.split('|', 1)
                    if len(parts) == 2:
                        result.append({'filepath': fp_str, 'filename': parts[1], 'date_str': parts[0]})
                    else:
                        result.append({'filepath': fp_str, 'filename': val_str})

            # 每批打印进度
            processed = min(i + batch_size, total)
            if i % (batch_size * 3) == 0 or processed == total:
                from datetime import datetime
                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{now}] [Valkey] 已获取 {processed}/{total} 个文件信息 ({processed*100//total}%)")

        from datetime import datetime
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{now}] [Valkey] 文件列表获取完成，共 {len(result)} 个文件")
        return result
    except Exception as e:
        print(f"  [Valkey Error] 获取文件缓存失败: {e}")
        import traceback
        traceback.print_exc()
        return []

def get_date_file_counts_from_redis() -> dict:
    """从 Valkey 获取每个日期的文件数量"""
    import os
    VALKEY_URI = os.environ.get('VALKEY_URI', '')
    if not VALKEY_URI:
        print(f"  [Valkey Error] VALKEY_URI 环境变量未设置")
        return {}
    try:
        import valkey
        r = valkey.from_url(VALKEY_URI)

        result = {}
        keys = r.keys('babycry:date:*')
        for key in keys:
            key_str = key.decode('utf-8') if isinstance(key, bytes) else key
            date_str = key_str.replace('babycry:date:', '')
            result[date_str] = r.scard(key)
        return result
    except Exception as e:
        print(f"  [Valkey Error] 获取日期文件数量失败: {e}")
        return {}

def get_file_count_from_redis() -> int:
    """从 Valkey 获取总文件数量"""
    import os
    VALKEY_URI = os.environ.get('VALKEY_URI', '')
    if not VALKEY_URI:
        print(f"  [Valkey Error] VALKEY_URI 环境变量未设置")
        return 0
    try:
        import valkey
        r = valkey.from_url(VALKEY_URI)
        return r.scard('babycry:files')
    except Exception as e:
        print(f"  [Valkey Error] 获取文件总数失败: {e}")
        return 0

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

def soft_delete_event_by_id(event_id: int) -> bool:
    """按 ID 软删除单条哭声事件记录"""
    conn = None
    try:
        conn = get_connection()
        if not conn: return False
        cursor = conn.cursor()
        cursor.execute("UPDATE baby_cry_events SET is_deleted = TRUE WHERE id = %s", (event_id,))
        conn.commit()
        deleted = cursor.rowcount > 0
        cursor.close()
        return deleted
    except Exception as e:
        print(f"  [DB Error] 软删除事件失败 (ID={event_id}): {e}")
        if conn: conn.rollback()
        return False
    finally:
        if conn: return_connection(conn)


def delete_cry_event_by_id(event_id: int) -> bool:
    """按 ID 删除单条哭声事件记录"""
    conn = None
    try:
        conn = get_connection()
        if not conn: return False
        cursor = conn.cursor()
        cursor.execute("DELETE FROM baby_cry_events WHERE id = %s", (event_id,))
        conn.commit()
        deleted = cursor.rowcount > 0
        cursor.close()
        return deleted
    except Exception as e:
        print(f"  [DB Error] 删除事件失败 (ID={event_id}): {e}")
        if conn: conn.rollback()
        return False
    finally:
        if conn: return_connection(conn)


def delete_cry_events_by_date(date_str: str) -> int:
    """删除指定日期（YYYY-MM-DD）的哭声分析事件
    
    注意：不再删除 processed_files_a 中的记录，保留处理进度用于智能续传。
    如果需要完全重跑某日期，应使用独立的清理方法。
    """
    conn = None
    deleted_count = 0
    try:
        conn = get_connection()
        if not conn: return 0
        cursor = conn.cursor()
        
        # 只删除哭声事件，保留 processed_files_a 处理进度
        cursor.execute(
            "DELETE FROM baby_cry_events WHERE recording_time::date = %s",
            (date_str,)
        )
        deleted_count = cursor.rowcount
        
        conn.commit()
        cursor.close()
        print(f"  [DB] 已清除日期 {date_str} 的 {deleted_count} 条哭声事件记录")
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
