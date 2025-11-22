#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import psycopg2
from psycopg2 import pool
import json
from typing import List, Dict, Optional

# PostgreSQL 连接配置
DATABASE_URL = "postgresql://postgres:difyai123456@192.168.1.188:5433/postgres"

# 连接池
connection_pool = None

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

def save_to_db(filename: str, full_text: str, segments_list: List[Dict]) -> bool:
    """保存转录记录到数据库"""
    conn = None
    try:
        conn = get_connection()
        if not conn:
            print("[DB Error] 无法获取数据库连接")
            return False
            
        cursor = conn.cursor()
        segments_json = json.dumps(segments_list, ensure_ascii=False)
        
        cursor.execute(
            "INSERT INTO transcriptions (filename, full_text, segments_json) VALUES (%s, %s, %s)",
            (filename, full_text, segments_json)
        )
        
        conn.commit()
        cursor.close()
        print(f"  [DB] Saved {filename}")
        return True
    except Exception as e:
        print(f"  [DB Error] {e}")
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
            "SELECT id, filename, created_at, full_text, segments_json FROM transcriptions ORDER BY created_at DESC LIMIT %s OFFSET %s",
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
                'segments_json': row[4]
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
