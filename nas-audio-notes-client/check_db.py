import sqlite3
import json

# 从web_viewer.py导入配置
try:
    import sys
    sys.path.append('.')
    from web_viewer import CONFIG
    db_path = CONFIG["DB_PATH"]
except:
    # 如果导入失败，使用默认路径
    db_path = "transcriptions.db"

print(f"检查数据库: {db_path}")

# 连接到数据库
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 查询所有表
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("Tables in database:")
for table in tables:
    print(f"- {table[0]}")

# 检查chat_history表是否存在
if any('chat_history' in table for table in tables):
    print("\nchat_history table structure:")
    cursor.execute("PRAGMA table_info(chat_history)")
    columns = cursor.fetchall()
    for col in columns:
        print(f"- {col[1]} ({col[2]})")
    
    # 查询chat_history表中的数据
    print("\nData in chat_history table:")
    cursor.execute("SELECT * FROM chat_history")
    rows = cursor.fetchall()
    for row in rows:
        print(f"- {row}")
else:
    print("\nchat_history table does not exist")

# 检查transcriptions表是否存在
if any('transcriptions' in table for table in tables):
    print("\ntranscriptions table structure:")
    cursor.execute("PRAGMA table_info(transcriptions)")
    columns = cursor.fetchall()
    for col in columns:
        print(f"- {col[1]} ({col[2]})")
    
    # 查询transcriptions表中的数据数量
    print("\nData count in transcriptions table:")
    cursor.execute("SELECT COUNT(*) FROM transcriptions")
    count = cursor.fetchone()
    print(f"- Total rows: {count[0]}")

conn.close()