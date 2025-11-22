import os

log_file = "asr-server.log"
if os.path.exists(log_file):
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"读取到 {len(lines)} 行日志")
            # 打印最后50行
            print("--- 最后50行日志 ---")
            for line in lines[-50:]:
                print(line.strip())
            
            # 搜索错误
            print("\n--- 数据库相关错误 ---")
            for line in lines:
                if "数据库" in line or "DB Error" in line or "Error" in line:
                    print(line.strip())
    except Exception as e:
        print(f"读取日志失败 (utf-8): {e}")
        try:
            with open(log_file, 'r', encoding='gbk') as f:
                lines = f.readlines()
                print(f"读取到 {len(lines)} 行日志 (GBK)")
                for line in lines[-20:]:
                    print(line.strip())
        except Exception as e2:
            print(f"读取日志失败 (gbk): {e2}")
else:
    print(f"日志文件不存在: {log_file}")
