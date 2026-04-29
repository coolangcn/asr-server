import os, time
from datetime import datetime, timedelta

source_dir = "/Volumes/download/records/Sony-2"
SUPPORTED_FORMATS = ['.m4a', '.mp3', '.wav', '.aac', '.flac', '.ogg', '.acc']

start = time.time()
latest_mtime = 0

today = datetime.now()
yesterday = today - timedelta(days=1)
recent_folders = [today.strftime("%Y-%m-%d"), yesterday.strftime("%Y-%m-%d")]

for item in os.listdir(source_dir):
    if item in ["processed", "failed", "audio_segments", "logs"] or item.startswith('.'):
        continue
    
    item_path = os.path.join(source_dir, item)
    items_to_check = []
    
    if os.path.isfile(item_path):
        items_to_check.append(item_path)
    elif os.path.isdir(item_path) and item in recent_folders:
        try:
            for subitem in os.listdir(item_path):
                if not subitem.startswith('.'):
                    subp = os.path.join(item_path, subitem)
                    if os.path.isfile(subp):
                        items_to_check.append(subp)
        except Exception:
            pass
            
    for filepath in items_to_check:
        ext = os.path.splitext(filepath)[1].lower()
        if ext in SUPPORTED_FORMATS:
            mtime = os.path.getmtime(filepath)
            if mtime > latest_mtime:
                latest_mtime = mtime

elapsed = time.time() - start
print(f"Latest mtime: {latest_mtime} ({datetime.fromtimestamp(latest_mtime)})")
print(f"Elapsed: {elapsed:.3f}s")
