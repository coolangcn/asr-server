import sys

# Read the file
with open('asr_server.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the line to modify
for i, line in enumerate(lines):
    # Change the global declaration
    if line.strip() == 'global asr_pipeline, sv_pipelines':
        lines[i] = line.replace('global asr_pipeline, sv_pipelines', 'global asr_pipeline, sv_pipelines, whisper_model')
        print(f"Modified line {i+1}: global declaration")
    
    # Find where to add Whisper loading code
    if 'print(f"✅ 服务就绪 | ASR: SenseVoice | SV: {list(sv_pipelines.keys())}\\n")' in line:
        # Insert Whisper loading code after this line
        indent = '    '
        whisper_code = [
            '\n',
            indent + '# 4. 加载 Whisper 模型 (可选)\n',
            indent + 'if Config.ENABLE_WHISPER_COMPARISON:\n',
            indent + '    print(f"🎤 加载 Whisper 模型...")\n',
            indent + '    try:\n',
            indent + '        whisper_model = whisper.load_model("base", device=Config.DEVICE.split(\':\')[0])\n',
            indent + '        print("✅ Whisper模型加载完成")\n',
            indent + '    except Exception as e:\n',
            indent + '        logger.warning(f"⚠️ Whisper模型加载失败: {e}，将禁用Whisper对比功能")\n',
            indent + '        whisper_model = None\n'
        ]
        lines[i+1:i+1] = whisper_code
        print(f"Added Whisper loading code after line {i+1}")
        break

# Write the file back
with open('asr_server.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("File modified successfully!")
