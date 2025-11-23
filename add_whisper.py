#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""添加Whisper配置和功能"""

with open('asr_server.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. 添加Whisper配置
old_config = """    # 情感检测配置
    EMOTION_MODEL = "iic/SenseVoiceSmall"
    ENABLE_EMOTION_DETECTION = True  # 是否启用情感检测"""

new_config = """    # 情感检测配置
    EMOTION_MODEL = "iic/SenseVoiceSmall"
    ENABLE_EMOTION_DETECTION = True  # 是否启用情感检测
    
    # Whisper对比配置
    WHISPER_MODEL = "small"  # tiny/base/small/medium/large-v3
    ENABLE_WHISPER_COMPARISON = True  # 是否启用Whisper对比识别"""

content = content.replace(old_config, new_config)

# 2. 添加whisper导入
old_import = "from db_manager import save_to_db\nfrom logging.handlers import TimedRotatingFileHandler"
new_import = "from db_manager import save_to_db\nfrom logging.handlers import TimedRotatingFileHandler\nimport whisper"

content = content.replace(old_import, new_import)

# 3. 添加whisper_model全局变量
old_globals = "emotion_pipeline = None  # 情感检测模型"
new_globals = "emotion_pipeline = None  # 情感检测模型\nwhisper_model = None  # Whisper对比模型"

content = content.replace(old_globals, new_globals)

# 4. 在load_models()中添加Whisper加载
old_load = """    else:
        print("⏭️  跳过情感检测模型加载（已禁用）")

    # 4. 加载 SV 模型"""

new_load = """    else:
        print("⏭️  跳过情感检测模型加载（已禁用）")

    # 4. 加载Whisper对比模型
    global whisper_model
    if Config.ENABLE_WHISPER_COMPARISON:
        print(f"🎤 加载Whisper对比模型: {Config.WHISPER_MODEL} ...")
        whisper_model = whisper.load_model(Config.WHISPER_MODEL, device=Config.DEVICE.split(':')[0])
        print("✅ Whisper对比模型加载完成")
    else:
        print("⏭️  跳过Whisper对比模型加载（已禁用）")

    # 5. 加载 SV 模型"""

content = content.replace(old_load, new_load)

# 5. 添加Whisper识别函数（在detect_emotion_for_segment之后）
whisper_function = '''

def transcribe_with_whisper(audio_path):
    """
    使用Whisper识别音频片段（作为FunASR的对比参考）
    
    Args:
        audio_path: 音频片段路径
        
    Returns:
        str: Whisper识别的文本，如果失败返回None
    """
    if not Config.ENABLE_WHISPER_COMPARISON or whisper_model is None:
        return None
    
    try:
        result = whisper_model.transcribe(
            audio_path,
            language='zh',
            fp16=True,  # GPU加速
            verbose=False
        )
        whisper_text = result['text'].strip()
        logger.info(f"      [Whisper对比] {whisper_text}")
        return whisper_text
    except Exception as e:
        logger.warning(f"      [Whisper对比] 识别失败: {e}")
        return None
'''

# 找到detect_emotion_for_segment函数结束的位置
detect_end = content.find("# =================== 提取 embedding")
if detect_end > 0:
    content = content[:detect_end] + whisper_function + "\n" + content[detect_end:]

with open('asr_server.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ 成功添加Whisper配置和功能!")
print("  - 添加了WHISPER_MODEL和ENABLE_WHISPER_COMPARISON配置")
print("  - 添加了whisper_model全局变量")
print("  - 在load_models()中加载Whisper模型")
print("  - 创建了transcribe_with_whisper()函数")
