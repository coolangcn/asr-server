import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.header import Header
import threading
import logging
import os
import re

logger = logging.getLogger("EmailUtils")
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(console_handler)
    logger.propagate = False

class EmailConfig:
    SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER", "")
    SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "465"))
    SENDER_EMAIL = os.getenv("EMAIL_SENDER", "")
    SMTP_AUTH_CODE = os.getenv("EMAIL_AUTH_CODE", "")
    RECEIVER_EMAIL = os.getenv("EMAIL_RECEIVER", "")

def send_email_sync(subject, content, image_data=None):
    """
    同步发送邮件（推荐在单独线程中调用）
    支持带图片：image_data 为 data:image/png;base64,... 格式的字符串
    """
    if not EmailConfig.SENDER_EMAIL or not EmailConfig.SMTP_AUTH_CODE:
        logger.warning("⚠️ 邮件配置未完成，跳过邮件发送")
        return False
    
    try:
        if image_data:
            # 带图片的邮件
            message = MIMEMultipart('related')
            message['From'] = f"BabyCry Monitor <{EmailConfig.SENDER_EMAIL}>"
            message['To'] = EmailConfig.RECEIVER_EMAIL or EmailConfig.SENDER_EMAIL
            message['Subject'] = str(Header(subject, 'utf-8'))
            
            # HTML 正文
            html_content = f"""
            <html>
                <body>
                    <pre style="white-space: pre-wrap; font-family: Arial; font-size: 14px;">{content}</pre>
                    <br>
                    <img src="cid:generated_image" alt="生成的插图" style="max-width: 800px; height: auto;">
                </body>
            </html>
            """
            msg_alternative = MIMEMultipart('alternative')
            msg_text = MIMEText(content, 'plain', 'utf-8')
            msg_html = MIMEText(html_content, 'html', 'utf-8')
            msg_alternative.attach(msg_text)
            msg_alternative.attach(msg_html)
            message.attach(msg_alternative)
            
            # 解析 base64 图片
            try:
                # 去掉 data:image/xxx;base64, 前缀
                match = re.match(r'data:image/[^;]+;base64,(.*)', image_data)
                if match:
                    image_base64 = match.group(1)
                    import base64
                    image_bytes = base64.b64decode(image_base64)
                    
                    msg_image = MIMEImage(image_bytes)
                    msg_image.add_header('Content-ID', '<generated_image>')
                    message.attach(msg_image)
            except Exception as img_error:
                logger.error(f"解析图片失败: {img_error}")
        else:
            # 纯文本邮件
            message = MIMEText(content, 'plain', 'utf-8')
            message['From'] = f"BabyCry Monitor <{EmailConfig.SENDER_EMAIL}>"
            message['To'] = EmailConfig.RECEIVER_EMAIL or EmailConfig.SENDER_EMAIL
            message['Subject'] = str(Header(subject, 'utf-8'))
        
        # QQ 邮箱必须使用 SSL
        with smtplib.SMTP_SSL(EmailConfig.SMTP_SERVER, EmailConfig.SMTP_PORT) as server:
            server.login(EmailConfig.SENDER_EMAIL, EmailConfig.SMTP_AUTH_CODE)
            server.sendmail(EmailConfig.SENDER_EMAIL, [EmailConfig.RECEIVER_EMAIL or EmailConfig.SENDER_EMAIL], message.as_string())
        
        logger.info(f"📧 邮件已成功发送: {subject}")
        return True
    except Exception as e:
        logger.error(f"❌ 邮件发送失败: {e}")
        import traceback
        logger.error(f"❌ 异常堆栈: {traceback.format_exc()}")
        return False

def send_email_async(subject, content, image_data=None):
    """
    异步发送邮件（立即返回，后台线程处理）
    """
    threading.Thread(target=send_email_sync, args=(subject, content, image_data), daemon=True).start()

def send_cry_alert_email(filename, confidence, details=None, reason=None, advice=None, category=None, image_data=None, time_range=None):
    """
    专门发送哭声警报（增强版，支持带分析结果和图片）
    """
    subject = f"🚨 宝宝哭声警报！{f'（{time_range}）' if time_range else ''}"
    details_str = "\n".join(details) if details else "无详细模型得分"
    
    content_parts = [f"检测文件: {filename}", f"置信度: {confidence:.3f}", f"模型详情:\n{details_str}"]
    
    if time_range:
        content_parts.insert(0, f"时间范围: {time_range}")
        content_parts.insert(0, "=" * 40)
    
    if category or reason or advice:
        content_parts.append("\n========== 深度分析 ==========")
        if category:
            content_parts.append(f"原因分类: {category}")
        if reason:
            content_parts.append(f"原因分析: {reason}")
        if advice:
            content_parts.append(f"安抚建议: {advice}")
    
    content_parts.extend([
        "",
        "系统已检测到宝宝哭声，并记录到数据库。",
        "针对该事件的深度分析通常在事件发生的 5 分钟后（收集完整上下文后）生成。"
    ])
    
    content = "\n".join(content_parts)
    send_email_async(subject, content, image_data)


def _post_cry_payload_async(payload):
    """统一异步推送哭声类 Webhook（URL/Token 来自 .env），失败仅记日志，绝不阻塞主流程"""
    url = (os.getenv("CRY_WEBHOOK_URL") or "").strip()
    if not url:
        logger.info("🔗 [Webhook] 未配置 CRY_WEBHOOK_URL，跳过推送")
        return False
    token = (os.getenv("CRY_WEBHOOK_TOKEN") or "").strip()

    def _post():
        try:
            import requests
            headers = {"Content-Type": "application/json"}
            if token:
                headers["X-Gitlab-Token"] = token
            resp = requests.post(url, json=payload, headers=headers, timeout=5)
            logger.info(f"🔗 [Webhook] {payload.get('event_type', '')}推送{'成功' if resp.ok else '失败'}: HTTP {resp.status_code} -> {url}")
            return resp.ok
        except Exception as e:
            logger.error(f"🔗 [Webhook] 推送失败: {e}")
            return False

    threading.Thread(target=_post, daemon=True).start()
    return True


def _resolve_room(device, room=None):
    """设备名 → 房间中文名（.env 的 RECOVERY_SONY_X_ROOM：living=客厅 / bedroom=卧室）"""
    if room:
        return room
    if not device:
        return None
    _code = (os.getenv(f"RECOVERY_{device.upper().replace('-', '_')}_ROOM") or "").strip().lower()
    return {"living": "客厅", "bedroom": "卧室"}.get(_code, _code or None)


def _preview_links(event_id):
    """返回 (preview_url, 无) 与公开插图直链所需的基础信息；未配置 VIEWER_URL 则返回 (None, None)"""
    viewer = (os.getenv("CRY_WEBHOOK_VIEWER_URL") or "").strip().rstrip('/')
    if not viewer:
        return None, None
    ptok = (os.getenv("CRY_PREVIEW_TOKEN") or "").strip()
    suffix = f"?t={ptok}" if ptok else ""
    return f"{viewer}/preview/cry/{event_id}{suffix}", f"{viewer}/preview/cry/{event_id}/illustration{suffix}"


def send_cry_webhook(confidence, message="检测到婴儿啼哭", advice="请尽快前往查看",
                     timestamp=None, event_id=None, filename=None, recording_time=None,
                     time_range=None, audio_duration=None, device=None, room=None,
                     models=None, viewer_url=None):
    """
    哭声报警时向外部 Webhook 推送即时报警消息（GitLab 风格 X-Gitlab-Token 请求头）。
    尽量携带系统已有的全部参数，可选字段为空时自动省略。异步发送。
    """
    url = (os.getenv("CRY_WEBHOOK_URL") or "").strip()
    if not url:
        logger.info("🔗 [Webhook] 未配置 CRY_WEBHOOK_URL，跳过哭声报警推送")
        return False
    room = _resolve_room(device, room)
    viewer_url = viewer_url or (os.getenv("CRY_WEBHOOK_VIEWER_URL") or "").strip() or None

    from datetime import datetime as _dt
    payload = {
        "event_type": "宝宝大哭报警",
        "timestamp": timestamp or _dt.now().strftime("%Y-%m-%d %H:%M:%S"),
        "confidence": f"{confidence * 100:.0f}%",
        "confidence_line": f" (置信度: {confidence * 100:.0f}%)",
        "message": message,
        "advice": advice,
        "analysis_block": "\n• **当前状态**：⏳ 深度分析中（录音合并窗口关闭后将自动更新原因与插图）\n• **安抚建议**：" + (advice or "请尽快前往查看"),
        "illustration_block": "",
    }
    # 可选扩展字段：有值才携带，方便对接方按需取用
    if event_id is not None:
        payload["event_id"] = event_id
    if filename:
        payload["filename"] = filename
    if recording_time:
        payload["recording_time"] = recording_time
    if time_range:
        payload["time_range"] = time_range
    if audio_duration is not None:
        payload["audio_duration_sec"] = round(float(audio_duration), 1)
    if device:
        payload["device"] = device
    if room:
        payload["room"] = room
    if models:
        payload["models"] = list(models)
    if viewer_url:
        payload["viewer_url"] = viewer_url
    # 免登录预览页（可经 CF 隧道公网访问）：接收方点开即可查看插图/详情并试听哭声片段
    if event_id is not None and viewer_url:
        preview_url, _ = _preview_links(event_id)
        if preview_url:
            payload["preview_url"] = preview_url
    return _post_cry_payload_async(payload)


def send_cry_analysis_webhook(event_id, status="ok", category=None, reason=None, advice=None,
                              confidence=None, filename=None, recording_time=None, time_range=None,
                              device=None, room=None, illustration_path=None):
    """
    深度分析 + AI 插图生成完成后，推送"分析报告"Webhook（第二推）。
    illustration_path 为 5008 侧相对路径（/api/illustration/xxx），自动转换为带令牌的公网直链。
    """
    from datetime import datetime as _dt
    room = _resolve_room(device, room)
    preview_url, illu_url = _preview_links(event_id)

    cat_str = category or "情绪发泄/未满足"
    rsn_str = reason or "结合上下文深度分析完毕"
    adv_str = advice or "请结合实际情境予以安抚"
    payload = {
        "event_type": "宝宝哭声分析报告",
        "timestamp": _dt.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event_id": event_id,
        "status": status,  # ok=分析完成 / failed=分析失败
        "category": cat_str,
        "reason": rsn_str,
        "advice": adv_str,
        "confidence_line": f" (置信度: {confidence * 100:.0f}%)" if confidence is not None else "",
        "analysis_block": f"\n• **分类判决**：{cat_str}\n• **原因分析**：{rsn_str}\n• **安抚建议**：{adv_str}",
        "illustration_block": "",
    }
    if confidence is not None:
        payload["confidence"] = f"{confidence * 100:.0f}%"
    if filename:
        payload["filename"] = filename
    if recording_time:
        payload["recording_time"] = recording_time
    if time_range:
        payload["time_range"] = time_range
    if device:
        payload["device"] = device
    if room:
        payload["room"] = room
    if preview_url:
        payload["preview_url"] = preview_url
    if illustration_path and illu_url and str(illustration_path).startswith("/api/illustration/"):
        payload["illustration_url"] = illu_url
        payload["illustration_block"] = f"\n• **分析插图**：[点击查看分析插图]({illu_url})"
    return _post_cry_payload_async(payload)
