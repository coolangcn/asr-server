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

def send_cry_alert_email(filename, confidence, details=None, reason=None, advice=None, category=None, image_data=None):
    """
    专门发送哭声警报（增强版，支持带分析结果和图片）
    """
    subject = "🚨 宝宝哭声警报！"
    details_str = "\n".join(details) if details else "无详细模型得分"
    
    content_parts = [f"检测文件: {filename}", f"置信度: {confidence:.3f}", f"模型详情:\n{details_str}"]
    
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
