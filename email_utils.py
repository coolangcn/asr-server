import smtplib
from email.mime.text import MIMEText
from email.header import Header
import threading
import logging
import os

logger = logging.getLogger("EmailUtils")

class EmailConfig:
    SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER", "smtp.qq.com")
    SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "465"))
    SENDER_EMAIL = os.getenv("EMAIL_SENDER", "403904552@qq.com")
    SMTP_AUTH_CODE = os.getenv("EMAIL_AUTH_CODE", "tsklsaxwaithbiac")
    RECEIVER_EMAIL = os.getenv("EMAIL_RECEIVER", "403904552@qq.com")

def send_email_sync(subject, content):
    """
    同步发送邮件（推荐在单独线程中调用）
    """
    if not EmailConfig.SENDER_EMAIL or not EmailConfig.SMTP_AUTH_CODE:
        logger.warning("⚠️ 邮件配置未完成，跳过邮件发送")
        return False
    
    try:
        message = MIMEText(content, 'plain', 'utf-8')
        # 设置邮件头，显式处理编码防止静默失败
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
        return False

def send_email_async(subject, content):
    """
    异步发送邮件（立即返回，后台线程处理）
    """
    threading.Thread(target=send_email_sync, args=(subject, content), daemon=True).start()

def send_cry_alert_email(filename, confidence, details=None):
    """
    专门发送哭声警报（增强版）
    """
    subject = "🚨 宝宝哭声警报！"
    details_str = "\n".join(details) if details else "无详细模型得分"
    content = (
        f"检测文件: {filename}\n"
        f"置信度: {confidence:.3f}\n"
        f"模型详情:\n{details_str}\n\n"
        "系统已检测到宝宝哭声，并记录到数据库。\n"
        "针对该事件的 Gemini 深度分析通常在事件发生的 5 分钟后（收集完整上下文后）生成。"
    )
    send_email_async(subject, content)
