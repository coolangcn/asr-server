import smtplib
from email.mime.text import MIMEText
from email.header import Header
import threading
import logging

logger = logging.getLogger("EmailUtils")

class EmailConfig:
    SMTP_SERVER = "smtp.qq.com"
    SMTP_PORT = 465
    SENDER_EMAIL = "403904552@qq.com"      # 用户 QQ 邮箱
    SMTP_AUTH_CODE = "tsklsaxwaithbiac"    # QQ 邮箱生成的 SMTP 授权码
    RECEIVER_EMAIL = "403904552@qq.com"    # 默认接收人相同

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

def send_cry_alert_email(filename, time_str):
    """
    专门发送哭声警报
    """
    subject = "🚨 宝宝哭声警报！"
    content = (
        f"时间: {time_str}\n"
        f"文件: {filename}\n\n"
        "系统已检测到宝宝哭声，正在开始 5 分钟的观察期以搜集完整分析上下文。\n"
        "分析报告将在 5 分钟后生成并存入系统。"
    )
    send_email_async(subject, content)
