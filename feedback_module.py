import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import hashlib
import logging
import os
import base64
from pathlib import Path
from storage import storage

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 常量定义
DB_FILE = "feedback.db"
KEY_FILE = "secret.key"
ADMIN_EMAIL = "3694224048@qq.com"
DAILY_LIMIT = 5

class FeedbackManager:
    def __init__(self):
        self.key = self._load_or_create_key()
        self.cipher = Fernet(self.key)
        self.use_supabase = storage.client is not None
        
        if not self.use_supabase:
            self._init_sqlite()

    def _load_or_create_key(self):
        """加载或创建加密密钥"""
        # 1. Try env var (for Cloud)
        env_key = os.environ.get("FEEDBACK_ENCRYPTION_KEY")
        if env_key:
            return env_key.encode()

        # 2. Try local file
        if os.path.exists(KEY_FILE):
            with open(KEY_FILE, "rb") as key_file:
                return key_file.read()
        else:
            key = Fernet.generate_key()
            try:
                with open(KEY_FILE, "wb") as key_file:
                    key_file.write(key)
            except Exception:
                pass # Might be read-only filesystem
            return key

    def _init_sqlite(self):
        """初始化SQLite数据库"""
        try:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS feedbacks
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          type TEXT,
                          content BLOB,
                          contact BLOB,
                          timestamp DATETIME,
                          ip_hash TEXT,
                          status TEXT DEFAULT 'pending',
                          admin_reply BLOB)''')
            
            # Check if new columns exist, if not add them
            c.execute("PRAGMA table_info(feedbacks)")
            columns = [column[1] for column in c.fetchall()]
            
            if 'status' not in columns:
                c.execute("ALTER TABLE feedbacks ADD COLUMN status TEXT DEFAULT 'pending'")
            if 'admin_reply' not in columns:
                c.execute("ALTER TABLE feedbacks ADD COLUMN admin_reply BLOB")
                
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"SQLite init failed (might be ephemeral fs): {e}")

    def _encrypt(self, text):
        """加密文本"""
        if not text:
            return b""
        return self.cipher.encrypt(text.encode())

    def _decrypt(self, ciphertext):
        """解密文本"""
        if not ciphertext:
            return ""
        if isinstance(ciphertext, str):
            # If loaded from JSON (Supabase), it might be base64 string
            try:
                ciphertext = base64.b64decode(ciphertext)
            except:
                pass
        return self.cipher.decrypt(ciphertext).decode()

    def _get_ip_hash(self):
        """获取用户IP的哈希值（模拟）"""
        user_id = "unknown"
        if 'user_id' not in st.session_state:
            st.session_state['user_id'] = hashlib.sha256(os.urandom(32)).hexdigest()
        user_id = st.session_state['user_id']
        return hashlib.sha256(user_id.encode()).hexdigest()

    def check_rate_limit(self):
        """检查每日提交限制"""
        ip_hash = self._get_ip_hash()
        one_day_ago = (datetime.now() - timedelta(days=1)).isoformat()
        
        if self.use_supabase:
            try:
                response = storage.client.table("feedbacks")\
                    .select("count", count="exact")\
                    .eq("ip_hash", ip_hash)\
                    .gt("timestamp", one_day_ago)\
                    .execute()
                return response.count < DAILY_LIMIT
            except Exception as e:
                logger.error(f"Rate limit check failed: {e}")
                return True # Fail open
        else:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("SELECT count(*) FROM feedbacks WHERE ip_hash=? AND timestamp > ?", 
                      (ip_hash, one_day_ago))
            count = c.fetchone()[0]
            conn.close()
            return count < DAILY_LIMIT

    def submit_feedback(self, fb_type, content, contact):
        """提交反馈"""
        if not self.check_rate_limit():
            return False, "您今日的反馈次数已达上限，请明天再试。"

        try:
            encrypted_content = self._encrypt(content)
            encrypted_contact = self._encrypt(contact) if contact else b""
            ip_hash = self._get_ip_hash()
            timestamp = datetime.now().isoformat()
            
            # Prepare data
            # Store bytes as base64 string for JSON compatibility
            content_b64 = base64.b64encode(encrypted_content).decode('utf-8')
            contact_b64 = base64.b64encode(encrypted_contact).decode('utf-8') if contact else ""

            if self.use_supabase:
                data = {
                    "type": fb_type,
                    "content": content_b64,
                    "contact": contact_b64,
                    "timestamp": timestamp,
                    "ip_hash": ip_hash,
                    "status": "pending"
                }
                storage.client.table("feedbacks").insert(data).execute()
            else:
                conn = sqlite3.connect(DB_FILE)
                c = conn.cursor()
                c.execute("INSERT INTO feedbacks (type, content, contact, timestamp, ip_hash) VALUES (?, ?, ?, ?, ?)",
                          (fb_type, encrypted_content, encrypted_contact, timestamp, ip_hash))
                conn.commit()
                conn.close()
            
            self._send_email_notification(fb_type, content)
            return True, "反馈提交成功！我们会尽快处理。"
        except Exception as e:
            logger.error(f"提交反馈失败: {e}")
            return False, f"提交失败: {str(e)}"

    def _send_email_notification(self, fb_type, content):
        """发送邮件通知"""
        smtp_server = "smtp.qq.com"
        smtp_port = 465
        default_sender = "3694224048@qq.com"
        sender_email = os.getenv("SMTP_SENDER_EMAIL", default_sender) 
        sender_password = os.getenv("SMTP_SENDER_PASSWORD", "")
        
        if not sender_password:
            try:
                from dotenv import load_dotenv
                load_dotenv()
                sender_password = os.getenv("SMTP_SENDER_PASSWORD", "")
            except ImportError:
                pass
        
        if not sender_email or not sender_password:
            return

        try:
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = ADMIN_EMAIL
            msg['Subject'] = f"【新反馈】{fb_type}"
            
            body = f"""
            收到新的用户反馈：
            类型：{fb_type}
            时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            内容：
            {content}
            """
            msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
                server.login(sender_email, sender_password)
                server.send_message(msg)
        except Exception as e:
            logger.error(f"邮件发送失败: {e}")

    def cleanup_old_data(self, days=7):
        """清理旧数据"""
        # Implement cleanup logic if needed
        pass

    def get_user_feedbacks(self):
        """获取当前用户的所有反馈"""
        ip_hash = self._get_ip_hash()
        
        rows = []
        if self.use_supabase:
            try:
                response = storage.client.table("feedbacks")\
                    .select("*")\
                    .eq("ip_hash", ip_hash)\
                    .order("timestamp", desc=True)\
                    .execute()
                rows = response.data
            except Exception as e:
                logger.error(f"Fetch feedbacks failed: {e}")
        else:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("SELECT id, type, content, timestamp, status, admin_reply FROM feedbacks WHERE ip_hash=? ORDER BY timestamp DESC", (ip_hash,))
            sqlite_rows = c.fetchall()
            conn.close()
            # Convert to dict
            for r in sqlite_rows:
                rows.append({
                    "id": r[0], "type": r[1], "content": r[2], "timestamp": r[3], "status": r[4], "admin_reply": r[5]
                })

        decrypted_rows = []
        for row in rows:
            try:
                # Handle SQLite blob vs Supabase b64 string
                content_raw = row.get("content")
                reply_raw = row.get("admin_reply")
                
                # If Supabase (string), decode to bytes for _decrypt? 
                # _decrypt now handles base64 string decoding internally
                
                content = self._decrypt(content_raw)
                reply = self._decrypt(reply_raw) if reply_raw else ""
                
                decrypted_rows.append({
                    "ID": row.get("id"),
                    "类型": row.get("type"),
                    "内容": content,
                    "时间": row.get("timestamp"),
                    "状态": row.get("status"),
                    "管理员回复": reply
                })
            except Exception:
                continue
                
        return pd.DataFrame(decrypted_rows)

    def get_all_feedbacks(self):
        """获取所有反馈（仅限管理员解密查看）"""
        rows = []
        if self.use_supabase:
            try:
                response = storage.client.table("feedbacks").select("*").order("timestamp", desc=True).execute()
                rows = response.data
            except Exception as e:
                logger.error(f"Fetch all feedbacks failed: {e}")
        else:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("SELECT id, type, content, contact, timestamp, status, admin_reply FROM feedbacks ORDER BY timestamp DESC")
            sqlite_rows = c.fetchall()
            conn.close()
            for r in sqlite_rows:
                rows.append({
                    "id": r[0], "type": r[1], "content": r[2], "contact": r[3], "timestamp": r[4], "status": r[5], "admin_reply": r[6]
                })

        decrypted_rows = []
        for row in rows:
            try:
                content = self._decrypt(row.get("content"))
                contact = self._decrypt(row.get("contact"))
                reply = self._decrypt(row.get("admin_reply")) if row.get("admin_reply") else ""
                
                decrypted_rows.append({
                    "ID": row.get("id"),
                    "类型": row.get("type"),
                    "内容": content,
                    "联系方式": contact,
                    "时间": row.get("timestamp"),
                    "状态": row.get("status"),
                    "回复内容": reply
                })
            except Exception:
                continue
                
        return pd.DataFrame(decrypted_rows)

    def update_feedback_status(self, feedback_id, status, reply=None):
        """更新反馈状态和回复"""
        try:
            encrypted_reply = self._encrypt(reply) if reply else None
            # Store as b64 string if supabase
            reply_val = encrypted_reply
            if self.use_supabase and reply_val:
                reply_val = base64.b64encode(reply_val).decode('utf-8')
            
            if self.use_supabase:
                data = {"status": status}
                if reply:
                    data["admin_reply"] = reply_val
                storage.client.table("feedbacks").update(data).eq("id", feedback_id).execute()
            else:
                conn = sqlite3.connect(DB_FILE)
                c = conn.cursor()
                if reply:
                    c.execute("UPDATE feedbacks SET status=?, admin_reply=? WHERE id=?", (status, encrypted_reply, feedback_id))
                else:
                    c.execute("UPDATE feedbacks SET status=? WHERE id=?", (status, feedback_id))
                conn.commit()
                conn.close()
            return True
        except Exception as e:
            logger.error(f"更新反馈失败: {e}")
            return False

def show_feedback_page():
    st.header("🛡️ 安全反馈中心")
    st.info("我们非常重视您的反馈与隐私。所有提交的内容均经过加密处理，仅管理员可见。")
    
    manager = FeedbackManager()
    
    tab1, tab2 = st.tabs(["✍️ 提交反馈", "📋 我的反馈记录"])

    with tab1:
        st.subheader("提交新反馈")
        with st.form("feedback_form"):
            col1, col2 = st.columns(2)
            with col1:
                fb_type = st.selectbox("反馈类型", ["功能建议", "错误报告", "商务合作", "其他"])
            with col2:
                contact = st.text_input("联系方式 (选填)", placeholder="邮箱或QQ，仅用于回复")
            content = st.text_area("反馈内容 (必填)", height=150)
            
            # Simple Captcha
            if 'captcha_num1' not in st.session_state:
                st.session_state['captcha_num1'] = 3
                st.session_state['captcha_num2'] = 5
                
            captcha_answer = st.number_input(f"验证: {st.session_state['captcha_num1']} + {st.session_state['captcha_num2']} = ?", min_value=0)
            
            submitted = st.form_submit_button("🔒 安全提交")
            
            if submitted:
                if not content.strip():
                    st.error("请输入反馈内容")
                elif captcha_answer != st.session_state['captcha_num1'] + st.session_state['captcha_num2']:
                    st.error("验证码错误")
                else:
                    success, message = manager.submit_feedback(fb_type, content, contact)
                    if success:
                        st.success(message)
                    else:
                        st.warning(message)

    with tab2:
        st.subheader("我的历史反馈")
        user_feedbacks = manager.get_user_feedbacks()
        if not user_feedbacks.empty:
            st.dataframe(user_feedbacks)
        else:
            st.info("您还没有提交过反馈记录")

if __name__ == "__main__":
    show_feedback_page()
