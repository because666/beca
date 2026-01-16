# 项目部署指南 (Streamlit Cloud + Supabase)

本指南介绍如何将该量化选股系统部署到 **Streamlit Community Cloud**，并使用 **Supabase** 进行数据持久化，实现完全免费的云端运行方案。

## 1. 部署架构方案

由于本项目是基于 Streamlit 的动态 Python 应用（需要后端 Python 运行时进行模型推理和数据获取），传统的静态托管平台（如 Vercel/GitHub Pages）无法直接运行。因此，我们采用以下最佳免费方案：

*   **应用托管**: [Streamlit Community Cloud](https://streamlit.io/cloud)
    *   **优势**: 专为 Streamlit 设计，永久免费，支持 Python 环境，一键部署。
    *   **限制**: 容器会休眠，文件系统是临时的（重启后文件丢失）。
*   **数据存储**: [Supabase](https://supabase.com/) (PostgreSQL)
    *   **优势**: 永久免费套餐 (500MB 数据库)，提供 Python 客户端，用于存储用户配置和回测参数。
*   **CDN & HTTPS**: 由 Streamlit Cloud 内置提供全球加速和 SSL 证书。

## 2. 准备工作

### 2.1 注册 Supabase 并创建项目
1.  访问 [Supabase](https://supabase.com/) 并注册账号。
2.  点击 "New Project"，创建一个新项目（选择离你最近的节点，如 Singapore 或 Tokyo）。
3.  设置数据库密码并等待项目初始化完成。
4.  进入项目 Dashboard，点击左侧 "Table Editor"，创建一个新表：
    *   Name: `app_storage`
    *   Name: `app_storage`
    *   Columns:
        *   `key` (Text, Primary Key)
        *   `value` (JSONB)
    *   保存创建。
    *   **(可选) 创建反馈表**: 点击 "New Table"，Name: `feedbacks`
        *   Columns:
            *   `id` (int8, Primary Key, Identity)
            *   `type` (text)
            *   `content` (text)
            *   `contact` (text)
            *   `timestamp` (text)
            *   `ip_hash` (text)
            *   `status` (text)
            *   `admin_reply` (text)
5.  进入 "Project Settings" -> "API"，获取以下信息：
    *   **Project URL** (`SUPABASE_URL`)
    *   **Project API keys** (`service_role` secret 或 `anon` public key，建议使用 `service_role` 以便读写，但在生产环境中应注意权限。对于个人项目，`anon` key 配合开启 RLS (Row Level Security) 策略或者直接使用 `service_role` key (注意保密))。

### 2.2 准备代码库
1.  确保代码已上传到 GitHub 仓库（Streamlit Cloud 从 GitHub 拉取代码）。
2.  项目中已包含 `requirements.txt` 和 `storage.py`。
3.  **(重要)** 如果使用反馈功能，请生成一个随机密钥并在部署时配置。

## 3. 部署步骤

1.  访问 [Streamlit Community Cloud](https://share.streamlit.io/) 并使用 GitHub 登录。
2.  点击 "New app"。
3.  选择你的 GitHub 仓库、分支（如 `main`）和主文件路径（`app.py`）。
4.  **关键步骤**: 点击 "Advanced settings..." 配置环境变量。
    *   在 "Secrets" 输入框中添加以下内容：
        ```toml
        SUPABASE_URL = "你的Supabase项目URL"
        SUPABASE_KEY = "你的Supabase API Key"
        GA_TRACKING_ID = "你的Google Analytics ID (可选)"
        FEEDBACK_ENCRYPTION_KEY = "生成的随机密钥 (可选，用于加密反馈数据)"
        SMTP_SENDER_EMAIL = "发送邮件的邮箱 (可选)"
        SMTP_SENDER_PASSWORD = "邮箱授权码 (可选)"
        ```
    *   可以使用 Python 生成密钥: `from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())`
5.  点击 "Deploy"。

## 4. 成本控制与监控

*   **Streamlit Cloud**: 免费版无固定成本。如果资源使用超限（1GB RAM），应用会重启。建议在处理大数据量时监控内存。
*   **Supabase**: 免费版包含 500MB 数据库存储和 2GB 带宽。对于配置文件的存储绰绰有余。
    *   **监控**: 在 Supabase Dashboard 中可以查看数据库用量。设置 "Spend Cap" 为开启（默认开启），防止意外产生费用。
*   **流量**: Streamlit Cloud 不限流量，适合个人或小团队使用。

## 5. 推广与优化

*   **SEO**: `app.py` 中已配置页面标题和图标。
*   **Analytics**: 在 Secrets 中配置 `GA_TRACKING_ID` 即可开启 Google Analytics 访问统计。
*   **性能**:
    *   项目已使用 Streamlit 的缓存机制 (`@st.cache_data`) 优化数据加载。
    *   Supabase 仅用于存取轻量级配置，不影响主流程性能。

## 6. 常见问题

*   **Q: 为什么不用 Vercel?**
    *   A: Vercel 主要用于静态网站和 Node.js 函数。虽然支持 Python Runtime，但对 `pandas`, `numpy`, `sklearn` 等大型科学计算库的支持有限（包体积限制），且不支持 Streamlit 的长连接特性。
*   **Q: 数据会丢失吗？**
    *   A: 用户设置和回测配置已通过 Supabase 持久化，不会丢失。但 `data/stock_data.csv` 等缓存文件在应用重启后会重置，需要重新点击"获取数据"（这也是为了节省云端存储空间）。

---
**现在，请将代码推送到 GitHub 并按照上述步骤部署！**
