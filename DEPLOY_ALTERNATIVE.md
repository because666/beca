# 替代部署方案指南

如果您发现 Streamlit Community Cloud 访问困难（即使使用代理），推荐使用以下替代方案。这些方案在中国大陆的访问稳定性通常更好，且依然保持免费。

## 方案一：Hugging Face Spaces (推荐)

Hugging Face 是全球最大的 AI 社区，其 Spaces 服务对中国用户访问较为友好，且完全免费。

### 1. 注册与创建 Space
1.  访问 [Hugging Face](https://huggingface.co/) 并注册账号。
2.  点击右上角头像 -> **New Space**。
3.  填写信息：
    *   **Space Name**: `quant-stock-app` (或任意名称)
    *   **License**: `MIT`
    *   **Select the Space SDK**: 选择 **Streamlit**
    *   **Space Hardware**: 选择 **CPU basic (Free)**
4.  点击 **Create Space**。

### 2. 部署代码
您有两种方式上传代码：

#### 方式 A: 通过网页上传 (最简单)
1.  在创建好的 Space 页面，点击 **Files** 标签页。
2.  点击 **+ Add file** -> **Upload files**。
3.  将本项目文件夹中的所有文件（除了 `.git`, `.venv`, `data` 等不需要的）拖进去。
    *   **必选文件**: `app.py`, `requirements.txt`, `config.py`, `data_fetcher.py`, `ml_models.py`, `backtest.py`, `storage.py`, `help_module.py`, `feedback_module.py`, `backtest_config.json`, `README.md`
4.  点击 **Commit changes to main**。
5.  等待几分钟，点击 **App** 标签页即可看到运行的应用。

#### 方式 B: 通过 Git 推送 (推荐开发者)
1.  在 Space 页面获取 Git 地址（如 `https://huggingface.co/spaces/username/space-name`）。
2.  在本地执行：
    ```bash
    git remote add space https://huggingface.co/spaces/您的用户名/您的Space名
    git push space main
    ```

### 3. 配置环境变量 (可选，用于数据保存)
1.  在 Space 页面，点击 **Settings** 标签页。
2.  滚动到 **Variables and secrets** 部分。
3.  添加 **Secrets** (用于敏感信息):
    *   `SUPABASE_URL`: 您的 Supabase URL
    *   `SUPABASE_KEY`: 您的 Supabase Key
    *   `FEEDBACK_ENCRYPTION_KEY`: 反馈加密密钥
4.  应用会自动重启并加载这些变量。

---

## 方案二：本地部署 + 内网穿透 (速度最快)

如果您有一台可以一直开机的电脑（Windows/Mac/Linux），这是访问速度最快且最稳定的方案。您将在本地运行程序，并通过“隧道”让外网访问。

### 1. 启动本地应用
双击运行 `启动系统.bat` 或在终端运行 `streamlit run app.py`，确保本地可以通过 `http://localhost:8501` 访问。

### 2. 使用 Cloudflare Tunnel (免费且无需公网IP)
1.  下载 `cloudflared` 工具: [下载地址](https://github.com/cloudflare/cloudflared/releases) (Windows下载 `.exe` 版本)。
2.  打开终端 (PowerShell 或 CMD)，运行命令：
    ```bash
    # 假设 cloudflared.exe 在当前目录，且本地应用运行在 8501 端口
    .\cloudflared.exe tunnel --url http://localhost:8501
    ```
3.  终端会显示一个以 `trycloudflare.com` 结尾的链接（例如 `https://funny-name-123.trycloudflare.com`）。
4.  **复制这个链接**，发给任何人即可访问。
    *   *注意：每次重启命令，链接会变。如需固定域名需在 Cloudflare 后台配置（免费）。*

### 3. 使用 Cpolar 或 Natapp (国内服务，速度更优)
如果 Cloudflare 依然慢，可以使用国内的穿透服务：
1.  注册 [Cpolar](https://www.cpolar.com/) (有免费版)。
2.  下载并安装客户端。
3.  运行命令映射端口：
    ```bash
    cpolar http 8501
    ```
4.  复制生成的公网链接即可。

---

## 方案三：Zeabur (亚太节点)

Zeabur 是一家对中文用户友好的云平台，服务器位于亚太地区，访问速度通常优于欧美平台。

1.  访问 [Zeabur](https://zeabur.com/) 并使用 GitHub 登录。
2.  点击 **Create Project** -> **Deploy New Service** -> **GitHub**。
3.  选择您的仓库。
4.  Zeabur 会自动识别 Python 项目。
5.  在 **Settings** -> **Variables** 中添加环境变量（同方案一）。
6.  Zeabur 提供免费额度（每月 $5），对于个人小流量项目通常够用，但需留意额度消耗。

---

## 总结建议

*   **首选**: **Hugging Face Spaces** (稳定、免费、不用开自己电脑)。
*   **备选**: **本地部署 + Cloudflare Tunnel/Cpolar** (如果完全无法访问外网，或者追求极致操作速度)。
