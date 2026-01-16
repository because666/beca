import sys
import os
import platform
import subprocess
import logging
import time
from pathlib import Path
from datetime import datetime

# 配置日志
LOG_FILE = "startup.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def print_header():
    print("=" * 60)
    print("        基于机器学习的量化投资选股系统 - 启动程序")
    print("=" * 60)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {sys.version.split()[0]}")
    print("-" * 60)

def check_python_version():
    logger.info("正在检查Python环境...")
    if sys.version_info < (3, 8):
        logger.error("Python版本过低，需要Python 3.8或更高版本")
        return False
    return True

def check_dependencies():
    logger.info("正在检查系统依赖...")
    req_file = Path("requirements.txt")
    if not req_file.exists():
        logger.error("找不到 requirements.txt 文件")
        return False
    
    # 尝试导入关键包来检测是否需要安装
    required_packages = ['streamlit', 'pandas', 'numpy', 'scikit-learn', 'plotly']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
            
    if missing_packages:
        logger.warning(f"发现缺失依赖: {', '.join(missing_packages)}")
        logger.info("正在自动安装缺失依赖，这可能需要几分钟...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            logger.info("依赖安装完成")
        except subprocess.CalledProcessError as e:
            logger.error(f"依赖安装失败: {e}")
            return False
    else:
        logger.info("核心依赖检查通过")
    
    return True

def check_env_file():
    logger.info("检查环境配置...")
    env_file = Path(".env")
    if not env_file.exists():
        logger.warning("未找到 .env 配置文件，将使用默认设置")
        # 可以选择自动创建 .env 模板
        # with open(".env", "w") as f:
        #     f.write("SMTP_SENDER_EMAIL=\nSMTP_SENDER_PASSWORD=\n")
    else:
        logger.info("配置环境加载正常")

def find_available_port(start_port=8501, max_attempts=10):
    import socket
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
    return None

def run_application(port=8501):
    logger.info("正在启动应用程序...")
    app_path = Path("app.py")
    if not app_path.exists():
        logger.error("找不到 app.py 文件")
        return False
    
    # 自动寻找可用端口
    available_port = find_available_port(port)
    if available_port is None:
        logger.error(f"无法找到可用端口 (尝试范围: {port}-{port+10})")
        return False
    
    if available_port != port:
        logger.warning(f"端口 {port} 被占用，自动切换到端口 {available_port}")
        port = available_port
    
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(port)]
    
    logger.info(f"执行命令: {' '.join(cmd)}")
    print("\n" + "*" * 40)
    print(f"系统即将启动，请在浏览器中访问: http://localhost:{port}")
    print("按 Ctrl+C 可停止运行")
    print("*" * 40 + "\n")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("用户停止了程序")
    except Exception as e:
        logger.error(f"程序运行出错: {e}")
        return False
    
    return True

def main():
    # 确保在脚本所在目录运行
    os.chdir(Path(__file__).parent)
    
    print_header()
    
    if not check_python_version():
        input("按回车键退出...")
        sys.exit(1)
        
    if not check_dependencies():
        input("按回车键退出...")
        sys.exit(1)
        
    check_env_file()
    
    # 允许通过命令行参数指定端口 (简单的参数解析)
    port = 8501
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
        
    if not run_application(port):
        input("按回车键退出...")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"发生未预期的错误: {e}", exc_info=True)
        input("按回车键退出...")
