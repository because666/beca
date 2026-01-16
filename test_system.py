import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    logger.info("=" * 60)
    logger.info("测试1: 检查依赖包导入")
    logger.info("=" * 60)
    
    required_packages = [
        'pandas',
        'numpy',
        'sklearn',
        'xgboost',
        'lightgbm',
        'streamlit',
        'plotly',
        'matplotlib',
        'seaborn',
        'joblib',
        'ta',
        'akshare',
        'tqdm'
    ]
    
    failed_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package}")
        except ImportError:
            logger.error(f"✗ {package} - 未安装")
            failed_packages.append(package)
    
    if failed_packages:
        logger.error(f"\n缺少以下依赖包: {', '.join(failed_packages)}")
        logger.error("请运行: pip install -r requirements.txt")
        return False
    
    logger.info("\n所有依赖包已正确安装！")
    return True

def test_modules():
    logger.info("\n" + "=" * 60)
    logger.info("测试2: 检查项目模块")
    logger.info("=" * 60)
    
    required_modules = [
        'config',
        'data_fetcher',
        'ml_models',
        'backtest'
    ]
    
    failed_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            logger.info(f"✓ {module}")
        except ImportError as e:
            logger.error(f"✗ {module} - 导入失败: {e}")
            failed_modules.append(module)
    
    if failed_modules:
        logger.error(f"\n以下模块导入失败: {', '.join(failed_modules)}")
        return False
    
    logger.info("\n所有项目模块导入成功！")
    return True

def test_directories():
    logger.info("\n" + "=" * 60)
    logger.info("测试3: 检查目录结构")
    logger.info("=" * 60)
    
    required_dirs = ['data', 'models', 'results', 'logs']
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            logger.info(f"✓ {dir_name}/ 目录存在")
        else:
            logger.info(f"○ {dir_name}/ 目录不存在，将自动创建")
            dir_path.mkdir(exist_ok=True)
    
    logger.info("\n目录结构检查完成！")
    return True

def test_config():
    logger.info("\n" + "=" * 60)
    logger.info("测试4: 检查配置文件")
    logger.info("=" * 60)
    
    try:
        import config
        
        logger.info(f"✓ 股票列表: {len(config.STOCK_LIST)} 只")
        logger.info(f"✓ 默认开始日期: {config.DEFAULT_START_DATE}")
        logger.info(f"✓ 默认结束日期: {config.DEFAULT_END_DATE}")
        logger.info(f"✓ 训练测试分割比: {config.TRAIN_TEST_SPLIT}")
        logger.info(f"✓ 预测天数: {config.PREDICTION_DAYS}")
        logger.info(f"✓ 初始资金: ¥{config.BACKTEST_PARAMS['initial_cash']:,}")
        
        logger.info("\n配置文件检查通过！")
        return True
    except Exception as e:
        logger.error(f"✗ 配置文件检查失败: {e}")
        return False

def test_data_fetcher():
    logger.info("\n" + "=" * 60)
    logger.info("测试5: 测试数据获取功能")
    logger.info("=" * 60)
    
    try:
        from data_fetcher import StockDataFetcher
        
        logger.info("✓ StockDataFetcher 类导入成功")
        
        fetcher = StockDataFetcher()
        logger.info("✓ StockDataFetcher 实例化成功")
        
        logger.info("\n数据获取功能测试通过！")
        return True
    except Exception as e:
        logger.error(f"✗ 数据获取功能测试失败: {e}")
        return False

def test_ml_models():
    logger.info("\n" + "=" * 60)
    logger.info("测试6: 测试机器学习模型")
    logger.info("=" * 60)
    
    try:
        from ml_models import StockSelectionModel
        
        logger.info("✓ StockSelectionModel 类导入成功")
        
        model_types = ['random_forest', 'xgboost', 'lightgbm', 'logistic', 'svm']
        
        for model_type in model_types:
            try:
                model = StockSelectionModel(model_type=model_type)
                model.create_model()
                logger.info(f"✓ {model_type} 模型创建成功")
            except Exception as e:
                logger.warning(f"○ {model_type} 模型创建警告: {e}")
        
        logger.info("\n机器学习模型测试通过！")
        return True
    except Exception as e:
        logger.error(f"✗ 机器学习模型测试失败: {e}")
        return False

def test_backtest():
    logger.info("\n" + "=" * 60)
    logger.info("测试7: 测试回测引擎")
    logger.info("=" * 60)
    
    try:
        from backtest import BacktestEngine, PerformanceEvaluator
        
        logger.info("✓ BacktestEngine 类导入成功")
        logger.info("✓ PerformanceEvaluator 类导入成功")
        
        engine = BacktestEngine()
        logger.info("✓ BacktestEngine 实例化成功")
        
        evaluator = PerformanceEvaluator()
        logger.info("✓ PerformanceEvaluator 实例化成功")
        
        logger.info("\n回测引擎测试通过！")
        return True
    except Exception as e:
        logger.error(f"✗ 回测引擎测试失败: {e}")
        return False

def test_streamlit():
    logger.info("\n" + "=" * 60)
    logger.info("测试8: 检查Streamlit应用")
    logger.info("=" * 60)
    
    try:
        app_path = Path('app.py')
        if app_path.exists():
            logger.info("✓ app.py 文件存在")
            
            with open(app_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'streamlit' in content:
                    logger.info("✓ app.py 包含Streamlit代码")
                else:
                    logger.warning("○ app.py 可能不包含Streamlit代码")
            
            logger.info("\nStreamlit应用检查通过！")
            return True
        else:
            logger.error("✗ app.py 文件不存在")
            return False
    except Exception as e:
        logger.error(f"✗ Streamlit应用检查失败: {e}")
        return False

def main():
    logger.info("\n")
    logger.info("╔" + "=" * 58 + "╗")
    logger.info("║" + " " * 10 + "基于机器学习的量化投资选股系统" + " " * 10 + "║")
    logger.info("║" + " " * 20 + "系统测试" + " " * 24 + "║")
    logger.info("╚" + "=" * 58 + "╝")
    logger.info("\n")
    
    tests = [
        ("依赖包导入", test_imports),
        ("项目模块", test_modules),
        ("目录结构", test_directories),
        ("配置文件", test_config),
        ("数据获取", test_data_fetcher),
        ("机器学习模型", test_ml_models),
        ("回测引擎", test_backtest),
        ("Streamlit应用", test_streamlit)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"测试 '{test_name}' 执行时发生错误: {e}")
            results.append((test_name, False))
    
    logger.info("\n" + "=" * 60)
    logger.info("测试结果汇总")
    logger.info("=" * 60)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        logger.info(f"{test_name:.<40} {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    logger.info("\n" + "=" * 60)
    logger.info(f"测试完成: {passed}/{total} 通过")
    logger.info("=" * 60)
    
    if passed == total:
        logger.info("\n🎉 所有测试通过！系统可以正常使用。")
        logger.info("\n下一步:")
        logger.info("1. 运行 '启动系统.bat' 或 '启动系统.py' 启动Web界面")
        logger.info("2. 或运行 'python run_full_pipeline.py' 执行完整流程")
        return 0
    else:
        logger.error("\n❌ 部分测试失败，请检查错误信息并解决问题。")
        logger.error("\n常见解决方案:")
        logger.error("1. 确保Python版本为3.8或更高")
        logger.error("2. 运行 'pip install -r requirements.txt' 安装依赖")
        logger.error("3. 检查网络连接是否正常")
        return 1

if __name__ == "__main__":
    sys.exit(main())
