import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from data_fetcher import StockDataFetcher, FeatureEngineer, DataPreprocessor
from ml_models import StockSelectionModel, EnsembleModel, HyperparameterTuner
from backtest import BacktestEngine, PerformanceEvaluator, BenchmarkComparator
import config

def main():
    logger.info("=" * 60)
    logger.info("基于机器学习的量化投资选股系统 - 完整流程")
    logger.info("=" * 60)
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        logger.info("\n[步骤 1/6] 获取股票数据...")
        fetcher = StockDataFetcher()
        df = fetcher.fetch_multiple_stocks(
            config.STOCK_LIST,
            config.DEFAULT_START_DATE,
            config.DEFAULT_END_DATE
        )
        
        if df.empty:
            logger.error("获取股票数据失败")
            return
        
        fetcher.save_data(df, f'stock_data_{timestamp}.csv')
        logger.info(f"成功获取 {len(df)} 条数据")
        
        logger.info("\n[步骤 2/6] 处理数据特征...")
        engineer = FeatureEngineer()
        
        df = engineer.add_technical_indicators(df)
        logger.info("技术指标计算完成")
        
        df = engineer.add_return_features(df)
        logger.info("收益率特征计算完成")
        
        df = engineer.add_target_variable(df, config.PREDICTION_DAYS)
        logger.info("目标变量添加完成")
        
        fetcher.save_data(df, f'processed_data_{timestamp}.csv')
        logger.info(f"处理后数据: {len(df)} 条, {len(df.columns)} 列")
        
        logger.info("\n[步骤 3/6] 准备训练数据...")
        preprocessor = DataPreprocessor()
        
        X, y, feature_cols = engineer.prepare_features(df)
        logger.info(f"特征数量: {len(feature_cols)}")
        
        split_idx = int(len(df) * config.TRAIN_TEST_SPLIT)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        logger.info(f"训练集: {len(X_train)} 条, 测试集: {len(X_test)} 条")
        
        X_train_scaled, X_test_scaled = preprocessor.normalize_features(X_train, X_test)
        logger.info("特征标准化完成")
        
        logger.info("\n[步骤 4/6] 训练机器学习模型...")
        
        model_types = ['random_forest', 'xgboost', 'lightgbm']
        models = {}
        
        for model_type in model_types:
            logger.info(f"\n训练 {model_type} 模型...")
            model = StockSelectionModel(model_type=model_type)
            model.train(X_train_scaled, y_train, X_test_scaled, y_test)
            model.save_model(f'{model_type}_model_{timestamp}.pkl')
            
            metrics = model.evaluate(X_test_scaled, y_test)
            logger.info(f"{model_type} 模型评估结果:")
            logger.info(f"  准确率: {metrics['accuracy']:.4f}")
            logger.info(f"  精确率: {metrics['precision']:.4f}")
            logger.info(f"  召回率: {metrics['recall']:.4f}")
            logger.info(f"  F1分数: {metrics['f1_score']:.4f}")
            if metrics['roc_auc']:
                logger.info(f"  AUC: {metrics['roc_auc']:.4f}")
            
            models[model_type] = model
        
        logger.info("\n[步骤 5/6] 构建集成模型...")
        ensemble = EnsembleModel(voting='soft')
        
        for model_type in model_types:
            ensemble.add_model(models[model_type], weight=1.0)
        
        logger.info("集成模型构建完成")
        
        logger.info("\n[步骤 6/6] 策略回测...")
        backtest_engine = BacktestEngine(
            initial_cash=config.BACKTEST_PARAMS['initial_cash'],
            commission=config.BACKTEST_PARAMS['commission']
        )
        
        test_start_date = test_df['date'].min().strftime('%Y-%m-%d')
        test_end_date = test_df['date'].max().strftime('%Y-%m-%d')
        
        logger.info(f"回测期间: {test_start_date} 至 {test_end_date}")
        
        best_model = models['random_forest']
        backtest_results = backtest_engine.run_backtest(
            df,
            best_model,
            feature_cols,
            start_date=test_start_date,
            end_date=test_end_date
        )
        
        if backtest_results:
            logger.info("\n回测结果:")
            logger.info(f"  总收益率: {backtest_results['total_return']:.2f}%")
            logger.info(f"  最终资金: ¥{backtest_results['final_value']:.2f}")
            logger.info(f"  夏普比率: {backtest_results['sharpe_ratio']:.4f}")
            logger.info(f"  最大回撤: {backtest_results['max_drawdown']:.2f}%")
            logger.info(f"  交易次数: {backtest_results['total_trades']}")
            logger.info(f"  胜率: {backtest_results.get('win_rate', 0):.2f}%")
            
            evaluator = PerformanceEvaluator()
            metrics = evaluator.evaluate_strategy(backtest_results)
            
            report_path = results_dir / f'backtest_report_{timestamp}.txt'
            evaluator.generate_report(str(report_path))
            logger.info(f"\n回测报告已保存至: {report_path}")
            
            portfolio_df = backtest_results['portfolio']
            portfolio_df.to_csv(results_dir / f'portfolio_{timestamp}.csv', index=False, encoding='utf-8-sig')
            
            trades_df = backtest_results['trades']
            trades_df.to_csv(results_dir / f'trades_{timestamp}.csv', index=False, encoding='utf-8-sig')
            
            logger.info(f"回测数据已保存至: {results_dir}")
        
        logger.info("\n" + "=" * 60)
        logger.info("完整流程执行完成！")
        logger.info("=" * 60)
        
        logger.info("\n下一步:")
        logger.info("1. 运行 '启动系统.bat' 或 '启动系统.py' 启动Web界面")
        logger.info("2. 在Web界面中查看详细的分析结果和图表")
        logger.info("3. 使用选股预测功能获取最新的股票推荐")
        
    except Exception as e:
        logger.error(f"执行过程中发生错误: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
