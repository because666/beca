import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from data_fetcher import StockDataFetcher, FeatureEngineer, DataPreprocessor
from ml_models import StockSelectionModel
from backtest import BacktestEngine
import config

def diagnose_backtest():
    logger.info("=" * 60)
    logger.info("回测系统诊断工具")
    logger.info("=" * 60)
    
    try:
        logger.info("\n[步骤 1/6] 检查数据文件...")
        fetcher = StockDataFetcher()
        
        df = fetcher.load_data('processed_data.csv')
        if df is None:
            logger.error("未找到处理后的数据文件！请先运行数据获取和特征处理。")
            return
        
        logger.info(f"✓ 数据加载成功，共 {len(df)} 条记录")
        logger.info(f"  股票数量: {df['stock_code'].nunique()}")
        logger.info(f"  日期范围: {df['date'].min()} 至 {df['date'].max()}")
        logger.info(f"  列数量: {len(df.columns)}")
        
        if 'target' not in df.columns:
            logger.error("✗ 数据中缺少 'target' 列！请先运行特征处理。")
            return
        
        logger.info(f"  目标分布: 上涨={len(df[df['target']==1])}, 下跌={len(df[df['target']==0])}")
        logger.info(f"  上涨比例: {df['target'].mean():.2%}")
        
        logger.info("\n[步骤 2/6] 检查特征数据...")
        engineer = FeatureEngineer()
        X, y, feature_cols = engineer.prepare_features(df)
        
        logger.info(f"✓ 特征准备完成")
        logger.info(f"  特征数量: {len(feature_cols)}")
        logger.info(f"  特征形状: {X.shape}")
        logger.info(f"  目标形状: {y.shape}")
        logger.info(f"  缺失值数量: {X.isnull().sum().sum()}")
        
        if X.isnull().sum().sum() > 0:
            logger.warning("⚠ 特征中存在缺失值")
            missing_cols = X.columns[X.isnull().any()].tolist()
            logger.warning(f"  缺失值列: {missing_cols}")
        
        logger.info("\n[步骤 3/6] 检查模型文件...")
        model_dir = Path('models')
        model_files = list(model_dir.glob('*.pkl'))
        
        if not model_files:
            logger.error("✗ 未找到训练好的模型文件！请先训练模型。")
            return
        
        logger.info(f"✓ 找到 {len(model_files)} 个模型文件:")
        for model_file in model_files:
            logger.info(f"  - {model_file.name}")
        
        logger.info("\n[步骤 4/6] 加载模型并测试预测...")
        model = StockSelectionModel(model_type='random_forest')
        model.load_model('random_forest_model.pkl')
        
        test_sample = X.iloc[:100]
        predictions = model.predict(test_sample)
        probabilities = model.predict_proba(test_sample)[:, 1]
        
        logger.info(f"✓ 模型预测测试完成")
        logger.info(f"  预测结果: 上涨={sum(predictions)}, 下跌={len(predictions)-sum(predictions)}")
        logger.info(f"  上涨比例: {predictions.mean():.2%}")
        logger.info(f"  平均概率: {probabilities.mean():.4f}")
        logger.info(f"  最大概率: {probabilities.max():.4f}")
        logger.info(f"  最小概率: {probabilities.min():.4f}")
        logger.info(f"  概率>0.6: {sum(probabilities > 0.6)}")
        logger.info(f"  概率>0.5: {sum(probabilities > 0.5)}")
        
        high_prob_count = sum(probabilities > 0.6)
        if high_prob_count == 0:
            logger.warning("⚠ 没有预测概率大于0.6的样本！")
            logger.warning("  这可能导致回测中没有买入信号")
            logger.warning("  建议:")
            logger.warning("    1. 降低买入概率阈值（如改为0.5）")
            logger.warning("    2. 重新训练模型")
            logger.warning("    3. 检查特征数据质量")
        
        logger.info("\n[步骤 5/6] 模拟回测买入条件...")
        df_sample = df.head(1000).copy()
        
        buy_signals = 0
        for idx, row in df_sample.iterrows():
            try:
                features = row[feature_cols].values.reshape(1, -1)
                prediction = model.predict(features)[0]
                probability = model.predict_proba(features)[0, 1]
                
                if prediction == 1 and probability > 0.6:
                    buy_signals += 1
            except Exception as e:
                logger.warning(f"  第 {idx} 行预测失败: {e}")
        
        logger.info(f"✓ 买入信号统计:")
        logger.info(f"  测试样本数: {len(df_sample)}")
        logger.info(f"  买入信号数: {buy_signals}")
        logger.info(f"  买入信号率: {buy_signals/len(df_sample):.2%}")
        
        if buy_signals == 0:
            logger.error("✗ 在测试样本中没有找到任何买入信号！")
            logger.error("  这是回测没有交易的根本原因")
            logger.error("\n  可能的原因:")
            logger.error("    1. 模型预测概率普遍较低")
            logger.error("    2. 买入条件过于严格（probability > 0.6）")
            logger.error("    3. 特征数据存在问题")
            logger.error("    4. 模型训练数据不足或质量差")
            logger.error("\n  建议的解决方案:")
            logger.error("    1. 降低买入概率阈值到0.5或更低")
            logger.error("    2. 使用更多的历史数据训练模型")
            logger.error("    3. 尝试不同的模型类型")
            logger.error("    4. 优化特征工程")
        
        logger.info("\n[步骤 6/6] 检查回测参数...")
        logger.info(f"  初始资金: ¥{config.BACKTEST_PARAMS['initial_cash']:,}")
        logger.info(f"  手续费率: {config.BACKTEST_PARAMS['commission']:.4f}")
        logger.info(f"  滑点: {config.BACKTEST_PARAMS['slippage']:.4f}")
        logger.info(f"  买入概率阈值: 0.5 (默认)")
        logger.info(f"  卖出概率阈值: 0.5 (默认)")
        logger.info(f"  止损阈值: 0.1 (默认)")
        logger.info(f"  最大持仓天数: 5 (默认)")
        logger.info(f"  单只股票最大仓位: 20% (默认)")
        logger.info(f"  最大持仓数量: 5 (默认)")
        
        logger.info("\n[步骤 7/7] 模拟不同阈值下的买入信号...")
        test_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        df_sample = df.head(1000).copy()
        
        logger.info("  不同阈值下的买入信号统计:")
        for threshold in test_thresholds:
            buy_signals = 0
            for idx, row in df_sample.iterrows():
                try:
                    features = row[feature_cols].values.reshape(1, -1)
                    prediction = model.predict(features)[0]
                    probability = model.predict_proba(features)[0, 1]
                    
                    if prediction == 1 and probability > threshold:
                        buy_signals += 1
                except Exception as e:
                    pass
            
            logger.info(f"    阈值 {threshold:.1f}: {buy_signals} 个买入信号 ({buy_signals/len(df_sample):.1%})")
        
        logger.info("\n" + "=" * 60)
        logger.info("诊断完成！")
        logger.info("=" * 60)
        
        if buy_signals == 0:
            logger.info("\n🔧 修复建议:")
            logger.info("1. 修改 app.py 中的买入概率阈值参数")
            logger.info("2. 降低买入概率阈值到0.4或更低")
            logger.info("3. 增加回测日期范围")
            logger.info("4. 重新训练模型以提高预测准确率")
            logger.info("5. 检查特征数据质量")
        else:
            logger.info("\n✓ 系统检测正常，应该可以产生交易")
            logger.info("如果回测仍然没有交易，请检查:")
            logger.info("  - 回测日期范围是否包含有效数据")
            logger.info("  - 资金是否足够进行交易")
            logger.info("  - 查看调试日志了解详细情况")
        
    except Exception as e:
        logger.error(f"诊断过程中发生错误: {e}", exc_info=True)

if __name__ == "__main__":
    diagnose_backtest()
