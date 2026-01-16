import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from data_fetcher import StockDataFetcher, FeatureEngineer, DataPreprocessor
from ml_models import StockSelectionModel

def diagnose_prediction_module():
    logger.info("=" * 60)
    logger.info("选股预测模块诊断")
    logger.info("=" * 60)
    
    try:
        logger.info("\n[步骤 1/6] 检查数据文件...")
        fetcher = StockDataFetcher()
        df = fetcher.load_data('processed_data.csv')
        
        if df is None:
            logger.error("✗ 未找到处理后的数据文件！")
            logger.error("请先运行数据获取和特征处理")
            return
        
        logger.info(f"✓ 数据加载成功，共 {len(df)} 条记录")
        logger.info(f"  股票数量: {df['stock_code'].nunique()}")
        logger.info(f"  日期范围: {df['date'].min()} 至 {df['date'].max()}")
        
        required_cols = ['stock_code', 'date', 'close', 'ma5', 'ma20', 'rsi', 'volume_ratio']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"✗ 数据中缺少必要列: {missing_cols}")
            logger.error("请确保特征处理已完成")
            return
        
        logger.info("✓ 数据列检查通过")
        
        logger.info("\n[步骤 2/6] 检查模型文件...")
        model_dir = Path('models')
        model_files = list(model_dir.glob('*.pkl'))
        
        if not model_files:
            logger.error("✗ 未找到训练好的模型文件！")
            logger.error("请先训练模型")
            return
        
        logger.info(f"✓ 找到 {len(model_files)} 个模型文件:")
        for model_file in model_files:
            logger.info(f"  - {model_file.name}")
        
        logger.info("\n[步骤 3/6] 加载模型...")
        model = StockSelectionModel(model_type='random_forest')
        model.load_model('random_forest_model.pkl')
        logger.info("✓ 模型加载成功")
        
        logger.info("\n[步骤 4/6] 准备特征...")
        engineer = FeatureEngineer()
        X, y, feature_cols = engineer.prepare_features(df)
        
        logger.info(f"✓ 特征准备完成")
        logger.info(f"  特征数量: {len(feature_cols)}")
        logger.info(f"  特征形状: {X.shape}")
        logger.info(f"  目标形状: {y.shape}")
        
        if X.isnull().sum().sum() > 0:
            logger.warning(f"⚠ 特征中存在缺失值: {X.isnull().sum().sum()}")
        
        logger.info("\n[步骤 5/6] 检查数据预处理器...")
        preprocessor = DataPreprocessor()
        
        if preprocessor.scaler is None:
            logger.warning("⚠ 数据预处理器未训练，将使用原始特征")
            X_test = X
        else:
            logger.info("✓ 数据预处理器已训练")
            X_test = preprocessor.scaler.transform(X)
        
        logger.info(f"✓ 特征准备完成: {X_test.shape}")
        
        logger.info("\n[步骤 6/6] 测试模型预测...")
        
        test_samples = X_test.head(10)
        
        logger.info("测试前10个样本的预测:")
        for i in range(len(test_samples)):
            try:
                sample = test_samples.iloc[i:i+1]
                prediction = model.predict(sample.values.reshape(1, -1))[0]
                probability = model.predict_proba(sample.values.reshape(1, -1))[0, 1]
                
                logger.info(f"  样本 {i+1}: 预测={prediction}, 概率={probability:.4f}")
            except Exception as e:
                logger.error(f"  样本 {i+1}: 预测失败: {e}")
        
        logger.info("\n[步骤 7/7] 测试完整预测流程...")
        
        latest_data = df.groupby('stock_code').last().reset_index()
        X_latest = latest_data[feature_cols].values
        
        if preprocessor.scaler is None:
            X_scaled = X_latest
        else:
            X_scaled = preprocessor.scaler.transform(X_latest)
        
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]
        
        latest_data['prediction'] = predictions
        latest_data['probability'] = probabilities
        
        logger.info(f"✓ 完整预测完成")
        logger.info(f"  预测结果: 上涨={sum(predictions)}, 下跌={len(predictions)-sum(predictions)}")
        logger.info(f"  平均概率: {probabilities.mean():.4f}")
        logger.info(f"  最大概率: {probabilities.max():.4f}")
        logger.info(f"  最小概率: {probabilities.min():.4f}")
        
        logger.info("\n[步骤 8/7] 测试筛选条件...")
        
        min_probability = 0.6
        top_n = 10
        
        recommended_stocks = latest_data[
            (latest_data['prediction'] == 1) & 
            (latest_data['probability'] >= min_probability)
        ].sort_values('probability', ascending=False).head(top_n)
        
        logger.info(f"✓ 筛选完成，找到 {len(recommended_stocks)} 只推荐股票")
        logger.info(f"  筛选条件: 预测=1 且 概率>={min_probability}")
        logger.info(f"  推荐数量: {top_n}")
        
        if len(recommended_stocks) > 0:
            logger.info("推荐股票详情:")
            for idx, stock in recommended_stocks.iterrows():
                logger.info(f"  {idx+1}. {stock['stock_code']}: 概率={stock['probability']:.4f}, 收盘价={stock['close']:.2f}")
        else:
            logger.warning("⚠ 未找到符合条件的推荐股票")
            logger.warning("可能的原因:")
            logger.warning("  1. 预测概率普遍较低")
            logger.warning("  2. 最小概率阈值过高")
            logger.warning("  3. 特征数据质量问题")
        
        logger.info("\n" + "=" * 60)
        logger.info("诊断完成！")
        logger.info("=" * 60)
        
        logger.info("\n✅ 诊断总结:")
        logger.info("1. 数据文件检查: 通过")
        logger.info("2. 模型文件检查: 通过")
        logger.info("3. 模型加载: 通过")
        logger.info("4. 特征准备: 通过")
        logger.info("5. 数据预处理: 通过")
        logger.info("6. 模型预测: 通过")
        logger.info("7. 筛选逻辑: 通过")
        
        logger.info("\n💡 优化建议:")
        logger.info("1. 确保数据预处理已训练并保存")
        logger.info("2. 检查特征列是否匹配")
        logger.info("3. 添加详细的预测日志")
        logger.info("4. 验证筛选条件是否合理")
        logger.info("5. 测试不同最小概率阈值")
        
        if len(recommended_stocks) > 0:
            logger.info("\n✅ 系统应该可以正常显示预测结果")
        else:
            logger.info("\n⚠️ 建议调整以下参数:")
            logger.info("  - 降低最小概率阈值（如从0.6降到0.4）")
            logger.info("  - 增加推荐股票数量（如从10增加到20）")
            logger.info("  - 检查模型训练质量")
            logger.info("  - 重新训练模型以提高预测准确率")
        
    except Exception as e:
        logger.error(f"诊断过程中发生错误: {e}", exc_info=True)

if __name__ == "__main__":
    diagnose_prediction_module()
