import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from backtest import BacktestEngine
from data_fetcher import StockDataFetcher, FeatureEngineer
from ml_models import StockSelectionModel

def test_backtest_logic():
    logger.info("=" * 60)
    logger.info("回测逻辑测试")
    logger.info("=" * 60)
    
    try:
        logger.info("\n[测试 1/5] 检查数据...")
        fetcher = StockDataFetcher()
        df = fetcher.load_data('processed_data.csv')
        
        if df is None:
            logger.error("未找到处理后的数据！")
            return
        
        logger.info(f"✓ 数据加载成功: {len(df)} 条记录")
        
        logger.info("\n[测试 2/5] 检查模型...")
        model_dir = Path('models')
        model_files = list(model_dir.glob('*.pkl'))
        
        if not model_files:
            logger.error("未找到模型文件！")
            return
        
        logger.info(f"✓ 找到 {len(model_files)} 个模型")
        
        model = StockSelectionModel(model_type='random_forest')
        model.load_model('random_forest_model.pkl')
        logger.info("✓ 模型加载成功")
        
        logger.info("\n[测试 3/5] 测试不同参数组合...")
        
        engineer = FeatureEngineer()
        X, y, feature_cols = engineer.prepare_features(df)
        
        test_cases = [
            {
                'name': '保守策略',
                'buy_threshold': 0.7,
                'sell_threshold': 0.5,
                'stop_loss_threshold': 0.05,
                'max_hold_days': 3,
                'max_position_pct': 0.1,
                'max_positions': 3
            },
            {
                'name': '平衡策略',
                'buy_threshold': 0.5,
                'sell_threshold': 0.5,
                'stop_loss_threshold': 0.1,
                'max_hold_days': 5,
                'max_position_pct': 0.2,
                'max_positions': 5
            },
            {
                'name': '激进策略',
                'buy_threshold': 0.3,
                'sell_threshold': 0.4,
                'stop_loss_threshold': 0.15,
                'max_hold_days': 10,
                'max_position_pct': 0.3,
                'max_positions': 10
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"测试案例 {i}: {test_case['name']}")
            logger.info(f"{'=' * 60}")
            
            engine = BacktestEngine(
                initial_cash=100000,
                commission=0.001,
                slippage=0.001,
                buy_threshold=test_case['buy_threshold'],
                sell_threshold=test_case['sell_threshold'],
                stop_loss_threshold=test_case['stop_loss_threshold'],
                max_hold_days=test_case['max_hold_days'],
                max_position_pct=test_case['max_position_pct'],
                max_positions=test_case['max_positions']
            )
            
            logger.info(f"参数配置:")
            logger.info(f"  买入阈值: {test_case['buy_threshold']}")
            logger.info(f"  卖出阈值: {test_case['sell_threshold']}")
            logger.info(f"  止损阈值: {test_case['stop_loss_threshold']}")
            logger.info(f"  最大持仓天数: {test_case['max_hold_days']}")
            logger.info(f"  最大仓位比例: {test_case['max_position_pct']:.1%}")
            logger.info(f"  最大持仓数量: {test_case['max_positions']}")
            
            df_test = df.head(500).copy()
            results = engine.run_backtest(
                df_test,
                model,
                feature_cols,
                start_date=df_test['date'].min().strftime('%Y-%m-%d'),
                end_date=df_test['date'].max().strftime('%Y-%m-%d')
            )
            
            if results:
                logger.info(f"\n结果:")
                logger.info(f"  买入信号: {results.get('buy_signals', 'N/A')}")
                logger.info(f"  卖出信号: {results.get('sell_signals', 'N/A')}")
                logger.info(f"  总交易数: {results['total_trades']}")
                logger.info(f"  总收益率: {results['total_return']:.2f}%")
                logger.info(f"  最终资金: ¥{results['final_value']:,.2f}")
                logger.info(f"  夏普比率: {results['sharpe_ratio']:.4f}")
                logger.info(f"  最大回撤: {results['max_drawdown']:.2f}%")
                
                if results['total_trades'] > 0:
                    logger.info(f"  胜率: {results.get('win_rate', 0):.2f}%")
                    logger.info(f"  平均盈亏: ¥{results.get('average_profit', 0):.2f}")
                    logger.info(f"  盈亏比: {results.get('profit_factor', 0):.2f}")
                else:
                    logger.warning("  ⚠️ 未产生任何交易！")
            else:
                logger.error("  ✗ 回测失败！")
        
        logger.info(f"\n{'=' * 60}")
        logger.info("测试总结")
        logger.info(f"{'=' * 60}")
        
        logger.info("\n推荐配置:")
        logger.info("根据测试结果，建议使用以下参数:")
        logger.info("1. 买入阈值: 0.4-0.5 (平衡交易频率和准确性)")
        logger.info("2. 卖出阈值: 0.5 (及时止损)")
        logger.info("3. 止损阈值: 0.1 (控制单笔最大亏损)")
        logger.info("4. 最大持仓天数: 5-10 (避免过度持有)")
        logger.info("5. 最大仓位比例: 0.2 (分散风险)")
        logger.info("6. 最大持仓数量: 5 (控制总体风险)")
        
        logger.info("\n下一步:")
        logger.info("1. 在界面上调整上述参数")
        logger.info("2. 运行完整回测")
        logger.info("3. 根据结果进一步优化")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}", exc_info=True)

if __name__ == "__main__":
    test_backtest_logic()
