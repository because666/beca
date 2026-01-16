---
title: Quant Stock Selection App
emoji: 📈
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---

# 基于机器学习的量化投资选股系统

一个完整的基于机器学习算法的量化投资选股平台，提供股票数据获取与处理、机器学习选股模型构建、策略回测与评估等功能。

## 功能特点

- 📊 **股票数据获取**：自动获取A股历史数据，支持多只股票批量获取
- 🔧 **特征工程**：自动计算技术指标和收益率特征
- 🤖 **机器学习模型**：支持随机森林、XGBoost、LightGBM等多种算法
- ⏱️ **策略回测**：完整的历史回测功能，多维度评估策略表现
- 🔮 **选股预测**：基于训练好的模型预测股票未来走势
- 📈 **可视化分析**：丰富的图表展示，直观理解策略表现
- 🖥️ **友好界面**：基于Streamlit的Web界面，操作简单直观

## 快速开始

### 环境要求

- Python 3.8 或更高版本
- Windows / macOS / Linux
- 8GB+ 内存推荐
- 稳定的网络连接

### 安装步骤

1. **克隆或下载项目**

2. **安装依赖包**

Windows用户：
```bash
双击运行 "安装依赖.bat"
```

Mac/Linux用户：
```bash
pip install -r requirements.txt
```

3. **启动系统**

Windows用户：
```bash
双击运行 "启动系统.bat"
```

Mac/Linux用户：
```bash
python 启动系统.py
```

或直接运行：
```bash
streamlit run app.py
```

4. **使用系统**

浏览器会自动打开系统界面，按照用户操作手册进行操作。

## 项目结构

```
.
├── app.py                    # Streamlit主应用
├── config.py                 # 配置文件
├── data_fetcher.py           # 数据获取和处理模块
├── ml_models.py              # 机器学习模型模块
├── backtest.py               # 回测和评估模块
├── run_full_pipeline.py      # 完整流程脚本
├── requirements.txt          # 依赖包列表
├── build.spec                # PyInstaller配置
├── 启动系统.bat              # Windows启动脚本
├── 启动系统.py               # Python启动脚本
├── 安装依赖.bat              # Windows安装脚本
├── 用户操作手册.md           # 详细使用说明
├── README.md                 # 项目说明
├── data/                     # 数据目录
├── models/                   # 模型目录
├── results/                  # 结果目录
└── logs/                     # 日志目录
```

## 核心模块

### 1. 数据获取与处理 (data_fetcher.py)

- **StockDataFetcher**：获取股票历史数据
- **FeatureEngineer**：计算技术指标和特征
- **DataPreprocessor**：数据预处理和标准化

### 2. 机器学习模型 (ml_models.py)

- **StockSelectionModel**：单个选股模型
- **EnsembleModel**：集成学习模型
- **HyperparameterTuner**：超参数调优

### 3. 回测与评估 (backtest.py)

- **BacktestEngine**：策略回测引擎
- **PerformanceEvaluator**：性能评估器
- **BenchmarkComparator**：基准比较器

## 使用流程

1. **数据管理**：获取股票数据并处理特征
2. **模型训练**：选择模型并训练
3. **策略回测**：对策略进行历史回测
4. **选股预测**：获取最新的股票推荐
5. **性能分析**：多维度分析策略表现

详细操作请参考[用户操作手册.md](用户操作手册.md)

## 支持的模型

- 随机森林 (Random Forest)
- XGBoost
- LightGBM
- 逻辑回归 (Logistic Regression)
- 支持向量机 (SVM)

## 技术指标

系统自动计算以下技术指标：

- 移动平均线 (MA5, MA10, MA20, MA60)
- 指数移动平均 (EMA12, EMA26)
- MACD指标
- RSI指标
- 布林带 (Bollinger Bands)
- ATR (平均真实波幅)
- OBV (能量潮)
- 随机指标 (Stochastic)
- 动量指标 (Momentum)

## 回测指标

- 总收益率
- 夏普比率 (Sharpe Ratio)
- 最大回撤 (Maximum Drawdown)
- 胜率 (Win Rate)
- 盈亏比 (Profit Factor)
- Alpha / Beta

## 依赖包

```
akshare>=1.12.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
streamlit>=1.28.0
plotly>=5.17.0
backtrader>=1.9.78
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
ta>=0.11.0
openpyxl>=3.1.0
pyinstaller>=6.0.0
tqdm>=4.66.0
```

## 快速测试

运行完整流程脚本：

```bash
python run_full_pipeline.py
```

这将自动执行：
1. 获取股票数据
2. 处理数据特征
3. 训练多个机器学习模型
4. 进行策略回测
5. 生成回测报告

## 打包部署

使用PyInstaller打包为可执行文件：

```bash
pyinstaller build.spec
```

打包后的可执行文件位于 `dist/` 目录。

## 注意事项

1. 本系统仅供学习和研究使用，不构成投资建议
2. 投资有风险，入市需谨慎
3. 历史表现不代表未来收益
4. 使用前请确保网络连接正常
5. 首次使用需要下载依赖包，可能需要几分钟

## 常见问题

### Q: 系统启动失败？
A: 请检查Python版本是否为3.8+，并确认所有依赖包已正确安装。

### Q: 获取股票数据失败？
A: 请检查网络连接，确认股票代码格式正确（6位数字）。

### Q: 模型训练时间过长？
A: 可以减少股票数量、缩短日期范围、使用更简单的模型。

### Q: 回测结果不理想？
A: 可以尝试调整模型参数、优化特征工程、调整交易策略参数。

更多问题请参考[用户操作手册.md](用户操作手册.md)中的常见问题部分。

## 技术支持

如遇到问题，请：
1. 查看日志文件（logs/目录）
2. 检查错误信息
3. 参考用户操作手册
4. 联系技术支持

## 版本历史

### v1.0.0 (2026-01-10)
- 初始版本发布
- 实现核心功能：数据获取、模型训练、策略回测、选股预测
- 支持多种机器学习算法
- 完整的Web界面
- 一键启动功能

## 许可证

本项目仅供学习和研究使用。

## 免责声明

本系统提供的所有信息、分析和预测结果仅供参考，不构成任何投资建议。股票投资存在风险，投资者应根据自身情况谨慎决策，并对自己的投资行为负责。开发者不对使用本系统造成的任何损失承担责任。

---

**祝您使用愉快！**
