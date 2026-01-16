import streamlit as st
import pandas as pd

def show_help_page():
    st.header("📘 帮助中心")
    
    # 搜索功能
    search_term = st.text_input("🔍 搜索帮助内容（术语、功能、问题）", placeholder="例如：夏普比率、如何回测...")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ 项目概述", "📚 术语解释", "📖 操作指南", "❓ 常见问题"])
    
    with tab1:
        st.subheader("项目概述")
        st.markdown("""
        **核心目标**：利用机器学习算法分析历史股票数据，挖掘潜在的交易规律，构建自动化的选股策略，并通过回测验证策略的有效性。
        
        **主要价值**：
        1.  **客观决策**：基于数据和模型，克服人为情绪干扰。
        2.  **效率提升**：自动处理海量数据，快速筛选潜力股。
        3.  **风险验证**：通过历史回测评估策略风险，避免盲目投资。
        """)
        
        st.subheader("工作原理")
        st.graphviz_chart("""
            digraph {
                rankdir=LR;
                node [shape=box, style=filled, fillcolor=lightblue];
                
                Data [label="数据获取\n(AkShare)", fillcolor="#e1f5fe"];
                Feature [label="特征工程\n(技术指标/收益率)", fillcolor="#b3e5fc"];
                Model [label="模型训练\n(RF/XGB/LGBM)", fillcolor="#81d4fa"];
                Backtest [label="策略回测\n(历史模拟)", fillcolor="#4fc3f7"];
                Predict [label="选股预测\n(未来推荐)", fillcolor="#29b6f6"];
                
                Data -> Feature -> Model;
                Model -> Backtest;
                Model -> Predict;
            }
        """)
        
        st.subheader("应用场景")
        st.info("""
        - **量化研究**：验证新的选股因子和机器学习算法。
        - **策略开发**：开发并优化中短线选股策略。
        - **实盘辅助**：每日盘后筛选次日重点关注的股票池。
        """)

    with tab2:
        st.subheader("术语解释")
        
        terms = {
            "夏普比率 (Sharpe Ratio)": "衡量策略在承担单位风险下获得的超额回报。**值越高越好**。>1 表示优秀，<0 表示不如无风险理财。",
            "最大回撤 (Max Drawdown)": "策略在某一段时间内资金从最高点跌落到最低点的最大跌幅。**值越低越好**。衡量策略的最大潜在亏损风险。",
            "准确率 (Accuracy)": "模型预测正确的样本占总样本的比例。在股市预测中，单纯准确率可能误导（如牛市全猜涨准确率也高），需结合其他指标。",
            "精确率 (Precision)": "模型预测为“涨”的样本中，实际真的“涨”的比例。**对实盘最重要**，因为我们只买模型预测涨的股票。",
            "召回率 (Recall)": "实际为“涨”的样本中，被模型正确预测出来的比例。衡量模型捕捉机会的能力。",
            "RSI (相对强弱指标)": "衡量股票超买超卖情况。>70 通常认为超买（可能回调），<30 通常认为超卖（可能反弹）。",
            "MACD (平滑异同移动平均线)": "趋势指标。DIF线上穿DEA线为金叉（买入信号），下穿为死叉（卖出信号）。",
            "量比": "衡量相对成交量的指标。量比 > 1 说明成交量放大，< 1 说明成交量萎缩。",
            "混淆矩阵": "展示模型预测结果与真实结果对比的表格。包含TP（真阳性）、FP（假阳性）、TN（真阴性）、FN（假阴性）。",
            "随机森林 (Random Forest)": "一种集成学习算法，通过构建多棵决策树并投票来决定最终结果，具有较好的抗过拟合能力。",
            "XGBoost/LightGBM": "梯度提升决策树算法，通常比随机森林精度更高、速度更快，是量化竞赛中的常用模型。"
        }
        
        # 搜索过滤
        filtered_terms = {k: v for k, v in terms.items() if not search_term or search_term.lower() in k.lower() or search_term.lower() in v.lower()}
        
        if not filtered_terms:
            st.warning("没有找到匹配的术语。")
        
        for term, desc in filtered_terms.items():
            with st.expander(f"📌 {term}", expanded=bool(search_term)):
                st.markdown(desc)
                # 简单的关联图示例 (静态)
                if "夏普" in term or "回撤" in term:
                    st.caption("关联：风险评估指标")
                elif "准确率" in term or "精确率" in term:
                    st.caption("关联：模型评估指标")

    with tab3:
        st.subheader("操作指南")
        
        steps = [
            ("第一步：数据准备", "进入【数据管理】页面 -> 输入股票代码 -> 点击【获取股票数据】 -> 点击【处理数据特征】。"),
            ("第二步：模型训练", "进入【模型训练】页面 -> 选择测试集比例 -> (可选)勾选超参数调优 -> 点击【开始训练模型】。关注评估结果中的【精确率】。"),
            ("第三步：策略回测", "进入【策略回测】页面 -> 设置回测日期和初始资金 -> 调整买卖阈值 -> 点击【开始回测】。观察资金曲线和交易记录。"),
            ("第四步：实盘预测", "进入【选股预测】页面 -> 确保已有最新数据 -> 点击【开始预测】 -> 查看推荐股票列表。")
        ]
        
        for title, content in steps:
            st.markdown(f"#### {title}")
            st.write(content)
            st.divider()
            
        st.subheader("典型场景示例")
        st.markdown("""
        **场景：构建一个稳健的短线策略**
        1. **选股**：选择沪深300成分股或流动性好的热门股。
        2. **模型**：使用 Random Forest 或 LightGBM。
        3. **回测参数**：
           - 买入阈值：0.60 (追求高胜率)
           - 卖出阈值：0.50
           - 止损：8%
           - 最大持仓：5天
        4. **优化**：如果回测交易太少，降低买入阈值；如果回撤太大，减小单只股票仓位或收紧止损。
        """)

    with tab4:
        st.subheader("常见问题 (FAQ)")
        
        faqs = {
            "Q: 为什么回测没有交易？": "A: 可能是买入阈值设置过高，模型预测概率达不到要求。或者回测时间段内数据缺失。尝试降低买入阈值（如设为0.5）或检查数据。",
            "Q: 模型训练需要多久？": "A: 取决于数据量和是否开启网格搜索。普通训练通常几秒到几分钟。开启网格搜索可能需要几分钟到几十分钟。",
            "Q: 预测结果准确吗？": "A: 机器学习预测基于历史概率，不能保证未来100%准确。建议将其作为辅助参考，结合基本面分析。",
            "Q: 如何导入自己的股票列表？": "A: 在【数据管理】页面的文本框中直接粘贴股票代码，每行一个即可。",
            "Q: 支持哪些市场数据？": "A: 目前主要支持A股日线数据（通过AkShare接口获取）。"
        }
        
        filtered_faqs = {k: v for k, v in faqs.items() if not search_term or search_term.lower() in k.lower() or search_term.lower() in v.lower()}
        
        for q, a in filtered_faqs.items():
            st.markdown(f"**{q}**")
            st.write(a)
            st.write("")

if __name__ == "__main__":
    show_help_page()
