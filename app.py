import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 页面配置 ---
st.set_page_config(page_title="AI 投资小助手", layout="wide")

st.title("🤖 AI 智能投顾助手 (AI Investment Assistant)")
st.markdown("基于 **现代投资组合理论 (MPT)** 与 **动量策略** 的自动化分析系统")

# --- 侧边栏：用户输入 ---
st.sidebar.header("⚙️ 投资组合设置")
user_tickers = st.sidebar.text_input("输入 ETF 代码 (用逗号分隔)", "QQQ, VGT, SPMO, GLD")
period = st.sidebar.selectbox("分析时间范围", ["1y", "2y", "5y", "ytd"], index=0)

# 解析用户输入的代码
tickers = [t.strip().upper() for t in user_tickers.split(",")]

# --- 核心函数 (带缓存，提高速度) ---
@st.cache_data
def get_data(tickers, period):
    data = yf.download(tickers, period=period, auto_adjust=True)['Close']
    # 修复列名对齐问题
    if len(tickers) > 1:
        data = data[tickers] 
    return data

# --- 主逻辑 ---
if st.sidebar.button("🚀 开始分析"):
    with st.spinner('正在拉取华尔街数据...'):
        try:
            # 1. 获取数据
            df = get_data(tickers, period)
            
            # 检查数据有效性
            if df.empty:
                st.error("无法获取数据，请检查代码拼写！")
            else:
                # 2. 展示基础走势
                st.subheader("📈 历史价格走势 (归一化)")
                normalized_df = df / df.iloc[0]
                st.line_chart(normalized_df)

                # --- [新增功能] 智能择时信号 (RSI Analysis) ---
                st.subheader("🚦 市场温度计 (RSI Timing)")
                
                # 计算 RSI 的简单函数
                def calculate_rsi(data, window=14):
                    delta = data.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                    rs = gain / loss
                    return 100 - (100 / (1 + rs))

                # 获取最新一天的 RSI 值
                rsi_data = calculate_rsi(df)
                latest_rsi = rsi_data.iloc[-1] # 取最后一行

                # 使用列布局来展示每个 ETF 的信号
                cols = st.columns(len(tickers))
                for idx, ticker in enumerate(tickers):
                    rsi_val = latest_rsi[ticker]
                    
                    # 判断信号颜色和文字
                    if rsi_val > 70:
                        status = "🔥 过热 (Overbought)"
                        color = "normal" # Streamlit metric 红色可以用 inverse，但这里我们简单处理
                    elif rsi_val < 30:
                        status = "💰 捡漏机会 (Oversold)"
                    else:
                        status = "⚖️ 正常波动"
                    
                    with cols[idx]:
                        st.metric(
                            label=f"{ticker} RSI",
                            value=f"{rsi_val:.2f}",
                            delta=status,
                            delta_color="inverse" if rsi_val > 70 else "normal"
                        )
                
                st.info("💡 小贴士: RSI 低于 30 通常意味着短期被'错杀'，可能是补仓的好时机；高于 70 则意味着短期涨幅过大，要注意回调风险。")
                
                # ... (下面的代码不用动) ...

                # 3. 计算指标
                daily_returns = df.pct_change().dropna()
                mean_returns = daily_returns.mean() * 252
                cov_matrix = daily_returns.cov() * 252
                
                # 4. 蒙特卡洛模拟 (寻找最优解)
                num_portfolios = 3000
                results = np.zeros((3, num_portfolios))
                weights_record = []

                for i in range(num_portfolios):
                    weights = np.random.random(len(tickers))
                    weights /= np.sum(weights)
                    weights_record.append(weights)
                    
                    p_ret = np.sum(mean_returns * weights)
                    p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    
                    results[0,i] = p_ret
                    results[1,i] = p_vol
                    results[2,i] = (p_ret - 0.04) / p_vol # Sharpe Ratio

                # 找到夏普比率最高的点
                max_sharpe_idx = np.argmax(results[2])
                best_weights = weights_record[max_sharpe_idx]
                
                # --- 展示结果 ---
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🏆 AI 建议的最佳仓位")
                    # 做一个漂亮的饼图
                    fig1, ax1 = plt.subplots()
                    ax1.pie(best_weights, labels=tickers, autopct='%1.1f%%', startangle=90)
                    ax1.axis('equal') 
                    st.pyplot(fig1)

                with col2:
                    st.subheader("📊 预期表现 (年化)")
                    st.metric("预期年化收益", f"{results[0, max_sharpe_idx]*100:.2f}%")
                    st.metric("预期波动率 (风险)", f"{results[1, max_sharpe_idx]*100:.2f}%")
                    st.metric("夏普比率 (Sharpe)", f"{results[2, max_sharpe_idx]:.2f}")

                # 5. 有效前沿图
                st.subheader("🎯 有效前沿 (Efficient Frontier)")
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                sc = ax2.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis', s=10, alpha=0.5)
                plt.colorbar(sc, label='Sharpe Ratio')
                ax2.scatter(results[1, max_sharpe_idx], results[0, max_sharpe_idx], c='red', s=100, marker='*', label='Optimal')
                ax2.set_xlabel('Risk (Volatility)')
                ax2.set_ylabel('Return')
                ax2.legend()
                st.pyplot(fig2)
                
                st.success("分析完成！这就是数据科学的力量。")
                
# --- [新增功能] 蒙特卡洛未来财富模拟 ---
                st.markdown("---")
                st.subheader("🔮 水晶球：未来 1 年资产推演")
                
                # 假设我们投资 10,000 美元
                initial_capital = 10000
                st.info(f"假设当前投入资金: ${initial_capital:,.0f}，基于最佳仓位模拟未来走势...")

                # 模拟参数
                simulation_days = 252 # 一年
                num_simulations = 50  # 模拟 50 条可能的平行宇宙
                
                # 获取最佳组合的预期收益和波动率
                best_port_ret = results[0, max_sharpe_idx]
                best_port_vol = results[1, max_sharpe_idx]

                # 生成随机路径
                # 公式: S_t = S_0 * exp((mu - 0.5 * sigma^2) * t + sigma * W_t)
                # 这是一个几何布朗运动模型 (Geometric Brownian Motion)
                simulation_df = pd.DataFrame()
                
                for i in range(num_simulations):
                    # 生成每日的随机波动
                    daily_vol = np.random.normal(
                        (best_port_ret - 0.5 * best_port_vol**2) / 252, 
                        best_port_vol / np.sqrt(252), 
                        simulation_days
                    )
                    # 计算累计净值路径
                    price_series = initial_capital * (1 + daily_vol).cumprod()
                    simulation_df[f"Scenario {i+1}"] = price_series

                # 画图
                fig3, ax3 = plt.subplots(figsize=(10, 5))
                ax3.plot(simulation_df, color='green', alpha=0.1, linewidth=1)
                ax3.set_title(f'Monte Carlo Simulation: 50 Possible Futures (1 Year)')
                ax3.set_ylabel('Portfolio Value ($)')
                ax3.set_xlabel('Trading Days')
                ax3.grid(True, alpha=0.3)
                
                # 标出平均结果
                avg_end_price = simulation_df.iloc[-1].mean()
                ax3.axhline(avg_end_price, color='red', linestyle='--', label=f'Average Outcome: ${avg_end_price:,.0f}')
                ax3.legend()
                
                st.pyplot(fig3)
                
                st.warning(f"注：这是基于概率的数学模拟。最坏情况下，你的资产可能跌至 ${simulation_df.iloc[-1].min():,.0f}，最好情况下可能达到 ${simulation_df.iloc[-1].max():,.0f}。")
            
                # --- [新增功能] 智能建仓计算器 ---
                st.markdown("---")
                st.subheader("🛒 智能建仓计算器 (Position Sizing)")

                col_input, col_calc = st.columns([1, 2])
                
                with col_input:
                    # 让用户输入想投资的金额
                    total_investment = st.number_input("💰 请输入你的总投资金额 ($):", min_value=1000, value=10000, step=500)
                
                with col_calc:
                    st.write(f"基于当前最佳配置，${total_investment:,.2f} 的购买清单如下：")
                    
                    # 获取最新价格
                    latest_prices = df.iloc[-1]
                    
                    # 计算逻辑
                    plan = []
                    cash_remaining = total_investment
                    
                    for ticker, weight in zip(tickers, best_weights):
                        # 该股票理论上应该分到的钱
                        target_value = total_investment * weight
                        price = latest_prices[ticker]
                        
                        # 向下取整，算出能买多少股
                        shares = int(target_value / price)
                        cost = shares * price
                        
                        # 只有当需要买至少1股时才显示
                        if shares > 0:
                            plan.append({
                                "代码 (Ticker)": ticker,
                                "建议仓位": f"{weight*100:.1f}%",
                                "最新股价": f"${price:.2f}",
                                "应买股数": shares,
                                "预计花费": f"${cost:.2f}"
                            })
                            cash_remaining -= cost
                    
                    # 转成 DataFrame 展示
                    plan_df = pd.DataFrame(plan)
                    if not plan_df.empty:
                        st.table(plan_df)
                        
                        # 展示剩下的零钱
                        st.success(f"✅ 执行此计划后，你还会剩余现金: ${cash_remaining:.2f}")
                    else:
                        st.warning("你的资金太少，无法按此比例购买任何一股股票！")

        except Exception as e:
            st.error(f"发生错误: {e}")
else:
    st.info("👈 请在左侧输入你想分析的 ETF，然后点击“开始分析”")
