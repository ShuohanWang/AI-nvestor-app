import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 页面配置 ---
st.set_page_config(page_title="AI Investment Assistant", layout="wide")

st.title("🤖 AI Investment Assistant")
st.markdown("Automated Analysis System based on **Modern Portfolio Theory (MPT)** and **Momentum Strategy**")

# --- 侧边栏：用户输入 ---
st.sidebar.header("⚙️ Portfolio Settings")
user_tickers = st.sidebar.text_input("Enter ETF Tickers (comma separated)", "QQQ, VGT, SPMO, GLD")
period = st.sidebar.selectbox("Analysis Time Period", ["1y", "2y", "5y", "ytd"], index=0)

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
if st.sidebar.button("🚀 Start Analysis"):
    with st.spinner('Fetching Wall Street Data...'):
        try:
            # 1. 获取数据
            df = get_data(tickers, period)
            
            # 检查数据有效性
            if df.empty:
                st.error("Unable to fetch data. Please check ticker spelling!")
            else:
                # 2. 展示基础走势
                st.subheader("📈 Historical Price Trend (Normalized)")
                normalized_df = df / df.iloc[0]
                st.line_chart(normalized_df)

                # --- [新增功能] 智能择时信号 (RSI Analysis) ---
                st.subheader("🚦 Market Thermometer (RSI Timing)")
                
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
                        status = "🔥 Overbought"
                        color = "normal" # Streamlit metric 红色可以用 inverse，但这里我们简单处理
                    elif rsi_val < 30:
                        status = "💰 Oversold (Buy Opportunity)"
                    else:
                        status = "⚖️ Normal"
                    
                    with cols[idx]:
                        st.metric(
                            label=f"{ticker} RSI",
                            value=f"{rsi_val:.2f}",
                            delta=status,
                            delta_color="inverse" if rsi_val > 70 else "normal"
                        )
                
                st.info("💡 Tip: RSI below 30 often indicates 'oversold' conditions (potential buy); above 70 indicates 'overbought' (potential pullback risk).")
                
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
                    st.subheader("🏆 AI Recommended Optimal Allocation")
                    # 做一个漂亮的饼图
                    fig1, ax1 = plt.subplots()
                    ax1.pie(best_weights, labels=tickers, autopct='%1.1f%%', startangle=90)
                    ax1.axis('equal') 
                    st.pyplot(fig1)

                with col2:
                    st.subheader("📊 Expected Performance (Annualized)")
                    st.metric("Expected Annual Return", f"{results[0, max_sharpe_idx]*100:.2f}%")
                    st.metric("Expected Volatility (Risk)", f"{results[1, max_sharpe_idx]*100:.2f}%")
                    st.metric("Sharpe Ratio", f"{results[2, max_sharpe_idx]:.2f}")

                # 5. 有效前沿图
                st.subheader("🎯 Efficient Frontier")
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                sc = ax2.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis', s=10, alpha=0.5)
                plt.colorbar(sc, label='Sharpe Ratio')
                ax2.scatter(results[1, max_sharpe_idx], results[0, max_sharpe_idx], c='red', s=100, marker='*', label='Optimal')
                ax2.set_xlabel('Risk (Volatility)')
                ax2.set_ylabel('Return')
                ax2.legend()
                st.pyplot(fig2)
                
                st.success("Analysis Complete! This is the power of Data Science.")
                
                # --- [新增功能] 蒙特卡洛未来财富模拟 ---
                st.markdown("---")
                st.subheader("🔮 Crystal Ball: 1-Year Asset Projection")
                
                # 假设我们投资 10,000 美元
                initial_capital = 10000
                st.info(f"Assuming initial capital: ${initial_capital:,.0f}, simulating future trends based on optimal allocation...")

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
                
                st.warning(f"Note: This is a probabilistic simulation. Worst case: ${simulation_df.iloc[-1].min():,.0f}, Best case: ${simulation_df.iloc[-1].max():,.0f}.")
            
                # --- [新增功能] 智能建仓计算器 ---
                st.markdown("---")
                st.subheader("🛒 Smart Position Sizing Calculator")

                col_input, col_calc = st.columns([1, 2])
                
                with col_input:
                    # 让用户输入想投资的金额
                    total_investment = st.number_input("💰 Enter Total Investment Amount ($):", min_value=1000, value=10000, step=500)
                
                with col_calc:
                    st.write(f"Based on optimal allocation, buying list for ${total_investment:,.2f}:")
                    
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
                                "Ticker": ticker,
                                "Target Allocation": f"{weight*100:.1f}%",
                                "Latest Price": f"${price:.2f}",
                                "Shares to Buy": shares,
                                "Est. Cost": f"${cost:.2f}"
                            })
                            cash_remaining -= cost
                    
                    # 转成 DataFrame 展示
                    plan_df = pd.DataFrame(plan)
                    if not plan_df.empty:
                        st.table(plan_df)
                        
                        # 展示剩下的零钱
                        st.success(f"✅ Cash remaining after execution: ${cash_remaining:.2f}")
                    else:
                        st.warning("Capital too low to purchase any shares at this allocation!")

        except Exception as e:
            st.error(f"Error occurred: {e}")
else:
    st.info("👈 Please enter ETFs on the left sidebar and click 'Start Analysis'")
