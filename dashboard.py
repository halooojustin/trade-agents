import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from plotly.subplots import make_subplots
import glob
import os

st.set_page_config(page_title="Official Trade Analysis", layout="wide")

st.markdown("### 📊 Trading Agent Performance Dashboard")
st.caption("Hyperliquid Export Analysis")

# Load Data
@st.cache_data
def load_data(file_paths):
    if not file_paths:
        return pd.DataFrame()
        
    all_dfs = []
    try:
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            df['source_file'] = os.path.basename(file_path)
            all_dfs.append(df)
        
        if not all_dfs:
            return pd.DataFrame()
            
        df = pd.concat(all_dfs, ignore_index=True)
        
        # Drop duplicates if any (based on time, coin, px, sz, dir)
        df = df.drop_duplicates(subset=['time', 'coin', 'px', 'sz', 'dir'])
        
        # Clean and convert columns
        df['time'] = pd.to_datetime(df['time'])
        df['closedPnl'] = pd.to_numeric(df['closedPnl'], errors='coerce').fillna(0)
        df['fee'] = pd.to_numeric(df['fee'], errors='coerce').fillna(0)
        df['px'] = pd.to_numeric(df['px'], errors='coerce').fillna(0)
        df['sz'] = pd.to_numeric(df['sz'], errors='coerce').fillna(0)
        
        # Correct PnL Logic based on Hyperliquid Export
        # closedPnl is ALREADY Net PnL (includes fees)
        df['netPnl'] = df['closedPnl'] 
        df['grossPnl'] = df['closedPnl'] + df['fee']
        
        # Sort by time
        df = df.sort_values('time')
        
        # Calculate Cumulative Metrics
        df['cum_gross_pnl'] = df['grossPnl'].cumsum()
        df['cum_net_pnl'] = df['netPnl'].cumsum()
        df['cum_fees'] = df['fee'].cumsum()
        
        # Calculate Cumulative PnL per Coin
        df['coin_cum_net_pnl'] = df.groupby('coin')['netPnl'].cumsum()
        
        return df
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return pd.DataFrame()

# Sidebar File Selection
st.sidebar.header("Data Selection")
available_files = glob.glob("agent_*.csv")
selected_files = st.sidebar.multiselect("Select Data Files", options=available_files, default=available_files)

# Display Agent Prompts
st.sidebar.header("📝 Agent Prompts")
for csv_file in selected_files:
    # Get corresponding .txt file
    txt_file = csv_file.replace('.csv', '.txt')
    if os.path.exists(txt_file):
        agent_name = os.path.basename(csv_file).replace('.csv', '')
        with st.sidebar.expander(f"🤖 {agent_name}"):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    prompt_content = f.read()
                st.code(prompt_content, language='text')
            except Exception as e:
                st.error(f"Error reading {txt_file}: {e}")

df = load_data(selected_files)

if df.empty:
    st.stop()

# Use all data without filtering
filtered_df = df

# Key Metrics
total_gross_pnl = filtered_df['grossPnl'].sum()
total_fees = filtered_df['fee'].sum()
total_net_pnl = filtered_df['netPnl'].sum()
total_trades = len(filtered_df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Net PnL", f"${total_net_pnl:.2f}", delta_color="normal")
col2.metric("Gross PnL", f"${total_gross_pnl:.2f}")
col3.metric("Total Fees", f"${total_fees:.2f}")
col4.metric("Total Trades", total_trades)

# Detailed Performance Metrics
st.markdown("##### 📊 Performance Metrics")

# Calculate metrics
initial_balance = 476.66  # From equity_curve.csv
total_return_net = total_net_pnl
total_return_gross = total_gross_pnl
return_pct_net = (total_return_net / initial_balance * 100) if initial_balance > 0 else 0
return_pct_gross = (total_return_gross / initial_balance * 100) if initial_balance > 0 else 0

# Sharpe Ratio calculation (both net and gross)
returns_net = filtered_df['netPnl'].values
returns_gross = filtered_df['grossPnl'].values
sharpe_ratio_net = (returns_net.mean() / returns_net.std() * (252 ** 0.5)) if returns_net.std() > 0 else 0
sharpe_ratio_gross = (returns_gross.mean() / returns_gross.std() * (252 ** 0.5)) if returns_gross.std() > 0 else 0

# Profit Factor (both net and gross)
winning_trades_net = filtered_df[filtered_df['netPnl'] > 0]
losing_trades_net = filtered_df[filtered_df['netPnl'] < 0]
winning_trades_gross = filtered_df[filtered_df['grossPnl'] > 0]
losing_trades_gross = filtered_df[filtered_df['grossPnl'] < 0]

gross_profit_net = winning_trades_net['netPnl'].sum()
gross_loss_net = abs(losing_trades_net['netPnl'].sum())
profit_factor_net = (gross_profit_net / gross_loss_net) if gross_loss_net > 0 else 0

gross_profit_gross = winning_trades_gross['grossPnl'].sum()
gross_loss_gross = abs(losing_trades_gross['grossPnl'].sum())
profit_factor_gross = (gross_profit_gross / gross_loss_gross) if gross_loss_gross > 0 else 0

# Max Drawdown (both net and gross) - Relative to Initial Balance
# Calculate equity curves (initial balance + cumulative PnL)
equity_net = initial_balance + filtered_df['cum_net_pnl'].values
equity_gross = initial_balance + filtered_df['cum_gross_pnl'].values

# Net Max Drawdown (relative to initial balance)
min_equity_net = equity_net.min()
max_drawdown_usd_net = min_equity_net - initial_balance
max_drawdown_pct_net = (max_drawdown_usd_net / initial_balance * 100) if initial_balance > 0 else 0

# Gross Max Drawdown (relative to initial balance)
min_equity_gross = equity_gross.min()
max_drawdown_usd_gross = min_equity_gross - initial_balance
max_drawdown_pct_gross = (max_drawdown_usd_gross / initial_balance * 100) if initial_balance > 0 else 0

# Direction Analysis
long_trades = filtered_df[filtered_df['dir'].str.contains('Long', na=False)]
short_trades = filtered_df[filtered_df['dir'].str.contains('Short', na=False)]
long_wins = len(long_trades[long_trades['netPnl'] > 0])
short_wins = len(short_trades[short_trades['netPnl'] > 0])
long_accuracy = (long_wins / len(long_trades) * 100) if len(long_trades) > 0 else 0
short_accuracy = (short_wins / len(short_trades) * 100) if len(short_trades) > 0 else 0

# Total Volume
total_volume = (filtered_df['sz'] * filtered_df['px']).sum()

# Display metrics in organized sections with smaller text
col_a, col_b, col_c, col_d = st.columns(4)

with col_a:
    st.markdown("#### 📊 NET METRICS")
    st.markdown(f"**Return:** ${total_return_net:.2f} ({return_pct_net:.1f}%)")
    st.markdown(f"**Sharpe:** {sharpe_ratio_net:.2f}")
    st.markdown(f"**Profit Factor:** {profit_factor_net:.2f}")
    st.markdown(f"**Max DD:** ${max_drawdown_usd_net:.2f} ({max_drawdown_pct_net:.1f}%)")
    st.markdown(f"**Win Rate:** {len(winning_trades_net) / total_trades * 100:.1f}%" if total_trades > 0 else "0%")

with col_b:
    st.markdown("#### 💎 GROSS METRICS")
    st.markdown(f"**Return:** ${total_return_gross:.2f} ({return_pct_gross:.1f}%)")
    st.markdown(f"**Sharpe:** {sharpe_ratio_gross:.2f}")
    st.markdown(f"**Profit Factor:** {profit_factor_gross:.2f}")
    st.markdown(f"**Max DD:** ${max_drawdown_usd_gross:.2f} ({max_drawdown_pct_gross:.1f}%)")
    st.markdown(f"**Fees Impact:** ${total_fees:.2f}")

with col_c:
    st.markdown("#### 📈 ACTIVITY")
    st.markdown(f"**Total Trades:** {total_trades}")
    st.markdown(f"**Positions:** {len(filtered_df[filtered_df['dir'].str.contains('Open', na=False)])}")
    st.markdown(f"**Volume:** ${total_volume:,.0f}")
    st.markdown(f"**Avg Size:** ${total_volume / total_trades:,.0f}" if total_trades > 0 else "$0")
    st.markdown(f"**Win/Loss:** {len(winning_trades_net)}/{len(losing_trades_net)}")

with col_d:
    st.markdown("#### 🎯 DIRECTION")
    st.markdown(f"**Long:** {len(long_trades)} ({long_accuracy:.1f}%)")
    st.markdown(f"**Short:** {len(short_trades)} ({short_accuracy:.1f}%)")
    st.markdown(f"**Avg Win:** ${gross_profit_net / len(winning_trades_net) if len(winning_trades_net) > 0 else 0:.2f}")
    st.markdown(f"**Avg Loss:** ${gross_loss_net / len(losing_trades_net) if len(losing_trades_net) > 0 else 0:.2f}")

st.markdown("---")

# Charts
st.subheader("📈 Cumulative PnL Over Time")
fig_cum = go.Figure()
fig_cum.add_trace(go.Scatter(x=filtered_df['time'], y=filtered_df['cum_net_pnl'], mode='lines', name='Net PnL'))
fig_cum.add_trace(go.Scatter(x=filtered_df['time'], y=filtered_df['cum_gross_pnl'], mode='lines', name='Gross PnL', line=dict(dash='dot')))
fig_cum.add_trace(go.Scatter(x=filtered_df['time'], y=filtered_df['cum_fees'], mode='lines', name='Cumulative Fees', fill='tozeroy'))
st.plotly_chart(fig_cum, use_container_width=True)

# Per Coin Cumulative PnL
st.subheader("📈 Cumulative Net PnL by Coin")
fig_coin_cum = px.line(filtered_df, x='time', y='coin_cum_net_pnl', color='coin', title="Net PnL Trend per Coin", markers=True)
st.plotly_chart(fig_coin_cum, use_container_width=True)

# Price vs PnL Analysis
st.subheader("🔬 Price Action vs PnL Analysis")
selected_coin = st.selectbox("Select Coin for Detailed Analysis", options=df['coin'].unique())

if selected_coin:
    coin_data = df[df['coin'] == selected_coin].sort_values('time')
    
    if not coin_data.empty:
        # Create Dual Axis Chart
        fig_analysis = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Match Open-Close pairs
        open_positions = {}  # Track open positions by direction
        trade_pairs = []  # Store (open_row, close_row) pairs
        
        for idx, row in coin_data.iterrows():
            if 'Open' in row['dir']:
                # Store open position
                direction = 'Long' if 'Long' in row['dir'] else 'Short'
                if direction not in open_positions:
                    open_positions[direction] = []
                open_positions[direction].append(row)
            elif 'Close' in row['dir']:
                # Match with corresponding open
                direction = 'Long' if 'Long' in row['dir'] else 'Short'
                if direction in open_positions and len(open_positions[direction]) > 0:
                    open_row = open_positions[direction].pop(0)  # FIFO matching
                    trade_pairs.append((open_row, row))
        
        # Determine Colors and Sizes
        # Open Long -> Blue, Open Short -> Orange (no PnL, fixed small size)
        # Close Long/Short -> Green (profit) or Red (loss), size based on PnL magnitude
        
        def get_color(row):
            if 'Open' in row['dir']:
                # Open positions: Blue for Long, Orange for Short
                return 'blue' if 'Long' in row['dir'] else 'orange'
            else:
                # Close positions: Green for profit, Red for loss
                return 'green' if row['closedPnl'] > 0 else 'red'
        
        colors = coin_data.apply(get_color, axis=1)
        
        # Determine Marker Size
        # Open positions: fixed small size (6)
        # Close positions: scaled by PnL magnitude (6-30)
        close_trades = coin_data[coin_data['dir'].str.contains('Close', na=False)]
        max_pnl_abs = close_trades['closedPnl'].abs().max() if not close_trades.empty else 1
        if max_pnl_abs == 0: max_pnl_abs = 1
        
        def get_size(row):
            if 'Open' in row['dir']:
                return 6  # Fixed small size for Open
            else:
                return 6 + (abs(row['closedPnl']) / max_pnl_abs) * 24  # Scaled for Close
        
        sizes = coin_data.apply(get_size, axis=1)
        
        hover_text = coin_data.apply(lambda row: f"<b>{row['dir']}</b><br>" +
                                                 f"Price: ${row['px']:.4f}<br>" +
                                                 f"Size: {row['sz']}<br>" +
                                                 f"PnL: ${row['closedPnl']:.2f}<br>" +
                                                 f"Fee: ${row['fee']:.2f}<br>" +
                                                 f"Time: {row['time']}", axis=1)

        # Add Price Line (gray, no markers)
        fig_analysis.add_trace(
            go.Scatter(x=coin_data['time'], y=coin_data['px'], 
                    mode='lines', 
                    name=f"{selected_coin} Price",
                    line=dict(color='gray', width=1, dash='dot'),
                    hoverinfo='skip'
            ),
            secondary_y=False
        )
        
        # Add connecting lines for Open-Close pairs
        for open_row, close_row in trade_pairs:
            line_color = 'green' if close_row['closedPnl'] > 0 else 'red'
            line_width = min(1 + abs(close_row['closedPnl']) / max_pnl_abs * 3, 4)  # Width 1-4
            
            # Calculate hold duration
            hold_duration = (pd.to_datetime(close_row['time']) - pd.to_datetime(open_row['time'])).total_seconds() / 60
            
            fig_analysis.add_trace(
                go.Scatter(
                    x=[open_row['time'], close_row['time']],
                    y=[open_row['px'], close_row['px']],
                    mode='lines',
                    line=dict(color=line_color, width=line_width, dash='solid'),
                    opacity=0.4,
                    showlegend=False,
                    hovertemplate=f"<b>Trade Pair</b><br>" +
                                 f"Open: {open_row['dir']} @ ${open_row['px']:.2f}<br>" +
                                 f"Close: {close_row['dir']} @ ${close_row['px']:.2f}<br>" +
                                 f"PnL: ${close_row['closedPnl']:.2f}<br>" +
                                 f"Hold: {hold_duration:.1f} min<br>" +
                                 f"Price Change: {((close_row['px'] - open_row['px']) / open_row['px'] * 100):.2f}%<extra></extra>",
                    hoverinfo='text'
                ),
                secondary_y=False
            )
        
        # Add Trade Markers (colored with size variation)
        fig_analysis.add_trace(
            go.Scatter(x=coin_data['time'], y=coin_data['px'], 
                    mode='markers', 
                    name='Trades',
                    marker=dict(color=colors, size=sizes, line=dict(width=1, color='DarkSlateGrey')),
                    text=hover_text,
                    hoverinfo='text'
            ),
            secondary_y=False
        )
        
        # Add PnL Line
        fig_analysis.add_trace(
            go.Scatter(x=coin_data['time'], y=coin_data['coin_cum_net_pnl'], 
                    mode='lines+markers', name='Cumulative Net PnL',
                    line=dict(color='blue', width=2)),
            secondary_y=True
        )
        
        fig_analysis.update_layout(
            title=f"{selected_coin}: Trade Execution Price vs Cumulative PnL",
            xaxis_title="Time",
            yaxis_title="Price (USD)",
            yaxis2_title="Cumulative PnL (USD)"
        )
        
        st.plotly_chart(fig_analysis, use_container_width=True)
        
        # Legend Explanation
        st.markdown("""
        **📊 Chart Legend:**
        - 🔵 **Blue** = Open Long (entry point, small marker)
        - 🟠 **Orange** = Open Short (entry point, small marker)
        - 🟢 **Green** = Close with Profit (marker size = profit amount)
        - 🔴 **Red** = Close with Loss (marker size = loss amount)
        - **Marker Size**: Larger markers indicate bigger PnL (profit or loss)
        """)
        
        st.info("Note: Price data is based on your trade execution prices, not external market data. This ensures it matches your simulation timeframe.")
    else:
        st.warning(f"No trade data found for {selected_coin}")

# PnL by Coin
st.subheader("💰 PnL by Coin (Net vs Gross)")
coin_stats = df.groupby('coin')[['grossPnl', 'fee', 'netPnl']].sum().sort_values('netPnl', ascending=True)
fig_coin = go.Figure()
fig_coin.add_trace(go.Bar(y=coin_stats.index, x=coin_stats['netPnl'], name='Net PnL', orientation='h'))
fig_coin.add_trace(go.Bar(y=coin_stats.index, x=coin_stats['grossPnl'], name='Gross PnL', orientation='h'))
fig_coin.add_trace(go.Bar(y=coin_stats.index, x=coin_stats['fee'], name='Fees', orientation='h'))
fig_coin.update_layout(barmode='group', title="Net PnL vs Gross PnL vs Fees per Coin")
st.plotly_chart(fig_coin, use_container_width=True)

# Trade Distribution
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Trade Count by Coin")
    trade_counts = df['coin'].value_counts()
    fig_counts = px.pie(values=trade_counts.values, names=trade_counts.index, title="Trade Volume Distribution")
    st.plotly_chart(fig_counts, use_container_width=True)

with col_b:
    st.subheader("Fee Impact")
    # Scatter plot of Trade Size vs Fee
    fig_fee = px.scatter(filtered_df, x='sz', y='fee', color='coin', title="Trade Size vs Fee", hover_data=['px', 'time'])
    st.plotly_chart(fig_fee, use_container_width=True)

# Detailed Data
st.subheader("📝 Detailed Trade History")
st.dataframe(filtered_df.sort_values('time', ascending=False), use_container_width=True)
