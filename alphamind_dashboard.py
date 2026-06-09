import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Page Global Setup Configuration
st.set_page_config(
    page_title="AlphaMind: Intelligent Agentic Terminal",
    page_icon="📈",
    layout="wide"
)

# 2. Optimized Processing Framework
@st.cache_data
def load_and_process_agentic_data(filepath):
    df = pd.read_csv(filepath)
    processed_df = df.copy()
    
    # Unsupervised Structural Clustering Configuration
    regime_features = ['trading_volume_change_pct', 'volatility_score', 'social_media_buzz_score']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(processed_df[regime_features])
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    processed_df['market_regime_cluster'] = kmeans.fit_predict(scaled_features)
    
    # Multimodal Linguistic Density Mapping Matrix
    negative_tokens = ['weak', 'negative', 'drop', 'sell', 'elevated', 'risk']
    positive_tokens = ['growth', 'breakout', 'buy', 'positive', 'strong', 'momentum']
    
    processed_df['reasoning_clean'] = processed_df['agent_reasoning_summary'].astype(str).str.lower()
    processed_df['neg_density'] = processed_df['reasoning_clean'].apply(lambda x: sum(1 for t in negative_tokens if t in x))
    processed_df['pos_density'] = processed_df['reasoning_clean'].apply(lambda x: sum(1 for t in positive_tokens if t in x))
    processed_df['net_text_velocity'] = processed_df['pos_density'] - processed_df['neg_density']
    
    # Agentic Stance Categorization Control
    conditions = [
        (processed_df['market_regime_cluster'] == 0) & (processed_df['net_text_velocity'] >= 0) & (processed_df['news_sentiment_score'] > 0.2),
        (processed_df['market_regime_cluster'] == 2) | (processed_df['volatility_score'] > 60) | (processed_df['risk_label'].str.lower() == 'high'),
        (processed_df['net_text_velocity'] < 0) & (processed_df['news_sentiment_score'] < -0.2)
    ]
    choices = [
        'CONFIDENCE HIGH: PROCEED/EXECUTE',
        'ELEVATED RISK: HOLD POSITION/STAY CAUTIOUS',
        'CRITICAL RISK: MARKET PANIC/LIQUIDATE'
    ]
    processed_df['agentic_action_signal'] = np.select(conditions, choices, default='NEUTRAL MARKET: MONITOR TREND')
    
    return processed_df

# Load System Environment Data
dataset_path = "finsight_ai_market_signals_dataset.csv"
data = load_and_process_agentic_data(dataset_path)

# 3. Main Dashboard Title Frame
st.title("AlphaMind Intelligence Analytics Terminal")
st.subheader("Multimodal Regime-Aware Trading Agent Framework")
st.markdown("---")

# 4. Global Core Performance Index Metric Blocks
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Total Tracked Assets", value=f"{data['asset_symbol'].nunique()}")
with col2:
    high_conf = len(data[data['agentic_action_signal'] == 'CONFIDENCE HIGH: PROCEED/EXECUTE'])
    st.metric(label="Confidence High Signals", value=high_conf)
with col3:
    panic_triggers = len(data[data['agentic_action_signal'] == 'CRITICAL RISK: MARKET PANIC/LIQUIDATE'])
    st.metric(label="Market Panic Triggers", value=panic_triggers)
with col4:
    st.metric(label="Mean Index Volatility", value=f"{data['volatility_score'].mean():.2f}")

st.markdown("---")

# 5. LIVE TESTING SELECTION INTERFACE (Restructured)
st.header("Live Agentic Testing & Asset Inspection Controls")

# Box 1: Signal Category Selection Dropdown
signal_options = sorted(list(data['agentic_action_signal'].unique()))
selected_signal = st.selectbox(
    "1. Choose Market Signal to Filter Dataset:", 
    options=signal_options,
    index=0
)

# Filter data based on first dropdown selection
filtered_data = data[data['agentic_action_signal'] == selected_signal]

# Box 2: NEW DROPDOWN - Specific Asset Selection inside that Signal Group
asset_options = sorted(list(filtered_data['asset_symbol'].unique()))
selected_asset = st.selectbox(
    "2. Select Specific Asset from filtered list to Inspect & Test Vibe:",
    options=asset_options
)

# Extract only that selected asset's specific row data
asset_data = filtered_data[filtered_data['asset_symbol'] == selected_asset].iloc[0]

# Displaying Individual Asset Core Insights Inside an Alert/Card layout
st.markdown(f"### Detailed Agentic Vibe Inspection for: **{selected_asset}**")

inspect_col1, inspect_col2 = st.columns([1, 1])

with inspect_col1:
    st.info(f"**News Headline:** {asset_data['news_headline']}")
    st.success(f"**Agent Reasoning Summary:** {asset_data['agent_reasoning_summary']}")

with inspect_col2:
    st.metric(label="Calculated News Sentiment Score", value=f"{asset_data['news_sentiment_score']}")
    st.metric(label="Market Price Change", value=f"{asset_data['market_price_change_pct']:.4f}%")
    st.write(f"**Asset Risk Level:** {asset_data['risk_label']}")
    st.write(f"**Market Cluster Group:** Cluster {asset_data['market_regime_cluster']}")

st.markdown("---")

# 6. Global Stats for the overall selected signal category
st.subheader("Group Summary Distribution for Selected Signal")
stat_col1, stat_col2, stat_col3 = st.columns(3)
with stat_col1:
    st.info(f"**Total Events in this Signal Group:** {len(filtered_data)}")
with stat_col2:
    st.success(f"**Group Average Price Deviation:** {filtered_data['market_price_change_pct'].mean():.4f}%")
with stat_col3:
    st.warning(f"**Group Average Social Buzz:** {filtered_data['social_media_buzz_score'].mean():.2f}")

# Data Table display layer
st.dataframe(filtered_data[['asset_symbol', 'asset_type', 'news_sentiment_score', 'market_price_change_pct', 'risk_label', 'agentic_action_signal']].reset_index(drop=True), use_container_width=True, height=200)

st.markdown("---")

# 7. Production-Grade Visualization Tabs Matrix Layer
st.header("Engineered Analytics Visualization Matrices")
tab1, tab2, tab3 = st.tabs(["Market Regime Mapping", "Price-Regime Bounds", "Text Flow Dynamics"])

with tab1:
    if os.path.exists('market_regime_clusters.png'):
        st.image('market_regime_clusters.png', caption='KMeans Multidimensional Structural Clustering Map', use_container_width=True)
    else:
        st.info("Run your visual EDA notebook to output 'market_regime_clusters.png' asset.")

with tab2:
    if os.path.exists('regime_price_distribution.png'):
        st.image('regime_price_distribution.png', caption='Price Variance Bounds and Signal Categories Across Clusters', use_container_width=True)
    else:
        st.info("Run your visual EDA notebook to output 'regime_price_distribution.png' asset.")

with tab3:
    if os.path.exists('text_signal_impact_matrix.png'):
        st.image('text_signal_impact_matrix.png', caption='Impact of Extracted Token Velocity Density on Returns Matrix', use_container_width=True)
    else:
        st.info("Run your text analytical engine block to output 'text_signal_impact_matrix.png' asset.")
