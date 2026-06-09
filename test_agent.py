import pandas as pd
import numpy as np

# Ek dummy fake test scenario banate hain (Jaise kal ko market mein ye news aayi)
test_scenarios = pd.DataFrame([
    {
        'asset_symbol': 'BTC',
        'volatility_score': 22.5, # Kam darr/risk
        'trading_volume_change_pct': 35.0, # Steady volume
        'social_media_buzz_score': 55.0, # Normal buzz
        'news_sentiment_score': 0.85, # Bohot solid positive sentiment
        'agent_reasoning_summary': "Strong growth expected after institutional breakout above key resistance.",
        'market_regime_cluster': 0, # Stable organic cluster
        'risk_label': 'Low'
    },
    {
        'asset_symbol': 'ETH',
        'volatility_score': 72.0, # Bohot zyada darr aur crash ka mahool
        'trading_volume_change_pct': 85.0, # Heavy panic volume
        'social_media_buzz_score': 90.0, # Extreme crowd hysteria
        'news_sentiment_score': -0.75, # Negative sentiment
        'agent_reasoning_summary': "Elevated risk detected as regulatory drop creates massive negative pressure and weak support.",
        'market_regime_cluster': 1, # Speculative chaotic cluster
        'risk_label': 'High'
    }
])

# Token calculation functions jo humne pipeline mein banaye the
negative_tokens = ['weak', 'negative', 'drop', 'sell', 'elevated', 'risk']
positive_tokens = ['growth', 'breakout', 'buy', 'positive', 'strong', 'momentum']

test_scenarios['reasoning_clean'] = test_scenarios['agent_reasoning_summary'].astype(str).str.lower()
test_scenarios['neg_density'] = test_scenarios['reasoning_clean'].apply(lambda x: sum(1 for t in negative_tokens if t in x))
test_scenarios['pos_density'] = test_scenarios['reasoning_clean'].apply(lambda x: sum(1 for t in positive_tokens if t in x))
test_scenarios['net_text_velocity'] = test_scenarios['pos_density'] - test_scenarios['neg_density']

# Rule Matrix Trigger
conditions = [
    (test_scenarios['market_regime_cluster'] == 0) & (test_scenarios['net_text_velocity'] >= 0) & (test_scenarios['news_sentiment_score'] > 0.2),
    (test_scenarios['market_regime_cluster'] == 2) | (test_scenarios['volatility_score'] > 60) | (test_scenarios['risk_label'].str.lower() == 'high'),
    (test_scenarios['net_text_velocity'] < 0) & (test_scenarios['news_sentiment_score'] < -0.2)
]
choices = [
    'CONFIDENCE HIGH: PROCEED/EXECUTE',
    'ELEVATED RISK: HOLD POSITION/STAY CAUTIOUS',
    'CRITICAL RISK: MARKET PANIC/LIQUIDATE'
]
test_scenarios['agentic_action_signal'] = np.select(conditions, choices, default='NEUTRAL MARKET: MONITOR TREND')

# Output Check
print("=== LIVE SCENARIO TESTING OUTPUT ===")
for idx, row in test_scenarios.iterrows():
    print(f"\nAsset: {row['asset_symbol']}")
    print(f"News Headline Vibe: {row['agent_reasoning_summary']}")
    print(f"Model Intelligent Output Signal -> {row['agentic_action_signal']}")
