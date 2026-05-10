import pandas as pd
import numpy as np

def add_features(df):
    df = df.copy()

    # 1. PDR drop rate
    df['pdr_drop'] = 1 - df['PDR']

    # 2. Normalized control msg rate per node (z-score within mote)
    df['control_msg_rate_norm'] = df.groupby('mote_id')['control_msg_rate'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-5)
    )

    # 3. CMR spike ratio (vs rolling mean)
    df['cmr_spike'] = df['control_msg_rate'] / (
        df['control_msg_rate'].rolling(3, min_periods=1).mean() + 1e-5
    )

    # 4. Rank instability: combined rank + parent volatility
    df['topology_instability'] = (
        df['rank_change_rate'] * 0.5 + df['parent_change_rate'] * 0.5
    )

    # 5. Hop-weighted PDR pressure
    if 'hop_count' in df.columns:
        df['hop_pdr_pressure'] = df['pdr_drop'] * (df['hop_count'] + 1)
    else:
        df['hop_pdr_pressure'] = df['pdr_drop']

    # 6. Anomaly risk composite (heuristic, not used in training — used in trust)
    df['risk_composite'] = (
        df['pdr_drop'] * 0.4 +
        df['cmr_spike'].clip(0, 5) / 5 * 0.3 +
        df['topology_instability'].clip(0, 1) * 0.3
    )

    return df

FEATURES = [
    'PDR', 'parent_change_rate', 'rank_change_rate',
    'control_msg_rate', 'control_msg_rate_norm', 'cmr_spike', 'pdr_drop',
    'topology_instability', 'hop_pdr_pressure'
]
