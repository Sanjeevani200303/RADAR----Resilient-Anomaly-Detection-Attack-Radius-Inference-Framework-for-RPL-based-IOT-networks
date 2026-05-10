import pandas as pd

REQUIRED_COLUMNS = [
    'mote_id', 'PDR', 'control_msg_rate',
    'parent_change_rate', 'rank_change_rate', 'label'
]

def load_data(path):
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"[ERROR] Missing required columns: {missing}")

    # Rename hop_to_attacker -> hop_count for consistency
    if 'hop_to_attacker' in df.columns and 'hop_count' not in df.columns:
        df = df.rename(columns={'hop_to_attacker': 'hop_count'})

    # Drop source_file if present (not useful for ML)
    df = df.drop(columns=['source_file'], errors='ignore')

    # Drop rows with nulls
    df = df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)

    print(f"[INFO] Loaded {len(df)} rows, {df['mote_id'].nunique()} unique nodes.")
    print(f"[INFO] Label distribution: {df['label'].value_counts().to_dict()}")

    return df
