"""
acr.py — Attack Containment Radius (ACR).

Computes ACR as the maximum hop distance of confirmed attack nodes
from the root (hop_count = 0), giving a meaningful radius of spread.
Falls back to unique attacker node count if hop_count unavailable.
"""

def compute_acr(df):
    attack_df = df[df['anomaly'] == 1]

    if attack_df.empty:
        print("[INFO] ACR: No attack nodes detected.")
        return 0

    if 'hop_count' in df.columns:
        acr   = int(attack_df['hop_count'].max())
        label = "ACR (max hop distance from root)"
    else:
        acr   = int(attack_df['mote_id'].nunique())
        label = "ACR (unique attacker nodes)"

    # Also report attack density per hop level
    if 'hop_count' in df.columns:
        hop_dist = attack_df.groupby('hop_count')['mote_id'].nunique()
        print(f"[INFO] Attack nodes per hop level:\n{hop_dist.to_string()}")

    print(f"[RESULT] {label}: {acr}")
    return acr
