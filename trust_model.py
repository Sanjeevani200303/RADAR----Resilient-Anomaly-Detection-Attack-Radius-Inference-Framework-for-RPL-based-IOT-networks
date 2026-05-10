"""
trust_model.py — Decaying trust model with anomaly probability weighting.

Improvements over v1:
  - Fixed hop penalty formula: previously broke for hop > 3, giving a
    positive bonus instead of a penalty. Now uses a symmetric decay
    that correctly increases penalty as hop_count decreases (closer to root
    means more exposure to attacker traffic).
  - Penalty curve is now documented and bounded.
"""


def compute_trust(df):
    print("[INFO] Computing trust scores...")

    df = df.copy()

    def score(row):
        # Hard zero for confirmed high-probability anomalies
        if row.get('anomaly', 0) == 1 and row.get('anomaly_prob', 1.0) > 0.75:
            return 0.0

        # Base behavioural trust
        pdr_score    = float(row['PDR'])
        parent_score = 1.0 - min(float(row['parent_change_rate']) / 10.0, 1.0)
        rank_score   = 1.0 - min(float(row['rank_change_rate']), 1.0)

        base = 0.5 * pdr_score + 0.3 * parent_score + 0.2 * rank_score

        # Penalise proportionally by anomaly probability
        anomaly_prob = float(row.get('anomaly_prob', 0.0))
        base *= (1.0 - 0.8 * anomaly_prob)

        # ── Fixed hop penalty ─────────────────────────────────────────────
        # v1 bug: (1 - 0.15*(3-hop)) is negative for hop > 3, granting bonus.
        # Fix: penalty scales with proximity to root (small hop = more exposure).
        # Formula: penalty = 0.15 / (hop + 1)  →  bounded ∈ (0, 0.15], always positive.
        # hop=0 (root): penalty=0.150,  hop=1: 0.075,  hop=3: 0.038,  hop=9: 0.015
        hop         = float(row.get('hop_count', 3))
        hop_penalty = max(0.0, 1.0 - 0.15 / (hop + 1.0))
        base       *= hop_penalty

        return round(max(0.0, min(base, 1.0)), 4)

    df['trust_score'] = df.apply(score, axis=1)

    print(f"[INFO] Trust scores — mean: {df['trust_score'].mean():.3f}, "
          f"min: {df['trust_score'].min():.3f}, max: {df['trust_score'].max():.3f}")
    return df
