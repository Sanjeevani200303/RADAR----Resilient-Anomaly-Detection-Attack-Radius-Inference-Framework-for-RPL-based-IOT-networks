"""
intent_inference.py — Hybrid attack intent classifier.

Improvements over v1:
  - Rule thresholds are no longer magic numbers — derived from training
    data percentiles so they adapt to each dataset.
  - A lightweight RandomForest is trained as a secondary classifier when
    labelled attack sub-types are available in the dataset ('attack_type' col).
  - Falls back cleanly to rule-based classification when sub-type labels
    are absent (preserving backward compatibility).
  - All thresholds are logged so they're inspectable and reproducible.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

INTENT_LABELS = [
    'Normal', 'Topology Disruption', 'Packet Drop',
    'Flooding', 'Traffic Attraction', 'Sinkhole', 'Stealth/Mixed-Attack'
]

# ── Threshold derivation ───────────────────────────────────────────────────────

def _derive_thresholds(df):
    """
    Derive classification thresholds from the attack sub-population
    using percentiles instead of hardcoded values.
    Returns a dict of thresholds logged for transparency.
    """
    attack = df[df['anomaly'] == 1]
    if attack.empty:
        # Sensible defaults if no attacks yet
        return {
            'pcr_high':  5.0,
            'pdr_low':   0.5,
            'cmr_spike': 2.0,
            'rcr_sink':  0.01,
            'rcr_attr':  0.02,
            'hop_sink':  1,
        }

    pcr_high  = float(np.percentile(attack['parent_change_rate'], 75))
    pdr_low   = float(np.percentile(attack['PDR'], 25))
    cmr_spike = float(np.percentile(attack.get('cmr_spike', attack['control_msg_rate']), 75))
    rcr_sink  = float(np.percentile(attack['rank_change_rate'], 10))
    rcr_attr  = float(np.percentile(attack['rank_change_rate'], 20))
    hop_sink  = int(np.percentile(attack.get('hop_count', pd.Series([1])), 25)) if 'hop_count' in attack.columns else 1

    thresholds = {
        'pcr_high': pcr_high, 'pdr_low': pdr_low,
        'cmr_spike': cmr_spike, 'rcr_sink': rcr_sink,
        'rcr_attr': rcr_attr, 'hop_sink': hop_sink,
    }
    print(f"[INFO] Intent thresholds derived from data: {thresholds}")
    return thresholds


# ── Rule-based classifier ──────────────────────────────────────────────────────

def _rule_classify(row, t):
    pcr = float(row['parent_change_rate'])
    pdr = float(row['PDR'])
    cmr = float(row.get('cmr_spike', 0))
    rcr = float(row['rank_change_rate'])
    hop = float(row.get('hop_count', 3))

    if rcr < t['rcr_sink'] and hop <= t['hop_sink']:
        return 'Sinkhole'
    if pcr > t['pcr_high']:
        return 'Topology Disruption'
    if pdr < t['pdr_low']:
        return 'Packet Drop'
    if cmr > t['cmr_spike']:
        return 'Flooding'
    if rcr < t['rcr_attr']:
        return 'Traffic Attraction'
    return 'Stealth/Mixed-Attack'


# ── Learned classifier (optional) ─────────────────────────────────────────────

def _try_train_learned(df, features):
    """
    If 'attack_type' column exists with multi-class labels, train a
    lightweight RandomForest to classify intent.
    Returns (model, le, features) or None if not possible.
    """
    if 'attack_type' not in df.columns:
        return None

    labelled = df[df['anomaly'] == 1].dropna(subset=['attack_type'])
    if labelled['attack_type'].nunique() < 2 or len(labelled) < 20:
        print("[INFO] Not enough labelled attack types for learned classifier.")
        return None

    feats = [f for f in features if f in labelled.columns]
    X = labelled[feats].fillna(0).values
    le = LabelEncoder()
    y = le.fit_transform(labelled['attack_type'].values)

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=6,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    clf.fit(X, y)
    print(f"[INFO] Learned intent classifier trained on {len(labelled)} samples, "
          f"classes: {list(le.classes_)}")
    return clf, le, feats


# ── Public API ─────────────────────────────────────────────────────────────────

_FEATURES_FOR_LEARNED = [
    'PDR', 'parent_change_rate', 'rank_change_rate',
    'control_msg_rate', 'cmr_spike', 'topology_instability',
    'hop_pdr_pressure', 'hop_count'
]


def infer(df):
    df = df.copy()

    thresholds    = _derive_thresholds(df)
    learned_model = _try_train_learned(df, _FEATURES_FOR_LEARNED)

    def classify_row(row):
        if row['anomaly'] != 1:
            return 'Normal'
        # Prefer learned classifier when available
        if learned_model is not None:
            clf, le, feats = learned_model
            x = np.array([[float(row.get(f, 0)) for f in feats]])
            return str(le.inverse_transform(clf.predict(x))[0])
        return _rule_classify(row, thresholds)

    df['intent'] = df.apply(classify_row, axis=1)
    print("[INFO] Intent distribution:", df['intent'].value_counts().to_dict())
    return df
