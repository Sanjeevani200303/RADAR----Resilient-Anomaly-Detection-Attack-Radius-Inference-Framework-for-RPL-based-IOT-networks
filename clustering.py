"""
clustering.py — Node clustering using KMeans + DBSCAN ensemble.

Improvements over v1:
  - Optimal k selected via silhouette scoring (not hardcoded 4).
  - DBSCAN eps tuned via k-distance elbow (not hardcoded 1.2).
  - PCA(2) applied before DBSCAN to reduce distance distortion in 6D.
  - All parameters exposed and logged for reproducibility.
  - Cluster-to-ACO feedback remains via 'is_safe_cluster'.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

CLUSTER_FEATURES = [
    'PDR', 'control_msg_rate', 'parent_change_rate',
    'rank_change_rate', 'trust_score', 'anomaly_prob'
]


def _best_k(X_scaled, k_range=range(2, 8)):
    """Select optimal KMeans k via silhouette score."""
    best_k, best_score = 2, -1
    for k in k_range:
        if k >= len(X_scaled):
            break
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_score, best_k = score, k
    print(f"[INFO] Optimal k={best_k} (silhouette={best_score:.4f})")
    return best_k


def _best_eps(X_pca, min_samples=2):
    """
    Estimate DBSCAN eps via k-distance elbow.
    Uses the knee of the sorted k-NN distances — a principled approach
    instead of a hardcoded magic number.
    """
    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X_pca)
    distances, _ = nbrs.kneighbors(X_pca)
    k_distances = np.sort(distances[:, -1])

    # Simple elbow: largest second-derivative point
    if len(k_distances) < 4:
        return float(np.percentile(k_distances, 75))

    diffs = np.diff(k_distances)
    diffs2 = np.diff(diffs)
    elbow_idx = int(np.argmax(diffs2)) + 1
    eps = float(k_distances[elbow_idx])
    eps = max(eps, 0.1)   # floor
    print(f"[INFO] DBSCAN eps (k-distance elbow) = {eps:.4f}")
    return eps


def cluster_nodes(df, n_clusters=None):
    """
    Aggregate per-node metrics and cluster nodes.

    Steps:
      1. Aggregate rows per mote_id (mean of behavioral features).
      2. StandardScaler normalisation.
      3. Silhouette-selected KMeans for broad cluster assignment.
      4. PCA(2) → DBSCAN to flag outlier nodes with tuned eps.
      5. Final cluster label: DBSCAN outliers get cluster = -1.

    Returns df with 'cluster', 'is_outlier', 'is_safe_cluster' merged in.
    """
    print("[INFO] Clustering nodes...")

    avail = [f for f in CLUSTER_FEATURES if f in df.columns]
    if len(avail) < 2:
        print("[WARNING] Not enough features for clustering. Skipping.")
        df['cluster'] = 0
        df['is_outlier'] = False
        df['is_safe_cluster'] = True
        return df

    node_df = df.groupby('mote_id')[avail].mean().reset_index()
    X = node_df[avail].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- KMeans with silhouette-selected k ---
    if n_clusters is None:
        k = _best_k(X_scaled, k_range=range(2, min(8, len(node_df))))
    else:
        k = max(2, min(n_clusters, len(node_df) - 1))

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    node_df['kmeans_cluster'] = kmeans.fit_predict(X_scaled)

    # --- PCA(2) before DBSCAN to avoid curse of dimensionality ---
    n_components = min(2, X_scaled.shape[1], X_scaled.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_.sum()
    print(f"[INFO] PCA({n_components}) explains {explained*100:.1f}% of variance")

    # --- DBSCAN with tuned eps ---
    min_samples = max(2, len(node_df) // 20)   # scale with dataset size
    eps = _best_eps(X_pca, min_samples=min_samples)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    node_df['dbscan_label'] = dbscan.fit_predict(X_pca)
    node_df['is_outlier'] = node_df['dbscan_label'] == -1

    # Final cluster: outliers = -1, others keep KMeans label
    node_df['cluster'] = np.where(
        node_df['is_outlier'], -1, node_df['kmeans_cluster']
    )

    # --- Identify safest cluster ---
    non_outliers = node_df[node_df['cluster'] >= 0]
    if not non_outliers.empty and 'trust_score' in avail and 'anomaly_prob' in avail:
        cluster_scores = (
            non_outliers.groupby('cluster')
            .apply(lambda g: g['trust_score'].mean() - g['anomaly_prob'].mean())
        )
        best_cluster = int(cluster_scores.idxmax())
    else:
        best_cluster = int(node_df['cluster'].mode()[0])

    node_df['is_safe_cluster'] = node_df['cluster'] == best_cluster

    print(f"[INFO] Clusters: {node_df['cluster'].value_counts().to_dict()}")
    print(f"[INFO] Outlier nodes: {node_df['is_outlier'].sum()}")
    print(f"[INFO] Safest cluster id: {best_cluster}")

    df = df.merge(
        node_df[['mote_id', 'cluster', 'is_outlier', 'is_safe_cluster']],
        on='mote_id', how='left'
    )
    return df


def get_cluster_summary(df):
    """Returns a per-cluster summary DataFrame for display."""
    if 'cluster' not in df.columns:
        return pd.DataFrame()

    cols = ['cluster', 'trust_score', 'anomaly_prob', 'PDR']
    avail = [c for c in cols if c in df.columns]

    summary = df.groupby('cluster')[avail[1:]].mean().round(3)
    summary['node_count']   = df.groupby('cluster')['mote_id'].nunique()
    summary['attack_nodes'] = df.groupby('cluster')['anomaly'].sum()
    summary = summary.reset_index()
    summary['cluster_type'] = summary['cluster'].apply(
        lambda x: 'Outlier/Suspicious' if x == -1 else f'Cluster {x}'
    )
    return summary
