"""
app.py — Streamlit dashboard for RPL Security System.
"""

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from feature_engineering import add_features, FEATURES
from anomaly_detection import tune_and_detect
from intent_inference import infer
from trust_model import compute_trust
from clustering import cluster_nodes, get_cluster_summary
from acr import compute_acr
from aco_routing import run_aco

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="RPL Security Dashboard", layout="wide")
st.title(" RPL Attack Detection & Secure Routing Dashboard")
st.caption("Anomaly detection · Intent inference · Trust scoring · Clustering · ACO routing")

# ── Upload ─────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(" Upload Dataset (.csv)", type=["csv"])

if uploaded_file is not None:
    st.success("Dataset uploaded ")

    try:
        with st.spinner("Running full pipeline... ⏳"):

            df = pd.read_csv(uploaded_file)

            if 'hop_to_attacker' in df.columns and 'hop_count' not in df.columns:
                df = df.rename(columns={'hop_to_attacker': 'hop_count'})
            df = df.drop(columns=['source_file'], errors='ignore')

            df = add_features(df)
            df = tune_and_detect(df, FEATURES)
            df = infer(df)
            df = compute_trust(df)
            df = cluster_nodes(df)          # silhouette k + PCA+DBSCAN eps

        # ── Alert banner ───────────────────────────────────────────────────────
        n_attacks = int(df['anomaly'].sum())
        if n_attacks > 0:
            st.error(f" {n_attacks} ATTACK NODE(S) DETECTED IN NETWORK")
        else:
            st.success(" NETWORK SECURE — No attacks detected")

        st.markdown("---")

        # ── Top metrics ────────────────────────────────────────────────────────
        st.subheader(" System Overview")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Records",  len(df))
        c2.metric("Unique Nodes",   df['mote_id'].nunique())
        c3.metric(" Attacks",     n_attacks)
        c4.metric(" Normal",      int((df['anomaly'] == 0).sum()))
        c5.metric("Avg Trust",      f"{df['trust_score'].mean():.3f}")

        st.markdown("---")

        # ── Row 1: Attack distribution | Trust histogram ───────────────────────
        row1_l, row1_r = st.columns(2)

        with row1_l:
            st.subheader(" Attack Intent Distribution")
            counts = df['intent'].value_counts()
            threshold   = 0.04 * counts.sum()
            small       = counts[counts < threshold]
            counts_clean = counts[counts >= threshold].copy()
            if len(small) > 0:
                counts_clean['Others'] = small.sum()

            fig, ax = plt.subplots(figsize=(5, 4))
            counts_clean.plot.pie(
                autopct='%1.1f%%', startangle=90,
                pctdistance=0.82,
                wedgeprops={'edgecolor': 'white'},
                ax=ax
            )
            ax.set_ylabel("")
            ax.set_title("Intent Distribution")
            st.pyplot(fig)
            plt.close()

        with row1_r:
            st.subheader(" Trust Score Distribution")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(df[df['anomaly'] == 0]['trust_score'], bins=20,
                    color='steelblue', alpha=0.7, label='Normal')
            ax.hist(df[df['anomaly'] == 1]['trust_score'], bins=20,
                    color='crimson', alpha=0.7, label='Attack')
            ax.set_xlabel("Trust Score")
            ax.set_ylabel("Node Count")
            ax.set_title("Trust Scores by Class")
            ax.legend()
            st.pyplot(fig)
            plt.close()

        st.markdown("---")

        # ── Row 2: Clustering | ACO Routing ───────────────────────────────────
        row2_l, row2_r = st.columns(2)

        with row2_l:
            st.subheader(" Node Cluster Summary")
            cluster_summary = get_cluster_summary(df)
            if not cluster_summary.empty:
                st.dataframe(cluster_summary.set_index('cluster_type'), use_container_width=True)

            st.subheader("Cluster Scatter (Trust vs PDR)")
            node_agg = df.groupby(['mote_id', 'cluster'])[['trust_score', 'PDR']].mean().reset_index()
            fig, ax = plt.subplots(figsize=(5, 4))
            clusters = node_agg['cluster'].unique()
            cmap = plt.cm.get_cmap('tab10', len(clusters))
            for i, c in enumerate(sorted(clusters)):
                sub    = node_agg[node_agg['cluster'] == c]
                label  = 'Outlier' if c == -1 else f'Cluster {c}'
                marker = 'x'      if c == -1 else 'o'
                ax.scatter(sub['PDR'], sub['trust_score'],
                           label=label, color=cmap(i), marker=marker, s=60)
            ax.set_xlabel("PDR")
            ax.set_ylabel("Trust Score")
            ax.set_title("Node Clusters (silhouette-selected k, PCA+DBSCAN outliers)")
            ax.legend(fontsize=8)
            st.pyplot(fig)
            plt.close()

        with row2_r:
            st.subheader(" ACO Secure Route")
            st.caption("Route order is the ACO-discovered hop sequence. "
                       "Score = min trust along path (weakest-link).")
            route, graph_data = run_aco(df)
            if not route.empty:
                st.dataframe(route, use_container_width=True)

                # ── 1. Network topology graph with pheromone trails ────────
                st.subheader(" Network Graph — Pheromone Trails & Best Path")
                st.caption(
                    "Edge thickness = pheromone strength. "
                    "🟢 Best-path nodes  🔴 High-risk / attack nodes  ⚪ Normal nodes"
                )

                candidates  = graph_data['candidates']
                adj         = graph_data['adj']
                tau         = graph_data['tau']
                best_path   = graph_data['best_path']
                n_nodes     = len(candidates)

                # Hierarchical layout: x = hop level, y = spread within hop
                hop_col = 'hop_count' if 'hop_count' in candidates.columns else None
                pos_x, pos_y = {}, {}
                if hop_col:
                    from collections import defaultdict as ddict
                    hop_buckets = ddict(list)
                    for i in range(n_nodes):
                        hop_buckets[int(candidates.iloc[i][hop_col])].append(i)
                    for hop, nodes in hop_buckets.items():
                        for rank, nidx in enumerate(nodes):
                            pos_x[nidx] = hop
                            pos_y[nidx] = rank - len(nodes) / 2
                else:
                    # Fallback: circle layout
                    for i in range(n_nodes):
                        angle = 2 * np.pi * i / n_nodes
                        pos_x[i], pos_y[i] = np.cos(angle), np.sin(angle)

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.set_facecolor('#0e1117')
                fig.patch.set_facecolor('#0e1117')

                # Draw all graph edges — thickness proportional to pheromone
                tau_max = tau.max() if tau.max() > 0 else 1.0
                drawn_edges = set()
                for i, neighbours in adj.items():
                    for j in neighbours:
                        edge = (min(i, j), max(i, j))
                        if edge in drawn_edges or i >= n_nodes or j >= n_nodes:
                            continue
                        drawn_edges.add(edge)
                        strength = float(tau[i, j]) / tau_max
                        lw    = 0.3 + 3.5 * strength
                        alpha = 0.15 + 0.55 * strength
                        ax.plot([pos_x[i], pos_x[j]], [pos_y[i], pos_y[j]],
                                color='#4a9eff', linewidth=lw, alpha=alpha, zorder=1)

                # Highlight best-path edges in bright gold
                best_path_set = set(best_path)
                for k in range(len(best_path) - 1):
                    i, j = best_path[k], best_path[k + 1]
                    ax.plot([pos_x[i], pos_x[j]], [pos_y[i], pos_y[j]],
                            color='#ffd700', linewidth=3.5, alpha=0.95,
                            zorder=3, solid_capstyle='round')
                    # Ant direction arrow
                    mx = (pos_x[i] + pos_x[j]) / 2
                    my = (pos_y[i] + pos_y[j]) / 2
                    dx = (pos_x[j] - pos_x[i]) * 0.01
                    dy = (pos_y[j] - pos_y[i]) * 0.01
                    ax.annotate("", xy=(mx + dx, my + dy), xytext=(mx - dx, my - dy),
                                arrowprops=dict(arrowstyle='->', color='#ffd700',
                                                lw=1.8), zorder=4)

                # Draw nodes
                trust_vals = candidates['trust_score'].values
                for i in range(n_nodes):
                    trust = float(trust_vals[i])
                    in_path = i in best_path_set

                    if in_path:
                        color  = '#00e676'   # bright green — best path
                        size   = 180
                        zorder = 5
                        edge_c = 'white'
                    elif trust < 0.3:
                        color  = '#ff4444'   # red — low trust / near-attack
                        size   = 100
                        zorder = 4
                        edge_c = '#ff4444'
                    else:
                        # Colour by trust: blue → teal → green gradient
                        r = 1.0 - trust
                        color  = (r * 0.2, 0.5 + trust * 0.4, 0.8)
                        size   = 60 + trust * 60
                        zorder = 3
                        edge_c = 'none'

                    ax.scatter(pos_x[i], pos_y[i], s=size, c=[color],
                               edgecolors=edge_c, linewidths=0.8,
                               zorder=zorder, alpha=0.92)

                    # Label best-path nodes with mote_id
                    if in_path:
                        rank = best_path.index(i) + 1
                        ax.text(pos_x[i], pos_y[i] + 0.35,
                                f"#{rank}\n{candidates.iloc[i]['mote_id']}",
                                ha='center', va='bottom', fontsize=6.5,
                                color='#ffd700', fontweight='bold', zorder=6)

                # Axis labels & legend
                if hop_col:
                    ax.set_xlabel("Hop Level (0 = root)", color='#cccccc', fontsize=9)
                ax.set_ylabel("Node spread within hop", color='#cccccc', fontsize=9)
                ax.tick_params(colors='#888888')
                ax.set_title("RPL Network — ACO Pheromone Map", color='white', fontsize=11)
                for spine in ax.spines.values():
                    spine.set_edgecolor('#333333')

                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='#00e676',
                           markersize=9, label='Best-path node'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff4444',
                           markersize=7, label='Low-trust node'),
                    Line2D([0], [0], color='#ffd700', linewidth=2.5, label='Best path (gold)'),
                    Line2D([0], [0], color='#4a9eff', linewidth=1.5,
                           alpha=0.7, label='Pheromone edge'),
                ]
                ax.legend(handles=legend_elements, loc='upper right',
                          facecolor='#1a1a2e', edgecolor='#444', labelcolor='white',
                          fontsize=7.5)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # ── 2. Pheromone heatmap (top active nodes) ───────────────
                st.subheader(" Pheromone Heatmap")
                st.caption("Shows which node-pairs accumulated the most pheromone trail.")
                top_n = min(15, n_nodes)
                top_idx = np.argsort(tau.sum(axis=1))[::-1][:top_n]
                tau_sub = tau[np.ix_(top_idx, top_idx)]
                labels  = [str(candidates.iloc[i]['mote_id']) for i in top_idx]

                fig2, ax2 = plt.subplots(figsize=(6, 5))
                im = ax2.imshow(tau_sub, cmap='YlOrRd', aspect='auto')
                plt.colorbar(im, ax=ax2, label='Pheromone τ')
                ax2.set_xticks(range(top_n))
                ax2.set_yticks(range(top_n))
                ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
                ax2.set_yticklabels(labels, fontsize=7)
                ax2.set_title(f"Top-{top_n} Nodes — Pheromone Matrix", fontsize=10)
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()

                # ── 3. Convergence curve ──────────────────────────────────
                st.subheader(" ACO Convergence Curve")
                iter_scores = graph_data['iter_scores']
                fig3, ax3 = plt.subplots(figsize=(6, 2.5))
                ax3.plot(range(1, len(iter_scores) + 1), iter_scores,
                         color='teal', linewidth=2, marker='o', markersize=3)
                ax3.fill_between(range(1, len(iter_scores) + 1), iter_scores,
                                 alpha=0.15, color='teal')
                ax3.set_xlabel("Iteration")
                ax3.set_ylabel("Best Path Score\n(min trust)")
                ax3.set_title("ACO Convergence — Best Score per Iteration")
                ax3.set_ylim(0, 1)
                plt.tight_layout()
                st.pyplot(fig3)
                plt.close()

                # ── 4. Trust profile ──────────────────────────────────────
                st.subheader(" Trust Profile Along Best Path")
                fig4, ax4 = plt.subplots(figsize=(6, 2.5))
                ax4.plot(route['aco_rank'], route['trust_score'],
                         marker='o', color='teal', linewidth=2)
                ax4.fill_between(route['aco_rank'], route['trust_score'],
                                 alpha=0.2, color='teal')
                ax4.axhline(route['trust_score'].min(), color='crimson',
                            linestyle='--', linewidth=1,
                            label=f"Min trust = {route['trust_score'].min():.3f}")
                ax4.set_xlabel("ACO Hop Order")
                ax4.set_ylabel("Trust Score")
                ax4.set_ylim(0, 1)
                ax4.legend(fontsize=8)
                plt.tight_layout()
                st.pyplot(fig4)
                plt.close()

            else:
                st.warning("No route found — all nodes compromised.")

        st.markdown("---")

        # ── Row 3: ACR | High-risk nodes ──────────────────────────────────────
        row3_l, row3_r = st.columns(2)

        with row3_l:
            st.subheader(" Attack Containment Radius (ACR)")
            acr_value = compute_acr(df)
            st.metric("ACR (max hop distance)", acr_value)

            if 'hop_count' in df.columns and n_attacks > 0:
                hop_dist = (
                    df[df['anomaly'] == 1]
                    .groupby('hop_count')['mote_id'].nunique()
                    .reset_index()
                    .rename(columns={'mote_id': 'attack_nodes'})
                )
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.bar(hop_dist['hop_count'].astype(str),
                       hop_dist['attack_nodes'], color='crimson')
                ax.set_xlabel("Hop from Root")
                ax.set_ylabel("Attack Nodes")
                ax.set_title("Attack Spread by Hop Level")
                st.pyplot(fig)
                plt.close()

        with row3_r:
            st.subheader(" Highest-Risk Nodes")
            cols  = ['mote_id', 'trust_score', 'anomaly_prob', 'intent', 'cluster']
            avail = [c for c in cols if c in df.columns]
            st.dataframe(
                df.sort_values('trust_score')[avail].head(12),
                use_container_width=True
            )

        st.markdown("---")

        with st.expander(" How the system works"):
            st.markdown("""
            | Component | Description |
            |-----------|-------------|
            | **Feature Engineering** | PDR drop, CMR spike, topology instability, hop-weighted pressure |
            | **Anomaly Detection** | Random Forest (300 trees, balanced, cached after first run) |
            | **Intent Inference** | Data-derived thresholds (percentile-based); learned RF if `attack_type` labels present |
            | **Trust Model** | Probabilistic decay + fixed hop penalty (bounded, correct for all hop values) |
            | **Clustering** | Silhouette-selected KMeans k · PCA(2) + k-distance elbow DBSCAN |
            | **ACO Routing** | RPL topology graph · min-trust path score · hop-order preserved · early stopping |
            | **ACR** | Max hop distance of confirmed attack nodes from root |
            """)

        st.success(" Analysis Complete")

    except Exception as e:
        st.error(f" Error: {e}")
        st.exception(e)
