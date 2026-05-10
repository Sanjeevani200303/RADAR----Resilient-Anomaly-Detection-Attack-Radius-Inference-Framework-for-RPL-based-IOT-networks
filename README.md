# RPL Attack Detection & Secure Routing

End-to-end security pipeline for RPL (Routing Protocol for Low-Power and Lossy Networks).

## File Structure

```
rpl_security/
├── preprocessing.py        # CSV loading & validation
├── feature_engineering.py  # Feature derivation (PDR drop, CMR spike, etc.)
├── anomaly_detection.py    # Random Forest classifier (cached)
├── intent_inference.py     # Hybrid rule+learned intent classifier
├── trust_model.py          # Decaying trust model (fixed hop penalty)
├── clustering.py           # KMeans (silhouette k) + PCA+DBSCAN ensemble
├── aco_routing.py          # ACO on RPL topology graph (min-trust scoring)
├── acr.py                  # Attack Containment Radius
├── evaluation.py           # Confusion matrix + AUC metrics
├── main.py                 # CLI pipeline entry point
├── app.py                  # Streamlit dashboard
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt

# CLI
python main.py --dataset radar_dataset_clean_with_hops.csv

# Dashboard
streamlit run app.py
```

## Key Improvements (v2)

| Component | Change |
|-----------|--------|
| **Clustering** | Silhouette scoring selects optimal k; PCA(2) before DBSCAN; eps via k-distance elbow |
| **ACO** | Real RPL hop-topology graph; min-trust path scoring; hop order preserved in output; early stopping |
| **Trust model** | Hop penalty formula fixed — correct and bounded for all hop values |
| **Intent** | Thresholds derived from data percentiles; optional learned RF classifier |
