"""
main.py — CLI entry point for the full RPL security pipeline.
"""

from preprocessing import load_data
from feature_engineering import add_features, FEATURES
from anomaly_detection import tune_and_detect
from intent_inference import infer
from trust_model import compute_trust
from clustering import cluster_nodes
from evaluation import evaluate
from acr import compute_acr
from aco_routing import run_aco

def run_pipeline(dataset_path="radar_dataset_clean_with_hops.csv", retrain=False):
    print("=" * 60)
    print("  RPL Attack Detection & Secure Routing Pipeline")
    print("=" * 60)

    df = load_data(dataset_path)
    df = add_features(df)
    df = tune_and_detect(df, FEATURES, retrain=retrain)
    df = infer(df)
    df = compute_trust(df)
    df = cluster_nodes(df)

    print("\n[INFO] Sample attack nodes:")
    print(df[df['anomaly'] == 1][['mote_id', 'intent', 'trust_score', 'cluster']].head(10))

    evaluate(df)
    compute_acr(df)
    run_aco(df)

    df.to_csv("final_results.csv", index=False)
    print("\n[DONE] Pipeline complete. Results saved to final_results.csv")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="radar_dataset_clean_with_hops.csv")
    parser.add_argument("--retrain", action="store_true", help="Force model retrain")
    args = parser.parse_args()
    run_pipeline(args.dataset, retrain=args.retrain)
