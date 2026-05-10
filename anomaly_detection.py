import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, f1_score

MODEL_PATH = "model_cache.pkl"

def tune_and_detect(df, features, retrain=False):
    X = df[features].values
    y = df['label'].values

    # --- Load cached model if available ---
    if not retrain and os.path.exists(MODEL_PATH):
        print("[INFO] Loading cached model...")
        bundle = joblib.load(MODEL_PATH)
        scaler = bundle['scaler']
        model  = bundle['model']
    else:
        print("[INFO] Training RandomForest anomaly detector...")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_s, y_train)

        y_pred = model.predict(X_test_s)
        print("\n[RESULT] Model Performance (held-out test set):")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))

        # Cross-validation for robustness estimate
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            model, scaler.transform(X), y, cv=cv, scoring='f1_macro', n_jobs=-1
        )
        print(f"[INFO] 5-Fold CV F1 (macro): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Cache model
        joblib.dump({'scaler': scaler, 'model': model}, MODEL_PATH)
        print(f"[INFO] Model cached to {MODEL_PATH}")

    # Apply to full dataset
    X_full = scaler.transform(X)
    df = df.copy()
    df['anomaly']      = model.predict(X_full)
    df['anomaly_prob'] = model.predict_proba(X_full)[:, 1]

    return df
