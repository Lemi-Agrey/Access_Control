"""
Step 4: Anomaly Detection on Access Requests
=============================================
Even with privacy-preserving ABAC, attackers may try to gain access by
submitting *adversarial* requests — e.g., claiming an inflated role,
bursts of requests, accesses from unusual locations at unusual hours,
or requesting unusually sensitive objects.

We inject synthetic "adversarial" requests into the test set and measure
how well three unsupervised anomaly detectors can flag them, having
been trained only on the (un-labelled) normal traffic.

Detectors compared:
  - Isolation Forest
  - One-Class SVM
  - Local Outlier Factor

Outputs:
  - results/anomaly_results.csv
  - results/anomaly_detection.png
"""

import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing  import StandardScaler, OneHotEncoder
from sklearn.compose        import ColumnTransformer
from sklearn.ensemble       import IsolationForest
from sklearn.svm            import OneClassSVM
from sklearn.neighbors      import LocalOutlierFactor
from sklearn.metrics        import (precision_score, recall_score,
                                    f1_score, roc_auc_score)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Build synthetic adversarial requests.  These deliberately violate the
# statistical norms of the dataset — they look "weird" without literally
# being labelled as attacks.
# ---------------------------------------------------------------------------
def synthesise_adversarial(n, columns):
    """Generate n adversarial rows with extreme/unusual attribute combos."""
    rows = []
    for _ in range(n):
        attack_type = np.random.choice(["burst", "off_hours_remote",
                                        "low_clearance_high_sens",
                                        "role_object_mismatch"])

        # Defaults
        row = {
            "subj_role":                 np.random.choice(
                ["doctor", "nurse", "patient", "researcher",
                 "admin", "pharmacist"]),
            "subj_department":           np.random.choice(
                ["cardiology", "oncology", "general", "research", "admin_office"]),
            "subj_clearance":            np.random.randint(1, 6),
            "subj_years_exp":            np.random.randint(0, 41),
            "subj_cert_active":          bool(np.random.randint(0, 2)),
            "subj_age":                  np.random.randint(22, 66),
            "obj_type":                  np.random.choice(
                ["medical_record", "lab_result", "prescription",
                 "billing", "research_data", "medical_image"]),
            "obj_sensitivity":           np.random.randint(1, 6),
            "obj_department":            np.random.choice(
                ["cardiology", "oncology", "general", "research", "admin_office"]),
            "obj_contains_pii":          bool(np.random.randint(0, 2)),
            "env_hour":                  np.random.randint(0, 24),
            "env_is_emergency":          False,
            "env_location":              np.random.choice(
                ["hospital_network", "remote", "office_network"]),
            "env_access_freq_last_hour": np.random.randint(0, 20),
            "action":                    np.random.choice(
                ["read", "write", "delete", "share"]),
        }

        # Apply attack-specific anomalies
        if attack_type == "burst":
            row["env_access_freq_last_hour"] = np.random.randint(60, 200)
        elif attack_type == "off_hours_remote":
            row["env_hour"]     = np.random.choice([1, 2, 3, 4, 23])
            row["env_location"] = "remote"
            row["subj_cert_active"] = False
        elif attack_type == "low_clearance_high_sens":
            row["subj_clearance"]  = 1
            row["obj_sensitivity"] = 5
            row["action"]          = np.random.choice(["delete", "share"])
        elif attack_type == "role_object_mismatch":
            row["subj_role"] = "patient"
            row["obj_type"]  = np.random.choice(["billing", "research_data"])
            row["action"]    = "write"

        rows.append(row)

    return pd.DataFrame(rows)[columns]


def build_preprocessor(X):
    cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(),                       num_cols),
    ])


def evaluate(name, y_true, y_pred, score=None):
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score   (y_true, y_pred, zero_division=0)
    f1  = f1_score       (y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, score) if score is not None else np.nan
    print(f"{name:<22}  prec={pre:.4f}  recall={rec:.4f}  "
          f"F1={f1:.4f}  AUC={auc:.4f}")
    return {"Detector": name, "Precision": pre, "Recall": rec,
            "F1": f1, "AUC": auc}


def main():
    print("=" * 60)
    print("STEP 4 — Anomaly / Adversarial Access Detection")
    print("=" * 60)

    df = pd.read_csv("data/abac_dataset.csv")
    X  = df.drop(columns=["decision"])
    columns = X.columns.tolist()

    # Use 80% of normal traffic to fit detectors (unsupervised — no labels)
    n_train = int(0.8 * len(X))
    X_train = X.iloc[:n_train].copy()

    # Test set: remaining normal traffic + injected adversarial requests
    X_normal_test = X.iloc[n_train:].copy()
    n_adv = 500
    X_adv = synthesise_adversarial(n_adv, columns)
    X_test  = pd.concat([X_normal_test, X_adv], ignore_index=True)
    y_test  = np.concatenate([np.zeros(len(X_normal_test)),  # 0 = normal
                              np.ones (len(X_adv))])         # 1 = anomaly
    print(f"Train (normal only): {len(X_train):,}  "
          f"Test: {len(X_normal_test):,} normal + {n_adv} adversarial = "
          f"{len(X_test):,}\n")

    # Preprocess (fit on training only to avoid leakage)
    pre = build_preprocessor(X_train)
    pre.fit(X_train)
    Xtr_e = pre.transform(X_train)
    Xte_e = pre.transform(X_test)
    if hasattr(Xtr_e, "toarray"):
        Xtr_e = Xtr_e.toarray()
        Xte_e = Xte_e.toarray()

    results = []

    # All three detectors are tuned to flag ~15% of inputs as anomalous,
    # so the comparison is on equal terms.  In practice you would tune
    # this to your expected attack rate.
    CONTAMINATION = 0.15

    # -- Isolation Forest ---------------------------------------------------
    iso = IsolationForest(n_estimators=200, contamination=CONTAMINATION,
                          random_state=RANDOM_SEED, n_jobs=-1)
    iso.fit(Xtr_e)
    pred  = (iso.predict(Xte_e) == -1).astype(int)
    # Lower score_samples = more anomalous, so negate for AUC
    score = -iso.score_samples(Xte_e)
    results.append(evaluate("Isolation Forest", y_test, pred, score))

    # -- One-Class SVM ------------------------------------------------------
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=CONTAMINATION)
    ocsvm.fit(Xtr_e)
    pred  = (ocsvm.predict(Xte_e) == -1).astype(int)
    score = -ocsvm.score_samples(Xte_e)
    results.append(evaluate("One-Class SVM", y_test, pred, score))

    # -- Local Outlier Factor (with novelty=True for prediction on new data)
    lof = LocalOutlierFactor(n_neighbors=20, contamination=CONTAMINATION,
                             novelty=True)
    lof.fit(Xtr_e)
    pred  = (lof.predict(Xte_e) == -1).astype(int)
    score = -lof.score_samples(Xte_e)
    results.append(evaluate("Local Outlier Factor", y_test, pred, score))

    df_res = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df_res.to_csv("results/anomaly_results.csv", index=False)

    # ----------------------------------------------------------------------
    # Plot detector comparison
    # ----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df_res)); w = 0.2
    ax.bar(x - 1.5 * w, df_res["Precision"], w, label="Precision")
    ax.bar(x - 0.5 * w, df_res["Recall"],    w, label="Recall")
    ax.bar(x + 0.5 * w, df_res["F1"],        w, label="F1")
    ax.bar(x + 1.5 * w, df_res["AUC"],       w, label="ROC-AUC")
    ax.set_xticks(x); ax.set_xticklabels(df_res["Detector"])
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("Adversarial Access Detection — Detector Comparison")
    ax.legend(loc="lower right"); ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(df_res["F1"]):
        ax.text(i + 0.5 * w, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("results/anomaly_detection.png", dpi=140)
    plt.close()

    best = df_res.loc[df_res["F1"].idxmax()]
    print(f"\nBest anomaly detector by F1: {best['Detector']} "
          f"(F1 = {best['F1']:.4f}, AUC = {best['AUC']:.4f})")
    print("\nSaved: results/anomaly_results.csv")
    print("Saved: results/anomaly_detection.png")


if __name__ == "__main__":
    main()
