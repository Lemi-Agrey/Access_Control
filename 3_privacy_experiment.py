"""
Step 3: Privacy–Utility Tradeoff
=================================
The three reference papers all address the same fundamental tension:

  * Fine-grained ABAC needs detailed attributes to make good decisions.
  * Those same attributes (health data, role, age, smoking habits, ...)
    are exactly what users want to keep private.

We measure this tradeoff empirically.  We re-train the Random Forest from
Step 2 under four privacy regimes and compare its F1 score:

  Regime A  — FULL                      All attributes visible (no privacy).
  Regime B  — ANONYMISE SUBJECT IDs     Subject role + dept hidden; only
                                        clearance / experience kept.
                                        (mirrors anonymous-access in
                                        Chaturvedi & Shirole 2024)
  Regime C  — COARSEN SENSITIVE         Age binned into ranges, clearance
                                        bucketed, hour -> "day/night".
                                        (mirrors k-anonymity-style
                                        coarsening, also in paper 3)
  Regime D  — REMOVE SENSITIVE          All subject-identifying and
                                        location attributes dropped.
                                        (maximum privacy, minimum utility)

Outputs:
  - results/privacy_tradeoff.csv
  - results/privacy_tradeoff.png
"""

import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler, OneHotEncoder
from sklearn.compose         import ColumnTransformer
from sklearn.pipeline        import Pipeline
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import accuracy_score, f1_score, precision_score, recall_score

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# The four privacy regimes — each is a function that transforms X
# ---------------------------------------------------------------------------
def regime_full(X):
    """Regime A — no transformation, full information."""
    return X.copy()


def regime_anonymise_ids(X):
    """Regime B — drop identifying categorical attributes of the subject."""
    Xn = X.copy()
    for col in ["subj_role", "subj_department"]:
        if col in Xn.columns:
            Xn = Xn.drop(columns=col)
    return Xn


def regime_coarsen(X):
    """Regime C — coarsen sensitive numerical attributes into bins."""
    Xn = X.copy()
    # Age -> bands
    Xn["subj_age"] = pd.cut(Xn["subj_age"], bins=[0, 30, 45, 65],
                            labels=["young", "mid", "senior"]).astype(str)
    # Clearance -> low / mid / high
    Xn["subj_clearance"] = pd.cut(Xn["subj_clearance"], bins=[0, 2, 3, 5],
                                  labels=["low", "mid", "high"]).astype(str)
    # Experience -> bands
    Xn["subj_years_exp"] = pd.cut(Xn["subj_years_exp"], bins=[-1, 5, 15, 40],
                                  labels=["junior", "mid", "senior"]).astype(str)
    # Hour -> day / night
    Xn["env_hour"] = np.where((Xn["env_hour"] >= 6) & (Xn["env_hour"] <= 21),
                              "day", "night")
    return Xn


def regime_remove_sensitive(X):
    """Regime D — drop all identifying & contextual subject information."""
    Xn = X.copy()
    drop = ["subj_role", "subj_department", "subj_age",
            "subj_years_exp", "subj_cert_active", "env_location"]
    Xn = Xn.drop(columns=[c for c in drop if c in Xn.columns])
    return Xn


REGIMES = [
    ("A — Full attributes (no privacy)",      regime_full),
    ("B — Anonymise subject identity",        regime_anonymise_ids),
    ("C — Coarsen sensitive attributes",      regime_coarsen),
    ("D — Remove sensitive attributes",       regime_remove_sensitive),
]


# ---------------------------------------------------------------------------
# Build the preprocessor dynamically from the transformed X
# ---------------------------------------------------------------------------
def build_preprocessor(X):
    cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(),                       num_cols),
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("STEP 3 — Privacy–Utility Tradeoff Experiment")
    print("=" * 60)

    df = pd.read_csv("data/abac_dataset.csv")
    y  = (df["decision"] == "ALLOW").astype(int)
    X  = df.drop(columns=["decision"])

    # We use the SAME train/test split for all regimes so the comparison is fair
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y)

    rows = []
    print(f"{'Privacy regime':<40} {'#feats':>7} {'Acc':>7} "
          f"{'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("-" * 80)

    for label, transform in REGIMES:
        Xtr = transform(X_train)
        Xte = transform(X_test)

        pre = build_preprocessor(Xtr)
        clf = RandomForestClassifier(n_estimators=100,
                                     random_state=RANDOM_SEED,
                                     n_jobs=-1)
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(Xtr, y_train)
        y_pred = pipe.predict(Xte)

        acc  = accuracy_score (y_test, y_pred)
        pre_ = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score   (y_test, y_pred, zero_division=0)
        f1   = f1_score       (y_test, y_pred, zero_division=0)

        rows.append({"Regime": label,
                     "Features": Xtr.shape[1],
                     "Accuracy":  acc,
                     "Precision": pre_,
                     "Recall":    rec,
                     "F1":        f1})

        print(f"{label:<40} {Xtr.shape[1]:>7} {acc:>7.4f} "
              f"{pre_:>7.4f} {rec:>7.4f} {f1:>7.4f}")

    df_res = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df_res.to_csv("results/privacy_tradeoff.csv", index=False)

    # ----------------------------------------------------------------------
    # Plot the tradeoff
    # ----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df_res)); w = 0.25
    ax.bar(x - w, df_res["Accuracy"], w, label="Accuracy")
    ax.bar(x,     df_res["F1"],       w, label="F1")
    ax.bar(x + w, df_res["Recall"],   w, label="Recall")
    short_labels = ["A: Full", "B: Anon ID", "C: Coarsen", "D: Remove"]
    ax.set_xticks(x); ax.set_xticklabels(short_labels)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("Privacy–Utility Tradeoff (Random Forest, same train/test split)")
    ax.legend(loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    # Annotate F1 values for clarity
    for i, v in enumerate(df_res["F1"]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("results/privacy_tradeoff.png", dpi=140)
    plt.close()

    drop_f1 = df_res["F1"].iloc[0] - df_res["F1"].iloc[-1]
    print(f"\nF1 drop from full privacy to max privacy: {drop_f1:.4f} "
          f"({drop_f1 / df_res['F1'].iloc[0]:.1%} relative).")
    print("This quantifies the cost of privacy in ABAC — the same tradeoff")
    print("that homomorphic encryption (Kerl 2025), blockchain anonymity")
    print("(Ghorbel 2022) and HABS (Chaturvedi 2024) attempt to mitigate.\n")
    print("Saved: results/privacy_tradeoff.csv")
    print("Saved: results/privacy_tradeoff.png")


if __name__ == "__main__":
    main()
