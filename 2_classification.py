"""
Step 2: ML Classification of ABAC Decisions
============================================
Train and compare 7 ML classifiers on the synthetic ABAC dataset.

Question: Can ML models accurately learn the ABAC policy from labelled
          access requests, without being given the rules?

We compare:
  - Logistic Regression  (linear baseline)
  - Decision Tree        (interpretable, rule-like)
  - Random Forest        (ensemble of trees)
  - Gradient Boosting    (boosted ensemble)
  - k-Nearest Neighbours (instance-based)
  - Neural Network (MLP) (non-linear)
  - SVM (RBF kernel)     (margin-based)

For each model we report Accuracy, Precision, Recall, F1, plus training
and inference time.  Confusion matrices are plotted.

Outputs:
  - results/classification_results.csv
  - results/classifier_comparison.png
  - results/confusion_matrices.png
"""

import os
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection      import train_test_split
from sklearn.preprocessing        import StandardScaler, OneHotEncoder
from sklearn.compose              import ColumnTransformer
from sklearn.pipeline             import Pipeline
from sklearn.linear_model         import LogisticRegression
from sklearn.tree                 import DecisionTreeClassifier
from sklearn.ensemble             import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors            import KNeighborsClassifier
from sklearn.neural_network       import MLPClassifier
from sklearn.svm                  import SVC
from sklearn.metrics              import (accuracy_score, precision_score,
                                          recall_score, f1_score,
                                          confusion_matrix)

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Load the dataset and split it into features (X) and label (y)
# ---------------------------------------------------------------------------
def load_data(path="data/abac_dataset.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found.  Run 1_generate_dataset.py first.")
    df = pd.read_csv(path)
    y = (df["decision"] == "ALLOW").astype(int)        # 1 = ALLOW, 0 = DENY
    X = df.drop(columns=["decision"])
    return X, y


# ---------------------------------------------------------------------------
# Build a preprocessor:
#   - one-hot encode categorical / boolean columns
#   - standardise numerical columns
# ---------------------------------------------------------------------------
def build_preprocessor(X):
    cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(),                       num_cols),
    ])


# ---------------------------------------------------------------------------
# Main: train every model, collect metrics, plot results
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("STEP 2 — ML Classification of ABAC Decisions")
    print("=" * 60)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y)
    print(f"Train set: {len(X_train):,} requests   "
          f"Test set: {len(X_test):,} requests")
    print(f"Features after one-hot encoding: ~{X_train.shape[1]} raw columns\n")

    pre = build_preprocessor(X)

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000,
                                                  random_state=RANDOM_SEED),
        "Decision Tree":       DecisionTreeClassifier(random_state=RANDOM_SEED),
        "Random Forest":       RandomForestClassifier(n_estimators=100,
                                                      random_state=RANDOM_SEED,
                                                      n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(random_state=RANDOM_SEED),
        "k-NN":                KNeighborsClassifier(n_neighbors=5),
        "Neural Network":      MLPClassifier(hidden_layer_sizes=(64, 32),
                                             max_iter=300,
                                             random_state=RANDOM_SEED),
        "SVM (RBF)":           SVC(kernel="rbf", random_state=RANDOM_SEED),
    }

    results       = []
    conf_matrices = {}
    print(f"{'Model':<22} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} "
          f"{'Train (s)':>10} {'Pred (s)':>10}")
    print("-" * 75)

    for name, clf in classifiers.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])

        t0 = time.time(); pipe.fit(X_train, y_train);  train_t = time.time() - t0
        t0 = time.time(); y_pred = pipe.predict(X_test); pred_t  = time.time() - t0

        acc  = accuracy_score (y_test, y_pred)
        pre_ = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score   (y_test, y_pred, zero_division=0)
        f1   = f1_score       (y_test, y_pred, zero_division=0)

        results.append({"Model": name,
                        "Accuracy":  acc,
                        "Precision": pre_,
                        "Recall":    rec,
                        "F1":        f1,
                        "Train_s":   train_t,
                        "Pred_s":    pred_t})
        conf_matrices[name] = confusion_matrix(y_test, y_pred)

        print(f"{name:<22} {acc:>7.4f} {pre_:>7.4f} {rec:>7.4f} {f1:>7.4f} "
              f"{train_t:>10.3f} {pred_t:>10.3f}")

    df_res = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df_res.to_csv("results/classification_results.csv", index=False)

    # ----------------------------------------------------------------------
    # Bar chart: compare classifiers on the four classification metrics
    # ----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(df_res)); w = 0.2
    ax.bar(x - 1.5 * w, df_res["Accuracy"],  w, label="Accuracy")
    ax.bar(x - 0.5 * w, df_res["Precision"], w, label="Precision")
    ax.bar(x + 0.5 * w, df_res["Recall"],    w, label="Recall")
    ax.bar(x + 1.5 * w, df_res["F1"],        w, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(df_res["Model"], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("ABAC Policy Learning — Classifier Comparison")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/classifier_comparison.png", dpi=140)
    plt.close()

    # ----------------------------------------------------------------------
    # 2x4 grid of confusion matrices
    # ----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    axes = axes.flatten()
    for i, (name, cm) in enumerate(conf_matrices.items()):
        ax = axes[i]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name, fontsize=10)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["DENY", "ALLOW"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["DENY", "ALLOW"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        # Annotate cells
        for r in range(2):
            for c in range(2):
                ax.text(c, r, f"{cm[r, c]}", ha="center", va="center",
                        color="white" if cm[r, c] > cm.max() / 2 else "black",
                        fontsize=10)
    axes[-1].axis("off")     # leave one slot blank in the 2x4 grid
    fig.suptitle("Confusion Matrices on Test Set", fontsize=13)
    plt.tight_layout()
    plt.savefig("results/confusion_matrices.png", dpi=140)
    plt.close()

    print("\nSaved: results/classification_results.csv")
    print("Saved: results/classifier_comparison.png")
    print("Saved: results/confusion_matrices.png")

    best = df_res.loc[df_res["F1"].idxmax()]
    print(f"\nBest model by F1 score: {best['Model']} "
          f"(F1 = {best['F1']:.4f}, Accuracy = {best['Accuracy']:.4f})")


if __name__ == "__main__":
    main()
