# Experimental Validation of Machine Learning Approaches for Privacy-Preserving Attribute-Based Access Control (ABAC)

A complete mini-project that uses Machine Learning to study three things about ABAC:

1. **Can ML learn ABAC policies from labelled access requests?**
2. **What happens to accuracy when sensitive attributes are hidden for privacy?**
3. **Can ML detect adversarial / malicious access attempts?**

All three questions are directly motivated by the three reference papers (see *References* at the bottom).

---

## Table of contents
1. [What's in this project](#whats-in-this-project)
2. [How it relates to the papers](#how-it-relates-to-the-papers)
3. [Quick start](#quick-start)
4. [Project structure](#project-structure)
5. [The three experiments](#the-three-experiments)
6. [Expected results](#expected-results)
7. [How to present this](#how-to-present-this)
8. [Frequently asked questions](#frequently-asked-questions)
9. [References](#references)

---

## What's in this project

This project simulates an **e-health ABAC system** (patients, doctors, nurses, researchers, etc. requesting access to medical records, lab results, prescriptions, billing data) and runs three ML experiments on it:

| # | Experiment | What it shows |
|---|---|---|
| 1 | **Policy learning** | 7 ML classifiers (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, k-NN, Neural Network, SVM) are compared on how well they learn ABAC allow/deny decisions. |
| 2 | **Privacy–utility tradeoff** | The same model is trained 4 times — full attributes, anonymised, coarsened, and privacy-removed — to **quantify the cost of privacy** on decision quality. |
| 3 | **Anomaly detection** | 3 unsupervised detectors (Isolation Forest, One-Class SVM, LOF) try to flag synthetic adversarial requests injected into the test stream. |

Everything is reproducible (fixed random seeds) and runs in **under 60 seconds** on a normal laptop.

---

## How it relates to the papers

| Reference paper | What it proposes | What we validate experimentally |
|---|---|---|
| **Kerl et al. 2025** — Privacy-preserving ABAC via Paillier homomorphic encryption + ZKP | Hide attribute *values* so the ABAC engine can decide without seeing them | We measure **what the engine would lose** if those attributes were hidden (Experiment 2). |
| **Ghorbel et al. 2022** — Blockchain ABAC with anonymous users | Decentralised, anonymous, accountable access control | The use case (e-health, multiple roles, sensitive attributes) is the basis of our synthetic dataset. Experiment 3 simulates the kind of "malicious user" their threat model addresses. |
| **Chaturvedi & Shirole 2024** — ABAC-PA² with HABS and privacy levels | Anonymous attribute-based signatures, privacy levels 0–2 | Their Regimes "no privacy / partial / full" map directly onto our Regimes A, B/C, and D in Experiment 2. |

So this isn't a re-implementation of those papers (they're cryptographic, not ML) — it's an **empirical study of the tradeoffs** their cryptographic mechanisms are designed to navigate.

---

## Quick start

### 1. Install Python 3.9+ if you don't have it
On Windows: download from <https://python.org>. On macOS: `brew install python`. On Linux: `sudo apt install python3 python3-pip`.

### 2. (Recommended) Create a virtual environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
```

### 3. Install the dependencies
```bash
pip install -r requirements.txt
```

### 4. Run everything
```bash
python run_all.py
```

That will:
* generate `data/abac_dataset.csv` (10 000 access requests)
* train and compare 7 classifiers
* run the privacy-utility experiment
* run the anomaly detection experiment
* save 4 CSVs and 4 plots under `results/`

You'll see progress printed to your terminal. The whole thing takes ~30–60 seconds.

### 5. (Alternative) Run experiments one by one
```bash
python 1_generate_dataset.py
python 2_classification.py
python 3_privacy_experiment.py
python 4_anomaly_detection.py
```

---

## Project structure

```
abac_ml_project/
├── README.md                      ← you are here
├── requirements.txt               ← Python dependencies
├── run_all.py                     ← runs every script in order
│
├── 1_generate_dataset.py          ← Step 1: build the synthetic ABAC dataset
├── 2_classification.py            ← Step 2: ML classification (Experiment 1)
├── 3_privacy_experiment.py        ← Step 3: privacy–utility (Experiment 2)
├── 4_anomaly_detection.py         ← Step 4: anomaly detection (Experiment 3)
│
├── data/                          ← created by Step 1
│   └── abac_dataset.csv
│
└── results/                       ← created by Steps 2–4
    ├── classification_results.csv
    ├── classifier_comparison.png
    ├── confusion_matrices.png
    ├── privacy_tradeoff.csv
    ├── privacy_tradeoff.png
    ├── anomaly_results.csv
    └── anomaly_detection.png
```

---

## The three experiments

### Experiment 1 — Can ML learn ABAC decisions? (`2_classification.py`)

We split the 10 000 access requests into 75% train and 25% test (stratified) and train **seven different ML classifiers** to predict the ALLOW/DENY label from the 14 attributes.

For each classifier we report:
* **Accuracy** — fraction of test requests labelled correctly
* **Precision** — when the model says ALLOW, how often is it right?
* **Recall** — of all true ALLOWs, how many did the model catch?
* **F1** — harmonic mean of precision and recall (the best single number when classes are imbalanced)
* **Training time** and **inference time**

A confusion matrix is plotted for every model so you can see exactly where the mistakes are.

### Experiment 2 — Privacy–utility tradeoff (`3_privacy_experiment.py`)

This is the **most important experiment** for the report — it directly engages the problem the papers are trying to solve.

We train the same Random Forest four times, each time with a different level of privacy:

| Regime | What is hidden | Privacy level (Chaturvedi 2024 notation) |
|---|---|---|
| **A — Full** | Nothing — all 14 attributes used | 0 (no privacy) |
| **B — Anonymise** | `subj_role` and `subj_department` dropped | 1 (partial) |
| **C — Coarsen** | Age → {young/mid/senior}, clearance → {low/mid/high}, experience → bands, hour → day/night | 1 (partial, k-anonymity-style) |
| **D — Remove** | All identifying & contextual subject info dropped | 2 (maximum privacy) |

You'll see how F1 drops as privacy goes up — this is exactly the tradeoff that homomorphic encryption (Kerl), blockchain anonymity (Ghorbel), and HABS (Chaturvedi) try to mitigate by letting the system make good decisions *without* actually seeing the sensitive values.

### Experiment 3 — Anomaly detection (`4_anomaly_detection.py`)

We inject 500 synthetic adversarial access requests into the test stream. The four attack types are:
* **Burst** — abnormally high request frequency in the last hour
* **Off-hours remote** — 1–4 am access from `remote` location with no active certificate
* **Low-clearance, high-sensitivity** — clearance=1 user asking to delete/share a sensitivity-5 object
* **Role/object mismatch** — patient trying to write to billing or research data

Three unsupervised detectors are trained **only on the normal traffic** (they never see the adversarial examples during training):
* **Isolation Forest**
* **One-Class SVM**
* **Local Outlier Factor (LOF)**

We measure precision, recall, F1 and ROC-AUC of each detector at flagging the injected anomalies.

---

## Expected results

(Your numbers may vary by ±1–2% depending on numpy/sklearn version, but the ordering and story should be the same.)

### Experiment 1 — classifier comparison
| Model | Accuracy | F1 |
|---|---|---|
| Logistic Regression | ~0.89 | ~0.81 |
| Decision Tree | ~0.97 | ~0.96 |
| Random Forest | ~0.98 | ~0.97 |
| Gradient Boosting | ~0.98 | ~0.96 |
| k-NN | ~0.89 | ~0.84 |
| **Neural Network** | **~0.99** | **~0.98** |
| SVM (RBF) | ~0.98 | ~0.96 |

**Takeaway:** Non-linear models (trees, NN, SVM) cleanly learn the policy. The linear baseline (LogReg) lags because the rules involve interactions (role × department × emergency).

### Experiment 2 — privacy–utility tradeoff
| Regime | F1 | Relative drop |
|---|---|---|
| A: Full | ~0.97 | — |
| B: Anonymise IDs | ~0.92 | ~5% |
| C: Coarsen | ~0.97 | ~0% (almost free!) |
| D: Remove sensitive | ~0.81 | ~16% |

**Takeaway:** *Coarsening* (Regime C) buys meaningful privacy almost for free — this is a real insight. *Removing* sensitive attributes outright (Regime D) costs ~16% F1 — that's the gap that cryptographic ABAC (Paillier, ABE, HABS) tries to close.

### Experiment 3 — anomaly detection
| Detector | F1 | ROC-AUC |
|---|---|---|
| Isolation Forest | ~0.31 | ~0.70 |
| One-Class SVM | ~0.55 | ~0.84 |
| **Local Outlier Factor** | **~0.64** | **~0.90** |

**Takeaway:** Density-based LOF wins because the adversarial examples are statistically isolated in feature space. AUC ≈ 0.90 means the detector ranks anomalies above normal traffic 90% of the time — useful as a *second line of defence* on top of the ABAC engine itself.

---


## References

The three reference papers are the source of all motivating examples in this project:

1. Kerl, M., Bodin, U., Schelén, O. (2025). Privacy-preserving attribute-based access control using homomorphic encryption. *Cybersecurity*, **8**:5. doi:10.1186/s42400-024-00323-8
2. Ghorbel, A., Ghorbel, M., Jmaiel, M. (2022). Accountable privacy preserving attribute-based access control for cloud services enforced using blockchain. *International Journal of Information Security*, **21**:489–508. doi:10.1007/s10207-021-00565-4
3. Chaturvedi, G. K., Shirole, M. (2024). ABAC-PA²: Attribute-Based Access Control Model with Privacy aware Anonymous Access. *Procedia Computer Science*, **237**:147–154. doi:10.1016/j.procs.2024.05.090

---

*Generated as a teaching mini-project. All code is heavily commented for beginners — read through each `*.py` file from top to bottom and you should be able to understand and explain it.*
