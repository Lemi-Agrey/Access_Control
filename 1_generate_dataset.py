"""
Step 1: Generate Synthetic ABAC Dataset
========================================
Creates a synthetic dataset of access-control requests for an
Attribute-Based Access Control (ABAC) system, inspired by:

  - The e-health use case in Ghorbel et al. (2022) — patients, doctors,
    researchers, pharmacists, medical records, prescriptions, etc.
  - The mining/asthma use case in Kerl et al. (2025) — using health
    attributes and environmental context to make access decisions.
  - The ABAC formalism in Chaturvedi & Shirole (2024) — subject, object,
    environment, action -> permission.

Each access request contains:
  - 6 SUBJECT attributes (who is asking)
  - 4 OBJECT  attributes (what is being accessed)
  - 4 ENVIRONMENT attributes (context: time, location, emergency...)
  - 1 ACTION (read / write / delete / share)
  - 1 DECISION (ALLOW or DENY)  <-- the ground-truth label

The DECISION is computed from a fixed policy function below.  Our ML models
will later try to learn that policy from data, without being shown the rules.

Output: data/abac_dataset.csv
"""

import os
import random
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Reproducibility — fix the random seed so the dataset is identical every run
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

N_REQUESTS = 10_000

# ---------------------------------------------------------------------------
# Vocabulary of attribute values
# ---------------------------------------------------------------------------
ROLES        = ["doctor", "nurse", "patient", "researcher", "admin", "pharmacist"]
DEPARTMENTS  = ["cardiology", "oncology", "general", "research", "admin_office"]
OBJECT_TYPES = ["medical_record", "lab_result", "prescription",
                "billing", "research_data", "medical_image"]
LOCATIONS    = ["hospital_network", "remote", "office_network"]
ACTIONS      = ["read", "write", "delete", "share"]


# ---------------------------------------------------------------------------
# Random generators for each entity
# ---------------------------------------------------------------------------
def generate_subject():
    role = random.choice(ROLES)
    dept = random.choice(DEPARTMENTS)
    # Some roles are tied to a specific department
    if role == "researcher":
        dept = "research"
    if role == "admin":
        dept = "admin_office"
    return {
        "subj_role":         role,
        "subj_department":   dept,
        "subj_clearance":    random.randint(1, 5),       # 1=lowest, 5=highest
        "subj_years_exp":    random.randint(0, 40),
        "subj_cert_active":  random.choice([True, False]),
        "subj_age":          random.randint(22, 65),
    }


def generate_object():
    obj_type = random.choice(OBJECT_TYPES)
    # Different object types have different baseline sensitivities
    sens_range = {
        "medical_record": (3, 5),
        "lab_result":     (2, 4),
        "prescription":   (2, 4),
        "billing":        (1, 3),
        "research_data":  (2, 4),
        "medical_image":  (3, 5),
    }[obj_type]
    return {
        "obj_type":          obj_type,
        "obj_sensitivity":   random.randint(*sens_range),
        "obj_department":    random.choice(DEPARTMENTS),
        "obj_contains_pii":  random.choices([True, False], weights=[0.7, 0.3])[0],
    }


def generate_environment():
    return {
        "env_hour":                   random.randint(0, 23),
        "env_is_emergency":           random.choices([True, False], weights=[0.15, 0.85])[0],
        "env_location":               random.choice(LOCATIONS),
        "env_access_freq_last_hour":  random.randint(0, 50),
    }


# ---------------------------------------------------------------------------
# THE POLICY  —  this function is the "ground truth" the ML models try to learn
# ---------------------------------------------------------------------------
def evaluate_policy(s, o, e, action):
    """Return True if the access should be ALLOWED, else False.

    The rules below combine role-based logic with contextual constraints,
    mimicking how a real ABAC policy engine would behave.
    """
    role      = s["subj_role"]
    obj_type  = o["obj_type"]
    same_dept = (s["subj_department"] == o["obj_department"])
    emergency = e["env_is_emergency"]

    # ---- Universal contextual constraints --------------------------------
    # Rule U1: deletes require high clearance
    if action == "delete" and s["subj_clearance"] < 4:
        return False
    # Rule U2: remote access requires an active certification
    if e["env_location"] == "remote" and not s["subj_cert_active"]:
        return False
    # Rule U3: late-night (22:00-05:00) needs clearance >= 3
    if (e["env_hour"] >= 22 or e["env_hour"] <= 5) and s["subj_clearance"] < 3:
        return False
    # Rule U4: high access frequency in the past hour is suspicious
    if e["env_access_freq_last_hour"] > 30:
        return False

    # ---- Role-based logic ------------------------------------------------
    if role == "doctor":
        if obj_type in {"medical_record", "lab_result",
                        "medical_image", "prescription"}:
            if same_dept or emergency:
                return action in {"read", "write"}
        return False

    if role == "nurse":
        if obj_type in {"medical_record", "lab_result"} and same_dept:
            return action == "read"
        return False

    if role == "patient":
        # Patients can read their own (low-sensitivity) records only
        if obj_type == "medical_record" and o["obj_sensitivity"] <= 3:
            return action == "read"
        return False

    if role == "researcher":
        # Researchers can read anonymised research data only
        if obj_type == "research_data" and not o["obj_contains_pii"]:
            return action == "read"
        return False

    if role == "admin":
        if obj_type == "billing":
            return action in {"read", "write"}
        return False

    if role == "pharmacist":
        if obj_type == "prescription":
            return action == "read"
        return False

    return False


# ---------------------------------------------------------------------------
# Main: build the dataset and write it to CSV
# ---------------------------------------------------------------------------
def generate_biased_request():
    """Generate a request that is more likely (but not guaranteed) to be
    authorised — by aligning department, choosing reasonable hours, etc.
    This prevents an extreme class imbalance in the final dataset."""
    s = generate_subject()
    o = generate_object()
    # Bias toward same-department resource accesses
    o["obj_department"] = s["subj_department"]
    # Bias toward a plausible object type for the role
    role_to_objs = {
        "doctor":     ["medical_record", "lab_result", "medical_image", "prescription"],
        "nurse":      ["medical_record", "lab_result"],
        "patient":    ["medical_record"],
        "researcher": ["research_data"],
        "admin":      ["billing"],
        "pharmacist": ["prescription"],
    }
    o["obj_type"] = random.choice(role_to_objs[s["subj_role"]])
    # Researchers need non-PII data to gain access
    if s["subj_role"] == "researcher":
        o["obj_contains_pii"] = False
    # Patient records need low sensitivity for patient access
    if s["subj_role"] == "patient":
        o["obj_sensitivity"] = random.randint(1, 3)
    e = generate_environment()
    e["env_hour"] = random.randint(8, 18)            # business hours
    e["env_access_freq_last_hour"] = random.randint(0, 20)
    # Pick a sensible action — avoid "delete" too often
    a = random.choices(ACTIONS, weights=[0.6, 0.25, 0.05, 0.10])[0]
    return s, o, e, a


def main():
    rows = []
    # Mix of "biased" (likely allowed) and fully random requests
    # so we end up with a healthier class balance (~30-40% ALLOW).
    for i in range(N_REQUESTS):
        if random.random() < 0.5:
            s, o, e, a = generate_biased_request()
        else:
            s = generate_subject()
            o = generate_object()
            e = generate_environment()
            a = random.choice(ACTIONS)
        allowed = evaluate_policy(s, o, e, a)
        rows.append({**s, **o, **e,
                     "action":   a,
                     "decision": "ALLOW" if allowed else "DENY"})

    df = pd.DataFrame(rows)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/abac_dataset.csv", index=False)

    n_allow = (df["decision"] == "ALLOW").sum()
    n_deny  = (df["decision"] == "DENY").sum()
    print("=" * 60)
    print("STEP 1 — Dataset Generation")
    print("=" * 60)
    print(f"Generated {len(df):,} access requests.")
    print(f"  ALLOW : {n_allow:,}  ({n_allow / len(df):.1%})")
    print(f"  DENY  : {n_deny:,}  ({n_deny  / len(df):.1%})")
    print(f"\nSaved to: data/abac_dataset.csv")
    print("\nPreview of the first 5 rows:\n")
    print(df.head().to_string())
    print()


if __name__ == "__main__":
    main()
