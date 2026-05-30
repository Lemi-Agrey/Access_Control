"""
run_all.py — Run every experiment in order.

This is just a convenience wrapper.  You can also run each numbered script
individually if you want to see one experiment at a time.

    python run_all.py
"""

import subprocess
import sys
import time


STEPS = [
    ("Step 1: Generate Dataset",      "1_generate_dataset.py"),
    ("Step 2: ML Classification",     "2_classification.py"),
    ("Step 3: Privacy Tradeoff",      "3_privacy_experiment.py"),
    ("Step 4: Anomaly Detection",     "4_anomaly_detection.py"),
]


def main():
    overall = time.time()
    for title, script in STEPS:
        print(f"\n{'#' * 65}")
        print(f"# {title}")
        print(f"# Running: {script}")
        print(f"{'#' * 65}")
        t0 = time.time()
        rc = subprocess.call([sys.executable, script])
        if rc != 0:
            print(f"\n[!] '{script}' failed with return code {rc}.  Stopping.")
            return rc
        print(f"\n[OK] '{script}' finished in {time.time() - t0:.1f} s.")

    print(f"\n{'=' * 65}")
    print(f"All experiments finished in {time.time() - overall:.1f} s.")
    print(f"Open the 'results/' folder to see CSVs and plots.")
    print(f"{'=' * 65}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
