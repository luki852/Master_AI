import argparse 
import os
import shutil
import sys
import tempfile
import time

parser = argparse.ArgumentParser()
parser.add_argument("--submission", type=str)
parser.add_argument("--target", type=str, default="evaluation.tar")

args = parser.parse_args()

print("Python:", sys.executable, flush=True)
print("Submission:", args.submission, flush=True)
print("Target:", args.target, flush=True)
print("Submission exists:", os.path.exists(args.submission), flush=True)
print("Target exists:", os.path.exists(args.target), flush=True)

with open(args.submission, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]
print("Submission lines:", len(lines), flush=True)
print("First line:", lines[0] if lines else None, flush=True)

t0 = time.time()

with tempfile.TemporaryDirectory() as tmpdir:
    print(f"Unpacking evaluation code/data to {tmpdir}", flush=True)
    shutil.unpack_archive(args.target, tmpdir)
    print(f"Unpacked after {time.time() - t0:.1f}s", flush=True)

    sys.path.insert(0, tmpdir)

    print("Importing get_metric...", flush=True)
    from evaluation.evaluate_submission import get_metric
    print(f"Imported get_metric after {time.time() - t0:.1f}s", flush=True)

    args.trainset = os.path.join(tmpdir, "evaluation/data/smiles_train.txt")
    args.teststats = os.path.join(tmpdir, "evaluation/data/test_stats.p")

    print("Trainset exists:", os.path.exists(args.trainset), flush=True)
    print("Teststats exists:", os.path.exists(args.teststats), flush=True)

    metric_name = "fcd"

    print("Starting metric calculation...", flush=True)
    metric_value = get_metric(args, metric_name)
    print(f"Finished metric after {time.time() - t0:.1f}s", flush=True)

    print("#################################################################################################################################################")
    print(f"Metric name: {metric_name}")
    print(metric_value)