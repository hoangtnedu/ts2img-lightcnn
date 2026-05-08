from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run(cmd):
    print("\n", " ".join(cmd), "\n")
    subprocess.run(cmd, check=True)


def _safe_int(value, default=None):
    try:
        return int(value)
    except Exception:
        return default


def load_completed_experiments(result_file):
    """
    Read results/summary_results.csv and return completed experiment keys.

    Key fields:
    dataset, representation, model_type, seed, epochs_requested, batch_size, image_size

    This prevents a full run with epochs=50 from being confused with a quick
    test run with epochs=2.
    """

    result_file = Path(result_file)

    if not result_file.exists():
        print(f"No existing result file found: {result_file}")
        return set()

    try:
        df = pd.read_csv(result_file)
    except Exception as exc:
        print(f"WARNING: Cannot read {result_file}: {exc}")
        return set()

    required = {
        "dataset",
        "representation",
        "model_type",
        "seed",
        "epochs_requested",
        "batch_size",
        "image_size",
    }

    missing = required - set(df.columns)
    if missing:
        print(f"WARNING: Result file is missing columns: {sorted(missing)}")
        print("Resume skip will be disabled for incomplete result rows.")
        return set()

    completed = set()

    for _, row in df.iterrows():
        key = (
            str(row["dataset"]),
            str(row["representation"]),
            str(row["model_type"]),
            _safe_int(row["seed"]),
            _safe_int(row["epochs_requested"]),
            _safe_int(row["batch_size"]),
            _safe_int(row["image_size"]),
        )
        completed.add(key)

    print(f"Loaded {len(completed)} completed experiment keys from {result_file}")
    return completed


def make_key(dataset, representation, model_type, seed, epochs, batch_size, image_size):
    return (
        str(dataset),
        str(representation),
        str(model_type),
        int(seed),
        int(epochs),
        int(batch_size),
        int(image_size),
    )


def main(args):
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    representations = [x.strip() for x in args.representations.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    completed = load_completed_experiments(args.results_file) if args.resume else set()

    total_planned = 0
    total_skipped = 0
    total_launched = 0

    for dataset in datasets:
        for seed in seeds:
            total_planned += 1
            baseline_key = make_key(
                dataset=dataset,
                representation="none",
                model_type="cnn1d",
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                image_size=args.image_size,
            )

            if args.resume and baseline_key in completed:
                total_skipped += 1
                print(f"[SKIP completed] {baseline_key}")
            else:
                total_launched += 1
                run(
                    [
                        sys.executable,
                        "-m",
                        "src.train",
                        "--dataset",
                        dataset,
                        "--representation",
                        "none",
                        "--model_type",
                        "cnn1d",
                        "--epochs",
                        str(args.epochs),
                        "--batch_size",
                        str(args.batch_size),
                        "--image_size",
                        str(args.image_size),
                        "--seed",
                        str(seed),
                    ]
                )
                completed.add(baseline_key)

            for rep in representations:
                total_planned += 1
                exp_key = make_key(
                    dataset=dataset,
                    representation=rep,
                    model_type=args.model_type,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    image_size=args.image_size,
                )

                if args.resume and exp_key in completed:
                    total_skipped += 1
                    print(f"[SKIP completed] {exp_key}")
                    continue

                total_launched += 1
                run(
                    [
                        sys.executable,
                        "-m",
                        "src.train",
                        "--dataset",
                        dataset,
                        "--representation",
                        rep,
                        "--model_type",
                        args.model_type,
                        "--epochs",
                        str(args.epochs),
                        "--batch_size",
                        str(args.batch_size),
                        "--image_size",
                        str(args.image_size),
                        "--seed",
                        str(seed),
                    ]
                )
                completed.add(exp_key)

    print("\n===== RUN EXPERIMENTS SUMMARY =====")
    print(f"Planned experiments:  {total_planned}")
    print(f"Skipped experiments:  {total_skipped}")
    print(f"Launched experiments: {total_launched}")
    print(f"Resume mode:          {args.resume}")
    print(f"Result file:          {args.results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        type=str,
        default="GunPoint,ECG200,Coffee,FordA,Wafer",
    )
    parser.add_argument("--representations", type=str, default="gaf,mtf,rp,stft")
    parser.add_argument("--model_type", type=str, default="light2dcnn")
    parser.add_argument("--seeds", type=str, default="42,2024,2026")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=64)

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip experiments already recorded in results/summary_results.csv. "
            "Use this when Colab stops unexpectedly and you want to continue."
        ),
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default="results/summary_results.csv",
        help="CSV file used to detect completed experiments in resume mode.",
    )

    main(parser.parse_args())
