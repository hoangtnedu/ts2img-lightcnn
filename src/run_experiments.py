from __future__ import annotations

import argparse
import subprocess
import sys


def run(cmd):
    print("\n", " ".join(cmd), "\n")
    subprocess.run(cmd, check=True)


def main(args):
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    representations = [x.strip() for x in args.representations.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    for dataset in datasets:
        for seed in seeds:
            # 1D-CNN baseline
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
                    "--seed",
                    str(seed),
                ]
            )

            # 2D representations
            for rep in representations:
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

    main(parser.parse_args())