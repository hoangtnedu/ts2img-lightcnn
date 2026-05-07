from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, f1_score

try:
    from src.data_ucr import load_dataset
    from src.transforms import transform_1d_to_2d
except ModuleNotFoundError:
    from data_ucr import load_dataset
    from transforms import transform_1d_to_2d


def main(args):
    X_train, X_test, y_train, y_test, num_classes, encoder = load_dataset(
        args.dataset, data_dir=args.data_dir
    )

    if args.model_type == "cnn1d":
        X_test_input = X_test[..., np.newaxis].astype("float32")
    else:
        X_test_input = transform_1d_to_2d(X_test, args.representation, args.image_size)

    model = tf.keras.models.load_model(args.model_path)
    y_prob = model.predict(X_test_input, batch_size=args.batch_size, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Macro F1:", f1_score(y_true, y_pred, average="macro"))
    print(classification_report(y_true, y_pred, zero_division=0))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(classification_report(y_true, y_pred, output_dict=True, zero_division=0)).transpose().to_csv(out_path)
    print("Saved report to:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="GunPoint")
    parser.add_argument("--data_dir", type=str, default="data/UCR")
    parser.add_argument("--representation", type=str, default="gaf")
    parser.add_argument("--model_type", type=str, default="light2dcnn")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/evaluation_report.csv")
    main(parser.parse_args())
