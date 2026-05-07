from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, f1_score

try:
    from src.data_ucr import load_dataset
    from src.models import build_model
    from src.transforms import transform_1d_to_2d
    from src.utils import get_project_root, prepare_run_dirs, print_environment, save_json, set_seed
except ModuleNotFoundError:
    from data_ucr import load_dataset
    from models import build_model
    from transforms import transform_1d_to_2d
    from utils import get_project_root, prepare_run_dirs, print_environment, save_json, set_seed


def compile_model(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def prepare_inputs(X_train, X_test, model_type: str, representation: str, image_size: int):
    if model_type.lower() == "cnn1d":
        X_train_input = X_train[..., np.newaxis].astype("float32")
        X_test_input = X_test[..., np.newaxis].astype("float32")
    else:
        X_train_input = transform_1d_to_2d(X_train, representation, image_size=image_size)
        X_test_input = transform_1d_to_2d(X_test, representation, image_size=image_size)
    return X_train_input, X_test_input


def append_result(result_file: Path, row: dict) -> None:
    if result_file.exists():
        old = pd.read_csv(result_file)
        new = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
    else:
        new = pd.DataFrame([row])
    new.to_csv(result_file, index=False)


def main(args):
    set_seed(args.seed)
    print_environment()

    project_root = get_project_root()
    dirs = prepare_run_dirs(
        project_root=project_root,
        dataset=args.dataset,
        representation=args.representation,
        model_type=args.model_type,
        seed=args.seed,
    )

    print("Project artifact root:", project_root)
    print("Run directory:", dirs["run"])

    save_json(vars(args), dirs["run"] / "args.json")

    X_train, X_test, y_train, y_test, num_classes, encoder = load_dataset(
        args.dataset, data_dir=args.data_dir
    )

    X_train_input, X_test_input = prepare_inputs(
        X_train=X_train,
        X_test=X_test,
        model_type=args.model_type,
        representation=args.representation,
        image_size=args.image_size,
    )

    input_shape = X_train_input.shape[1:]
    print("Input shape:", input_shape)
    print("Number of classes:", num_classes)

    model = build_model(args.model_type, input_shape=input_shape, num_classes=num_classes)
    model = compile_model(model, learning_rate=args.learning_rate)
    model.summary()

    best_model_path = dirs["checkpoints"] / "best_model.keras"
    last_model_path = dirs["checkpoints"] / "last_model.keras"
    csv_log_path = dirs["logs"] / "training_log.csv"

    callbacks = [
        tf.keras.callbacks.BackupAndRestore(
            backup_dir=str(dirs["backup"]),
            save_freq="epoch",
            delete_checkpoint=False,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor=args.monitor,
            mode=args.monitor_mode,
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(last_model_path),
            save_best_only=False,
            save_weights_only=False,
            verbose=0,
        ),
        tf.keras.callbacks.CSVLogger(str(csv_log_path), append=True),
        tf.keras.callbacks.EarlyStopping(
            monitor=args.monitor,
            mode=args.monitor_mode,
            patience=args.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=args.reduce_lr_patience,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    start = time.time()
    model.fit(
        X_train_input,
        y_train,
        validation_split=args.validation_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    training_time = time.time() - start

    if best_model_path.exists():
        print("Loading best model:", best_model_path)
        model = tf.keras.models.load_model(best_model_path)

    inference_start = time.time()
    y_prob = model.predict(X_test_input, batch_size=args.batch_size, verbose=0)
    inference_time = time.time() - inference_start

    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_prob, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    params = int(model.count_params())
    inference_time_per_sample = inference_time / max(len(X_test_input), 1)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(dirs["run"] / "classification_report.csv")

    result_row = {
        "dataset": args.dataset,
        "representation": args.representation,
        "model_type": args.model_type,
        "seed": args.seed,
        "image_size": args.image_size,
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "params": params,
        "training_time_sec": training_time,
        "inference_time_total_sec": inference_time,
        "inference_time_per_sample_sec": inference_time_per_sample,
        "best_model_path": str(best_model_path),
        "run_dir": str(dirs["run"]),
    }

    append_result(dirs["results"] / "summary_results.csv", result_row)
    save_json(result_row, dirs["run"] / "result.json")

    print("\n===== FINAL RESULT =====")
    for k, v in result_row.items():
        print(f"{k}: {v}")
    print("Saved summary to:", dirs["results"] / "summary_results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train time-series CNN experiments.")

    parser.add_argument("--dataset", type=str, default="GunPoint")
    parser.add_argument("--data_dir", type=str, default="data/UCR")
    parser.add_argument("--representation", type=str, default="gaf")
    parser.add_argument("--model_type", type=str, default="light2dcnn")

    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--validation_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--monitor", type=str, default="val_accuracy")
    parser.add_argument("--monitor_mode", type=str, default="max")
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--reduce_lr_patience", type=int, default=5)

    main(parser.parse_args())
