import json
import os

import pandas as pd


def extract_features(full_path):
    parts = full_path.split("\\")
    features = {}
    features["Dataset"] = parts[1].split("-")[0]
    features["Epochs"] = parts[1].split("-")[1].split("_")[1]
    features["Attention Type"] = parts[2]
    features["Optimizer"] = "AdamW" if "adamw" in parts[3] else "Adam"
    features["Window Size"] = int(parts[4].split("_")[-1])
    return features


def load_json(file):
    with open(file, "r") as f:
        data = json.load(f)
    return data


def collect_results(folder_path):
    data = []
    hyperparameters = [(0.25, 0.75), (0.5, 0.5), (0.75, 0.25)]
    for root, dirs, _ in os.walk(folder_path):
        features = {}
        for dir_name in dirs:
            full_path = os.path.join(root, dir_name)
            depth = full_path.count(os.path.sep) - folder_path.count(os.path.sep)
            if depth == 4:
                features = extract_features(full_path)
                summary_data = load_json(os.path.join(full_path, "summary.json"))
                test_results_data = load_json(
                    os.path.join(full_path, "test_results.json")
                )

                features["GPU Memory Usage"] = summary_data["gpu_memory"]
                features["Training Time"] = summary_data["time"]
                features["samples/sec"] = summary_data["samples/sec"]
                features["test_loss"] = test_results_data["test_loss"]
                features["Test Accuracy"] = test_results_data["test_accuracy"]
                features["path_to_train_results"] = os.path.join(
                    full_path, "train_losses.csv"
                )
                features["path_to_val_results"] = os.path.join(
                    full_path, "val_losses.csv"
                )
                for alpha, betha in hyperparameters:
                    cost_function = alpha * features["GPU Memory Usage"] + betha * (
                        1 - features["Test Accuracy"]
                    )
                    features[f"Cost Function ({alpha}, {betha})"] = cost_function
                data.append(features)
    data = pd.DataFrame(data)
    data.to_csv("extracted_features.csv", index=False)
    print('Data saved into "extracted_features.csv".')


if __name__ == "__main__":
    results_folder = "results"
    results_data = collect_results(results_folder)
