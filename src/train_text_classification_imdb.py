import json
import os
import shutil
from argparse import ArgumentParser
from copy import deepcopy

import evaluate
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset, concatenate_datasets
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    logging,
)

logging.set_verbosity_error()


torch.cuda.empty_cache()

MAX_LENGTH = 512

OPTIMIZERS = {
    "adamw": "adamw_torch",
    "adafactor": "adafactor",
}


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")
    return info.used // 1024**2


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    MB = print_gpu_utilization()

    return {
        "time": result.metrics["train_runtime"],
        "samples/sec": result.metrics["train_samples_per_second"],
        "gpu_memory": MB,
    }


class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(
                eval_dataset=self._trainer.train_dataset, metric_key_prefix="train"
            )
            return control_copy


def main(args):
    if args.model == 'longformer':
        output_dir = f"longformer-base-512-text-classification-imdb-{args.optimizer}-att_win_size_{args.attention_window_size}-epochs-{args.num_epochs}"
    elif args.model == 'roberta':
        output_dir = f"roberta-base-text-classification-imdb-{args.optimizer}-epochs-{args.num_epochs}"
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model_name = args.model_name
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs

    data_dir = args.data_dir
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    dataset = load_dataset("imdb", cache_dir=data_dir)
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    def stratified_split(dataset, n_samples):
        df = dataset.to_pandas()
        df_0 = df[df['label'] == 0]
        df_1 = df[df['label'] == 1]

        sample_0, _ = train_test_split(df_0, train_size=n_samples//2, stratify=df_0['label'], random_state=42)
        sample_1, _ = train_test_split(df_1, train_size=n_samples//2, stratify=df_1['label'], random_state=42)

        sample = pd.concat([sample_0, sample_1])
        remaining = df.drop(sample.index)

        return Dataset.from_pandas(sample), Dataset.from_pandas(remaining)

    val_from_train, new_train = stratified_split(train_dataset, 7500)
    val_from_test, new_test = stratified_split(test_dataset, 7500)

    validation_dataset = concatenate_datasets([val_from_train, val_from_test])

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)

    tokenized_train = new_train.map(preprocess_function, batched=True, remove_columns=['text'])
    tokenized_test = new_test.map(preprocess_function, batched=True, remove_columns=['text'])
    tokenized_eval = validation_dataset.map(preprocess_function, batched=True, remove_columns=['text'])

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    id2label = {
        0: "SADNESS",
        1: "JOY",
    }
    label2id = {v: k for k, v in id2label.items()}

    cfg = AutoConfig.from_pretrained(model_name)
    if args.model == 'longformer':
        cfg.attention_window = args.attention_window_size
        cfg.max_position_embeddings = MAX_LENGTH + 2
    cfg.num_labels = 2
    cfg.id2label = id2label
    cfg.label2id = label2id

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_config(cfg)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy='epoch',
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=10,
        weight_decay=0.01,
        fp16=True,
        optim=OPTIMIZERS[args.optimizer],
        push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.add_callback(CustomCallback(trainer))

    result = trainer.train()

    summary = print_summary(result)

    with open(f"{output_dir}/summary.json", "w") as f:
        json.dump(summary, f)
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(vars(args), f)

    train_losses = [
        {
            k: v
            for k, v in dictionary.items()
            if k in ("train_loss", "train_accuracy", "epoch")
        }
        for dictionary in trainer.state.log_history
        if "train_accuracy" in dictionary
    ]
    train_losses = pd.DataFrame(train_losses)
    train_losses.epoch = train_losses.epoch.astype(int)

    val_losses = [
        {
            k: v
            for k, v in dictionary.items()
            if k in ("eval_loss", "eval_accuracy", "epoch")
        }
        for dictionary in trainer.state.log_history
        if "eval_accuracy" in dictionary
    ]
    val_losses = pd.DataFrame(val_losses)
    val_losses.epoch = val_losses.epoch.astype(int)

    train_losses.to_csv(f"{output_dir}/train_losses.csv", index=False)
    val_losses.to_csv(f"{output_dir}/val_losses.csv", index=False)

    output = trainer.predict(tokenized_test)

    with open(f"{output_dir}/test_results.json", "w") as f:
        json.dump(output.metrics, f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="allenai/longformer-base-4096"
    )
    parser.add_argument("--model", type=str, default='longformer')
    parser.add_argument("--num_epochs", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--scheduler", type=str, default="linear")
    parser.add_argument("--data_dir", type=str, default="model_path")
    parser.add_argument("--attention_window_size", type=int, default=64)

    args = parser.parse_args()
    main(args)
