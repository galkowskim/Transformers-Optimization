import collections
import json
import os

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

N_BEST = 20
MAX_ANSWER_LENGTH = 30

metric = evaluate.load("squad")


def compute_metrics(start_logits, end_logits, features, examples):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -N_BEST - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -N_BEST - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > MAX_ANSWER_LENGTH
                    ):
                        continue

                    answer = {
                        "text": context[
                            offsets[start_index][0] : offsets[end_index][1]
                        ],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [
        {"id": ex["id"], "answers": ex["answers"]} for ex in examples
    ]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)


def main(config):

    MAX_LENGTH = int(config["max_length"] * 0.75)
    STRIDE = 128 * (MAX_LENGTH // 512)

    def preprocess_training_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=MAX_LENGTH,
            truncation="only_second",
            stride=STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if (
                offset[context_start][0] > start_char
                or offset[context_end][1] < end_char
            ):
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    def preprocess_validation_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=MAX_LENGTH,
            truncation="only_second",
            stride=STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return inputs

    torch.cuda.empty_cache()
    model_name = config["model_name"]
    optimizer_type = config["optimizer"]
    model_path = (
        model_name.split("/")[-1].replace("-", "_") + "_QA_SQUAD_" + optimizer_type
    )
    BATCH_SIZE = config["batch_size"]
    NUM_EPOCHS = config["num_epochs"]

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    squad = load_dataset("squad", cache_dir="squad_data")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = squad["train"].map(
        preprocess_training_examples,
        batched=True,
        remove_columns=squad["train"].column_names,
    )
    validation_dataset = squad["validation"].map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=squad["validation"].column_names,
    )

    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    args = TrainingArguments(
        model_path,
        evaluation_strategy="no",
        per_device_train_batch_size=BATCH_SIZE,
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        fp16=True,
        optim=optimizer_type,
        push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()

    predictions, _, _ = trainer.predict(validation_dataset)
    start_logits, end_logits = predictions
    results = compute_metrics(
        start_logits, end_logits, validation_dataset, squad["validation"]
    )

    with open(f"{model_path}/test_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    train_losses = [
        {k: v for k, v in dictionary.items() if k in ("loss", "epoch")}
        for dictionary in trainer.state.log_history
    ]

    train_losses = [
        dictionary for dictionary in train_losses if len(dictionary.keys()) == 2
    ]

    train_losses_int_epoch = [
        {k: v for k, v in dictionary.items() if k in ("train_loss", "epoch")}
        for dictionary in trainer.state.log_history
    ]

    train_losses_int_epoch = [
        dictionary
        for dictionary in train_losses_int_epoch
        if len(dictionary.keys()) == 2
    ]

    for dictionary in train_losses_int_epoch:
        dictionary["loss"] = dictionary.pop("train_loss")

    train_losses += train_losses_int_epoch
    train_losses = pd.DataFrame(train_losses)
    train_losses.to_csv(f"{model_path}/train_losses.csv", index=False)

    with open(f"{model_path}/config.json", "w") as f:
        json.dump(config, f, indent=4)

    trainer.push_to_hub(commit_message="Training complete")


if __name__ == "__main__":
    with open("config.json") as f:
        config = json.load(f)
    main(config)
