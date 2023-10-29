import json
from pathlib import Path
from evaluate import evaluator
from datasets import load_dataset
from transformers import (
    RobertaTokenizerFast,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)
from utils.ner_utils import compute_ner_metrics
from utils.memory import MemoryCallback


def tokenize_and_align_labels(examples):
    tokenizer = RobertaTokenizerFast.from_pretrained(
        "distilroberta-base", add_prefix_space=True
    )

    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(
            batch_index=i
        )  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif (
                word_idx != previous_word_idx
            ):  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def roberta_pipeline():
    set_seed(42)

    conll2003 = load_dataset("conll2003")
    tokenizer = RobertaTokenizerFast.from_pretrained(
        "distilroberta-base", add_prefix_space=True
    )
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    label_list = conll2003["train"].features["ner_tags"].feature.names

    tokenized_conll = conll2003.map(tokenize_and_align_labels, batched=True)
    id2label = {idx: label for idx, label in zip(range(len(label_list)), label_list)}
    label2id = {label: idx for label, idx in zip(label_list, range(len(label_list)))}

    model = AutoModelForTokenClassification.from_pretrained(
        "distilroberta-base",
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir="conll2003-distill-roberta-large",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=True,
        report_to="wandb",
        run_name="ner-distill-roberta-base",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_conll["train"],
        eval_dataset=tokenized_conll["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_ner_metrics,
        callbacks=[MemoryCallback],
    )

    trainer.train()
    trainer.push_to_hub()

    task_evaluator = evaluator("token-classification")

    eval_results = task_evaluator.compute(
        model_or_pipeline=model,
        data=conll2003["test"],
        metric="seqeval",
        tokenizer=tokenizer,
    )

    print(eval_results)

    json_object = json.dumps(eval_results, indent=4, default=str)

    with open(
        Path.cwd() / "results" / "ner_distill-roberta_finetuning.json", "w"
    ) as result_file:
        result_file.write(json_object)
