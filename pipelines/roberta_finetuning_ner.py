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
from utils.ner_utils import tokenize_and_align_labels, compute_ner_metrics
from utils.memory import MemoryCallback


def roberta_pipeline():
    set_seed(42)

    conll2003 = load_dataset("conll2003")
    tokenizer = RobertaTokenizerFast.from_pretrained(
        "roberta-large", add_prefix_space=True
    )
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    label_list = conll2003["train"].features["ner_tags"].feature.names

    tokenized_conll = conll2003.map(tokenize_and_align_labels, batched=True)
    id2label = {idx: label for idx, label in zip(range(len(label_list)), label_list)}
    label2id = {label: idx for label, idx in zip(label_list, range(len(label_list)))}

    model = AutoModelForTokenClassification.from_pretrained(
        "roberta-large",
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir="conll2003-roberta-large",
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
        run_name="ner-roberta",
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

    task_evaluator = evaluator("token-classification")

    eval_results = task_evaluator.compute(
        model_or_pipeline=model,
        data=conll2003["test"],
        metric="seqeval",
        tokenizer=tokenizer,
    )

    with open(
        Path.cwd() / "results" / "ner_roberta_finetuning.log", "w"
    ) as result_file:
        result_file.write(eval_results)