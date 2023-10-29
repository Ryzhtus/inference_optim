import json
import os
from pathlib import Path
import torch
from evaluate import evaluator
from datasets import load_dataset
from transformers import (
    RobertaTokenizerFast,
    AutoModelForTokenClassification,
    set_seed,
)
from utils.memory import print_size_of_model

os.environ["OPENMP_NUM_THREADS"] = "4"


def roberta_dynamic_qunatization():
    torch.set_num_threads(4)
    print(torch.__config__.parallel_info())

    torch.backends.quantized.engine = "qnnpack"
    set_seed(42)

    conll2003 = load_dataset("conll2003")
    tokenizer = RobertaTokenizerFast.from_pretrained(
        "Ryzhtus/conll2003-roberta-large", add_prefix_space=True
    )

    label_list = conll2003["train"].features["ner_tags"].feature.names
    id2label = {idx: label for idx, label in zip(range(len(label_list)), label_list)}
    label2id = {label: idx for label, idx in zip(label_list, range(len(label_list)))}

    model = AutoModelForTokenClassification.from_pretrained(
        "Ryzhtus/conll2003-roberta-large",
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    task_evaluator = evaluator("token-classification")

    eval_results = task_evaluator.compute(
        model_or_pipeline=quantized_model,
        data=conll2003["test"],
        metric="seqeval",
        tokenizer=tokenizer,
    )

    print(eval_results)
    print_size_of_model(model)

    json_object = json.dumps(eval_results, indent=4, default=str)

    with open(
        Path.cwd() / "results" / "ner_roberta_dynamic_quantized_4_threads.json", "w"
    ) as result_file:
        result_file.write(json_object)
