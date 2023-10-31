import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from datasets import load_dataset


def generate_prompt(train_examples, test_example):
    propmt_template_head = "Your task is to choose the correct completion for a given sentence from 4 available options.\n"
    prompt_templeate_example = "Sentence: '{}', Return the most likely ending for this sentence from these 4 options: {}. Answer: {}"

    prompt = propmt_template_head

    for idx in range(10):
        sample = train_examples[idx]
        ctx = sample["ctx"]
        endings = sample["endings"]
        answer = endings[int(sample["label"])]

        prompt += prompt_templeate_example.format(ctx, endings, answer)

    prompt += prompt_templeate_example.format(
        test_example["ctx"], test_example["endings"], ""
    )

    return prompt


def mistral_few_shot_pipeline():
    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

    hellaswag = load_dataset("Rowan/hellaswag")
    train_examples = hellaswag["train"]
    test_examples = hellaswag["test"]

    prompt = generate_prompt(train_examples, test_examples[0])

    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate
    generate_ids = model.generate(inputs.input_ids, max_length=30)
    answer = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print(answer)


if __name__ == "__main__":
    mistral_few_shot_pipeline()
