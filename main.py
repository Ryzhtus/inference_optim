from pipelines.distilled_roberta_finetuning_ner import distilled_roberta_pipeline
from pipelines.roberta_finetuning_ner import roberta_pipeline

if __name__ == "__main__":
    roberta_pipeline()
    distilled_roberta_pipeline()
