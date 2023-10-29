from transformers import TrainerCallback
import torch


class MemoryCallback(TrainerCallback):
    "A callback that prints a message at the beginning of training"

    def on_train_end(self, args, state, control, **kwargs):
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")
