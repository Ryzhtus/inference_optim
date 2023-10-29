import os
import torch
from transformers import TrainerCallback


class MemoryCallback(TrainerCallback):
    "A callback that prints a message at the beginning of training"

    def on_train_end(self, args, state, control, **kwargs):
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print("Size (MB):", os.path.getsize("temp.p") / 1e6)
    os.remove("temp.p")
