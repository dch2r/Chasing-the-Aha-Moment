"""TRL helpers for the ablation runner. Requires a GPU environment with TRL.

Kept at module level: Trainer.save_model pickles the config, and pickle cannot
find classes defined inside a function.
"""

import time

from transformers import TrainerCallback
from trl import GRPOConfig


class CompatibleGRPOConfig(GRPOConfig):
    def __post_init__(self):
        # TRL 0.22.2 copies an initialized config in prepare_peft_model.
        # Clear the derived field so its parent can recompute it consistently.
        if self.generation_batch_size is not None and self.steps_per_generation is not None:
            self.steps_per_generation = None
        super().__post_init__()


class TrainingTimeBudget(TrainerCallback):
    """Stop at the end of the step on which the wall-clock budget runs out."""

    def __init__(self, seconds):
        self.seconds, self.started, self.triggered = seconds, None, False

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = time.perf_counter()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if time.perf_counter() - self.started >= self.seconds:
            self.triggered = True
            control.should_training_stop = True
        return control
