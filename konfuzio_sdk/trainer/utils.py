"""Add utility common functions and classes to be used for AI Training."""


from konfuzio_sdk.extras import torch, Trainer, TrainerCallback
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class LoggerCallback(TrainerCallback):
    """
    Custom callback for logger.info to be used in Trainer.

    This callback is called by `Trainer` at the end of every epoch to log metrics.
    It replaces calling `print` and `tqdm` and calls `logger.info` instead.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log losses and metrics when training or evaluating using Trainer."""
        _ = logs.pop('total_flos', None)
        if state.is_local_process_zero:
            logger.info(logs)


class BalancedLossTrainer(Trainer):
    """Custom trainer with custom loss to leverage class weights."""

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute weighted cross-entropy loss to recompensate for unbalanced datasets."""
        labels = inputs.pop('labels')
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(self.class_weights, device=model.device, dtype=torch.float)
        )
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        param: logs (`Dict[str, float]`): The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
            logs["percentage"] = round(100 * self.state.epoch / self.args.num_train_epochs, 2)

        # reformatting metrics

        ##########################################
        if "loss" in logs:
            logs["loss"] = round(logs["loss"], 4)
        if "eval_loss" in logs:
            logs["eval_loss"] = round(logs["eval_loss"], 4)
        if "train_loss" in logs:
            logs["train_loss"] = round(logs["train_loss"], 4)
        if "eval_samples_per_second" in logs:
            logs["eval_samples_per_second"] = round(logs["eval_samples_per_second"], 2)
        if "train_samples_per_second" in logs:
            logs["train_samples_per_second"] = round(logs["train_samples_per_second"], 2)
        if "eval_steps_per_second" in logs:
            logs["eval_steps_per_second"] = round(logs["eval_steps_per_second"], 2)
        if "train_steps_per_second" in logs:
            logs["train_steps_per_second"] = round(logs["train_steps_per_second"], 2)
        if "eval_runtime" in logs:
            logs["eval_runtime"] = round(logs["eval_runtime"], 2)
        if "train_runtime" in logs:
            logs["train_runtime"] = round(logs["train_runtime"], 2)
        if "total_flos" in logs:
            logs.pop("total_flos")
        if "learning_rate" in logs:
            learning_rate_formatted = f"{logs['learning_rate']:.2e}"
            logs["learning_rate"] = float(learning_rate_formatted)
        ##########################################
        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
