from typing import Literal, Optional

import torch as t
from torch.nn.functional import log_softmax

from auto_circuit.data import PromptPairBatch
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.utils.tensor_ops import batch_avg_answer_diff, batch_avg_answer_val, batch_avg_answer_max_diff


def compute_loss(
    model: PatchableModel,
    batch: PromptPairBatch,
    grad_function: Literal["logit", "prob", "logprob", "logit_exp"],
    answer_function: Literal["avg_diff", "max_diff", "avg_val", "mse"],
    out_slice: slice,
    logits: Optional[t.Tensor] = None,
) -> t.Tensor:
    if logits is None:
        logits = model(batch.clean)[out_slice]
    if grad_function == "logit":
        token_vals = logits
    elif grad_function == "prob":
        token_vals = t.softmax(logits, dim=-1)
    elif grad_function == "logprob":
        token_vals = log_softmax(logits, dim=-1)
    elif grad_function == "logit_exp":
        numerator = t.exp(logits)
        denominator = numerator.sum(dim=-1, keepdim=True)
        token_vals = numerator / denominator.detach()
    else:
        raise ValueError(f"Unknown grad_function: {grad_function}")

    if answer_function == "avg_diff":
        loss = -batch_avg_answer_diff(token_vals, batch)
    elif answer_function == "max_diff":
        loss = -batch_avg_answer_max_diff(token_vals, batch)
    elif answer_function == "avg_val":
        loss = -batch_avg_answer_val(token_vals, batch)
    elif answer_function == "mse":
        loss = t.nn.functional.mse_loss(token_vals, batch.answers)
    else:
        raise ValueError(f"Unknown answer_function: {answer_function}")
    return loss

