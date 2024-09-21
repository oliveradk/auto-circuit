from typing import Literal, Optional

import torch as t
from torch.nn.functional import log_softmax

from auto_circuit.data import PromptPairBatch
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.utils.tensor_ops import (
    batch_avg_answer_diff, 
    batch_avg_answer_val, 
    batch_avg_answer_max_diff, 
    multibatch_kl_div, 
    multibatch_js_div
)


def compute_loss(
    model: PatchableModel,
    batch: PromptPairBatch,
    grad_function: Literal["logit", "prob", "logprob", "logit_exp"],
    answer_function: Literal["avg_diff", "max_diff", "avg_val", "mse", "kl_div", "js_div"],
    logits: Optional[t.Tensor] = None,
    clean_out: Optional[t.Tensor] = None,
) -> t.Tensor:
    if answer_function in ["kl_div", "js_div"]:
        assert grad_function == "logprob"
    if logits is None:
        logits = model(batch.clean)[model.out_slice]

    # get output values
    if grad_function == "logit":
        vals = logits
    elif grad_function == "prob":
        vals = t.softmax(logits, dim=-1)
    elif grad_function == "logprob":
        vals = log_softmax(logits, dim=-1)
    elif grad_function == "logit_exp":
        numerator = t.exp(logits)
        denominator = numerator.sum(dim=-1, keepdim=True)
        vals = numerator / denominator.detach()
    else:
        raise ValueError(f"Unknown grad_function: {grad_function}")

    if answer_function == "avg_diff":
        loss = -batch_avg_answer_diff(vals, batch)
    elif answer_function == "max_diff":
        loss = -batch_avg_answer_max_diff(vals, batch)
    elif answer_function == "avg_val":
        loss = -batch_avg_answer_val(vals, batch)
    elif answer_function == "mse":
        loss = t.nn.functional.mse_loss(vals, batch.answers)
    elif answer_function == "kl_div":
        clean_logprobs = t.nn.functional.log_softmax(clean_out, dim=-1)
        loss = multibatch_kl_div(vals, clean_logprobs)
    elif answer_function == "js_div":
        clean_logprobs = t.nn.functional.log_softmax(clean_out, dim=-1)
        loss = multibatch_js_div(vals, clean_logprobs)
    else:
        raise ValueError(f"Unknown answer_function: {answer_function}")
    return loss

