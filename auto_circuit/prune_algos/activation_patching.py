from typing import Dict, Literal, Optional

import torch as t

from auto_circuit.data import PromptDataLoader 
from auto_circuit.types import AblationType, BatchKey, Edge, PruneScores
from auto_circuit.utils.ablation_activations import batch_src_ablations
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import (
    patch_mode,
    set_all_masks
)
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.prune_algos.utils import compute_loss


def act_patch_prune_scores(
    model: PatchableModel,
    dataloader: PromptDataLoader,
    grad_function: Literal["logit", "prob", "logprob", "logit_exp"],
    answer_function: Literal["avg_diff", "max_diff", "avg_val", "mse"],
    ablation_type: AblationType = AblationType.RESAMPLE,
    clean_corrupt: Optional[Literal["clean", "corrupt"]] = "corrupt",
) -> PruneScores:
    out_slice = model.out_slice
    src_outs: Dict[BatchKey, t.Tensor] = batch_src_ablations(
        model,
        dataloader,
        ablation_type=ablation_type,
        clean_corrupt=clean_corrupt,
    )
    prune_scores = model.new_prune_scores()
    # compute loss on full model 
    with t.no_grad():
        for batch in tqdm(dataloader, desc="Full model"):
            loss = compute_loss(model, batch, grad_function, answer_function, out_slice)
            for edge in model.edges:
                prune_scores[edge.dest.module_name][edge.patch_idx] += loss.sum().item()
    # compute loss on model with each edge ablated
    for edge in tqdm(model.edges, desc="Edge ablation"):
        edge: Edge
        set_all_masks(model, val=0)
        for batch in dataloader:
            patch_src_outs = src_outs[batch.key].clone().detach()
            with patch_mode(model, patch_src_outs, edges=[edge]):
                loss = compute_loss(model, batch, grad_function, answer_function, out_slice)
            prune_scores[edge.dest.module_name][edge.patch_idx] -= loss.sum().item()
    return prune_scores