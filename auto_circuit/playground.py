#%%
import os
import pickle
import random
from datetime import datetime
from functools import partial
from typing import Callable, Dict

import numpy as np
import torch as t
import torch.backends.mps
import transformer_lens as tl

import auto_circuit
import auto_circuit.data
import auto_circuit.prune
import auto_circuit.utils.graph_utils
from auto_circuit.prune import (
    measure_kl_div,
    run_pruned,
)
from auto_circuit.prune_functions.ACDC import acdc_edge_counts, acdc_prune_scores
from auto_circuit.prune_functions.activation_magnitude import (
    activation_magnitude_prune_scores,
)
from auto_circuit.prune_functions.parameter_integrated_gradients import (
    BaselineWeights,
    parameter_integrated_grads_prune_scores,
)
from auto_circuit.prune_functions.random_edges import random_prune_scores
from auto_circuit.types import ActType, Edge, EdgeCounts, ExperimentType
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import edge_counts_util
from auto_circuit.utils.misc import percent_gpu_mem_used
from auto_circuit.visualize import kl_vs_edges_plot

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)
os.environ["TOKENIZERS_PARALLELISM"] = "False"
#%%

device = (
    "cuda"
    if t.cuda.is_available()
    else "mps"
    if True and torch.backends.mps.is_available()
    else "cpu"
)
print("device", device)
toy_model = False
if toy_model:
    cfg = tl.HookedTransformerConfig(
        d_vocab=50257,
        n_layers=3,
        d_model=4,
        n_ctx=64,
        n_heads=2,
        d_head=2,
        act_fn="gelu",
        tokenizer_name="gpt2",
        device=device,
    )
    model = tl.HookedTransformer(cfg)
    model.init_weights()
else:
    model = tl.HookedTransformer.from_pretrained("gpt2-small", device=device)

model.cfg.use_attn_result = True
model.cfg.use_split_qkv_input = True
model.cfg.use_hook_mlp_in = True
# model = t.compile(model)

# repo_root = "/Users/josephmiller/Documents/auto-circuit"
repo_root = "/home/dev/auto-circuit"
data_file = "datasets/indirect_object_identification.json"
data_path = f"{repo_root}/{data_file}"
print(percent_gpu_mem_used())

#%%
# ---------- Config ----------
experiment_type = ExperimentType(input_type=ActType.CLEAN, patch_type=ActType.CORRUPT)
factorized = True
pig_baseline, pig_samples = BaselineWeights.ZERO, 50
edge_counts = EdgeCounts.LOGARITHMIC
acdc_tao_range, acdc_tao_step = (1e-6, 2e-5), 2e-6

train_loader, test_loader = auto_circuit.data.load_datasets_from_json(
    model.tokenizer,
    data_path,
    device=device,
    prepend_bos=True,
    batch_size=32,
    train_test_split=[0.5, 0.5],
    length_limit=64,
)
# ----------------------------
#%%
prune_funcs: Dict[str, Callable] = {
    f"PIG ({pig_baseline.name.lower()} Base, {pig_samples} iter)": partial(
        parameter_integrated_grads_prune_scores,
        baseline_weights=pig_baseline,
        samples=pig_samples,
    ),
    "Act Mag": activation_magnitude_prune_scores,
    "Random": random_prune_scores,
    f"ACDC (\u03C4={acdc_tao_range})": partial(
        acdc_prune_scores, tao_range=acdc_tao_range, tao_step=acdc_tao_step
    ),
    # "Subnetwork Probing": partial(
    #     subnetwork_probing_prune_scores, learning_rate=1e-2, epochs=200, max_lambda=30
    # ),
}
prune_scores_dict: Dict[str, Dict[Edge, float]] = {}
for name, prune_func in (prune_score_pbar := tqdm(prune_funcs.items())):
    prune_score_pbar.set_description_str(f"Computing prune scores: {name}")
    prune_scores_dict[name] = prune_func(model, factorized, train_loader)
#%%
# SAVE PRUNE SCORES DICT
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
with open(f"../.prune_scores_cache/prune_scores_dict-{dt_string}.pkl", "wb") as f:
    pickle.dump(prune_scores_dict, f)
#%%
# LOAD PRUNE SCORES DICT
date = "28-08-2023_23-38-19"
prune_scores_file_name = f"../.prune_scores_cache/prune_scores_dict-{date}.pkl"
if date is not None:
    with open(prune_scores_file_name, "rb") as f:
        prune_scores_dict = pickle.load(f)

#%%
test_edge_counts = edge_counts_util(model, factorized, edge_counts)
# pruned_outs_dict: Dict[str, Dict[int, List[t.Tensor]]] = {}
kl_divs: Dict[str, Dict[int, float]] = {}
for prune_func_str, prune_scores in (
    prune_func_pbar := tqdm(prune_scores_dict.items())
):
    prune_func_pbar.set_description_str(f"Pruning with {prune_func_str} scores")
    print("BEFORE prune_func_str", prune_func_str, percent_gpu_mem_used())
    test_edge = (
        acdc_edge_counts(model, factorized, prune_scores)
        if prune_func_str.startswith("ACDC")
        else test_edge_counts
    )
    pruned_outs = run_pruned(
        model, factorized, test_loader, experiment_type, test_edge, prune_scores
    )
    kl_clean, kl_corrupt = measure_kl_div(model, test_loader, pruned_outs)
    kl_divs[prune_func_str + " clean"] = kl_clean
    kl_divs[prune_func_str + " corr"] = kl_corrupt
    del pruned_outs
    t.cuda.empty_cache()
    print("AFTER prune_func_str", prune_func_str, percent_gpu_mem_used())
#%%
kl_vs_edges_plot(kl_divs, experiment_type, edge_counts, factorized).show()

#%%

# %%
