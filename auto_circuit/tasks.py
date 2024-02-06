from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch as t
import transformer_lens as tl

from auto_circuit.data import PromptDataLoader, load_datasets_from_json
from auto_circuit.metrics.official_circuits.docstring_official import (
    docstring_true_edges,
)
from auto_circuit.metrics.official_circuits.greaterthan_official import (
    greaterthan_true_edges,
)
from auto_circuit.metrics.official_circuits.ioi_official import ioi_true_edges
from auto_circuit.metrics.official_circuits.sports_players_official import (
    sports_players_true_edges,
)
from auto_circuit.model_utils.sparse_autoencoders.autoencoder_transformer import (
    AutoencoderTransformer,
    sae_model,
)
from auto_circuit.types import AutoencoderInput, Edge, OutputSlice, TaskKey
from auto_circuit.utils.graph_utils import patchable_model
from auto_circuit.utils.misc import repo_path_to_abs_path
from auto_circuit.utils.patchable_model import PatchableModel

MODEL_CACHE: Dict[Any, t.nn.Module] = {}
DATASET_CACHE: Dict[Any, Tuple[PromptDataLoader, PromptDataLoader]] = {}


@dataclass
class Task:
    key: TaskKey
    name: str
    batch_size: int | Tuple[int, int]  # (train, test) if tuple
    batch_count: int | Tuple[int, int]  # (train, test) if tuple
    token_circuit: bool
    _model_def: str | t.nn.Module
    _dataset_name: str
    separate_qkv: bool = True
    _true_edge_func: Optional[Callable[..., Set[Edge]]] = None
    slice_output: OutputSlice = "last_seq"
    autoencoder_input: Optional[AutoencoderInput] = None
    autoencoder_max_latents: Optional[int] = None
    autoencoder_pythia_size: Optional[str] = None
    autoencoder_prune_with_corrupt: Optional[bool] = None
    dtype: t.dtype = t.float32
    __init_complete__: bool = False

    @property
    def true_edges(self) -> Optional[Set[Edge]]:
        self.init_task() if not self.__init_complete__ else None
        return self._true_edges

    @property
    def true_edge_count(self) -> Optional[int]:
        return len(self.true_edges) if self.true_edges is not None else None

    @property
    def model(self) -> PatchableModel:
        self.init_task() if not self.__init_complete__ else None
        return self._model

    @property
    def train_loader(self) -> PromptDataLoader:
        self.init_task() if not self.__init_complete__ else None
        return self._train_loader

    @property
    def test_loader(self) -> PromptDataLoader:
        self.init_task() if not self.__init_complete__ else None
        return self._test_loader

    def init_task(self):
        if self.autoencoder_input is not None:
            # We prune autoencoders using the dataset, so we cache the dataset name
            assert isinstance(self._model_def, str)
            model_cache_key = (
                self._model_def,
                self.autoencoder_input,
                self.autoencoder_max_latents,
                self.autoencoder_pythia_size,
                self.autoencoder_prune_with_corrupt,
                self._dataset_name,
                self.dtype,
            )
        else:
            model_cache_key = (
                self._model_def,
                self.dtype,
            )

        using_cached_model = False
        if isinstance(self._model_def, t.nn.Module):
            model = self._model_def
            self.device: t.device = next(model.parameters()).device
        elif model_cache_key in MODEL_CACHE:
            model = MODEL_CACHE[model_cache_key]
            self.device: t.device = next(model.parameters()).device
            using_cached_model = True
        else:
            device_str = "cuda" if t.cuda.is_available() else "cpu"
            self.device: t.device = t.device(device_str)
            if self.dtype != t.float32:
                model = tl.HookedTransformer.from_pretrained_no_processing(
                    self._model_def, device=self.device, dtype=self.dtype
                )
            else:
                model = tl.HookedTransformer.from_pretrained(
                    self._model_def,
                    device=self.device,
                    fold_ln=True,
                    center_writing_weights=True,
                    center_unembed=True,
                )
            model.cfg.use_attn_result = True
            model.cfg.use_attn_in = True
            model.cfg.use_split_qkv_input = True
            model.cfg.use_hook_mlp_in = True
            assert model.tokenizer is not None
            model.eval()

            if self.autoencoder_input is not None:
                model = sae_model(
                    model=model,
                    sae_input=self.autoencoder_input,
                    load_pretrained=True,
                    pythia_size=self.autoencoder_pythia_size,
                    new_instance=False,
                )
            MODEL_CACHE[model_cache_key] = model

        for param in model.parameters():
            param.requires_grad = False

        dataset_cache_key = (
            self._dataset_name,
            self.batch_size,
            self.batch_count,
            self.token_circuit,
            self.device,
        )
        if dataset_cache_key in DATASET_CACHE:
            self._train_loader, self._test_loader = DATASET_CACHE[dataset_cache_key]
        else:
            bs, b_count = self.batch_size, self.batch_count
            bs_1 = bs[0] if isinstance(bs, tuple) else bs
            count_1 = b_count[0] if isinstance(b_count, tuple) else b_count
            bs_2 = bs[1] if isinstance(bs, tuple) else bs
            count_2 = b_count[1] if isinstance(b_count, tuple) else b_count
            has_tokenizer = hasattr(model, "tokenizer") and model.tokenizer is not None
            train_loader, test_loader = load_datasets_from_json(
                model=model if has_tokenizer else None,
                path=repo_path_to_abs_path(f"datasets/{self._dataset_name}.json"),
                device=self.device,
                prepend_bos=has_tokenizer,
                batch_size=self.batch_size,
                train_test_split=[bs_1 * count_1, bs_2 * count_2],
                length_limit=bs_1 * count_1 + bs_2 * count_2,
                return_seq_length=self.token_circuit,
                tail_divergence=True if has_tokenizer else False,
                pad=True,
            )
            DATASET_CACHE[dataset_cache_key] = (train_loader, test_loader)
            self._train_loader, self._test_loader = (train_loader, test_loader)

        if isinstance(model, AutoencoderTransformer) and not using_cached_model:
            assert self.autoencoder_prune_with_corrupt is not None
            model._prune_latents_with_dataset(
                dataloader=self._train_loader,
                max_latents=self.autoencoder_max_latents,
                include_corrupt=self.autoencoder_prune_with_corrupt,
                seq_len=self._train_loader.seq_len,
            )  # in place operation

        seq_len = self._train_loader.seq_len
        diverge_idx = self._train_loader.diverge_idx
        kv_caches = self._train_loader.kv_cache, self._test_loader.kv_cache
        self._model = patchable_model(
            model,
            True,
            self.separate_qkv,
            self.slice_output,
            seq_len,
            kv_caches,
            self.device,
        )

        if self._true_edge_func is not None:
            if self.token_circuit:
                self._true_edges = self._true_edge_func(self._model, True, diverge_idx)
            else:
                self._true_edges = self._true_edge_func(self._model)
        else:
            self._true_edges = None
        self.__init_complete__ = True


SPORTS_PLAYERS_TOKEN_CIRCUIT_TASK: Task = Task(
    key="Sports Players Token Circuit",
    name="Sports Players",
    _model_def="pythia-2.8b-deduped",
    _dataset_name="sports-players/sports_players_pythia-2.8b-deduped_prompts",
    batch_size=(10, 20),  # There are 3 sports (football, basketball, baseball),
    batch_count=(
        10,
        5,
    ),  # 70 prompts for each sport (210 total), 105 test and 105 train
    _true_edge_func=sports_players_true_edges,
    token_circuit=True,
    separate_qkv=False,
    dtype=t.bfloat16,
)
SPORTS_PLAYERS_COMPONENT_CIRCUIT_TASK: Task = Task(
    key="Sports Players Component Circuit",
    name="Sports Players",
    _model_def="pythia-2.8b-deduped",
    _dataset_name="sports-players/sports_players_pythia-2.8b-deduped_prompts",
    batch_size=15,
    batch_count=7,
    _true_edge_func=sports_players_true_edges,
    token_circuit=False,
    separate_qkv=False,
    dtype=t.bfloat16,
)
IOI_TOKEN_CIRCUIT_TASK: Task = Task(
    key="Indirect Object Identification Token Circuit",
    name="Indirect Object Identification",
    _model_def="gpt2-small",
    _dataset_name="ioi_single_template_prompts",
    batch_size=64,
    batch_count=2,
    _true_edge_func=ioi_true_edges,
    token_circuit=True,
)
IOI_COMPONENT_CIRCUIT_TASK: Task = Task(
    key="Indirect Object Identification Component Circuit",
    name="Indirect Object Identification",
    _model_def="gpt2-small",
    _dataset_name="ioi_prompts",
    batch_size=64,
    batch_count=2,
    _true_edge_func=ioi_true_edges,
    token_circuit=False,
)
IOI_GPT2_AUTOENCODER_COMPONENT_CIRCUIT_TASK: Task = Task(
    key="Indirect Object Identification GPT2 Autoencoder Component Circuit",
    name="Indirect Object Identification",
    _model_def="gpt2-small",
    _dataset_name="ioi_prompts",
    batch_size=1,
    batch_count=2,
    _true_edge_func=None,
    token_circuit=False,
    autoencoder_input="resid_delta_mlp",
    autoencoder_prune_with_corrupt=False,
)
DOCSTRING_TOKEN_CIRCUIT_TASK: Task = Task(
    key="Docstring Token Circuit",
    name="Docstring",
    _model_def="attn-only-4l",
    _dataset_name="docstring_prompts",
    batch_size=128,
    batch_count=2,
    _true_edge_func=docstring_true_edges,
    token_circuit=True,
)
DOCSTRING_COMPONENT_CIRCUIT_TASK: Task = Task(
    key="Docstring Component Circuit",
    name="Docstring",
    _model_def="attn-only-4l",
    _dataset_name="docstring_prompts",
    batch_size=128,
    batch_count=2,
    _true_edge_func=docstring_true_edges,
    token_circuit=False,
)
GREATERTHAN_COMPONENT_CIRCUIT_TASK: Task = Task(
    key="Greaterthan Component Circuit",
    name="Greaterthan",
    _model_def="gpt2-small",
    _dataset_name="greaterthan_gpt2-small_prompts",
    batch_size=64,
    batch_count=2,
    _true_edge_func=greaterthan_true_edges,
    token_circuit=False,
)
GREATERTHAN_GPT2_AUTOENCODER_COMPONENT_CIRCUIT_TASK: Task = Task(
    key="Greaterthan GPT2 Autoencoder Component Circuit",
    name="Greaterthan",
    _model_def="gpt2-small",
    _dataset_name="greaterthan_gpt2-small_prompts",
    batch_size=1,
    batch_count=2,
    _true_edge_func=None,
    token_circuit=False,
    autoencoder_input="resid_delta_mlp",
    autoencoder_prune_with_corrupt=False,
)
ANIMAL_DIET_GPT2_AUTOENCODER_COMPONENT_CIRCUIT_TASK: Task = Task(
    key="Animal Diet GPT2 Autoencoder Component Circuit",
    name="Animal Diet",
    _model_def="gpt2-small",
    _dataset_name="animal_diet_short_prompts",
    batch_size=2,
    batch_count=16,
    _true_edge_func=None,
    token_circuit=False,
    autoencoder_input="resid_delta_mlp",
    autoencoder_prune_with_corrupt=False,
)
CAPITAL_CITIES_PYTHIA_70M_AUTOENCODER_COMPONENT_CIRCUIT_TASK: Task = Task(
    key="Capital Cities Pythia 70M Deduped Autoencoder Component Circuit",
    name="Capital Cities",
    _model_def="pythia-70m-deduped",
    _dataset_name="capital_cities_pythia-70m-deduped_prompts",
    batch_size=2,
    batch_count=4,
    _true_edge_func=None,
    token_circuit=True,
    autoencoder_input="resid_delta_mlp",
    autoencoder_max_latents=200,
    autoencoder_pythia_size="2_32768",
    autoencoder_prune_with_corrupt=False,
)

TASKS: List[Task] = [
    SPORTS_PLAYERS_TOKEN_CIRCUIT_TASK,
    SPORTS_PLAYERS_COMPONENT_CIRCUIT_TASK,
    IOI_TOKEN_CIRCUIT_TASK,
    IOI_COMPONENT_CIRCUIT_TASK,
    IOI_GPT2_AUTOENCODER_COMPONENT_CIRCUIT_TASK,
    DOCSTRING_TOKEN_CIRCUIT_TASK,
    DOCSTRING_COMPONENT_CIRCUIT_TASK,
    GREATERTHAN_COMPONENT_CIRCUIT_TASK,
    GREATERTHAN_GPT2_AUTOENCODER_COMPONENT_CIRCUIT_TASK,
    ANIMAL_DIET_GPT2_AUTOENCODER_COMPONENT_CIRCUIT_TASK,
    CAPITAL_CITIES_PYTHIA_70M_AUTOENCODER_COMPONENT_CIRCUIT_TASK,
]
TASK_DICT: Dict[TaskKey, Task] = {task.key: task for task in TASKS}
