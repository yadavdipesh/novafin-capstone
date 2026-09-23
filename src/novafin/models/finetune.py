"""
novafin-capstone/src/novafin/models/finetune.py

D4 Level 3 - model-specific deep fine-tuning.

What Level 3 is, and why it is not just "more tuning"
-----------------------------------------------------
Level 2 searches *hyper-parameters* - it refits a model from scratch at each
trial. Level 3 adapts a model that has already been fitted: it continues
training, or it transfers learned representations to a new task. Three
techniques, each matched to a data type present in this project:

=========================  ==========================  ========================
Technique                  Applies to                  Implemented in
=========================  ==========================  ========================
Staged / warm-start        tabular boosting            :func:`staged_boosting`
boosting (``init_model``)  (M1, M2, M3, M4, M7, M10)
Self-supervised            pooled tabular -> per-head  :class:`FTTransformer`,
pre-training + LoRA        adapters (all modules)      :func:`attach_lora`
Layer-wise unfreezing      the HFT sequence net (M9)   :func:`layerwise_unfreeze_schedule`
with discriminative LRs
=========================  ==========================  ========================

The honest note about PEFT/LoRA in this project
------------------------------------------------
The D4 brief names "PEFT/LoRA for transformers". **There is no text or image
data in this package** - all eight files are tabular or panel numeric. Adapting
a pretrained language model here would be decorative: there is nothing for its
pretrained representations to transfer.

So LoRA is applied where it is genuinely meaningful instead. An FT-Transformer
(Gorishniy et al., 2021) is pre-trained *self-supervised* on the pooled
tabular data with a masked-feature objective - no labels - and each module then
attaches its own low-rank adapter and head. The shared trunk stays frozen. That
is exactly the PEFT pattern (one backbone, many cheap task adapters), applied
to the data that actually exists, and :func:`lora_parameter_report` quantifies
the saving.

Why LoRA rather than full fine-tuning
--------------------------------------
A weight update is approximated as a low-rank product:

.. math::
    W' = W + \\frac{\\alpha}{r} B A, \\quad
    A \\in \\mathbb{R}^{r \\times d}, \\; B \\in \\mathbb{R}^{d \\times r}, \\; r \\ll d

Only ``A`` and ``B`` train. For ``d = 192`` and ``r = 8`` that is 3,072
parameters against 36,864 - **8.3%** - per adapted matrix, and the frozen
trunk is shared across all eight modules. On a capstone-sized problem the
argument is not GPU memory; it is that a module with 89 positives (churn)
cannot fit a full transformer without memorising, but it can plausibly fit
a few thousand adapter weights on top of a representation learned from 190,000
unlabelled rows.

Reference: Hu EJ et al. (2021), "LoRA: Low-Rank Adaptation of Large Language
Models", arXiv:2106.09685. https://arxiv.org/abs/2106.09685

Reference: Gorishniy Y, Rubachev I, Khrulkov V, Babenko A (2021), "Revisiting
Deep Learning Models for Tabular Data", NeurIPS 34. https://arxiv.org/abs/2106.11959

Colab envelope
--------------
The FT-Transformer here is deliberately small (d_token 192, 3 blocks, ~1.2M
parameters). On a T4 that is seconds per epoch over 190,000 pooled rows, and it
fits in well under 1 GB of VRAM. If no GPU is present, ``compute.use_gpu: auto``
resolves to CPU and every function below still runs - slower, but the
LightGBM-only fallback path never needs it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from novafin.config import Config, load_config
from novafin.evaluate import EvaluationResult
from novafin.models.build import ModelSpec

__all__ = [
    "StageResult",
    "StagedBoostingResult",
    "staged_boosting",
    "LoRAConfig",
    "LoRALinear",
    "attach_lora",
    "lora_parameter_report",
    "FTTransformerConfig",
    "build_ft_transformer",
    "pretrain_masked_features",
    "layerwise_unfreeze_schedule",
    "discriminative_learning_rates",
]

LOGGER = logging.getLogger(__name__)


# =============================================================================
# Level 3a - staged / warm-start boosting
# =============================================================================
@dataclass
class StageResult:
    """One stage of a warm-start boosting run."""

    stage: int
    n_trees_added: int
    n_trees_total: int
    params: dict[str, Any]
    score: float
    improved: bool
    seconds: float = 0.0


@dataclass
class StagedBoostingResult:
    """The full staged-boosting trace."""

    module: str
    model_name: str
    objective: str
    stages: list[StageResult] = field(default_factory=list)
    best_stage: int = 0
    best_score: float = float("nan")
    baseline_score: float = float("nan")
    booster: Any | None = None
    notes: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        """The stage-by-stage table for the D7 guide."""
        return pd.DataFrame(
            [
                {
                    "stage": s.stage,
                    "trees_added": s.n_trees_added,
                    "trees_total": s.n_trees_total,
                    "score": round(s.score, 6),
                    "improved": s.improved,
                    "seconds": round(s.seconds, 2),
                    **{f"param.{k}": v for k, v in s.params.items()},
                }
                for s in self.stages
            ]
        )

    @property
    def improvement(self) -> float:
        if not np.isfinite(self.baseline_score) or not np.isfinite(self.best_score):
            return float("nan")
        return self.best_score - self.baseline_score


def staged_boosting(
    X: pd.DataFrame,
    y: pd.Series,
    splitter: Any,
    *,
    schedule: Sequence[dict[str, Any]],
    base_params: dict[str, Any] | None = None,
    scorer: Callable[[np.ndarray, np.ndarray], float],
    module: str = "",
    model_name: str = "lightgbm_staged",
    cfg: Config | None = None,
    split_kwargs: dict[str, Any] | None = None,
    objective_name: str = "score",
    task: str = "binary_classification",
    patience: int = 2,
) -> StagedBoostingResult:
    """Continue training a booster across stages with changing parameters.

    The technique, and why it is not the same as raising ``n_estimators``
    ---------------------------------------------------------------------
    LightGBM's ``init_model`` argument continues an existing booster rather
    than starting a new one. That allows the *learning schedule itself* to
    change between stages - which a single fit cannot do:

    * **stage 1** - a large learning rate and shallow trees learn the coarse
      structure quickly;
    * **stage 2** - the rate drops and capacity rises, refining the residual
      that stage 1 left behind;
    * **stage 3** - a very small rate with heavy regularisation polishes,
      without the capacity to memorise.

    This is the gradient-boosting analogue of a learning-rate schedule in deep
    learning, and it is the correct Level-3 technique for tabular data: it
    adapts an *already fitted* model rather than refitting from scratch, which
    is what separates Level 3 from Level 2.

    Each stage is scored on out-of-fold predictions and kept only if it
    improves; ``patience`` consecutive non-improving stages stop the run. Early
    stopping matters here because later stages have the whole earlier model to
    build on and can overfit quickly.

    Args:
        X, y: Training data.
        splitter: Phase-2 splitter for the module.
        schedule: One dict per stage. Recognised keys: ``n_estimators`` (trees
            to ADD in that stage) plus any LightGBM parameter to change.
        base_params: Parameters common to every stage.
        scorer: ``(y_true, y_pred) -> float``, higher is better.
        module, model_name: Labels for reporting.
        cfg: Project config.
        split_kwargs: Extra arguments for ``splitter.split``.
        objective_name: Metric name, for the report.
        task: Controls whether probabilities or point predictions are scored.
        patience: Consecutive non-improving stages before stopping.

    Returns:
        A :class:`StagedBoostingResult` with the per-stage trace.

    Raises:
        ImportError: If LightGBM is unavailable.
    """
    import time

    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "Staged boosting needs LightGBM: pip install -r requirements.txt"
        ) from exc

    cfg = cfg or load_config()
    split_kwargs = dict(split_kwargs or {})
    base_params = dict(base_params or {})
    y_values = np.asarray(y)

    folds = list(splitter.split(X, y_values, **split_kwargs))
    if not folds:
        raise ValueError(f"Splitter produced no folds for '{module}'.")

    numeric = X.select_dtypes(include=[np.number]).copy()
    numeric = numeric.fillna(numeric.median(numeric_only=True))

    is_classification = task.endswith("classification")
    objective = "binary" if is_classification else "regression"

    result = StagedBoostingResult(
        module=module, model_name=model_name, objective=objective_name
    )
    boosters: list[Any] = [None] * len(folds)
    total_trees = 0
    misses = 0

    for stage_index, stage_params in enumerate(schedule, start=1):
        started = time.perf_counter()
        params = {
            "objective": objective,
            "verbose": -1,
            "seed": cfg.reproducibility.seed,
            **base_params,
            **{k: v for k, v in stage_params.items() if k != "n_estimators"},
        }
        n_new = int(stage_params.get("n_estimators", 100))

        oof = np.full(len(numeric), np.nan, dtype="float64")
        for fold_index, (train_idx, test_idx) in enumerate(folds):
            dataset = lgb.Dataset(
                numeric.iloc[train_idx], label=y_values[train_idx], free_raw_data=False
            )
            # THE LEVEL-3 MECHANISM: continue the existing booster.
            booster = lgb.train(
                params,
                dataset,
                num_boost_round=n_new,
                init_model=boosters[fold_index],
                keep_training_booster=True,
            )
            boosters[fold_index] = booster
            oof[test_idx] = booster.predict(numeric.iloc[test_idx])

        mask = ~np.isnan(oof)
        score = float(scorer(y_values[mask], oof[mask]))
        total_trees += n_new
        improved = not result.stages or score > result.best_score

        result.stages.append(
            StageResult(
                stage=stage_index,
                n_trees_added=n_new,
                n_trees_total=total_trees,
                params={k: v for k, v in stage_params.items() if k != "n_estimators"},
                score=score,
                improved=improved,
                seconds=time.perf_counter() - started,
            )
        )

        if improved:
            result.best_stage = stage_index
            result.best_score = score
            result.booster = boosters[0]
            misses = 0
        else:
            misses += 1
            if misses >= patience:
                result.notes.append(
                    f"stopped after stage {stage_index}: {patience} consecutive "
                    "stages without improvement"
                )
                break

        LOGGER.info(
            "%s staged boosting, stage %d: +%d trees (%d total) -> %s = %.5f%s",
            module, stage_index, n_new, total_trees, objective_name, score,
            "  [best]" if improved else "",
        )

    if result.stages:
        result.baseline_score = result.stages[0].score
    return result


def default_schedule(n_stages: int = 3, trees_per_stage: int = 150) -> list[dict[str, Any]]:
    """A sensible three-stage schedule: coarse, refine, polish.

    The learning rate falls by roughly 3x per stage while capacity rises and
    regularisation tightens - the same shape as a step learning-rate decay in
    deep learning, for the same reason.
    """
    rates = [0.10, 0.03, 0.01]
    leaves = [15, 31, 63]
    lambdas = [0.1, 1.0, 5.0]
    return [
        {
            "n_estimators": trees_per_stage,
            "learning_rate": rates[i % len(rates)],
            "num_leaves": leaves[i % len(leaves)],
            "lambda_l2": lambdas[i % len(lambdas)],
        }
        for i in range(n_stages)
    ]


# =============================================================================
# Level 3b - LoRA
# =============================================================================
@dataclass(frozen=True)
class LoRAConfig:
    """Low-rank adapter settings.

    Attributes:
        r: Adapter rank. The whole parameter saving is governed by this - a
            ``d x d`` update becomes ``2 x d x r``. r=8 on d=192 trains 8.3% of
            the weights of the matrix it adapts.
        alpha: Scaling. The update is multiplied by ``alpha / r``, which keeps
            the effective step size roughly constant as r changes, so r can be
            tuned without re-tuning the learning rate.
        dropout: Applied to the adapter input during training.
        target_modules: Which linear layers to adapt. Attention projections are
            the conventional choice: they carry the task-specific interaction
            structure, while the feed-forward blocks carry more generic
            feature transformations worth keeping frozen.
    """

    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("query", "key", "value")

    @property
    def scaling(self) -> float:
        return self.alpha / self.r

    def parameters_for(self, in_features: int, out_features: int) -> int:
        """Trainable parameters an adapter adds to one linear layer."""
        return self.r * (in_features + out_features)

    def saving_vs_full(self, in_features: int, out_features: int) -> float:
        """Fraction of the full weight matrix that LoRA actually trains."""
        full = in_features * out_features
        return self.parameters_for(in_features, out_features) / full if full else float("nan")


def _torch():
    """Import torch with a message that names the fix."""
    try:
        import torch

        return torch
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "The Level-3 deep path needs PyTorch: pip install -r requirements.txt. "
            "The staged-boosting path (Level 3a) has no such dependency."
        ) from exc


def LoRALinear(base_layer: Any, config: LoRAConfig) -> Any:  # noqa: N802 - factory
    """Wrap a ``nn.Linear`` with a trainable low-rank update.

    The base weight is **frozen**. Only ``A`` (r x in) and ``B`` (out x r)
    receive gradients, and the forward pass computes

    .. math:: y = W_0 x + \\frac{\\alpha}{r} B A x

    ``B`` is initialised to zero so the adapter starts as an exact identity -
    the adapted model at step 0 is numerically the pretrained model, which
    means fine-tuning can never begin from a worse point than the backbone.

    Args:
        base_layer: A ``torch.nn.Linear`` to adapt.
        config: Adapter settings.

    Returns:
        A module with the same interface as ``base_layer``.
    """
    torch = _torch()
    import torch.nn as nn

    class _LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear, cfg: LoRAConfig) -> None:
            super().__init__()
            self.base = base
            self.cfg = cfg
            for parameter in self.base.parameters():
                parameter.requires_grad = False        # freeze the backbone

            self.lora_A = nn.Parameter(torch.zeros(cfg.r, base.in_features))
            self.lora_B = nn.Parameter(torch.zeros(base.out_features, cfg.r))
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
            # B stays zero -> the adapter is an exact identity at init.
            self.dropout = nn.Dropout(cfg.dropout)

        def forward(self, x):  # noqa: ANN001, ANN201
            update = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
            return self.base(x) + update * self.cfg.scaling

        def extra_repr(self) -> str:
            return (
                f"r={self.cfg.r}, alpha={self.cfg.alpha}, "
                f"trainable={self.lora_A.numel() + self.lora_B.numel()}, "
                f"frozen={sum(p.numel() for p in self.base.parameters())}"
            )

    return _LoRALinear(base_layer, config)


def attach_lora(model: Any, config: LoRAConfig | None = None) -> Any:
    """Freeze a model and attach LoRA adapters to its target modules.

    Every parameter is frozen first, then adapters are inserted - so anything
    trainable afterwards is trainable *by intent*. That ordering is what makes
    :func:`lora_parameter_report` a meaningful audit rather than a description.

    Args:
        model: A ``torch.nn.Module``.
        config: Adapter settings.

    Returns:
        The same model, modified in place.
    """
    torch = _torch()
    import torch.nn as nn

    config = config or LoRAConfig()

    for parameter in model.parameters():
        parameter.requires_grad = False

    replaced = 0
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and any(
                token in name for token in config.target_modules
            ):
                setattr(module, name, LoRALinear(child, config))
                replaced += 1

    # The task head must train - it is new, and has no pretrained weights.
    if hasattr(model, "head"):
        for parameter in model.head.parameters():
            parameter.requires_grad = True

    LOGGER.info("Attached %d LoRA adapter(s) at rank %d", replaced, config.r)
    return model


def lora_parameter_report(model: Any) -> pd.DataFrame:
    """Count trainable versus frozen parameters - the PEFT evidence table.

    This is the table that makes the LoRA claim checkable. A figure well under
    1% trainable is what "parameter-efficient" means, and it is the difference
    between a module with 89 positives being able to adapt a shared backbone or
    not.
    """
    _torch()
    rows: list[dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        rows.append(
            {
                "parameter": name,
                "shape": tuple(parameter.shape),
                "count": int(parameter.numel()),
                "trainable": bool(parameter.requires_grad),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    total = int(frame["count"].sum())
    trainable = int(frame.loc[frame["trainable"], "count"].sum())
    frame.attrs["total_parameters"] = total
    frame.attrs["trainable_parameters"] = trainable
    frame.attrs["trainable_pct"] = 100.0 * trainable / total if total else float("nan")
    return frame


# =============================================================================
# Level 3b - the FT-Transformer backbone
# =============================================================================
@dataclass(frozen=True)
class FTTransformerConfig:
    """A deliberately small tabular transformer, sized for the Colab free tier.

    d_token 192 with 3 blocks is roughly 1.2M parameters - seconds per epoch on
    a T4 over the ~190,000 pooled rows available here, and well under 1 GB of
    VRAM. Scaling it up would be easy and pointless: the constraint in this
    project is labelled rows, not model capacity.
    """

    n_features: int
    d_token: int = 192
    n_blocks: int = 3
    n_heads: int = 8
    dropout: float = 0.1
    d_out: int = 1


def build_ft_transformer(config: FTTransformerConfig) -> Any:
    """Construct a feature-tokeniser transformer for tabular data.

    Each numeric feature is projected to its own token (a learned weight and
    bias per feature), a ``[CLS]`` token is prepended, and standard transformer
    blocks attend across features. The ``[CLS]`` representation feeds the head.

    Why this architecture rather than an MLP: the attention matrix models
    *pairwise feature interactions* explicitly, which is the representation we
    want to share across modules. An MLP's hidden layers entangle them.

    The linear projections inside attention are named ``query``/``key``/
    ``value`` deliberately - :func:`attach_lora` targets them by name.
    """
    torch = _torch()
    import torch.nn as nn

    class FeatureTokenizer(nn.Module):
        """One learned token per feature, plus a [CLS] token."""

        def __init__(self, n_features: int, d_token: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.empty(n_features, d_token))
            self.bias = nn.Parameter(torch.empty(n_features, d_token))
            self.cls = nn.Parameter(torch.empty(1, 1, d_token))
            for tensor in (self.weight, self.bias, self.cls):
                nn.init.uniform_(tensor, -(d_token**-0.5), d_token**-0.5)

        def forward(self, x):  # noqa: ANN001, ANN201
            tokens = x.unsqueeze(-1) * self.weight + self.bias
            cls = self.cls.expand(x.shape[0], -1, -1)
            return torch.cat([cls, tokens], dim=1)

    class Attention(nn.Module):
        def __init__(self, d_token: int, n_heads: int, dropout: float) -> None:
            super().__init__()
            self.n_heads = n_heads
            self.d_head = d_token // n_heads
            # Names matter: attach_lora targets "query"/"key"/"value".
            self.query = nn.Linear(d_token, d_token)
            self.key = nn.Linear(d_token, d_token)
            self.value = nn.Linear(d_token, d_token)
            self.proj = nn.Linear(d_token, d_token)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):  # noqa: ANN001, ANN201
            batch, seq, _ = x.shape
            def split(t):
                return t.view(batch, seq, self.n_heads, self.d_head).transpose(1, 2)

            q, k, v = split(self.query(x)), split(self.key(x)), split(self.value(x))
            scores = (q @ k.transpose(-2, -1)) / (self.d_head**0.5)
            attention = self.dropout(scores.softmax(dim=-1))
            out = (attention @ v).transpose(1, 2).reshape(batch, seq, -1)
            return self.proj(out)

    class Block(nn.Module):
        def __init__(self, d_token: int, n_heads: int, dropout: float) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(d_token)
            self.attention = Attention(d_token, n_heads, dropout)
            self.norm2 = nn.LayerNorm(d_token)
            self.ffn = nn.Sequential(
                nn.Linear(d_token, d_token * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_token * 2, d_token),
            )

        def forward(self, x):  # noqa: ANN001, ANN201
            x = x + self.attention(self.norm1(x))
            return x + self.ffn(self.norm2(x))

    class FTTransformer(nn.Module):
        def __init__(self, cfg: FTTransformerConfig) -> None:
            super().__init__()
            self.cfg = cfg
            self.tokenizer = FeatureTokenizer(cfg.n_features, cfg.d_token)
            self.blocks = nn.ModuleList(
                [Block(cfg.d_token, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_blocks)]
            )
            self.norm = nn.LayerNorm(cfg.d_token)
            self.head = nn.Linear(cfg.d_token, cfg.d_out)

        def encode(self, x):  # noqa: ANN001, ANN201
            """Return the [CLS] representation - the transferable part."""
            tokens = self.tokenizer(x)
            for block in self.blocks:
                tokens = block(tokens)
            return self.norm(tokens)[:, 0]

        def forward(self, x):  # noqa: ANN001, ANN201
            return self.head(self.encode(x))

    return FTTransformer(config)


def pretrain_masked_features(
    model: Any,
    X: np.ndarray,
    *,
    epochs: int = 5,
    batch_size: int = 512,
    mask_ratio: float = 0.15,
    lr: float = 1e-3,
    seed: int = 42,
    device: str | None = None,
    log_every: int = 1,
) -> pd.DataFrame:
    """Self-supervised pre-training: reconstruct masked feature values.

    **No labels are used.** A random ``mask_ratio`` of each row's features is
    zeroed and the model must reconstruct the original values. The objective is
    the tabular analogue of masked language modelling, and it is what makes the
    backbone transferable: it learns the joint structure of the features, which
    is shared across modules, rather than any one module's target.

    This is the step that makes the LoRA story coherent. Without a pretrained
    backbone there would be nothing to adapt, and "LoRA" would just be a
    randomly-initialised small model wearing a PEFT label.

    Args:
        model: An FT-Transformer from :func:`build_ft_transformer`.
        X: Standardised numeric matrix, ``(n_rows, n_features)``.
        epochs, batch_size, mask_ratio, lr: Training settings.
        seed: Seed for the masking pattern.
        device: ``cuda`` or ``cpu``. Auto-detected when omitted.
        log_every: Epoch logging interval.

    Returns:
        A per-epoch loss table.
    """
    torch = _torch()
    import torch.nn as nn

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator().manual_seed(seed)
    model = model.to(device)

    # A reconstruction head, discarded after pre-training - only the trunk
    # transfers, which is the point.
    reconstruct = nn.Linear(model.cfg.d_token, model.cfg.n_features).to(device)
    optimiser = torch.optim.AdamW(
        list(model.parameters()) + list(reconstruct.parameters()), lr=lr
    )
    loss_fn = nn.MSELoss()
    tensor = torch.tensor(np.asarray(X, dtype="float32"), device=device)

    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(len(tensor), generator=generator).to(device)
        total, batches = 0.0, 0
        for start in range(0, len(tensor), batch_size):
            batch = tensor[permutation[start : start + batch_size]]
            mask = (torch.rand(batch.shape, device=device) < mask_ratio).float()
            corrupted = batch * (1 - mask)

            predicted = reconstruct(model.encode(corrupted))
            # Score ONLY the masked positions - scoring the visible ones would
            # reward the identity function.
            loss = loss_fn(predicted * mask, batch * mask)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss.item())
            batches += 1

        history.append({"epoch": epoch, "masked_mse": total / max(batches, 1)})
        if epoch % log_every == 0:
            LOGGER.info("pretrain epoch %d/%d: masked MSE %.5f", epoch, epochs, history[-1]["masked_mse"])

    return pd.DataFrame(history)


# =============================================================================
# Level 3c - layer-wise unfreezing and discriminative learning rates
# =============================================================================
def layerwise_unfreeze_schedule(
    n_layers: int, *, epochs: int, warmup_epochs: int = 1
) -> list[dict[str, Any]]:
    """Gradual unfreezing: train the head, then release layers top-down.

    Introduced by Howard & Ruder (2018) for ULMFiT. The reasoning transfers
    directly to this project: a randomly-initialised head produces large,
    noisy gradients in its first epochs, and if the whole backbone is trainable
    those gradients destroy the pretrained representation - "catastrophic
    forgetting". Training the head alone first lets it reach a sensible region
    before any backbone weight moves.

    Layers are then released from the **top down** because later layers encode
    the most task-specific structure and the earliest layers the most generic -
    so the ones released first are the ones most worth adapting.

    Reference: Howard J, Ruder S (2018), "Universal Language Model Fine-tuning
    for Text Classification", ACL. https://arxiv.org/abs/1801.06146

    Args:
        n_layers: Number of backbone blocks.
        epochs: Total fine-tuning epochs.
        warmup_epochs: Epochs with the head only.

    Returns:
        One dict per epoch: ``epoch``, ``trainable_layers`` (block indices),
        ``head_trainable``, ``stage``.
    """
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    warmup = max(0, min(warmup_epochs, epochs))

    schedule: list[dict[str, Any]] = []
    for epoch in range(1, warmup + 1):
        schedule.append(
            {"epoch": epoch, "trainable_layers": [], "head_trainable": True, "stage": "head only"}
        )

    remaining = epochs - warmup
    if remaining <= 0:
        return schedule

    # Release one layer at a time, top-down, spread over the remaining epochs.
    per_layer = max(1, remaining // max(n_layers, 1))
    released: list[int] = []
    for offset in range(remaining):
        epoch = warmup + offset + 1
        target = min(n_layers, offset // per_layer + 1)
        released = list(range(n_layers - target, n_layers))
        schedule.append(
            {
                "epoch": epoch,
                "trainable_layers": released,
                "head_trainable": True,
                "stage": ("all layers" if len(released) == n_layers
                          else f"top {len(released)} layer(s)"),
            }
        )
    return schedule


def discriminative_learning_rates(
    n_layers: int, *, base_lr: float = 1e-3, decay: float = 2.6
) -> dict[str, float]:
    """Per-layer learning rates, decaying toward the input.

    .. math:: \\eta_{l-1} = \\eta_l / \\xi

    The head trains at ``base_lr``; each layer below trains ``decay`` times
    more slowly. The justification is the same as for gradual unfreezing:
    earlier layers hold generic feature structure that is already correct and
    should barely move, while later layers hold task-specific structure that
    should adapt.

    ``decay = 2.6`` is the ULMFiT value, arrived at empirically there; it is
    kept here rather than re-derived, and cited rather than presented as ours.

    Args:
        n_layers: Number of backbone blocks.
        base_lr: The head's learning rate.
        decay: Per-layer divisor.

    Returns:
        ``{group_name: learning_rate}`` for ``head``, ``layer_{i}``, ``tokenizer``.
    """
    if decay <= 1:
        raise ValueError("decay must exceed 1, or the rates do not discriminate")

    rates = {"head": base_lr}
    for index in range(n_layers - 1, -1, -1):
        steps = n_layers - index
        rates[f"layer_{index}"] = base_lr / (decay**steps)
    rates["tokenizer"] = base_lr / (decay ** (n_layers + 1))
    return rates


def build_param_groups(model: Any, rates: dict[str, float]) -> list[dict[str, Any]]:
    """Map a rate dictionary onto optimiser parameter groups.

    Only parameters with ``requires_grad`` are included, so a frozen backbone
    or a LoRA-adapted model produces exactly the groups it should.
    """
    _torch()
    groups: list[dict[str, Any]] = []

    for name, rate in rates.items():
        if name == "head":
            params = [p for n, p in model.named_parameters() if n.startswith("head") and p.requires_grad]
        elif name == "tokenizer":
            params = [p for n, p in model.named_parameters() if n.startswith("tokenizer") and p.requires_grad]
        else:
            index = name.split("_")[-1]
            prefix = f"blocks.{index}."
            params = [p for n, p in model.named_parameters() if n.startswith(prefix) and p.requires_grad]
        if params:
            groups.append({"params": params, "lr": rate, "name": name})

    return groups
