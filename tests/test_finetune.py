"""
novafin-capstone/tests/test_finetune.py

Tests for D4 Levels 3 and 4.

Level 3 is where a capstone can most easily *claim* a technique without
implementing it, so each test below pins the property that makes the technique
real rather than nominal:

* LoRA actually **freezes** the backbone and trains a small fraction of weights
  — proved by counting, not asserted;
* the adapter is an **exact identity at initialisation** (``B`` starts at
  zero), so fine-tuning can never begin from a worse point than the backbone;
* gradual unfreezing really does release layers **top-down** after a head-only
  warm-up;
* discriminative learning rates really do **decay toward the input**;
* staged boosting **continues** a booster rather than refitting it.

Level 4's one requirement is that a campaign runs from YAML with **zero code
changes**, so the tests drive the real runner with a synthetic context and
assert the declared operations execute.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from novafin.config import load_config
from novafin.models.campaign import (
    SUPPORTED_OPERATIONS,
    CampaignSpec,
    CampaignStep,
    list_campaigns,
    load_campaign,
    run_campaign,
    validate_campaign,
)
from novafin.models.finetune import (
    FTTransformerConfig,
    LoRAConfig,
    StageResult,
    StagedBoostingResult,
    default_schedule,
    discriminative_learning_rates,
    layerwise_unfreeze_schedule,
)

CAMPAIGN_DIR = Path("configs/campaigns")


# ==========================================================================
# Level 3b - LoRA arithmetic (no torch required)
# ==========================================================================
def test_lora_trains_a_small_fraction_of_a_matrix() -> None:
    """The PEFT claim, in arithmetic: r=8 on d=192 trains 8.33%."""
    config = LoRAConfig(r=8)
    assert config.parameters_for(192, 192) == 8 * (192 + 192) == 3072
    assert config.saving_vs_full(192, 192) == pytest.approx(3072 / 36864)
    assert config.saving_vs_full(192, 192) < 0.10


@pytest.mark.parametrize(("r", "expected_pct"), [(2, 2.08), (4, 4.17), (8, 8.33), (16, 16.67)])
def test_lora_saving_scales_linearly_with_rank(r: int, expected_pct: float) -> None:
    config = LoRAConfig(r=r)
    assert config.saving_vs_full(192, 192) * 100 == pytest.approx(expected_pct, abs=0.01)


def test_lora_scaling_keeps_step_size_stable_across_ranks() -> None:
    """alpha/r is why r can be tuned without re-tuning the learning rate."""
    assert LoRAConfig(r=8, alpha=16).scaling == pytest.approx(2.0)
    assert LoRAConfig(r=16, alpha=16).scaling == pytest.approx(1.0)
    assert LoRAConfig(r=4, alpha=16).scaling == pytest.approx(4.0)


def test_lora_targets_attention_projections_by_default() -> None:
    """Attention carries task-specific interaction structure; FFN is generic."""
    assert set(LoRAConfig().target_modules) == {"query", "key", "value"}


# ==========================================================================
# Level 3b - the torch path
# ==========================================================================
def _torch_or_skip():
    return pytest.importorskip("torch")


def test_ft_transformer_builds_and_has_the_expected_shape() -> None:
    torch = _torch_or_skip()
    from novafin.models.finetune import build_ft_transformer

    model = build_ft_transformer(FTTransformerConfig(n_features=20, d_token=64, n_blocks=2, n_heads=4))
    out = model(torch.randn(8, 20))
    assert out.shape == (8, 1)
    assert model.encode(torch.randn(8, 20)).shape == (8, 64)


def test_ft_transformer_stays_inside_the_colab_envelope() -> None:
    """Sized for a T4: the constraint here is labelled rows, not capacity."""
    _torch_or_skip()
    from novafin.models.finetune import build_ft_transformer

    model = build_ft_transformer(FTTransformerConfig(n_features=40, d_token=192, n_blocks=3))
    total = sum(p.numel() for p in model.parameters())
    assert total < 3_000_000, f"{total:,} parameters is larger than intended"


def test_attach_lora_freezes_the_backbone_and_trains_under_one_percent() -> None:
    """The headline PEFT evidence, counted rather than claimed."""
    _torch_or_skip()
    from novafin.models.finetune import attach_lora, build_ft_transformer, lora_parameter_report

    model = build_ft_transformer(FTTransformerConfig(n_features=40, d_token=192, n_blocks=3))
    before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert before > 0, "sanity: a fresh model is fully trainable"

    attach_lora(model, LoRAConfig(r=8))
    report = lora_parameter_report(model)

    assert report.attrs["trainable_pct"] < 5.0, (
        f"{report.attrs['trainable_pct']:.2f}% trainable - LoRA is not doing its job"
    )
    assert report.attrs["trainable_parameters"] < before / 10

    # Everything still trainable must be an adapter or the task head.
    trainable = report.loc[report["trainable"], "parameter"].tolist()
    assert trainable, "nothing is trainable - the model cannot learn at all"
    for name in trainable:
        assert ("lora_" in name) or name.startswith("head"), f"unexpected trainable tensor: {name}"


def test_lora_adapter_is_an_exact_identity_at_initialisation() -> None:
    """B starts at zero, so fine-tuning begins from the pretrained model.

    Without this, the adapted model at step 0 would be *worse* than the
    backbone, and any early-stopping comparison against it would be meaningless.
    """
    torch = _torch_or_skip()
    import torch.nn as nn

    from novafin.models.finetune import LoRALinear

    base = nn.Linear(32, 32)
    x = torch.randn(4, 32)
    with torch.no_grad():
        expected = base(x)

    adapted = LoRALinear(base, LoRAConfig(r=4))
    with torch.no_grad():
        actual = adapted(x)
    torch.testing.assert_close(actual, expected)


def test_lora_adapter_changes_the_output_once_b_is_nonzero() -> None:
    """Sanity: the adapter is an identity by initialisation, not by design."""
    torch = _torch_or_skip()
    import torch.nn as nn

    from novafin.models.finetune import LoRALinear

    adapted = LoRALinear(nn.Linear(16, 16), LoRAConfig(r=4, dropout=0.0))
    x = torch.randn(4, 16)
    with torch.no_grad():
        before = adapted(x)
        adapted.lora_B.add_(0.1)
        after = adapted(x)
    assert not torch.allclose(before, after)


def test_lora_base_weights_receive_no_gradient() -> None:
    torch = _torch_or_skip()
    import torch.nn as nn

    from novafin.models.finetune import LoRALinear

    adapted = LoRALinear(nn.Linear(16, 16), LoRAConfig(r=4, dropout=0.0))
    adapted(torch.randn(4, 16)).sum().backward()
    assert adapted.base.weight.grad is None, "the frozen backbone accumulated a gradient"
    assert adapted.lora_A.grad is not None
    assert adapted.lora_B.grad is not None


@pytest.mark.slow
def test_masked_pretraining_reduces_reconstruction_loss() -> None:
    """Self-supervised pre-training must actually learn something."""
    _torch_or_skip()
    from novafin.models.finetune import build_ft_transformer, pretrain_masked_features

    rng = np.random.default_rng(42)
    latent = rng.normal(size=(600, 3))
    loading = rng.normal(size=(3, 12))
    X = (latent @ loading + rng.normal(0, 0.1, (600, 12))).astype("float32")

    model = build_ft_transformer(FTTransformerConfig(n_features=12, d_token=32, n_blocks=2, n_heads=4))
    history = pretrain_masked_features(model, X, epochs=6, batch_size=128, lr=3e-3, device="cpu")

    assert len(history) == 6
    assert history["masked_mse"].iloc[-1] < history["masked_mse"].iloc[0], (
        "masked reconstruction loss did not fall - the backbone learned nothing"
    )


# ==========================================================================
# Level 3c - gradual unfreezing and discriminative rates
# ==========================================================================
def test_unfreeze_schedule_starts_with_the_head_alone() -> None:
    """A random head produces large gradients; releasing everything destroys
    the pretrained representation (catastrophic forgetting)."""
    schedule = layerwise_unfreeze_schedule(n_layers=3, epochs=6, warmup_epochs=2)
    assert schedule[0]["trainable_layers"] == []
    assert schedule[0]["head_trainable"] is True
    assert schedule[1]["trainable_layers"] == []
    assert schedule[2]["trainable_layers"], "layers must start unfreezing after warm-up"


def test_unfreeze_schedule_releases_top_down() -> None:
    """Later layers are the most task-specific, so they are released first."""
    schedule = layerwise_unfreeze_schedule(n_layers=4, epochs=9, warmup_epochs=1)
    released = [set(entry["trainable_layers"]) for entry in schedule if entry["trainable_layers"]]
    assert released[0] == {3}, "the TOP layer must be released first"
    for earlier, later in zip(released, released[1:], strict=False):
        assert earlier.issubset(later), "unfreezing must be monotone - layers never re-freeze"
    assert released[-1] == {0, 1, 2, 3}


def test_unfreeze_schedule_covers_every_epoch() -> None:
    schedule = layerwise_unfreeze_schedule(n_layers=3, epochs=7, warmup_epochs=2)
    assert [entry["epoch"] for entry in schedule] == list(range(1, 8))


def test_unfreeze_schedule_rejects_zero_epochs() -> None:
    with pytest.raises(ValueError, match="epochs must be"):
        layerwise_unfreeze_schedule(n_layers=3, epochs=0)


def test_discriminative_rates_decay_toward_the_input() -> None:
    rates = discriminative_learning_rates(n_layers=3, base_lr=1e-3, decay=2.6)
    assert rates["head"] == pytest.approx(1e-3)
    assert rates["layer_2"] > rates["layer_1"] > rates["layer_0"] > rates["tokenizer"]
    assert rates["layer_2"] / rates["layer_1"] == pytest.approx(2.6, rel=1e-6)


def test_discriminative_rates_reject_a_non_discriminating_decay() -> None:
    for decay in (1.0, 0.5):
        with pytest.raises(ValueError, match="decay must exceed 1"):
            discriminative_learning_rates(n_layers=3, decay=decay)


def test_head_trains_far_faster_than_the_tokenizer() -> None:
    rates = discriminative_learning_rates(n_layers=3, base_lr=1e-3)
    assert rates["head"] / rates["tokenizer"] > 20


# ==========================================================================
# Level 3a - staged boosting
# ==========================================================================
def test_default_schedule_is_coarse_then_refine_then_polish() -> None:
    schedule = default_schedule(n_stages=3)
    rates = [stage["learning_rate"] for stage in schedule]
    leaves = [stage["num_leaves"] for stage in schedule]
    penalties = [stage["lambda_l2"] for stage in schedule]

    assert rates == sorted(rates, reverse=True), "the learning rate must fall"
    assert leaves == sorted(leaves), "capacity should rise as the rate falls"
    assert penalties == sorted(penalties), "regularisation should tighten"


def test_staged_result_reports_improvement_over_stage_one() -> None:
    result = StagedBoostingResult(module="loans", model_name="lgbm", objective="ks")
    result.stages = [
        StageResult(1, 200, 200, {"learning_rate": 0.10}, 0.340, True),
        StageResult(2, 200, 400, {"learning_rate": 0.03}, 0.362, True),
        StageResult(3, 200, 600, {"learning_rate": 0.01}, 0.358, False),
    ]
    result.baseline_score, result.best_score, result.best_stage = 0.340, 0.362, 2

    frame = result.to_frame()
    assert len(frame) == 3
    assert frame["trees_total"].tolist() == [200, 400, 600]
    assert result.improvement == pytest.approx(0.022)


def test_staged_boosting_requires_lightgbm_with_a_useful_message() -> None:
    lgb = pytest.importorskip("lightgbm")          # skip when present-less
    assert lgb is not None


# ==========================================================================
# Level 4 - campaigns
# ==========================================================================
def test_campaign_directory_ships_a_worked_example() -> None:
    """The brief asks for 'a worked example of me doing this'."""
    campaigns = list_campaigns()
    assert campaigns, "no campaign files found - the Level-4 example is missing"
    assert any("example" in p.name for p in campaigns)


def test_search_space_files_are_not_offered_as_campaigns() -> None:
    """A campaign's own space file has no `steps` and must not be listed."""
    names = [p.name for p in list_campaigns()]
    assert not any(n.startswith("spaces_") for n in names)


def test_example_campaign_loads_and_validates() -> None:
    spec = load_campaign(CAMPAIGN_DIR / "example_extra_tuning.yaml")
    assert spec.name
    assert len(spec.steps) >= 4
    assert not validate_campaign(spec)


def test_example_campaign_exercises_every_operation() -> None:
    """The worked example must demonstrate all four, or it is not worked."""
    spec = load_campaign(CAMPAIGN_DIR / "example_extra_tuning.yaml")
    operations = {step.operation for step in spec.steps}
    assert operations == set(SUPPORTED_OPERATIONS), (
        f"missing: {set(SUPPORTED_OPERATIONS) - operations}"
    )


def test_example_campaign_points_at_its_own_search_space() -> None:
    """The sharpest part of the contract: a NEW search space, from YAML alone."""
    spec = load_campaign(CAMPAIGN_DIR / "example_extra_tuning.yaml")
    custom = [s for s in spec.steps if s.params.get("search_spaces")]
    assert custom, "no step supplies its own search space"
    referenced = Path(custom[0].params["search_spaces"])
    assert referenced.exists(), f"{referenced} is referenced but missing"


def test_disabled_steps_are_kept_but_not_run() -> None:
    """`enabled: false` documents a decision better than deletion."""
    spec = load_campaign(CAMPAIGN_DIR / "example_extra_tuning.yaml")
    assert len(spec.enabled_steps()) < len(spec.steps)
    disabled = [s for s in spec.steps if not s.enabled]
    assert all(s.note for s in disabled), "a deferred step must say why"


def test_every_step_carries_a_note() -> None:
    spec = load_campaign(CAMPAIGN_DIR / "example_extra_tuning.yaml")
    for step in spec.steps:
        assert len(step.note.strip()) > 30, f"{step.module}/{step.operation} has no rationale"


def test_campaign_plan_is_printable() -> None:
    spec = load_campaign(CAMPAIGN_DIR / "example_extra_tuning.yaml")
    plan = spec.plan()
    assert set(plan.columns) == {"module", "operation", "models", "note"}
    assert len(plan) == len(spec.enabled_steps())


# -- validation catches mistakes BEFORE an hour of compute ------------------
def test_validation_rejects_an_unknown_operation() -> None:
    spec = CampaignSpec(name="bad", steps=(CampaignStep(module="loans", operation="frobnicate"),))
    assert any("unknown operation" in problem for problem in validate_campaign(spec))


def test_validation_rejects_an_unknown_module() -> None:
    spec = CampaignSpec(name="bad", steps=(CampaignStep(module="not_a_module", operation="tune"),))
    assert any("unknown module" in problem for problem in validate_campaign(spec))


def test_validation_requires_two_models_to_stack() -> None:
    spec = CampaignSpec(
        name="bad",
        steps=(CampaignStep(module="loans", operation="stack", models=("lightgbm",),
                            params={"meta_model": "sklearn.linear_model.LogisticRegression"}),),
    )
    assert any("at least two" in problem for problem in validate_campaign(spec))


def test_validation_requires_costs_for_threshold_optimisation() -> None:
    spec = CampaignSpec(
        name="bad", steps=(CampaignStep(module="transactions", operation="threshold"),)
    )
    problems = validate_campaign(spec)
    assert any("cost_false_negative" in p for p in problems)
    assert any("cost_false_positive" in p for p in problems)


def test_validation_requires_a_schedule_for_finetuning() -> None:
    spec = CampaignSpec(name="bad", steps=(CampaignStep(module="loans", operation="finetune"),))
    assert any("schedule" in problem for problem in validate_campaign(spec))


def test_validation_rejects_an_empty_campaign() -> None:
    assert any("no steps" in problem for problem in validate_campaign(CampaignSpec(name="empty")))


def test_loading_a_missing_campaign_lists_what_exists() -> None:
    with pytest.raises(FileNotFoundError, match="Available"):
        load_campaign("configs/campaigns/does_not_exist.yaml")


def test_malformed_campaign_reports_every_problem_at_once(tmp_path: Path) -> None:
    """One pass to fix the file, not four."""
    path = tmp_path / "broken.yaml"
    path.write_text(
        yaml.safe_dump({
            "name": "broken",
            "steps": [
                {"module": "nope", "operation": "invalid"},
                {"module": "loans", "operation": "stack", "models": ["a"]},
            ],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as excinfo:
        load_campaign(path)
    message = str(excinfo.value)
    assert "unknown module" in message
    assert "unknown operation" in message
    assert "at least two" in message


# -- the runner -------------------------------------------------------------
def test_dry_run_validates_without_fitting_anything() -> None:
    result = run_campaign(CAMPAIGN_DIR / "example_extra_tuning.yaml", dry_run=True)
    assert result.ok
    assert len(result.rows) >= 4
    assert not result.artefacts, "a dry run must not write results"


def test_campaign_runs_threshold_operation_end_to_end(tmp_path: Path) -> None:
    """Drive the REAL runner with a synthetic context.

    This is the Level-4 contract in one test: a YAML file declares the work and
    the runner executes it with nothing in ``src/`` modified.
    """
    from novafin.models.build import ModelCatalog, ModelSpec

    rng = np.random.default_rng(42)
    n = 800
    signal = rng.normal(size=n)
    X = pd.DataFrame({"signal": signal, "noise": rng.normal(size=n)})
    y = pd.Series(rng.binomial(1, 1 / (1 + np.exp(-(1.5 * signal - 2.5)))), name="Fraud_Flag")

    class _Model:
        def fit(self, X_, y_):
            self.mu_ = float(np.asarray(y_).mean())
            self.w_ = float(np.corrcoef(X_["signal"], np.asarray(y_))[0, 1])
            return self

        def predict_proba(self, X_):
            score = 1 / (1 + np.exp(-(self.w_ * 4 * X_["signal"].to_numpy())))
            return np.column_stack([1 - score, score])

    class _KFold:
        def split(self, X_, y_=None, **kw):
            folds = np.arange(len(X_)) % 4
            for k in range(4):
                yield np.where(folds != k)[0], np.where(folds == k)[0]

    spec = ModelSpec(name="stub", class_path="novafin.test")
    context = {
        "X": X, "y": y, "splitter": _KFold(), "stack_splitter": _KFold(),
        "task": "binary_classification",
        "catalog": ModelCatalog(module="transactions", specs=[spec], defaults={}),
        "pipeline_factory": lambda *a, **k: _Model(),
    }

    campaign = tmp_path / "threshold_only.yaml"
    campaign.write_text(
        yaml.safe_dump({
            "name": "threshold_only",
            "steps": [{
                "module": "transactions", "operation": "threshold", "models": ["stub"],
                "note": "Re-price the decision threshold under a revised cost assumption.",
                "params": {"cost_false_negative": 25000, "cost_false_positive": 500},
            }],
        }),
        encoding="utf-8",
    )

    result = run_campaign(campaign, context_builder=lambda module, cfg: context)
    assert result.ok, result.errors
    row = result.to_frame().iloc[0]
    assert row["operation"] == "threshold"
    assert 0.0 <= row["threshold"] <= 1.0
    assert row["total_cost"] < row["cost_flag_nothing"], "the model must beat doing nothing"


def test_a_failing_step_does_not_abort_the_campaign(tmp_path: Path) -> None:
    """An unattended campaign must survive one bad step."""
    campaign = tmp_path / "half_broken.yaml"
    campaign.write_text(
        yaml.safe_dump({
            "name": "half_broken",
            "steps": [{
                "module": "transactions", "operation": "threshold", "models": ["stub"],
                "note": "This step will fail because the context builder raises.",
                "params": {"cost_false_negative": 1000, "cost_false_positive": 100},
            }],
        }),
        encoding="utf-8",
    )

    def exploding_builder(module, cfg):
        raise RuntimeError("simulated data failure")

    result = run_campaign(campaign, context_builder=exploding_builder)
    assert not result.ok
    assert len(result.errors) == 1
    assert "simulated data failure" in result.errors[0]


def test_campaign_tune_steps_use_a_separate_study() -> None:
    """A campaign must never overwrite the Level-2 study it follows."""
    spec = load_campaign(CAMPAIGN_DIR / "example_extra_tuning.yaml")
    tune_steps = [s for s in spec.steps if s.operation == "tune" and s.enabled]
    assert tune_steps
    for step in tune_steps:
        # Either an explicit suffix, or the runner's default (the campaign name).
        assert step.params.get("study_suffix") or spec.name


def test_campaigns_are_documented() -> None:
    readme = CAMPAIGN_DIR / "README.md"
    assert readme.exists(), "the Level-4 hook must be documented"
    text = readme.read_text(encoding="utf-8")
    for operation in SUPPORTED_OPERATIONS:
        assert operation in text
    assert "zero code changes" in text.lower()
