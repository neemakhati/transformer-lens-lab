import pytest
import torch

from tlab.model import load_gpt2_small, tokenize
from tlab.patching import patch_layer, patching_sweep, run_with_cache


@pytest.fixture(scope="module")
def bundle():
    return load_gpt2_small(device="cpu")


def test_mismatched_lengths_raise_clear_error(bundle) -> None:
    clean_ids = tokenize(bundle, "The capital of France is")
    corrupted_ids = tokenize(bundle, "The capital of the United Kingdom is")
    with pytest.raises(ValueError, match="same shape"):
        patching_sweep(bundle, clean_ids, corrupted_ids, position=0, target_token=" Paris")


def test_patching_own_run_is_a_no_op(bundle) -> None:
    """Patching a run with ITS OWN cached activations must reproduce that
    run's real output exactly -- the strongest possible sanity check that
    patch_layer's hook correctly overwrites (not corrupts) the residual
    stream. If this fails, the patching mechanism itself is broken."""
    input_ids = tokenize(bundle, "The capital of France is")
    last_pos = input_ids.shape[1] - 1

    real_logits, cache = run_with_cache(bundle, input_ids)

    for layer in range(bundle.n_layers):
        patched_logits = patch_layer(bundle, cache.resid_post, input_ids, layer, last_pos)
        assert torch.allclose(
            patched_logits[0, last_pos], real_logits[0, last_pos], atol=1e-4
        ), f"self-patching layer {layer} should be a no-op"


def test_recovery_bounds_at_extremes(bundle) -> None:
    """Early layers (0) should show near-zero recovery; the final layer (11)
    should show ~100% recovery, since patching the last layer's residual
    stream is patching in everything the clean run ever computed."""
    clean_ids = tokenize(bundle, "The capital of France is")
    corrupted_ids = tokenize(bundle, "The capital of Germany is")
    last_pos = clean_ids.shape[1] - 1

    results = patching_sweep(
        bundle, clean_ids, corrupted_ids, position=last_pos, target_token=" Paris"
    )

    assert results[0].recovery < 0.1
    assert results[-1].recovery > 0.9


def test_recovery_increases_toward_later_layers(bundle) -> None:
    clean_ids = tokenize(bundle, "The capital of France is")
    corrupted_ids = tokenize(bundle, "The capital of Germany is")
    last_pos = clean_ids.shape[1] - 1

    results = patching_sweep(
        bundle, clean_ids, corrupted_ids, position=last_pos, target_token=" Paris"
    )
    early_avg = sum(r.recovery for r in results[:6]) / 6
    late_avg = sum(r.recovery for r in results[6:]) / 6
    assert late_avg > early_avg
