import pytest
import torch

from tlab.hooks import capture_activations
from tlab.model import load_gpt2_small, tokenize


@pytest.fixture(scope="module")
def bundle():
    return load_gpt2_small(device="cpu")


def test_captures_all_layers(bundle) -> None:
    input_ids = tokenize(bundle, "The capital of France is")
    with capture_activations(bundle) as cache:
        bundle.model(input_ids)

    assert sorted(cache.resid_post.keys()) == list(range(bundle.n_layers))
    assert sorted(cache.attn_patterns.keys()) == list(range(bundle.n_layers))


def test_resid_post_shape(bundle) -> None:
    input_ids = tokenize(bundle, "The capital of France is")
    seq_len = input_ids.shape[1]
    with capture_activations(bundle) as cache:
        bundle.model(input_ids)

    for layer_resid in cache.resid_post.values():
        assert layer_resid.shape == (1, seq_len, bundle.d_model)


def test_attention_rows_sum_to_one(bundle) -> None:
    input_ids = tokenize(bundle, "The capital of France is")
    with capture_activations(bundle) as cache:
        bundle.model(input_ids)

    pattern = cache.attn_patterns[0]  # (batch, heads, seq, seq)
    row_sums = pattern.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_causal_mask_blocks_future_tokens(bundle) -> None:
    input_ids = tokenize(bundle, "The capital of France is")
    with capture_activations(bundle) as cache:
        bundle.model(input_ids)

    pattern = cache.attn_patterns[0]  # (batch, heads, seq, seq)
    # token 0 has nothing before it, so it must attend entirely to itself
    assert torch.allclose(pattern[0, :, 0, 0], torch.ones(bundle.n_heads))
    assert torch.allclose(pattern[0, :, 0, 1:], torch.zeros(bundle.n_heads, pattern.shape[-1] - 1))


def test_hooks_are_removed_after_context(bundle) -> None:
    input_ids = tokenize(bundle, "hello")
    with capture_activations(bundle):
        bundle.model(input_ids)

    for block in bundle.model.transformer.h:
        assert len(block._forward_hooks) == 0
        assert len(block.attn._forward_hooks) == 0
