import torch

from tlab.attention_viz import (
    find_first_token_heads,
    find_previous_token_heads,
    plot_attention_pattern,
)
from tlab.hooks import capture_activations
from tlab.model import load_gpt2_small, token_strings, tokenize


def _make_synthetic_pattern(seq_len: int, n_heads: int, sink_head: int, prev_head: int):
    """Build a hand-crafted attention pattern with one known sink head and one
    known previous-token head, so detector tests don't depend on real model
    weights happening to contain these circuits."""
    pattern = torch.zeros(1, n_heads, seq_len, seq_len)
    for head in range(n_heads):
        for i in range(seq_len):
            if head == sink_head:
                pattern[0, head, i, 0] = 1.0
            elif head == prev_head and i > 0:
                pattern[0, head, i, i - 1] = 1.0
            else:
                # uniform causal attention as a "boring" default head
                pattern[0, head, i, : i + 1] = 1.0 / (i + 1)
    return pattern


def test_find_first_token_heads_detects_synthetic_sink() -> None:
    pattern = _make_synthetic_pattern(seq_len=6, n_heads=4, sink_head=1, prev_head=2)
    found = find_first_token_heads({0: pattern}, threshold=0.5)
    assert (0, 1) in found
    assert (0, 2) not in found


def test_find_previous_token_heads_detects_synthetic_prev_head() -> None:
    pattern = _make_synthetic_pattern(seq_len=6, n_heads=4, sink_head=1, prev_head=2)
    found = find_previous_token_heads({0: pattern}, threshold=0.5)
    assert (0, 2) in found
    assert (0, 1) not in found


def test_plot_attention_pattern_runs_on_real_model() -> None:
    bundle = load_gpt2_small(device="cpu")
    input_ids = tokenize(bundle, "hello there friend")
    tokens = token_strings(bundle, input_ids)
    with capture_activations(bundle) as cache:
        bundle.model(input_ids)

    fig = plot_attention_pattern(bundle, cache.attn_patterns[0], tokens, layer=0, head=0)
    assert fig is not None
