"""Capture the residual stream and attention patterns at any layer using forward hooks.

The core idea: a `GPT2Block` computes `x = x + attn(x)` then `x = x + mlp(x)`.
That running `x` — one vector per token position, added to at every layer —
is called the *residual stream*. It's the shared "workspace" every layer
reads from and writes to. Everything else in this project (attention viz,
logit lens, activation patching) is built on being able to read or rewrite
that stream at a chosen layer.

PyTorch's `register_forward_hook(fn)` lets us attach `fn(module, input, output)`
to any submodule; PyTorch calls it automatically during the forward pass with
that submodule's actual input/output tensors, before they're discarded.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

from tlab.model import ModelBundle


@dataclass
class ActivationCache:
    """Holds every activation captured during one forward pass, keyed by layer.

    `resid_post[i]` is the residual stream *after* block i has added both its
    attention and MLP contributions — i.e. the state of the "workspace" as it
    hands off to block i+1. `attn_patterns[i]` is the post-softmax attention
    weights for block i, shape (batch, n_heads, seq_len, seq_len).
    """

    resid_post: dict[int, torch.Tensor] = field(default_factory=dict)
    attn_patterns: dict[int, torch.Tensor] = field(default_factory=dict)


@contextmanager
def capture_activations(bundle: ModelBundle) -> Iterator[ActivationCache]:
    """Run a forward pass with hooks attached, yielding a cache you can inspect after.

    Usage:
        with capture_activations(bundle) as cache:
            bundle.model(input_ids)
        cache.resid_post[5]  # residual stream after block 5, shape (1, seq_len, 768)

    We register the hooks, yield control back to the caller so they can run
    their own forward pass, then remove the hooks in a `finally` block. Hooks
    left attached would keep firing (and leaking memory) on every future
    forward pass, so cleanup is not optional.
    """
    cache = ActivationCache()
    handles = []

    for layer_idx, block in enumerate(bundle.model.transformer.h):
        handles.append(block.register_forward_hook(_make_resid_hook(cache, layer_idx)))
        handles.append(
            block.attn.register_forward_hook(_make_attn_pattern_hook(bundle, cache, layer_idx))
        )

    try:
        yield cache
    finally:
        for handle in handles:
            handle.remove()


def _make_resid_hook(cache: ActivationCache, layer_idx: int) -> Callable:
    """Build a hook that stores a GPT2Block's output (the post-block residual stream)."""

    def hook(module, inputs, output):
        # GPT2Block returns a tuple; element 0 is the updated residual stream.
        hidden_state = output[0] if isinstance(output, tuple) else output
        cache.resid_post[layer_idx] = hidden_state.detach()

    return hook


def _make_attn_pattern_hook(
    bundle: ModelBundle, cache: ActivationCache, layer_idx: int
) -> Callable:
    """Build a hook that recomputes attention weights for one block from its own Q/K.

    HuggingFace's GPT2Attention doesn't return attention weights unless you
    pass `output_attentions=True` through the whole model call, which is
    awkward to thread through generically. Recomputing Q @ K^T ourselves from
    the module's own weights is more work but keeps this hook self-contained
    and, more importantly, teaches the actual mechanics — see attention_viz.py
    for the annotated version of this same math.
    """

    def hook(module, inputs, output):
        hidden_states = inputs[0]  # the block's input to self-attention, pre-attn
        qkv = module.c_attn(hidden_states)  # (batch, seq, 3 * d_model)
        query, key, _value = qkv.split(module.embed_dim, dim=2)

        batch, seq_len, _ = query.shape
        n_heads = bundle.n_heads
        head_dim = bundle.d_model // n_heads

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, seq_len, n_heads, head_dim).transpose(1, 2)

        query, key = split_heads(query), split_heads(key)

        attn_scores = query @ key.transpose(-1, -2) / (head_dim**0.5)

        # GPT-2 is causal: token i can only attend to tokens <= i.
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        attn_scores = attn_scores.masked_fill(~causal_mask, float("-inf"))

        cache.attn_patterns[layer_idx] = torch.softmax(attn_scores, dim=-1).detach()

    return hook
