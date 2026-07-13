"""Render attention patterns as heatmaps, and summarize where each head 'looks'.

An attention pattern for one (layer, head) is a (seq_len, seq_len) matrix:
row i, column j is "how much does token i attend to token j when updating
its own representation." Because GPT-2 is causal, this matrix is lower
triangular (a token can never attend to a future token) — you'll see that
as a solid white upper-right triangle in every plot this module produces.
"""

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure

from tlab.model import ModelBundle


def plot_attention_pattern(
    bundle: ModelBundle,
    pattern: torch.Tensor,
    tokens: list[str],
    layer: int,
    head: int,
) -> Figure:
    """Plot one head's attention pattern as a heatmap, tokens labeled on both axes.

    `pattern` is the full (batch, n_heads, seq_len, seq_len) tensor from
    ActivationCache.attn_patterns[layer] — we index out the batch and head
    dimensions here so callers don't have to repeat that indexing everywhere.
    """
    matrix = pattern[0, head].numpy()  # (seq_len, seq_len)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right")
    ax.set_yticklabels(tokens)
    ax.set_xlabel("key (attended-to token)")
    ax.set_ylabel("query (attending-from token)")
    ax.set_title(f"Layer {layer}, Head {head} attention pattern")
    fig.colorbar(im, ax=ax, label="attention weight")
    fig.tight_layout()
    return fig


def find_previous_token_heads(
    attn_patterns: dict[int, torch.Tensor], threshold: float = 0.5
) -> list[tuple[int, int]]:
    """Find heads that mostly attend to the token immediately before the current one.

    "Previous-token heads" are one of the first documented, well-understood
    circuit components in GPT-2 (see Anthropic's Transformer Circuits work) —
    a head that has learned the simple, useful rule "copy information from
    one position back." This function detects them heuristically: average,
    across all query positions (except position 0, which has no previous
    token), how much attention weight lands exactly one position back.
    """
    found = []
    for layer, pattern in attn_patterns.items():
        n_heads = pattern.shape[1]
        seq_len = pattern.shape[2]
        if seq_len < 2:
            continue
        for head in range(n_heads):
            # weight each query position i>=1 puts on position i-1
            prev_token_weight = torch.stack(
                [pattern[0, head, i, i - 1] for i in range(1, seq_len)]
            ).mean()
            if prev_token_weight.item() > threshold:
                found.append((layer, head))
    return found


def find_first_token_heads(
    attn_patterns: dict[int, torch.Tensor], threshold: float = 0.5
) -> list[tuple[int, int]]:
    """Find heads that mostly attend back to the very first token in the sequence.

    These are sometimes called "attention sink" heads — documented behavior
    where a head dumps most of its weight onto position 0 regardless of
    content, seemingly using it as a no-op / rest position rather than
    retrieving meaningful information from it.
    """
    found = []
    for layer, pattern in attn_patterns.items():
        n_heads = pattern.shape[1]
        seq_len = pattern.shape[2]
        if seq_len < 2:
            continue
        for head in range(n_heads):
            first_token_weight = torch.stack(
                [pattern[0, head, i, 0] for i in range(1, seq_len)]
            ).mean()
            if first_token_weight.item() > threshold:
                found.append((layer, head))
    return found
