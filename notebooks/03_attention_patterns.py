"""Lesson 3: visualize attention patterns and search for known head 'archetypes'.

Run with:
    python notebooks/03_attention_patterns.py

Saves plots to assets/ so they can be embedded in the README.
"""

from pathlib import Path

from tlab.attention_viz import (
    find_first_token_heads,
    find_previous_token_heads,
    plot_attention_pattern,
)
from tlab.hooks import capture_activations
from tlab.model import load_gpt2_small, token_strings, tokenize

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def main() -> None:
    bundle = load_gpt2_small()
    text = "The quick brown fox jumps over the lazy dog"
    input_ids = tokenize(bundle, text)
    tokens = token_strings(bundle, input_ids)
    print(f"Input: {text!r}")
    print(f"Tokens: {tokens}\n")

    with capture_activations(bundle) as cache:
        bundle.model(input_ids)

    # Search every layer/head for two well-documented "archetypes" from the
    # interpretability literature.
    prev_token_heads = find_previous_token_heads(cache.attn_patterns, threshold=0.5)
    first_token_heads = find_first_token_heads(cache.attn_patterns, threshold=0.5)

    print(f"Previous-token heads found (layer, head): {prev_token_heads}")
    print(f"Attention-sink heads found (layer, head):  {first_token_heads}\n")

    # Plot one example of each, if we found any, plus one arbitrary head for
    # contrast so the reader can see what a "normal," less interpretable
    # head's pattern looks like.
    ASSETS_DIR.mkdir(exist_ok=True)

    if prev_token_heads:
        layer, head = prev_token_heads[0]
        fig = plot_attention_pattern(bundle, cache.attn_patterns[layer], tokens, layer, head)
        out_path = ASSETS_DIR / "attention_previous_token_head.png"
        fig.savefig(out_path, dpi=150)
        print(f"Saved previous-token head example -> {out_path}")

    if first_token_heads:
        layer, head = first_token_heads[0]
        fig = plot_attention_pattern(bundle, cache.attn_patterns[layer], tokens, layer, head)
        out_path = ASSETS_DIR / "attention_sink_head.png"
        fig.savefig(out_path, dpi=150)
        print(f"Saved attention-sink head example -> {out_path}")

    # A contrasting "ordinary" head for comparison
    fig = plot_attention_pattern(bundle, cache.attn_patterns[8], tokens, layer=8, head=3)
    out_path = ASSETS_DIR / "attention_layer8_head3.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved layer 8 / head 3 for comparison -> {out_path}")


if __name__ == "__main__":
    main()
