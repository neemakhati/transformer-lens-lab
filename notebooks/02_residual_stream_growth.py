"""Lesson 2: use hooks to watch the residual stream accumulate across layers.

Each GPT2Block *adds* its attention and MLP output to the incoming stream —
it never replaces it. A natural question: does the stream's magnitude grow
as we go deeper? (Spoiler: yes, substantially — this is a well-documented
property of trained transformers and one reason later layers use LayerNorm
so aggressively.)

Run with:
    python notebooks/02_residual_stream_growth.py
"""

import torch

from tlab.hooks import capture_activations
from tlab.model import load_gpt2_small, tokenize


def main() -> None:
    bundle = load_gpt2_small()
    text = "The capital of France is"
    input_ids = tokenize(bundle, text)

    with capture_activations(bundle) as cache:
        bundle.model(input_ids)

    print(f"Input: {text!r}\n")
    print("Residual stream norm at the LAST token position, by layer:")
    print("(norm = length of the 768-dim vector -- a proxy for 'how much signal is here')\n")

    last_token_idx = input_ids.shape[1] - 1
    for layer in range(bundle.n_layers):
        resid = cache.resid_post[layer][0, last_token_idx]  # shape (768,)
        norm = torch.linalg.norm(resid).item()
        bar = "#" * int(norm / 3)
        print(f"  layer {layer:2d}: norm={norm:7.2f}  {bar}")

    print("\nWhat to notice: the norm grows roughly monotonically with depth.")
    print("Every block ADDS to the stream and nothing removes from it, so this")
    print("is architecturally guaranteed to trend upward -- LayerNorm at the start")
    print("of each block is what keeps the *computation* stable despite this growth.")

    # A second observation: which layer's attention looks most "diffuse" vs "sharp"?
    print("\n" + "=" * 60)
    print("Attention entropy by layer (head 0, from the last token):")
    print("Low entropy = attention concentrated on one token. High = spread out.\n")
    for layer in range(bundle.n_layers):
        pattern = cache.attn_patterns[layer][0, 0, last_token_idx]  # (seq_len,)
        entropy = -(pattern * torch.log(pattern + 1e-10)).sum().item()
        print(f"  layer {layer:2d}: entropy={entropy:.3f}")


if __name__ == "__main__":
    main()
