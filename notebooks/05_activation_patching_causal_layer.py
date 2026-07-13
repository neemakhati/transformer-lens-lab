"""Lesson 5: activation patching -- causally test which layer's residual
stream is responsible for the France -> Paris association, rather than
just observing correlations (which is all logit lens gave us in lesson 4).

Run with:
    python notebooks/05_activation_patching_causal_layer.py
"""

from pathlib import Path

import matplotlib.pyplot as plt

from tlab.model import load_gpt2_small, tokenize
from tlab.patching import patching_sweep

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def main() -> None:
    bundle = load_gpt2_small()

    clean_text = "The capital of France is"
    corrupted_text = "The capital of Germany is"
    clean_ids = tokenize(bundle, clean_text)
    corrupted_ids = tokenize(bundle, corrupted_text)
    last_pos = clean_ids.shape[1] - 1

    print(f"Clean prompt:     {clean_text!r}")
    print(f"Corrupted prompt: {corrupted_text!r}")
    print("(minimal pair: only the country name differs, same token count)\n")

    results = patching_sweep(
        bundle, clean_ids, corrupted_ids, position=last_pos, target_token=" Paris"
    )

    print(
        f"Baseline: clean P(Paris)={results[0].clean_baseline_prob:.2%}, "
        f"corrupted P(Paris)={results[0].corrupted_baseline_prob:.2%}\n"
    )
    print(f"{'Layer':<8}{'Patched P(Paris)':<20}{'Recovery'}")
    for r in results:
        print(f"{r.layer:<8}{r.patched_prob:<20.2%}{r.recovery:>7.1%}")

    # Plot recovery by layer -- the clearest possible picture of "which layer
    # causally matters," in contrast to lesson 4's purely observational plot.
    layers = [r.layer for r in results]
    recoveries = [r.recovery * 100 for r in results]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(layers, recoveries, color="#c9184a")
    ax.axhline(100, linestyle="--", color="gray", alpha=0.6, label="full recovery (100%)")
    ax.set_xlabel("Layer patched")
    ax.set_ylabel("% of France→Paris gap recovered")
    ax.set_title('Activation patching: which layer causally restores " Paris"?')
    ax.set_xticks(layers)
    ax.legend()
    fig.tight_layout()

    ASSETS_DIR.mkdir(exist_ok=True)
    out_path = ASSETS_DIR / "patching_recovery_by_layer.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved recovery plot -> {out_path}")


if __name__ == "__main__":
    main()
