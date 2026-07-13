"""Lesson 4: use the logit lens to answer session 1's open question —
at which layer does GPT-2 start "considering" Paris for
"The capital of France is ___"?

Run with:
    python notebooks/04_logit_lens_paris.py
"""

from pathlib import Path

import matplotlib.pyplot as plt

from tlab.hooks import capture_activations
from tlab.logit_lens import logit_lens_trajectory
from tlab.model import load_gpt2_small, tokenize

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def main() -> None:
    bundle = load_gpt2_small()
    text = "The capital of France is"
    input_ids = tokenize(bundle, text)
    last_pos = input_ids.shape[1] - 1

    with capture_activations(bundle) as cache:
        bundle.model(input_ids)

    traj = logit_lens_trajectory(bundle, cache.resid_post, position=last_pos, target_token=" Paris")

    print(f"Prompt: {text!r}\n")
    header = f"{'Layer':<6}{'Top prediction':<18}{'Top prob':<12}{'Paris rank':<12}{'Paris prob'}"
    print(header)
    for p in traj:
        print(
            f"{p.layer:<6}{p.top_token!r:<18}{p.top_prob:<12.1%}"
            f"{p.target_rank:<12}{p.target_prob:.2%}"
        )

    # Plot Paris's probability trajectory across layers -- this is the
    # single clearest way to see the "layer 9 phase transition, then partial
    # reversal by layer 11" finding from docs/04.
    layers = [p.layer for p in traj]
    paris_probs = [p.target_prob * 100 for p in traj]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(layers, paris_probs, marker="o", color="#2a6f97")
    ax.set_xlabel("Layer")
    ax.set_ylabel("P(' Paris') via logit lens (%)")
    ax.set_title('Logit lens: P(" Paris") by layer\n"The capital of France is ___"')
    ax.set_xticks(layers)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    ASSETS_DIR.mkdir(exist_ok=True)
    out_path = ASSETS_DIR / "logit_lens_paris_trajectory.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved trajectory plot -> {out_path}")


if __name__ == "__main__":
    main()
