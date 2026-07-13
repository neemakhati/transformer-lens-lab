"""Lesson 1: load GPT-2 small, tokenize a sentence, and watch one forward pass happen.

Run with:
    python notebooks/01_first_forward_pass.py
"""

import torch

from tlab.model import load_gpt2_small, token_strings, tokenize


def main() -> None:
    bundle = load_gpt2_small()
    print(
        f"Loaded GPT-2 small: {bundle.n_layers} layers, {bundle.n_heads} heads, "
        f"d_model={bundle.d_model}, running on {bundle.device}"
    )

    text = "The capital of France is"
    input_ids = tokenize(bundle, text)
    print(f"\nInput text: {text!r}")
    print(f"Token IDs:  {input_ids.tolist()}")
    print(f"As tokens:  {token_strings(bundle, input_ids)}")

    # A forward pass: feed token IDs in, get a probability distribution over
    # the *next* token out, for every position in the sequence at once.
    with torch.no_grad():  # we're not training, so skip building the backward graph
        output = bundle.model(input_ids)
        logits = output.logits  # shape: (batch, seq_len, vocab_size)

    print(f"\nLogits shape: {tuple(logits.shape)}")
    print("  -> (batch=1, seq_len={}, vocab_size={})".format(*logits.shape[1:]))

    # We only care about the prediction *after the last token* — i.e. what
    # comes next given the whole sentence so far.
    next_token_logits = logits[0, -1, :]
    top5 = torch.topk(next_token_logits, k=5)
    print("\nTop 5 predicted next tokens:")
    for rank, (logit_val, token_id) in enumerate(zip(top5.values, top5.indices), start=1):
        token_str = bundle.tokenizer.decode([token_id])
        prob = torch.softmax(next_token_logits, dim=-1)[token_id].item()
        print(f"  {rank}. {token_str!r:12s} (logit={logit_val:.2f}, prob={prob:.1%})")


if __name__ == "__main__":
    main()
