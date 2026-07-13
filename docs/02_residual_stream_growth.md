# Finding: the residual stream grows ~8x in norm across GPT-2 small's 12 layers

**Setup:** captured the residual stream (`resid_post`) at every layer for the last
token position, for the prompt `"The capital of France is"`. Code: `notebooks/02_residual_stream_growth.py`.

## Result

| Layer | Norm |
|---|---|
| 0 | 56.5 |
| 3 | 64.3 |
| 6 | 85.4 |
| 9 | 165.6 |
| 11 | 429.6 |

The growth is not linear — it accelerates in the final layers, nearly doubling from
layer 9 to layer 10 alone.

## Why this happens

Every `GPT2Block` computes `x = x + attn(x)` and `x = x + mlp(x)` — pure addition,
never replacement or subtraction of the existing stream. There is no architectural
mechanism to shrink the stream back down, so norm growth with depth is expected.
What's notable is the *rate*: acceleration late in the network suggests the last
few layers are writing large-magnitude, high-confidence updates — plausibly where
the model commits to its final answer, rather than exploring alternatives.

## Why it matters for interpretability

Comparing raw activation magnitudes *across layers* is misleading without
accounting for this — a "big" activation at layer 2 and a "big" activation at
layer 10 are on completely different scales. Any technique that compares
activations across depth (including logit lens and activation patching, both
coming up in this project) needs to either normalize for this or be aware of it
when interpreting results.

## Open question for later modules

Does the growth curve look different for a prompt where the model is *uncertain*
of its answer vs. one where it's confident? Worth revisiting once `logit_lens.py`
lets us track per-layer prediction confidence directly.
