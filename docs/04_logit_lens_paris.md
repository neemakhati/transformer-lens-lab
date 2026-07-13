# Finding: GPT-2 "considers" Paris at layer 9, then partially backs away from it by layer 11

**Setup:** prompt `"The capital of France is"`, logit lens applied to the residual
stream at the last token position, for every layer. Tracked both the layer's top
prediction and the specific rank/probability of `" Paris"`.
Code: `notebooks/04_logit_lens_paris.py`, mechanism in `tlab/logit_lens.py`.

## Result

| Layer | Top prediction | " Paris" rank | " Paris" prob |
|---|---|---|---|
| 0–8 | generic (`" not"`, `" now"`) | 3,000–18,000+ | ~0% |
| **9** | `" France"` | **2** | **18.2%** |
| 10 | `" France"` | 2 | 13.9% |
| 11 (real output) | `" the"` | 5 | 3.2% |

![Logit lens trajectory for "Paris"](../assets/logit_lens_paris_trajectory.png)

## Interpretation

For the first 8 layers, decoding the residual stream gives nothing
France-specific at all — generic continuations dominate, and "Paris" is
buried among tens of thousands of tokens. Then **layer 9 produces a sharp,
sudden jump**: "Paris" goes from effectively 0% to the second-most-likely
token in one layer. This is a clean example of what the interpretability
literature calls a late-layer "answer injection" — evidence the specific
factual association (France → Paris) is being introduced by a small number
of components, probably around layer 9, rather than gradually accumulated
across the whole network.

**Just as interesting: the probability partially reverses by layer 11.**
The model appears to "know" Paris is highly relevant, then de-prioritizes it
in favor of more syntactically generic completions (`" the"` ends up on top).
One plausible read: layers 10–11 are doing something more like "what's a
safe, grammatical continuation" rather than "what's the most factually
specific one" — i.e., factual recall and final output shaping may be
partially in tension in this model, at least for this prompt.

## This confirms a suspicion from session 1

Back in `docs/01` (implicitly, via the forward-pass notebook) we noted Paris
only ranked 5th in the real output — surprising for such a canonical fact.
This module explains *why*: it's not that the model doesn't know the
answer. It gets much closer at layer 9 (rank 2, 18%) than it ends up at
layer 11 (rank 5, 3%). Something after layer 9 pulls the final answer away
from its best internal guess.

## Caveat

The logit lens is an approximation the model was never trained to support —
early-layer "generic" predictions may partly reflect that intermediate
representations aren't *meant* to be decoded this way, not necessarily that
the model has "no idea" yet. Activation patching (next module) gives a more
rigorous, causal way to test which components actually matter, rather than
just observing correlational snapshots layer by layer.

## Follow-up

- Use activation patching to test: does *specifically* patching layer 9's
  MLP or attention output (not the whole residual stream) reproduce this
  jump? That would localize the "Paris injection" to a specific sub-component
  instead of just a layer.
- Try 2-3 other factual-recall prompts (different countries) and see if the
  "peak then partial reversal" pattern is a general phenomenon or specific
  to this example.
