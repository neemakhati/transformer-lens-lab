"""The logit lens: decode intermediate residual stream states into vocabulary space.

GPT-2's real final step is:
    logits = lm_head(ln_f(resid_stream_after_layer_11))

The logit lens (nostalgebraist, 2020) asks what happens if we apply that same
`ln_f` + `lm_head` to the residual stream *after every other layer* too — even
though the model was never trained for intermediate layers to be meaningfully
decodable that way. It's an approximation, not something the model "intends,"
but it empirically tends to surface a sensible partial guess, letting us watch
the predicted next token evolve layer by layer.

Two details matter for correctness here, both easy to get subtly wrong:
  1. `lm_head` and the token embedding share weights (tied embeddings) — see
     `tlab.model`'s docstring notes; using `model.lm_head` directly handles
     this correctly without us needing to know that internally.
  2. `ln_f` must be applied before unembedding at every layer, not just the
     last one. Skipping it doesn't just change the numbers slightly — since
     residual stream norm grows ~8x across layers (see docs/02), an
     un-normalized early-layer vector is a completely different scale than
     what `lm_head` was trained to operate on, and the resulting "logits"
     would be close to meaningless.
"""

from dataclasses import dataclass

import torch

from tlab.model import ModelBundle


@dataclass
class LayerPrediction:
    """The logit-lens decoding of one layer's residual stream at one token position."""

    layer: int
    top_token: str
    top_prob: float
    target_rank: int | None  # rank of a token of interest, if one was requested
    target_prob: float | None


def logit_lens_trajectory(
    bundle: ModelBundle,
    resid_post: dict[int, torch.Tensor],
    position: int,
    target_token: str | None = None,
) -> list[LayerPrediction]:
    """Decode the residual stream at `position`, for every captured layer.

    `resid_post` comes straight from `ActivationCache.resid_post` (see
    tlab.hooks). `target_token` lets you track one specific token's rank and
    probability across layers even when it's not the top prediction — e.g.
    tracking "Paris" even at layers where "the" or "a" currently rank higher.
    """
    target_id = None
    if target_token is not None:
        # encode without special tokens so we get exactly the token IDs GPT-2
        # would use mid-sequence, matching how token_strings/tokenize work
        ids = bundle.tokenizer.encode(target_token)
        if len(ids) != 1:
            raise ValueError(
                f"{target_token!r} encodes to {len(ids)} tokens ({ids}), "
                "logit lens tracking needs a single-token string (try a leading space, e.g. ' Paris')"
            )
        target_id = ids[0]

    ln_f = bundle.model.transformer.ln_f
    lm_head = bundle.model.lm_head

    trajectory = []
    with torch.no_grad():
        for layer in sorted(resid_post.keys()):
            hidden = resid_post[layer][0, position]  # (d_model,)
            normed = ln_f(hidden)
            logits = lm_head(normed)  # (vocab_size,)
            probs = torch.softmax(logits, dim=-1)

            top_id = torch.argmax(probs).item()
            top_token = bundle.tokenizer.decode([top_id])
            top_prob = probs[top_id].item()

            target_rank = None
            target_prob = None
            if target_id is not None:
                target_prob = probs[target_id].item()
                # rank = how many tokens have strictly higher probability, 1-indexed
                target_rank = int((probs > probs[target_id]).sum().item()) + 1

            trajectory.append(
                LayerPrediction(
                    layer=layer,
                    top_token=top_token,
                    top_prob=top_prob,
                    target_rank=target_rank,
                    target_prob=target_prob,
                )
            )
    return trajectory
