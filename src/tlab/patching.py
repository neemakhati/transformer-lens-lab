"""Activation patching: a causal test for which layer/position actually matters.

Everything else in this project is observational -- we run the model once and
look at what its internals show us. Patching is different: it's an
*intervention*. The recipe (from Meng et al.'s ROME paper and Anthropic's
circuits work) is:

  1. Run a "clean" prompt (e.g. "The capital of France is") and cache its
     residual stream at every layer.
  2. Run a "corrupted" prompt that changes one key fact (e.g. swap France
     for Germany) and cache ITS residual stream too.
  3. Re-run the corrupted prompt, but this time splice in the CLEAN run's
     residual stream at one specific (layer, position) -- everything else
     stays corrupted.
  4. Measure how much closer the output moves back toward the clean answer.

If patching in layer 9's residual stream alone recovers most of "Paris",
that's a causal claim: layer 9 carries the information that matters, not
just a layer that happens to correlate with it.

Both prompts must tokenize to the same length so a "position" means the same
grammatical slot in both sequences -- see the notebook for why "France" vs
"Germany" is a good minimal pair for this (same token count, difference
isolated to one position).
"""

from dataclasses import dataclass

import torch

from tlab.model import ModelBundle


@dataclass
class PatchResult:
    """The effect of patching one (layer, position) on the target token's probability."""

    layer: int
    corrupted_baseline_prob: float  # P(target) with no patching, corrupted run
    clean_baseline_prob: float  # P(target) on the clean run, for reference
    patched_prob: float  # P(target) after patching this layer in
    recovery: float  # 0.0 = no effect, 1.0 = fully recovered clean-level probability


def _target_prob(bundle: ModelBundle, logits: torch.Tensor, target_id: int) -> float:
    probs = torch.softmax(logits, dim=-1)
    return probs[target_id].item()


def run_with_cache(bundle: ModelBundle, input_ids: torch.Tensor):
    """Run a forward pass and return (logits, resid_post cache) together.

    Thin wrapper so patch_layer callers don't need to import capture_activations
    separately just to get both pieces they need from one forward pass.
    """
    from tlab.hooks import capture_activations

    with capture_activations(bundle) as cache:
        with torch.no_grad():
            logits = bundle.model(input_ids).logits
    return logits, cache


def patch_layer(
    bundle: ModelBundle,
    clean_resid: dict[int, torch.Tensor],
    corrupted_ids: torch.Tensor,
    layer: int,
    position: int,
) -> torch.Tensor:
    """Re-run `corrupted_ids`, but force layer `layer`'s residual stream at
    `position` to equal the clean run's value at that same coordinate.

    We do this by registering a temporary forward hook that OVERWRITES the
    block's output right as it happens, then removing the hook immediately
    after -- same cleanup discipline as tlab.hooks.capture_activations, and
    for the same reason (a leaked hook would corrupt every later forward pass).
    """
    clean_value = clean_resid[layer][0, position]  # (d_model,)

    def patch_hook(module, inputs, output):
        hidden_state = output[0] if isinstance(output, tuple) else output
        hidden_state = hidden_state.clone()
        hidden_state[0, position] = clean_value
        if isinstance(output, tuple):
            return (hidden_state,) + output[1:]
        return hidden_state

    block = bundle.model.transformer.h[layer]
    handle = block.register_forward_hook(patch_hook)
    try:
        with torch.no_grad():
            logits = bundle.model(corrupted_ids).logits
    finally:
        handle.remove()
    return logits


def patching_sweep(
    bundle: ModelBundle,
    clean_ids: torch.Tensor,
    corrupted_ids: torch.Tensor,
    position: int,
    target_token: str,
) -> list[PatchResult]:
    """Patch each layer one at a time and measure recovery of `target_token`'s probability.

    `recovery = (patched - corrupted_baseline) / (clean_baseline - corrupted_baseline)`
    -- 0 means the patch had no effect (still looks like the corrupted run),
    1 means the patch fully restored the clean run's confidence in the target.
    Recovery can go outside [0, 1] if a single layer overshoots the effect.
    """
    if clean_ids.shape != corrupted_ids.shape:
        raise ValueError(
            f"clean and corrupted prompts must tokenize to the same shape, "
            f"got {clean_ids.shape} vs {corrupted_ids.shape}"
        )

    target_ids = bundle.tokenizer.encode(target_token)
    if len(target_ids) != 1:
        raise ValueError(f"{target_token!r} must be a single token, got {target_ids}")
    target_id = target_ids[0]

    clean_logits, clean_cache = run_with_cache(bundle, clean_ids)
    corrupted_logits, _ = run_with_cache(bundle, corrupted_ids)

    clean_prob = _target_prob(bundle, clean_logits[0, position], target_id)
    corrupted_prob = _target_prob(bundle, corrupted_logits[0, position], target_id)

    results = []
    for layer in range(bundle.n_layers):
        patched_logits = patch_layer(bundle, clean_cache.resid_post, corrupted_ids, layer, position)
        patched_prob = _target_prob(bundle, patched_logits[0, position], target_id)

        denom = clean_prob - corrupted_prob
        recovery = (patched_prob - corrupted_prob) / denom if abs(denom) > 1e-9 else 0.0

        results.append(
            PatchResult(
                layer=layer,
                corrupted_baseline_prob=corrupted_prob,
                clean_baseline_prob=clean_prob,
                patched_prob=patched_prob,
                recovery=recovery,
            )
        )
    return results
