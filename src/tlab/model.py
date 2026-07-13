"""Loads GPT-2 small and exposes the internals we need for interpretability work.

HuggingFace's `GPT2LMHeadModel` is a fine model to *use*, but its internals are
buried behind generic module names (`transformer.h.0.attn.c_attn`, etc). This
wrapper loads the same weights and gives us a stable, documented handle onto
the pieces we'll be hooking into later: the token embedding, each block, and
the final unembedding ("language model head").
"""

from dataclasses import dataclass

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


@dataclass
class ModelBundle:
    """Everything downstream modules need: the model, its tokenizer, and its shape."""

    model: GPT2LMHeadModel
    tokenizer: GPT2TokenizerFast
    n_layers: int
    n_heads: int
    d_model: int
    device: torch.device


def load_gpt2_small(device: str | None = None) -> ModelBundle:
    """Download (or load from cache) GPT-2 small and wrap it for inspection.

    GPT-2 small is 124M parameters — small enough to run on a laptop CPU in
    under a second per forward pass, but a "real" trained transformer, not a
    toy. That combination (real weights, tiny size) is what makes it the
    standard model for interpretability research.
    """
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(resolved_device)
    model.eval()  # disables dropout — we want deterministic forward passes

    config = model.config
    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        n_layers=config.n_layer,
        n_heads=config.n_head,
        d_model=config.n_embd,
        device=resolved_device,
    )


def tokenize(bundle: ModelBundle, text: str) -> torch.Tensor:
    """Turn a string into the integer token IDs GPT-2 was trained on.

    Returns a tensor of shape (1, seq_len) — the leading 1 is the batch
    dimension. Every tensor in this codebase keeps an explicit batch dim,
    even for single-example use, because that's what the model expects.
    """
    encoding = bundle.tokenizer(text, return_tensors="pt")
    return encoding["input_ids"].to(bundle.device)


def token_strings(bundle: ModelBundle, token_ids: torch.Tensor) -> list[str]:
    """Decode each token ID individually so we can see the model's actual vocabulary units.

    GPT-2 uses byte-pair encoding: common words are one token, rarer words
    split into sub-word pieces. Seeing this split is often the first surprise
    for people new to how LLMs "see" text — e.g. "unbelievable" might become
    ["un", "believ", "able"].
    """
    ids = token_ids.squeeze(0).tolist()
    return [bundle.tokenizer.decode([tid]) for tid in ids]
