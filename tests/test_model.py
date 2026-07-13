import torch

from tlab.model import load_gpt2_small, token_strings, tokenize


def test_load_gpt2_small_shape() -> None:
    bundle = load_gpt2_small(device="cpu")
    assert bundle.n_layers == 12
    assert bundle.n_heads == 12
    assert bundle.d_model == 768


def test_tokenize_roundtrip() -> None:
    bundle = load_gpt2_small(device="cpu")
    input_ids = tokenize(bundle, "hello world")
    assert input_ids.shape[0] == 1  # batch dim
    decoded = bundle.tokenizer.decode(input_ids.squeeze(0))
    assert decoded == "hello world"


def test_token_strings_lengths_match() -> None:
    bundle = load_gpt2_small(device="cpu")
    input_ids = tokenize(bundle, "The capital of France is")
    strings = token_strings(bundle, input_ids)
    assert len(strings) == input_ids.shape[1]


def test_forward_pass_logit_shape() -> None:
    bundle = load_gpt2_small(device="cpu")
    input_ids = tokenize(bundle, "The capital of France is")
    with torch.no_grad():
        logits = bundle.model(input_ids).logits
    assert logits.shape == (1, input_ids.shape[1], 50257)
