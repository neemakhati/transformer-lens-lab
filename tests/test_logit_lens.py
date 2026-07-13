import pytest

from tlab.hooks import capture_activations
from tlab.logit_lens import logit_lens_trajectory
from tlab.model import load_gpt2_small, tokenize


@pytest.fixture(scope="module")
def bundle():
    return load_gpt2_small(device="cpu")


def test_trajectory_covers_all_layers(bundle) -> None:
    input_ids = tokenize(bundle, "The capital of France is")
    with capture_activations(bundle) as cache:
        bundle.model(input_ids)

    traj = logit_lens_trajectory(bundle, cache.resid_post, position=input_ids.shape[1] - 1)
    assert [p.layer for p in traj] == list(range(bundle.n_layers))


def test_last_layer_top_token_matches_real_model_output(bundle) -> None:
    """The logit lens at the LAST layer should reproduce the model's real
    output exactly, since that's literally what the model does — this is the
    key correctness check: no LayerNorm/weight-tying bug silently changing
    the final-layer answer."""
    import torch

    input_ids = tokenize(bundle, "The capital of France is")
    last_pos = input_ids.shape[1] - 1

    with capture_activations(bundle) as cache:
        with torch.no_grad():
            real_logits = bundle.model(input_ids).logits[0, last_pos]
        real_top_token = bundle.tokenizer.decode([torch.argmax(real_logits).item()])

    traj = logit_lens_trajectory(bundle, cache.resid_post, position=last_pos)
    assert traj[-1].top_token == real_top_token


def test_target_token_tracking(bundle) -> None:
    input_ids = tokenize(bundle, "The capital of France is")
    with capture_activations(bundle) as cache:
        bundle.model(input_ids)

    traj = logit_lens_trajectory(
        bundle, cache.resid_post, position=input_ids.shape[1] - 1, target_token=" Paris"
    )
    assert all(p.target_rank is not None for p in traj)
    assert all(p.target_prob is not None for p in traj)
    # rank 1 is the best possible rank -- sanity bound, not a specific value
    assert all(p.target_rank >= 1 for p in traj)


def test_multi_token_target_raises_clear_error(bundle) -> None:
    long_word = "extraordinarily" + "un" * 20 + "believable"
    with pytest.raises(ValueError, match="single-token string"):
        input_ids = tokenize(bundle, "hello")
        with capture_activations(bundle) as cache:
            bundle.model(input_ids)
        logit_lens_trajectory(bundle, cache.resid_post, position=0, target_token=long_word)
