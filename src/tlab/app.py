"""Interactive demo: type a prompt, see attention patterns and the logit lens
trajectory live. Ties together every module built in this project.

Run with:
    streamlit run src/tlab/app.py
"""

import streamlit as st

from tlab.attention_viz import plot_attention_pattern
from tlab.hooks import capture_activations
from tlab.logit_lens import logit_lens_trajectory
from tlab.model import ModelBundle, load_gpt2_small, token_strings, tokenize


@st.cache_resource
def get_bundle() -> ModelBundle:
    """Load GPT-2 once and cache it across Streamlit re-runs.

    Streamlit re-executes this whole script on every widget interaction --
    without caching, that would reload GPT-2's weights from disk on every
    keystroke. `st.cache_resource` is Streamlit's mechanism for exactly this:
    "compute once, reuse across re-runs," meant for objects like models and
    DB connections rather than plain data (which would use st.cache_data).
    """
    return load_gpt2_small()


def main() -> None:
    st.set_page_config(page_title="transformer-lens-lab", layout="wide")
    st.title("transformer-lens-lab")
    st.caption(
        "Look inside GPT-2 small: attention patterns, the logit lens, and what each layer "
        "'believes' about the next token — all from scratch, no TransformerLens dependency."
    )

    bundle = get_bundle()

    prompt = st.text_input("Prompt", value="The capital of France is")
    target_token = st.text_input(
        "Track a specific next token (must be a single GPT-2 token, e.g. ' Paris' with a leading space)",
        value=" Paris",
    )

    if not prompt.strip():
        st.stop()

    input_ids = tokenize(bundle, prompt)
    tokens = token_strings(bundle, input_ids)
    last_pos = input_ids.shape[1] - 1

    with capture_activations(bundle) as cache:
        import torch

        with torch.no_grad():
            logits = bundle.model(input_ids).logits

    top_id = int(torch.argmax(logits[0, last_pos]))
    top_prob = float(torch.softmax(logits[0, last_pos], dim=-1)[top_id])
    st.markdown(
        f"**Model's actual next-token prediction:** "
        f"`{bundle.tokenizer.decode([top_id])!r}` ({top_prob:.1%})"
    )

    tab_attn, tab_lens = st.tabs(["Attention patterns", "Logit lens"])

    with tab_attn:
        st.write("Pick a layer and head to see where each token attends.")
        col1, col2 = st.columns(2)
        layer = col1.slider("Layer", 0, bundle.n_layers - 1, value=9)
        head = col2.slider("Head", 0, bundle.n_heads - 1, value=0)

        fig = plot_attention_pattern(bundle, cache.attn_patterns[layer], tokens, layer, head)
        st.pyplot(fig)
        st.caption(
            "Rows = the token attending; columns = the token being attended to. "
            "GPT-2 is causal, so the upper-right triangle is always exactly zero."
        )

    with tab_lens:
        try:
            trajectory = logit_lens_trajectory(
                bundle, cache.resid_post, position=last_pos, target_token=target_token
            )
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        st.write(
            f"How the model's 'best guess' (decoded via the logit lens) evolves across layers, "
            f"and where {target_token!r} ranks at each one."
        )

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        layers = [p.layer for p in trajectory]
        probs = [p.target_prob * 100 for p in trajectory]
        ax.plot(layers, probs, marker="o", color="#2a6f97")
        ax.set_xlabel("Layer")
        ax.set_ylabel(f"P({target_token!r}) via logit lens (%)")
        ax.set_xticks(layers)
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        st.dataframe(
            {
                "Layer": [p.layer for p in trajectory],
                "Top prediction": [p.top_token for p in trajectory],
                "Top prob": [f"{p.top_prob:.1%}" for p in trajectory],
                f"{target_token!r} rank": [p.target_rank for p in trajectory],
                f"{target_token!r} prob": [f"{p.target_prob:.2%}" for p in trajectory],
            },
            hide_index=True,
        )


if __name__ == "__main__":
    main()
