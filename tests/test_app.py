"""Only smoke-test the non-Streamlit-specific logic in app.py; Streamlit's
own widgets (st.slider, st.text_input, etc.) require a running Streamlit
session to test properly and are out of scope for a unit test here."""

import ast
from pathlib import Path


def test_app_module_has_valid_syntax() -> None:
    source = Path("src/tlab/app.py").read_text()
    ast.parse(source)  # raises SyntaxError if anything is malformed


def test_app_module_defines_main() -> None:
    source = Path("src/tlab/app.py").read_text()
    tree = ast.parse(source)
    function_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert "main" in function_names
    assert "get_bundle" in function_names
