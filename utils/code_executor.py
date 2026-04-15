"""
Safe-ish code executor that runs agent-generated Python in an isolated
namespace, captures stdout, result DataFrames, and Plotly figures.

Not a full sandbox — assumes trusted code from the agent.
Retries are handled by the caller (agents pass errors back to Claude).
"""

import sys
import io
import traceback
import threading
import pandas as pd
import numpy as np

EXEC_TIMEOUT_SECONDS = 60


def execute_code(code: str, extra_globals: dict | None = None) -> dict:
    """
    Execute `code` in an isolated namespace.

    Pre-populated globals:
      pd, np, px, go, file_path (if provided via extra_globals)

    Returns a dict:
      {
        "success": bool,
        "stdout": str,
        "result_df": pd.DataFrame | None,
        "additional_results": dict | None,   # keyed DataFrames
        "figures": list,                     # plotly Figure objects
        "error": str | None,
      }
    """
    # Build the execution namespace
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        px = None
        go = None

    namespace = {
        "__builtins__": __builtins__,
        "pd": pd,
        "np": np,
        "px": px,
        "go": go,
    }
    if extra_globals:
        namespace.update(extra_globals)

    # Capture stdout
    stdout_capture = io.StringIO()

    result_container = {"done": False, "error": None}

    def _run():
        try:
            old_stdout = sys.stdout
            sys.stdout = stdout_capture
            exec(code, namespace)  # noqa: S102
            sys.stdout = old_stdout
        except Exception:
            sys.stdout = old_stdout
            result_container["error"] = traceback.format_exc()
        finally:
            result_container["done"] = True

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=EXEC_TIMEOUT_SECONDS)

    if not result_container["done"]:
        return {
            "success": False,
            "stdout": stdout_capture.getvalue(),
            "result_df": None,
            "additional_results": None,
            "figures": [],
            "error": f"Execution timed out after {EXEC_TIMEOUT_SECONDS}s",
        }

    if result_container["error"]:
        return {
            "success": False,
            "stdout": stdout_capture.getvalue(),
            "result_df": None,
            "additional_results": None,
            "figures": [],
            "error": result_container["error"],
        }

    # Harvest outputs from namespace
    result_df = namespace.get("result_df", None)
    if not isinstance(result_df, pd.DataFrame):
        result_df = None

    additional_results = namespace.get("additional_results", None)
    if not isinstance(additional_results, dict):
        additional_results = None

    # Collect any plotly figures (list named `figures`, or individual `fig`)
    figures = []
    if "figures" in namespace and isinstance(namespace["figures"], list):
        figures = namespace["figures"]
    elif "fig" in namespace:
        fig = namespace["fig"]
        if hasattr(fig, "to_json"):
            figures = [fig]

    return {
        "success": True,
        "stdout": stdout_capture.getvalue(),
        "result_df": result_df,
        "additional_results": additional_results,
        "figures": figures,
        "error": None,
    }


def dataframe_to_prompt_text(df: pd.DataFrame, max_rows: int = 50) -> str:
    """Render a DataFrame as a compact string for injecting into prompts."""
    if df is None or df.empty:
        return "(empty dataframe)"
    total = len(df)
    preview = df.head(max_rows)
    text = preview.to_string(index=True)
    if total > max_rows:
        text += f"\n... ({total - max_rows} more rows not shown)"
    return text
