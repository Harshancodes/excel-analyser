"""
Visualization Agent
-------------------
Given the prepared result data, the analysis summary, and the user's
original request, this agent generates Plotly code that creates
appropriate interactive charts.

The generated code must:
  - Use `result_df` which is already defined in the execution namespace
  - Store all figures in a list called `figures`
  - Use plotly.express (px) or plotly.graph_objects (go)
  - Include proper titles, axis labels, and color schemes

The agent will retry up to MAX_RETRIES times if code fails.
"""

import anthropic

MAX_RETRIES = 3

SYSTEM_PROMPT = """You are an expert data visualization engineer specializing in Plotly.

Your job is to write Python code that creates insightful, beautiful interactive charts using Plotly.

RULES:
1. `result_df` is already defined as a pandas DataFrame — use it directly. Do NOT re-read the Excel file.
2. `px` (plotly.express) and `go` (plotly.graph_objects) are already imported.
3. `pd` and `np` are already imported.
4. Store ALL figures in a Python list called `figures`. e.g. figures = [fig1, fig2]
5. Choose chart types that best fit the data and the user's question:
   - Comparisons → bar chart, grouped bar
   - Trends over time → line chart
   - Distributions → histogram, box plot
   - Correlations → scatter plot, heatmap
   - Part-to-whole → pie chart, treemap
6. Every chart must have: a descriptive title, labeled axes, and a clean color scheme.
7. Prefer plotly.express for simplicity. Use graph_objects only when express can't do it.
8. Return ONLY raw executable Python code. No markdown fences, no explanations.
9. If result_df has fewer than 2 columns, create a single chart only.
10. Do not create more than 4 charts — pick the most informative ones."""


def _build_user_prompt(
    result_table: str,
    user_request: str,
    analysis_summary: str,
    df_columns: list[str],
    df_dtypes: dict,
) -> str:
    dtype_info = "\n".join(f"  {col}: {dtype}" for col, dtype in df_dtypes.items())
    return f"""User's original request:
{user_request}

Analysis summary (for context):
{analysis_summary}

result_df columns and dtypes:
{dtype_info}

result_df preview:
{result_table}

Write the Plotly visualization code now."""


def _build_fix_prompt(code: str, error: str) -> str:
    return f"""The visualization code produced an error. Fix it.

Code that failed:
```python
{code}
```

Error:
{error}

Write the corrected Python code only."""


def generate_viz_code(
    client: anthropic.Anthropic,
    result_table: str,
    user_request: str,
    analysis_summary: str,
    df_columns: list[str],
    df_dtypes: dict,
    model: str = "claude-sonnet-4-6",
) -> tuple[str, list[dict]]:
    """
    Ask the agent to generate Plotly visualization code.

    Returns:
        (code: str, conversation_history: list[dict])
    """
    prompt = _build_user_prompt(
        result_table, user_request, analysis_summary, df_columns, df_dtypes
    )

    messages = [{"role": "user", "content": prompt}]

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    code = response.content[0].text.strip()
    code = _strip_fences(code)

    messages.append({"role": "assistant", "content": code})
    return code, messages


def fix_viz_code(
    client: anthropic.Anthropic,
    messages: list[dict],
    failed_code: str,
    error: str,
    model: str = "claude-sonnet-4-6",
) -> tuple[str, list[dict]]:
    """Fix broken visualization code."""
    fix_prompt = _build_fix_prompt(failed_code, error)
    messages.append({"role": "user", "content": fix_prompt})

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    fixed_code = response.content[0].text.strip()
    fixed_code = _strip_fences(fixed_code)

    messages.append({"role": "assistant", "content": fixed_code})
    return fixed_code, messages


def _strip_fences(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)
