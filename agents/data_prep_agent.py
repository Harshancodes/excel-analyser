"""
Data Prep Agent
---------------
Given an Excel file schema and the user's analysis request, this agent
generates and iteratively fixes pandas code that:
  - Reads the relevant sheet(s)
  - Cleans / transforms the data
  - Performs the requested analysis (pivot, groupby, filter, etc.)
  - Stores the primary result in `result_df`
  - Stores any secondary results in `additional_results` (dict of DataFrames)
  - Prints key findings to stdout

The agent will retry up to MAX_RETRIES times if the generated code fails.
"""

import anthropic

MAX_RETRIES = 3

SYSTEM_PROMPT = """You are an expert Python data analyst. Your job is to write clean, efficient pandas code to analyse Excel data based on a user's request.

RULES:
1. The variable `file_path` is already defined — use it to read the file.
2. Always import pandas as pd and numpy as np at the top of your code.
3. For reading sheets use: pd.read_excel(file_path, sheet_name='SheetName')
4. Store your PRIMARY result table in a variable called `result_df` (must be a DataFrame).
5. If you produce multiple result tables, store them in `additional_results` as a dict: {"label": df, ...}
6. Print a short summary of key findings using print() statements.
7. Handle missing / NaN values gracefully — don't let them crash the analysis.
8. Return ONLY raw executable Python code. No markdown fences, no explanations, no comments beyond what helps understand the code.
9. If the user asks for a pivot table, use pd.pivot_table(). For groupby aggregations use df.groupby(). Use the most appropriate pandas construct.
10. Column names may have spaces or special characters — always access them with df["column name"] syntax, not df.column_name."""


def _build_user_prompt(schema_text: str, user_request: str) -> str:
    return f"""Here is the schema of the uploaded Excel file:

{schema_text}

User's analysis request:
{user_request}

Write the Python code now."""


def _build_fix_prompt(code: str, error: str, user_request: str) -> str:
    return f"""The code you wrote produced an error. Fix it.

Original request: {user_request}

Code that failed:
```python
{code}
```

Error:
{error}

Write the corrected Python code only."""


def generate_data_prep_code(
    client: anthropic.Anthropic,
    schema_text: str,
    user_request: str,
    model: str = "claude-sonnet-4-6",
) -> tuple[str, list[str]]:
    """
    Ask the agent to generate data prep code.

    Returns:
        (code: str, conversation_history: list[dict])
    """
    messages = [
        {"role": "user", "content": _build_user_prompt(schema_text, user_request)}
    ]

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    code = response.content[0].text.strip()
    # Strip any accidental markdown fences the model might add
    code = _strip_fences(code)

    messages.append({"role": "assistant", "content": code})
    return code, messages


def fix_data_prep_code(
    client: anthropic.Anthropic,
    messages: list[dict],
    failed_code: str,
    error: str,
    user_request: str,
    model: str = "claude-sonnet-4-6",
) -> tuple[str, list[dict]]:
    """
    Ask the agent to fix broken code given the error message.

    Returns:
        (fixed_code: str, updated_messages: list[dict])
    """
    fix_prompt = _build_fix_prompt(failed_code, error, user_request)
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
    """Remove ```python ... ``` or ``` ... ``` fences if present."""
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)
