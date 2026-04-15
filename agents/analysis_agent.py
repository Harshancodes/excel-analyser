"""
Analysis Agent
--------------
Takes the output of the data prep step (result DataFrames + stdout)
and the user's original request, then produces a clear natural-language
summary: key findings, patterns, anomalies, direct answers to the question.
"""

import anthropic

SYSTEM_PROMPT = """You are a senior data analyst. You will be given:
1. The user's original analysis request.
2. The output produced by a pandas data analysis script (tables + printed stats).

Your job is to write a clear, insightful summary for a business audience.

Guidelines:
- Open with a direct answer to the user's question.
- Highlight the 3-5 most important findings with specific numbers.
- Call out any anomalies, outliers, or surprising patterns.
- Keep it concise — 3 to 5 short paragraphs.
- Use plain language, no jargon, no code.
- If the data is insufficient to answer the question, say so clearly."""


def _build_prompt(user_request: str, stdout_output: str, result_table: str) -> str:
    parts = [f"User's request:\n{user_request}\n"]

    if stdout_output.strip():
        parts.append(f"Script output (printed stats):\n{stdout_output}\n")

    if result_table.strip():
        parts.append(f"Result table:\n{result_table}\n")

    parts.append("Write your analysis summary now.")
    return "\n".join(parts)


def generate_analysis_summary(
    client: anthropic.Anthropic,
    user_request: str,
    stdout_output: str,
    result_table: str,
    model: str = "claude-sonnet-4-6",
) -> str:
    """
    Generate a natural-language analysis summary.

    Args:
        client:        Anthropic client
        user_request:  The original question from the user
        stdout_output: Captured stdout from the data prep code execution
        result_table:  The result_df rendered as a string
        model:         Claude model to use

    Returns:
        summary: str
    """
    prompt = _build_prompt(user_request, stdout_output, result_table)

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text.strip()
