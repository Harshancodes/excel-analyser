# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

# Set API key (copy and fill in)
cp .env.example .env
```

## Architecture

This is a Streamlit app that runs a sequential 3-agent pipeline powered by the Anthropic API. Each agent is a module in `agents/` that calls `client.messages.create()` and returns generated Python code or text.

**Pipeline flow (app.py):**
1. User uploads Excel → `utils/excel_utils.py` inspects sheets and builds a schema string for the prompt
2. **Data Prep Agent** (`agents/data_prep_agent.py`) — generates pandas code; `utils/code_executor.py` runs it; on failure, the agent is called again with the error to self-fix (up to `MAX_RETRIES=3`)
3. **Analysis Agent** (`agents/analysis_agent.py`) — receives the result DataFrame as a string + stdout and returns a plain-text summary (no code execution)
4. **Viz Agent** (`agents/viz_agent.py`) — generates Plotly code; same retry loop as Data Prep

**Code execution (`utils/code_executor.py`):**
- Runs in a `threading.Thread` with a 60 s timeout
- Pre-populates namespace with `pd`, `np`, `px`, `go`, and any `extra_globals` (e.g. `file_path`, `result_df`)
- Harvests `result_df` (DataFrame), `additional_results` (dict of DataFrames), and `figures` (list of Plotly figures) from the namespace after execution

**Key conventions:**
- All agents receive the Anthropic `client` and `model` as arguments — the client is created once in `app.py` using the key from the sidebar
- Agent functions return `(code/text, messages_list)` so the conversation history can be passed back for fix retries
- `_strip_fences()` in each agent removes accidental ` ```python ``` ` wrapping from model output
- `result_df` is the single contract between the Data Prep step and the Analysis/Viz steps
