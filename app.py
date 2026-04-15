"""
Excel Analyser — Streamlit App
================================
Pipeline:
  1. User uploads Excel + selects sheets + types analysis request
  2. [Data Prep Agent]  → writes pandas code → executes it → shows code + result table
  3. [Analysis Agent]   → reads result → writes natural-language summary
  4. [Viz Agent]        → writes Plotly code → executes it → shows code + interactive charts
"""

import os
import tempfile

import anthropic
import streamlit as st
from dotenv import load_dotenv

from agents.data_prep_agent import (
    MAX_RETRIES as DATA_MAX_RETRIES,
    fix_data_prep_code,
    generate_data_prep_code,
)
from agents.analysis_agent import generate_analysis_summary
from agents.viz_agent import (
    MAX_RETRIES as VIZ_MAX_RETRIES,
    fix_viz_code,
    generate_viz_code,
)
from utils.code_executor import dataframe_to_prompt_text, execute_code
from utils.excel_utils import (
    get_sheet_names,
    inspect_sheet,
    multi_schema_to_prompt_text,
    schema_to_prompt_text,
)

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Excel Analyser",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Excel Analyser")
st.caption("Upload an Excel file, describe what you want to know — agents do the rest.")

# ── Sidebar: API key ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input(
        "Anthropic API Key",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Get yours at console.anthropic.com",
    )
    model_choice = st.selectbox(
        "Model",
        ["claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5-20251001"],
        index=0,
    )
    st.divider()
    st.markdown("**Pipeline steps**")
    st.markdown("1. 🔍 Inspect Excel schema")
    st.markdown("2. 🤖 Code Writer Agent")
    st.markdown("3. ▶️ Execute analysis code")
    st.markdown("4. 📝 Analysis Agent")
    st.markdown("5. 📈 Viz Agent")
    st.markdown("6. ▶️ Execute chart code")

# ── File upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload your Excel file (.xlsx / .xls)",
    type=["xlsx", "xls"],
)

if not uploaded_file:
    st.info("Upload an Excel file to get started.")
    st.stop()

# Save to a temp file so pandas can read it by path
with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
    tmp.write(uploaded_file.read())
    tmp_path = tmp.name

# ── Sheet selection ───────────────────────────────────────────────────────────
try:
    sheet_names = get_sheet_names(tmp_path)
except Exception as e:
    st.error(f"Could not read Excel file: {e}")
    st.stop()

st.subheader("Sheets detected")
selected_sheets = st.multiselect(
    "Select sheet(s) to analyse",
    options=sheet_names,
    default=sheet_names[:1],
)

if not selected_sheets:
    st.warning("Select at least one sheet.")
    st.stop()

# ── Schema preview ────────────────────────────────────────────────────────────
with st.expander("File schema preview", expanded=True):
    for sheet in selected_sheets:
        inspection = inspect_sheet(tmp_path, sheet)
        st.markdown(f"**Sheet: `{sheet}`** — {inspection['shape'][0]} rows × {inspection['shape'][1]} cols")
        col_data = [
            {
                "Column": c["name"],
                "Type": c["dtype"],
                "Nulls": c["null_count"],
                "Samples": ", ".join(str(v) for v in c["sample_values"]),
            }
            for c in inspection["columns"]
        ]
        st.dataframe(col_data, use_container_width=True, hide_index=True)

# ── Analysis request ──────────────────────────────────────────────────────────
st.subheader("What do you want to analyse?")
user_request = st.text_area(
    "Describe your analysis",
    placeholder=(
        "e.g. Show me total sales by region and product category, "
        "sorted by revenue descending. Highlight the top 5 performers."
    ),
    height=120,
)

run_btn = st.button("🚀 Run Analysis", type="primary", disabled=not user_request.strip())

if not run_btn:
    st.stop()

if not api_key:
    st.error("Enter your Anthropic API key in the sidebar.")
    st.stop()

# ── Setup ─────────────────────────────────────────────────────────────────────
client = anthropic.Anthropic(api_key=api_key)

# Build schema text for selected sheets
all_inspections = {s: inspect_sheet(tmp_path, s) for s in selected_sheets}
if len(selected_sheets) == 1:
    schema_text = schema_to_prompt_text(all_inspections[selected_sheets[0]])
else:
    schema_text = multi_schema_to_prompt_text(all_inspections)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Data Prep Agent
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Step 1 — Data Prep Agent")

with st.status("Generating data preparation code...", expanded=True) as status:
    st.write("Sending schema + request to Code Writer Agent...")

    code, messages = generate_data_prep_code(
        client=client,
        schema_text=schema_text,
        user_request=user_request,
        model=model_choice,
    )
    st.write("Code generated. Executing...")

    exec_result = execute_code(code, extra_globals={"file_path": tmp_path})

    retries = 0
    while not exec_result["success"] and retries < DATA_MAX_RETRIES:
        retries += 1
        st.write(f"Execution failed (attempt {retries}/{DATA_MAX_RETRIES}). Asking agent to fix...")
        code, messages = fix_data_prep_code(
            client=client,
            messages=messages,
            failed_code=code,
            error=exec_result["error"],
            user_request=user_request,
            model=model_choice,
        )
        exec_result = execute_code(code, extra_globals={"file_path": tmp_path})

    if exec_result["success"]:
        status.update(label="Data prep complete!", state="complete")
    else:
        status.update(label="Data prep failed after retries", state="error")

# Show generated code
with st.expander("View generated data prep code", expanded=False):
    st.code(code, language="python")

if not exec_result["success"]:
    st.error(f"Data prep code failed after {DATA_MAX_RETRIES} retries:\n\n{exec_result['error']}")
    st.stop()

# Show stdout output
if exec_result["stdout"].strip():
    with st.expander("Script output (printed stats)", expanded=True):
        st.text(exec_result["stdout"])

# Show result DataFrame
result_df = exec_result["result_df"]
if result_df is not None and not result_df.empty:
    st.markdown("**Result table**")
    st.dataframe(result_df, use_container_width=True)
else:
    st.warning("No `result_df` was produced by the data prep code. Analysis may be limited.")

# Show additional result tables if any
if exec_result["additional_results"]:
    for label, df in exec_result["additional_results"].items():
        st.markdown(f"**{label}**")
        st.dataframe(df, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Analysis Agent
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Step 2 — Analysis Agent")

with st.status("Generating analysis summary...", expanded=True) as status:
    result_table_str = dataframe_to_prompt_text(result_df) if result_df is not None else ""

    summary = generate_analysis_summary(
        client=client,
        user_request=user_request,
        stdout_output=exec_result["stdout"],
        result_table=result_table_str,
        model=model_choice,
    )
    status.update(label="Analysis complete!", state="complete")

st.markdown(summary)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Visualization Agent
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("Step 3 — Visualization Agent")

if result_df is None or result_df.empty:
    st.warning("No result data available for visualization.")
    st.stop()

df_dtypes = {col: str(dtype) for col, dtype in result_df.dtypes.items()}

with st.status("Generating visualization code...", expanded=True) as status:
    st.write("Sending data + analysis to Viz Agent...")

    viz_code, viz_messages = generate_viz_code(
        client=client,
        result_table=dataframe_to_prompt_text(result_df, max_rows=30),
        user_request=user_request,
        analysis_summary=summary,
        df_columns=list(result_df.columns),
        df_dtypes=df_dtypes,
        model=model_choice,
    )
    st.write("Viz code generated. Executing...")

    viz_namespace = {"result_df": result_df, "file_path": tmp_path}
    viz_result = execute_code(viz_code, extra_globals=viz_namespace)

    retries = 0
    while not viz_result["success"] and retries < VIZ_MAX_RETRIES:
        retries += 1
        st.write(f"Execution failed (attempt {retries}/{VIZ_MAX_RETRIES}). Asking agent to fix...")
        viz_code, viz_messages = fix_viz_code(
            client=client,
            messages=viz_messages,
            failed_code=viz_code,
            error=viz_result["error"],
            model=model_choice,
        )
        viz_result = execute_code(viz_code, extra_globals=viz_namespace)

    if viz_result["success"]:
        status.update(label="Visualization complete!", state="complete")
    else:
        status.update(label="Visualization failed after retries", state="error")

# Show generated viz code
with st.expander("View generated visualization code", expanded=False):
    st.code(viz_code, language="python")

if not viz_result["success"]:
    st.error(f"Viz code failed after {VIZ_MAX_RETRIES} retries:\n\n{viz_result['error']}")
    st.stop()

# Render charts
figures = viz_result.get("figures", [])
if figures:
    for i, fig in enumerate(figures):
        st.plotly_chart(fig, use_container_width=True, key=f"chart_{i}")
else:
    st.warning("No figures were produced by the visualization code.")
