import pandas as pd
from pathlib import Path


def get_sheet_names(file_path: str) -> list[str]:
    """Return all sheet names in the Excel file."""
    xl = pd.ExcelFile(file_path)
    return xl.sheet_names


def inspect_sheet(file_path: str, sheet_name: str, sample_rows: int = 5) -> dict:
    """
    Return a rich schema description for one sheet:
      - shape
      - column names + dtypes
      - sample rows (as list of dicts)
      - basic null counts
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=200)

    col_info = []
    for col in df.columns:
        col_info.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isna().sum()),
            "sample_values": df[col].dropna().head(3).tolist(),
        })

    return {
        "sheet_name": sheet_name,
        "shape": df.shape,
        "columns": col_info,
        "sample_rows": df.head(sample_rows).to_dict(orient="records"),
    }


def inspect_all_sheets(file_path: str) -> dict:
    """Inspect every sheet in the workbook."""
    result = {}
    for sheet in get_sheet_names(file_path):
        result[sheet] = inspect_sheet(file_path, sheet)
    return result


def schema_to_prompt_text(inspection: dict) -> str:
    """
    Convert sheet inspection output into a compact, readable string
    suitable for injecting into an agent prompt.
    """
    lines = []
    sheet = inspection["sheet_name"]
    rows, cols = inspection["shape"]
    lines.append(f"Sheet: '{sheet}'  ({rows} rows × {cols} columns)")
    lines.append("Columns:")
    for c in inspection["columns"]:
        samples = ", ".join(str(v) for v in c["sample_values"])
        nulls = f"  [{c['null_count']} nulls]" if c["null_count"] else ""
        lines.append(f"  - {c['name']} ({c['dtype']}){nulls}  e.g. {samples}")
    lines.append("\nSample rows (first 5):")
    for i, row in enumerate(inspection["sample_rows"], 1):
        lines.append(f"  Row {i}: {row}")
    return "\n".join(lines)


def multi_schema_to_prompt_text(all_inspections: dict) -> str:
    """Convert inspections for multiple sheets into prompt text."""
    parts = []
    for sheet_name, inspection in all_inspections.items():
        parts.append(schema_to_prompt_text(inspection))
    return "\n\n---\n\n".join(parts)
