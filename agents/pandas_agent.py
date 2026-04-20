"""
Gemini-powered Data Scientist agent — code-generation approach.

Gemini receives ONLY schema metadata (column names, dtypes, shape, value ranges).
No row data is ever sent to the API.
Gemini writes pandas code; the code executes locally against the real DataFrames.
"""

from __future__ import annotations

import traceback
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import GEMINI_MODEL, PANDAS_AGENT_SYSTEM_PROMPT

# Execution sandbox — all analytics libs pre-injected so generated code
# never needs to call import (which is blocked by the restricted builtins).
_SAFE_GLOBALS: dict[str, Any] = {
    "__builtins__": {
        "len": len, "range": range, "enumerate": enumerate,
        "zip": zip, "list": list, "dict": dict, "set": set,
        "tuple": tuple, "str": str, "int": int, "float": float,
        "bool": bool, "round": round, "abs": abs, "min": min,
        "max": max, "sum": sum, "sorted": sorted, "print": print,
        "isinstance": isinstance, "hasattr": hasattr,
        "any": any, "all": all, "map": map, "filter": filter,
    },
    "pd": pd,
    "np": np,
    "px": px,
    "go": go,
    "make_subplots": make_subplots,
}


class PandasAnalysisAgent:
    """
    Schema-only Gemini agent.

    1. Builds a metadata-only prompt (no row values).
    2. Gemini returns executable pandas code.
    3. Code runs locally; result is captured and returned.
    """

    MODEL = GEMINI_MODEL

    def __init__(self, api_key: str):
        self.api_key = api_key

    # ── public interface ──────────────────────────────────────

    def analyze(
        self,
        query: str,
        performance_df: pd.DataFrame,
        alerts_df: pd.DataFrame,
        selected_profile: str = "",
        selected_model: str = "",
        valid_models: list[str] | None = None,
    ) -> dict:
        """
        Returns {success: bool, result: str, dataframe: pd.DataFrame | None}.

        selected_profile / selected_model / valid_models are passed as metadata
        only — they tell Gemini what the DataFrames already contain so it does
        NOT generate redundant or incorrect filter code.
        """
        try:
            code = self._generate_code(
                query, performance_df, alerts_df,
                selected_profile, selected_model, valid_models or [],
            )
            return self._execute_code(code, performance_df, alerts_df)
        except Exception:
            return {
                "success": False,
                "result": f"Agent error:\n```\n{traceback.format_exc()}\n```",
                "dataframe": None,
            }

    # ── code generation ───────────────────────────────────────

    def _generate_code(
        self,
        query: str,
        performance_df: pd.DataFrame,
        alerts_df: pd.DataFrame,
        selected_profile: str = "",
        selected_model: str = "",
        valid_models: list[str] | None = None,
    ) -> str:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)

        schema_prompt = self._build_schema_prompt(
            query, performance_df, alerts_df,
            selected_profile, selected_model, valid_models or [],
        )

        response = client.models.generate_content(
            model=self.MODEL,
            contents=schema_prompt,
            config=types.GenerateContentConfig(
                system_instruction=self._code_gen_instruction(query),
                temperature=0.0,
            ),
        )

        return self._extract_code(response.text)

    @staticmethod
    def _extract_code(text: str) -> str:
        """
        Pull only the Python code out of the model response.

        Priority:
        1. Content inside the first ```python … ``` fence.
        2. Content inside any ``` … ``` fence.
        3. The full response stripped of leading/trailing whitespace.
        """
        import re
        # Match ```python ... ``` (preferred)
        m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        # Match any ``` ... ```
        m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        # No fences — return as-is (Gemini obeyed the instruction)
        return text.strip()

    # ── local execution ───────────────────────────────────────

    def _execute_code(
        self,
        code: str,
        performance_df: pd.DataFrame,
        alerts_df: pd.DataFrame,
    ) -> dict:
        local_vars: dict[str, Any] = {
            "performance_df": performance_df.copy(),
            "alerts_df": alerts_df.copy(),
            "result": None,
            "result_df": None,
            "result_fig": None,   # assign a plotly Figure here to render a chart
        }
        sandbox = {**_SAFE_GLOBALS, **local_vars}

        # Capture stdout (print statements in generated code)
        import sys
        old_stdout = sys.stdout
        sys.stdout = captured = StringIO()

        try:
            exec(code, sandbox)  # noqa: S102
        except Exception as exc:
            sys.stdout = old_stdout
            # Return the code + error so the user can see what was attempted
            return {
                "success": False,
                "result": (
                    f"Generated code failed to execute:\n"
                    f"```python\n{code}\n```\n\n"
                    f"**Error:** `{exc}`"
                ),
                "dataframe": None,
            }
        finally:
            sys.stdout = old_stdout

        printed = captured.getvalue()
        result_val = sandbox.get("result")
        result_df  = sandbox.get("result_df")
        result_fig = sandbox.get("result_fig")

        # If result itself is a plotly figure, promote it
        if hasattr(result_val, "to_plotly_json"):
            result_fig = result_val
            result_val = None

        # Build the narrative output
        parts: list[str] = []

        if printed.strip():
            parts.append(printed.strip())

        if isinstance(result_val, pd.DataFrame):
            result_df = result_val
            parts.append(result_val.to_markdown(index=False))
        elif isinstance(result_val, pd.Series):
            parts.append(result_val.to_frame().to_markdown())
        elif result_val is not None:
            parts.append(str(result_val))

        if isinstance(result_df, pd.DataFrame) and result_df is not result_val:
            parts.append(result_df.to_markdown(index=False))

        # When a chart is the primary output suppress the raw table from the
        # narrative — the user asked for a plot, not a data dump.
        if result_fig is not None:
            parts = [p for p in parts if not p.startswith("|")]  # strip markdown tables
            if not parts:
                parts.append("Chart generated successfully.")

        narrative = "\n\n".join(parts) if parts else "Analysis complete — no output assigned to `result`, `result_df`, or `result_fig`."

        final_df = result_df if isinstance(result_df, pd.DataFrame) else \
                   (result_val if isinstance(result_val, pd.DataFrame) else None)

        return {
            "success": True,
            "result": narrative,
            "dataframe": final_df if result_fig is None else None,  # hide table when chart present
            "figure": result_fig,
        }

    # ── prompt / instruction helpers ──────────────────────────

    _PLOT_KEYWORDS = {
        "plot", "chart", "graph", "scatter", "bar", "line", "pie",
        "histogram", "heatmap", "visualize", "visualise", "show me a",
        "draw", "display chart", "display plot",
    }

    @classmethod
    def _code_gen_instruction(cls, query: str = "") -> str:
        q = query.lower()
        is_plot_query = any(kw in q for kw in cls._PLOT_KEYWORDS)

        chart_rule = (
            "- *** The user asked for a CHART/PLOT. You MUST create a Plotly figure "
            "and assign it to `result_fig`. Do NOT return a table as the primary output. ***\n"
            if is_plot_query else
            "- For charts assign a Plotly figure to `result_fig`.\n"
        )

        return (
            PANDAS_AGENT_SYSTEM_PROMPT
            + "\n\n"
            "You are a pandas code generator. Rules:\n"
            "- Output ONLY valid Python code — no markdown, no explanation text.\n"
            "- DO NOT write any import statements. All libraries are pre-injected:\n"
            "    pd (pandas), np (numpy), px (plotly.express),\n"
            "    go (plotly.graph_objects), make_subplots (plotly.subplots).\n"
            "- Two DataFrames are pre-loaded: `performance_df` and `alerts_df`.\n"
            "- Join key: performance_df['PIN'] == alerts_df['AssetID'].\n"
            "- Assign the final scalar/string/Series/DataFrame answer to `result`.\n"
            "- For tabular output also assign to `result_df` (a DataFrame).\n"
            + chart_rule +
            "- Never use matplotlib, plt, or any import statement.\n"
            "- Use print() for narrative text.\n"
        )

    @staticmethod
    def _build_schema_prompt(
        query: str,
        performance_df: pd.DataFrame,
        alerts_df: pd.DataFrame,
        selected_profile: str = "",
        selected_model: str = "",
        valid_models: list[str] | None = None,
    ) -> str:
        def schema_block(df: pd.DataFrame, name: str) -> str:
            lines = [f"### {name}"]
            lines.append(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
            lines.append("Columns (name | dtype | non-null count):")
            for col in df.columns:
                dtype = str(df[col].dtype)
                non_null = int(df[col].notna().sum())
                lines.append(f"  - {col!r:55s} {dtype:12s} non-null={non_null}")
            return "\n".join(lines)

        # Build active-filter context block — critical so Gemini does NOT
        # generate redundant or wrong filter code
        filter_lines = ["### Active Data Filter (already applied — DO NOT re-filter)"]
        filter_lines.append(f"Profile  : {selected_profile or 'All profiles'}")
        filter_lines.append(f"Model    : {selected_model or 'All valid models'}")
        if valid_models:
            filter_lines.append(f"Models in data: {', '.join(valid_models)}")
        filter_lines.append(
            "IMPORTANT: Both DataFrames contain ONLY rows matching these filters. "
            "Do not add .isin(), == model, or == profile filters — the data is already scoped."
        )
        filter_block = "\n".join(filter_lines)

        perf_block = schema_block(performance_df, "performance_df")
        alert_block = schema_block(alerts_df, "alerts_df")

        return (
            f"{filter_block}\n\n"
            f"{perf_block}\n\n"
            f"{alert_block}\n\n"
            f"---\n"
            f"User query: {query}\n\n"
            "Write pandas code that answers the query. "
            "Assign the final answer to `result` and (if tabular) to `result_df`."
        )
