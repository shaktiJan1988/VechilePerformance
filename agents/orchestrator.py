"""
IIoT Orchestrator — ADK-style routing via Gemini function calling (google-genai SDK).

Flow:
  User Query
      │
      ├─ keyword fast-path: intent matches ML Registry  → execute_ml_model()
      └─ Gemini function-calling router
              ├─ execute_ml_model  → MLModelTool.execute()
              └─ run_pandas_analysis → PandasAnalysisAgent.analyze()
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from config import GEMINI_MODEL, ML_PURPOSE_KEYWORDS, ORCHESTRATOR_SYSTEM_PROMPT
from tools.ml_model_tool import MLModelEntry, MLModelTool


# ──────────────────────────────────────────────────────────────
# Shared context — populated by the Streamlit app before each call
# ──────────────────────────────────────────────────────────────

@dataclass
class OrchestratorContext:
    performance_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    alerts_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    ml_registry: list[MLModelEntry] = field(default_factory=list)
    pandas_agent: Any = None
    selected_profile: str = ""
    selected_model: str = ""


# ──────────────────────────────────────────────────────────────
# Tool schemas for Gemini function calling
# ──────────────────────────────────────────────────────────────

_EXECUTE_ML_DECL = {
    "name": "execute_ml_model",
    "description": (
        "Execute a registered ML model. Use when the query clearly matches a "
        "model's stated purpose (Predictive Maintenance, Anomaly Detection, etc.)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "model_name": {"type": "string", "description": "Exact registered model name."},
            "reasoning": {"type": "string", "description": "Why this model was selected."},
        },
        "required": ["model_name", "reasoning"],
    },
}

_RUN_PANDAS_DECL = {
    "name": "run_pandas_analysis",
    "description": (
        "Default analysis tool for all analytical, comparative, trend, or "
        "cross-dataset correlation queries."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The user query to analyse."},
        },
        "required": ["query"],
    },
}


# ──────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────

class IIoTOrchestrator:
    """
    Gemini function-calling orchestrator — same routing behaviour as Google ADK,
    implemented with google-genai to avoid Windows MAX_PATH issues from
    google-cloud-aiplatform.
    """

    ROUTER_MODEL = GEMINI_MODEL

    def __init__(self, api_key: str):
        self.api_key = api_key

    # ── public entry point ────────────────────────────────────

    def run(self, query: str, context: OrchestratorContext) -> dict:
        """
        Returns {success, result, source, model_used?, dataframe_md?}
        """
        # Fast keyword path — skips LLM round-trip for obvious intent matches
        matched = self._keyword_match(query, context.ml_registry)
        if matched:
            return self._call_ml_model(matched.name, query, context)

        return self._route_via_gemini(query, context)

    # ── Gemini function-calling router ────────────────────────

    def _route_via_gemini(self, query: str, context: OrchestratorContext) -> dict:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)

            tools = types.Tool(function_declarations=[
                types.FunctionDeclaration(**_EXECUTE_ML_DECL),
                types.FunctionDeclaration(**_RUN_PANDAS_DECL),
            ])

            response = client.models.generate_content(
                model=self.ROUTER_MODEL,
                contents=self._enrich_query(query, context),
                config=types.GenerateContentConfig(
                    system_instruction=self._build_system_prompt(context),
                    tools=[tools],
                    tool_config=types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(mode="ANY")
                    ),
                    temperature=0.0,
                ),
            )

            for part in response.candidates[0].content.parts:
                if part.function_call:
                    name = part.function_call.name
                    args = dict(part.function_call.args)
                    if name == "execute_ml_model":
                        return self._call_ml_model(args["model_name"], query, context)
                    if name == "run_pandas_analysis":
                        return self._call_pandas(args.get("query", query), context)

            # No function call returned — fall back
            return self._call_pandas(query, context)

        except Exception:
            return self._call_pandas(query, context)

    # ── tool executors ────────────────────────────────────────

    def _call_ml_model(
        self, model_name: str, query: str, context: OrchestratorContext
    ) -> dict:
        entry = next((e for e in context.ml_registry if e.name == model_name), None)
        if entry is None:
            return self._call_pandas(query, context)

        out = MLModelTool(entry).execute(context.performance_df, context.alerts_df, query)
        if not out["success"]:
            return self._call_pandas(query, context)

        result: dict = {"success": True, "result": out["result"], "source": "ml_model", "model_used": model_name}
        if out.get("dataframe") is not None:
            result["dataframe_md"] = out["dataframe"].head(20).to_markdown(index=False)
        return result

    def _call_pandas(self, query: str, context: OrchestratorContext) -> dict:
        if context.pandas_agent is None:
            return {"success": False, "result": "Analysis agent not initialised.", "source": "error"}

        # Derive the list of models actually present in the filtered DataFrame
        from config import PERF_MODEL_COL
        valid_models_in_data = (
            context.performance_df[PERF_MODEL_COL].unique().tolist()
            if PERF_MODEL_COL in context.performance_df.columns
            else []
        )

        out = context.pandas_agent.analyze(
            query,
            context.performance_df,
            context.alerts_df,
            selected_profile=context.selected_profile,
            selected_model=context.selected_model,
            valid_models=valid_models_in_data,
        )
        result: dict = {"success": out.get("success", False), "result": out.get("result", ""), "source": "pandas_ai"}
        if out.get("dataframe") is not None:
            result["dataframe_md"] = out["dataframe"].head(20).to_markdown(index=False)
        if out.get("figure") is not None:
            result["figure"] = out["figure"]
        return result

    # ── helpers ───────────────────────────────────────────────

    @staticmethod
    def _keyword_match(query: str, registry: list[MLModelEntry]) -> Optional[MLModelEntry]:
        q = query.lower()
        for entry in registry:
            keywords = ML_PURPOSE_KEYWORDS.get(entry.purpose, [])
            if any(kw in q for kw in keywords) or entry.purpose.lower() in q:
                return entry
        return None

    @staticmethod
    def _build_system_prompt(context: OrchestratorContext) -> str:
        registry_lines = (
            "\n".join(f"  • {e.name} | Purpose: {e.purpose} | Target: {e.target_model}"
                      for e in context.ml_registry)
            if context.ml_registry else "  (none registered)"
        )
        return (
            ORCHESTRATOR_SYSTEM_PROMPT
            + f"\n\nRegistered ML Models:\n{registry_lines}"
            + f"\n\nActive context — Profile: {context.selected_profile or 'All'}, "
            + f"Model: {context.selected_model or 'All'}, "
            + f"Performance rows: {len(context.performance_df)}, "
            + f"Alert rows: {len(context.alerts_df)}"
        )

    @staticmethod
    def _enrich_query(query: str, context: OrchestratorContext) -> str:
        return f"{query}\n\n[Select the most appropriate tool based on the ML registry and query intent.]"
