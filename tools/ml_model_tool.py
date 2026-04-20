"""
Standardized wrapper for uploaded ML models.

Any model stored in the ML Registry must be callable through execute_model().
Supports sklearn-style (predict / predict_proba) and custom callable objects.
"""

from __future__ import annotations

import io
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd


@dataclass
class MLModelEntry:
    name: str
    purpose: str
    target_model: str
    model_obj: Any
    feature_columns: list[str] = field(default_factory=list)
    description: str = ""


class MLModelTool:
    """
    Wraps a registered ML model entry and exposes a standardised execute() interface
    the Orchestrator can call after preprocessing the DataFrames.
    """

    def __init__(self, entry: MLModelEntry):
        self.entry = entry

    def execute(
        self,
        performance_df: pd.DataFrame,
        alerts_df: pd.DataFrame,
        query: str,
    ) -> dict:
        """
        Run the ML model on the pre-filtered DataFrames.

        Returns a dict with keys:
            success (bool), result (str), metrics (dict), dataframe (pd.DataFrame | None)
        """
        try:
            model = self.entry.model_obj
            feature_cols = self.entry.feature_columns

            # Build the input DataFrame
            input_df = self._build_input(performance_df, feature_cols)

            if input_df.empty:
                return self._error("Input DataFrame is empty after feature selection.")

            # Dispatch based on sklearn interface
            if hasattr(model, "predict_proba"):
                raw = model.predict_proba(input_df)
                labels = getattr(model, "classes_", list(range(raw.shape[1])))
                result_df = pd.DataFrame(raw, columns=[f"prob_{l}" for l in labels])
                result_df.insert(0, "PIN", performance_df["PIN"].values[: len(result_df)])
                narrative = self._narrate_classification(result_df, labels)

            elif hasattr(model, "predict"):
                raw = model.predict(input_df)
                result_df = pd.DataFrame({"PIN": performance_df["PIN"].values[: len(raw)], "prediction": raw})
                narrative = self._narrate_regression(result_df, self.entry.purpose)

            elif callable(model):
                # Custom callable: receives (performance_df, alerts_df, query) → dict or str
                out = model(performance_df, alerts_df, query)
                if isinstance(out, dict):
                    return {
                        "success": True,
                        "result": str(out.get("result", out)),
                        "metrics": out.get("metrics", {}),
                        "dataframe": out.get("dataframe"),
                    }
                return {"success": True, "result": str(out), "metrics": {}, "dataframe": None}

            else:
                return self._error("Model object has no recognisable predict interface.")

            return {
                "success": True,
                "result": narrative,
                "metrics": self._summary_metrics(result_df),
                "dataframe": result_df,
            }

        except Exception:
            return self._error(traceback.format_exc())

    # ── helpers ──────────────────────────────────────────────

    def _build_input(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        if not feature_cols:
            # Use all numeric columns as default
            return df.select_dtypes(include=[np.number]).fillna(0)
        available = [c for c in feature_cols if c in df.columns]
        return df[available].select_dtypes(include=[np.number]).fillna(0)

    def _narrate_classification(self, result_df: pd.DataFrame, labels) -> str:
        dominant = result_df.drop(columns=["PIN"]).idxmax(axis=1).value_counts()
        lines = [f"**{self.entry.purpose} — Classification Results**"]
        for label, count in dominant.items():
            lines.append(f"- {label}: {count} machines ({count/len(result_df)*100:.1f}%)")
        return "\n".join(lines)

    def _narrate_regression(self, result_df: pd.DataFrame, purpose: str) -> str:
        preds = result_df["prediction"]
        return (
            f"**{purpose} — Regression Output**\n"
            f"- Mean: {preds.mean():.2f} | Min: {preds.min():.2f} | Max: {preds.max():.2f}\n"
            f"- Machines analysed: {len(preds)}"
        )

    def _summary_metrics(self, df: pd.DataFrame) -> dict:
        numeric = df.select_dtypes(include=[np.number])
        return {col: {"mean": round(numeric[col].mean(), 3)} for col in numeric.columns}

    @staticmethod
    def _error(msg: str) -> dict:
        return {"success": False, "result": f"ML model execution failed:\n{msg}", "metrics": {}, "dataframe": None}


def load_model_from_bytes(file_bytes: bytes) -> Any:
    """Deserialize a joblib/pickle model from raw bytes."""
    return joblib.load(io.BytesIO(file_bytes))
