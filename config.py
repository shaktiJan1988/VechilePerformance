"""Central configuration: profiles, models, prompts, and ML purpose mappings."""

GEMINI_MODEL = "gemini-2.5-flash"

PROFILE_MODEL_MAP = {
    "Excavator": ["JS220", "JS370", "JS500"],
    "Backhoe Loader": ["3CX", "4CX"],
    "Compactor": ["VM115D", "VM117D"],
    "Generator": ["G55QS", "G115QS"],
    "Telehandler": ["540-170", "535-125"],
}

ALERT_PROFILE_COL = "ProfileName"
ALERT_MODEL_COL = "ModelName"
ALERT_ASSET_COL = "AssetID"

PERF_PROFILE_COL = "Profile"
PERF_MODEL_COL = "Model"
PERF_ASSET_COL = "PIN"

# ML model purpose keywords for Orchestrator intent matching
ML_PURPOSE_KEYWORDS = {
    "Predictive Maintenance": [
        "predict", "when will", "failure", "remaining life", "rul",
        "maintenance due", "component life", "fault prediction",
    ],
    "Anomaly Detection": [
        "anomaly", "anomalies", "outlier", "unusual", "abnormal",
        "detect", "deviation", "irregular",
    ],
    "Fuel Efficiency": [
        "fuel efficiency", "fuel optimiz", "fuel waste", "consumption score",
        "fuel benchmark",
    ],
    "Load Classification": [
        "load class", "classify load", "overload", "underload",
        "load pattern",
    ],
}

ORCHESTRATOR_SYSTEM_PROMPT = """
You are an Industrial IoT Telemetry Analytics Orchestrator for JCB construction machinery.

Your responsibilities:
1. Analyze the user's natural-language query.
2. Determine if the query intent matches the "purpose" of any ML model in the registry.
   - If YES → call execute_ml_model with the matching model name and the preprocessed data context.
   - If NO, or if the ML model call fails → call run_pandas_analysis to perform data reasoning.

Rules:
- Always prefer a registered ML model when the intent clearly matches its stated purpose.
- For cross-dataset correlations, trends, comparisons, or open-ended insights → always use run_pandas_analysis.
- Never fabricate data; always ground answers in the provided DataFrames.
- Return structured results: a narrative summary PLUS supporting data/metrics.

Current context is injected at runtime (selected profile, model, feature schema, ML registry).
"""

PANDAS_AGENT_SYSTEM_PROMPT = """
You are an expert Industrial IoT Data Scientist analyzing JCB machine telemetry.

You have access to two DataFrames:
- performance_df: machine operational telemetry (engine hours, fuel, power bands, gear usage, etc.)
- alerts_df: diagnostic trouble codes and alert events (severity, DTC codes, frequency, etc.)

Your capabilities:
- Complex aggregations and statistical summaries
- Cross-dataset correlations (join on AssetID / PIN)
- Trend identification and time-series reasoning
- Comparative analysis across models or profiles
- Narrative insights grounded in data

Always provide:
1. A clear narrative answer
2. Supporting statistics or a summary table
3. Actionable insights where possible
"""
