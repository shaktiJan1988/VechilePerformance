"""
JCB IIoT Telemetry Analytics POC
Multi-agent Streamlit app (Google ADK + PandasAI)
"""

from __future__ import annotations

import io
import json
import os
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from agents.orchestrator import IIoTOrchestrator, OrchestratorContext
from agents.pandas_agent import PandasAnalysisAgent
from config import ALERT_MODEL_COL, PERF_MODEL_COL, PERF_PROFILE_COL, PROFILE_MODEL_MAP
from tools.ml_model_tool import MLModelEntry, load_model_from_bytes
from utils.data_processor import (
    filter_dataframes,
    generate_alerts_data,
    generate_performance_data,
    parse_feature_file,
)

load_dotenv()

# ──────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="JCB IIoT Analytics",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# Session state initialisation
# ──────────────────────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "performance_df": None,
        "alerts_df": None,
        "feature_files": {},        # {model_name: {"columns": [...], "filename": str}}
        "ml_registry": [],          # list[MLModelEntry]
        "chat_history": [],         # [{"role": str, "content": str, "source": str}]
        "chat_figures": [],         # parallel list — plotly Figure or None per message
        "orchestrator": None,
        "pandas_agent": None,
        "api_key": os.getenv("GOOGLE_API_KEY", ""),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ──────────────────────────────────────────────────────────────
# Sidebar – API key + data load
# ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/en/thumb/5/5e/JCB_logo.svg/320px-JCB_logo.svg.png",
        width=120,
    )
    st.title("JCB IIoT Analytics")
    st.caption("Multi-Agent Telemetry POC")
    st.divider()

    if not st.session_state.api_key:
        st.error("GOOGLE_API_KEY not found. Add it to your .env file.", icon="🔑")

    st.divider()
    st.subheader("Data Source")

    data_source = st.radio(
        "Load data from",
        ["Upload CSV files", "Generate sample data"],
        horizontal=True,
    )

    if data_source == "Upload CSV files":
        perf_file = st.file_uploader(
            "Performance CSV",
            type=["csv"],
            key="perf_csv",
            help="Must contain columns: PIN, Profile, Model",
        )
        alert_file = st.file_uploader(
            "Alerts CSV",
            type=["csv"],
            key="alert_csv",
            help="Must contain columns: AssetID, ProfileName, ModelName",
        )

        if perf_file and alert_file:
            if st.button("📂 Load CSV Files", use_container_width=True, type="primary"):
                try:
                    st.session_state.performance_df = pd.read_csv(perf_file)
                    st.session_state.alerts_df = pd.read_csv(alert_file)
                    st.success(
                        f"Loaded — {len(st.session_state.performance_df):,} performance rows, "
                        f"{len(st.session_state.alerts_df):,} alert rows."
                    )
                except Exception as e:
                    st.error(f"Failed to load CSVs: {e}")
        elif perf_file or alert_file:
            st.warning("Upload both CSV files to continue.")

    else:
        if st.button("🔄 Generate Sample Data", use_container_width=True):
            with st.spinner("Generating sample telemetry dataset…"):
                perf = generate_performance_data(300)
                st.session_state.performance_df = perf
                st.session_state.alerts_df = generate_alerts_data(perf, alerts_per_pin=5)
            st.success(
                f"Generated — {len(st.session_state.performance_df):,} performance rows, "
                f"{len(st.session_state.alerts_df):,} alert rows."
            )

    if st.session_state.performance_df is not None:
        st.caption(
            f"📊 Performance: {len(st.session_state.performance_df):,} rows  \n"
            f"🚨 Alerts: {len(st.session_state.alerts_df):,} rows"
        )
        # Download generated / loaded data as CSV
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "⬇️ Performance CSV",
                data=st.session_state.performance_df.to_csv(index=False).encode(),
                file_name="performance_data.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_dl2:
            st.download_button(
                "⬇️ Alerts CSV",
                data=st.session_state.alerts_df.to_csv(index=False).encode(),
                file_name="alerts_data.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.divider()
    st.caption("Feature Files Registered")
    if st.session_state.feature_files:
        for model, info in st.session_state.feature_files.items():
            st.markdown(f"- **{model}** — {len(info['columns'])} features")
    else:
        st.caption("None yet — upload in Tab 2")

    st.divider()
    st.caption("ML Model Registry")
    if st.session_state.ml_registry:
        for entry in st.session_state.ml_registry:
            st.markdown(f"- **{entry.name}** ({entry.purpose})")
    else:
        st.caption("None yet — upload in Tab 3")


# ──────────────────────────────────────────────────────────────
# Agent initialisation helper
# ──────────────────────────────────────────────────────────────

def _ensure_agents() -> bool:
    if not st.session_state.api_key:
        st.error("⚠️ GOOGLE_API_KEY missing. Add it to your .env file and restart the app.")
        return False
    if st.session_state.orchestrator is None:
        with st.spinner("Initialising agents…"):
            st.session_state.pandas_agent = PandasAnalysisAgent(st.session_state.api_key)
            st.session_state.orchestrator = IIoTOrchestrator(st.session_state.api_key)
    return True


# ──────────────────────────────────────────────────────────────
# Main tabs
# ──────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["💬 Query Interface", "🔧 Feature Engineering", "🤖 ML Model Registry"])


# ══════════════════════════════════════════════════════════════
# TAB 1 — Query Interface
# ══════════════════════════════════════════════════════════════

with tab1:
    st.header("Telemetry Query Interface")

    if st.session_state.performance_df is None:
        st.info("👈 Click **Load / Refresh Sample Data** in the sidebar to begin.")
        st.stop()

    # ── Profile / Model selectors ─────────────────────────────
    col_profile, col_model = st.columns(2)

    with col_profile:
        _perf_data = st.session_state.performance_df
        if PERF_PROFILE_COL in _perf_data.columns:
            all_profiles = sorted(_perf_data[PERF_PROFILE_COL].dropna().unique().tolist())
        else:
            all_profiles = list(PROFILE_MODEL_MAP.keys())
        selected_profile = st.selectbox("Vehicle Profile", ["All Profiles"] + all_profiles)
        profile_val = None if selected_profile == "All Profiles" else selected_profile

    with col_model:
        _perf_data = st.session_state.performance_df
        if PERF_MODEL_COL in _perf_data.columns:
            if profile_val and PERF_PROFILE_COL in _perf_data.columns:
                models_in_data = sorted(
                    _perf_data[_perf_data[PERF_PROFILE_COL] == profile_val][PERF_MODEL_COL]
                    .dropna().unique().tolist()
                )
            else:
                models_in_data = sorted(_perf_data[PERF_MODEL_COL].dropna().unique().tolist())
        else:
            models_in_data = []

        valid_models = [m for m in models_in_data if m in st.session_state.feature_files]
        model_options = ["All Valid Models"] + valid_models
        if not valid_models:
            st.warning(
                f"No feature files uploaded for {'**' + profile_val + '** ' if profile_val else ''}models yet.  \n"
                "Upload feature files in Tab 2 to enable model-level filtering."
            )

        # Key changes whenever valid_models changes → forces widget recreation
        model_select_key = "model_sel_" + "_".join(sorted(valid_models))
        selected_model_opt = st.selectbox("Machine Model", model_options, key=model_select_key)
        model_val = None if selected_model_opt == "All Valid Models" else selected_model_opt

    # Feature columns for a specific selected model
    feature_columns: Optional[list[str]] = None
    if model_val and model_val in st.session_state.feature_files:
        feature_columns = st.session_state.feature_files[model_val]["columns"]
        st.caption(f"🔍 Schema filter active: **{len(feature_columns)}** features from feature file.")

    # Always restrict data to feature-validated models only.
    # When a specific model is chosen, valid_models = [model_val];
    # when "All Valid Models" is chosen, valid_models already contains only
    # those models that have a feature file registered in Tab 2.
    perf_base = st.session_state.performance_df
    alerts_base = st.session_state.alerts_df

    if valid_models:
        if PERF_MODEL_COL in perf_base.columns:
            perf_base = perf_base[perf_base[PERF_MODEL_COL].isin(valid_models)]
        if ALERT_MODEL_COL in alerts_base.columns:
            alerts_base = alerts_base[alerts_base[ALERT_MODEL_COL].isin(valid_models)]

    perf_filtered, alerts_filtered = filter_dataframes(
        perf_base,
        alerts_base,
        profile_val,
        model_val,
        feature_columns,
    )

    col_info1, col_info2 = st.columns(2)
    col_info1.metric("Performance Rows (filtered)", len(perf_filtered))
    col_info2.metric("Alert Rows (filtered)", len(alerts_filtered))

    # Hard stop — agent cannot answer if there is nothing to analyse
    if perf_filtered.empty and alerts_filtered.empty:
        st.error(
            "⚠️ No data matches the current filters. "
            "Check that your CSV columns match the expected schema "
            "(`Profile`, `Model`, `PIN` for performance; `ProfileName`, `ModelName`, `AssetID` for alerts), "
            "or select a different Profile / Model combination."
        )
        with st.expander("🔍 Debug — raw data shapes"):
            st.write(f"Full performance_df: {st.session_state.performance_df.shape}")
            st.write(f"Full alerts_df:      {st.session_state.alerts_df.shape}")
            st.write("Performance columns:", list(st.session_state.performance_df.columns[:10]))
            st.write("Alerts columns:     ", list(st.session_state.alerts_df.columns))
        st.stop()

    st.divider()

    # ── Chat history header + clear button ───────────────────
    _ch_col1, _ch_col2 = st.columns([6, 1])
    _ch_col1.subheader("Conversation")
    if st.session_state.chat_history:
        if _ch_col2.button("🗑️ Clear", help="Clear conversation history"):
            st.session_state.chat_history = []
            st.session_state.chat_figures = []
            st.rerun()

    chat_container = st.container()
    with chat_container:
        for idx, msg in enumerate(st.session_state.chat_history):
            role = msg["role"]
            avatar = "🏗️" if role == "assistant" else "👤"
            fig = (st.session_state.chat_figures[idx]
                   if idx < len(st.session_state.chat_figures) else None)
            with st.chat_message(role, avatar=avatar):
                st.markdown(msg["content"])
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                if msg.get("source"):
                    badge = "🤖 ML Model" if msg["source"] == "ml_model" else "🧠 Gemini Analysis"
                    st.caption(f"Source: {badge}")
                if msg.get("dataframe_md"):
                    with st.expander("📋 Data Table"):
                        st.markdown(msg["dataframe_md"])

    # ── Query input ───────────────────────────────────────────
    st.divider()

    # Example queries
    with st.expander("💡 Example Queries"):
        examples = [
            "What is the average fuel consumption for the selected profile?",
            "Compare the duty cycle of models with high-frequency hydraulic alerts vs those without.",
            "Which machines have the highest idle time percentage?",
            "Show the distribution of alert severities across engine manufacturers.",
            "Predict which machines need maintenance soon.",
            "Detect anomalies in hydraulic pressure bands.",
            "What is the correlation between power band utilisation and fuel consumption?",
            "List top 10 machines by carbon emission in the last period.",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex[:20]}", use_container_width=False):
                st.session_state["_prefill_query"] = ex

    prefill = st.session_state.pop("_prefill_query", "")
    user_query = st.chat_input("Ask a telemetry question…", key="chat_input")
    if prefill and not user_query:
        user_query = prefill

    if user_query:
        if not _ensure_agents():
            st.stop()

        if perf_filtered.empty:
            st.warning("Performance data is empty for the current filter — cannot run analysis.")
            st.stop()

        with st.spinner("Orchestrating analysis…"):
            ctx = OrchestratorContext(
                performance_df=perf_filtered,
                alerts_df=alerts_filtered,
                ml_registry=st.session_state.ml_registry,
                pandas_agent=st.session_state.pandas_agent,
                selected_profile=profile_val or "",
                selected_model=model_val or "",
            )
            result = st.session_state.orchestrator.run(user_query, ctx)

        source = result.get("source", "")
        figure = result.get("figure")

        # Store user message + figure-less assistant msg first
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.chat_figures.append(None)

        assistant_msg = {
            "role": "assistant",
            "content": result.get("result", ""),
            "source": source,
            "dataframe_md": result.get("dataframe_md"),
        }
        st.session_state.chat_history.append(assistant_msg)
        st.session_state.chat_figures.append(figure)   # may be None or a Figure
        st.rerun()

    # ── Data preview ──────────────────────────────────────────
    with st.expander("🗃️ Filtered Data Preview"):
        preview_tab1, preview_tab2 = st.tabs(["Performance", "Alerts"])
        with preview_tab1:
            st.dataframe(perf_filtered.head(50), use_container_width=True)
        with preview_tab2:
            st.dataframe(alerts_filtered.head(50), use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — Feature Engineering
# ══════════════════════════════════════════════════════════════

with tab2:
    st.header("Feature Engineering — Schema Registry")
    st.markdown(
        """
        Upload a **Feature File** (CSV or Excel) for each machine model.
        The file's column names define the allowed feature schema.
        Only columns present in the feature file will be passed to agents.

        _A model is considered "validated" and appears in the Tab 1 Model dropdown
        only after its feature file has been uploaded here._
        """
    )
    # ── Sample feature file downloads (generated from loaded data) ──
    with st.expander("⬇️ Download Feature File Templates (all columns)"):
        st.caption(
            "One template per model — contains every column from the loaded dataset. "
            "Edit to keep only the features you need, then upload below."
        )
        _perf_tmpl = st.session_state.performance_df
        if _perf_tmpl is None:
            st.info("Load data in the sidebar first to generate templates.")
        else:
            _all_cols = list(_perf_tmpl.columns) + [
                c for c in st.session_state.alerts_df.columns
                if c not in _perf_tmpl.columns
            ]
            _template_csv = pd.DataFrame(columns=_all_cols).to_csv(index=False).encode()
            _tmpl_models = (
                sorted(_perf_tmpl[PERF_MODEL_COL].dropna().unique().tolist())
                if PERF_MODEL_COL in _perf_tmpl.columns else []
            )
            _cols_per_row = 4
            for _i in range(0, len(_tmpl_models), _cols_per_row):
                _row_models = _tmpl_models[_i:_i + _cols_per_row]
                _btn_cols = st.columns(len(_row_models))
                for _col, _m in zip(_btn_cols, _row_models):
                    _col.download_button(
                        f"📄 {_m}",
                        data=_template_csv,
                        file_name=f"{_m}_features.csv",
                        mime="text/csv",
                        key=f"dl_tmpl_{_m}",
                        use_container_width=True,
                    )

    st.divider()

    # ── Upload section ────────────────────────────────────────
    col_prof_sel, col_model_sel, col_uploader = st.columns([1, 1, 2])

    _perf_tab2 = st.session_state.performance_df

    with col_prof_sel:
        if _perf_tab2 is not None and PERF_PROFILE_COL in _perf_tab2.columns:
            _tab2_profiles = sorted(_perf_tab2[PERF_PROFILE_COL].dropna().unique().tolist())
        else:
            _tab2_profiles = list(PROFILE_MODEL_MAP.keys())
        target_profile_t2 = st.selectbox("Select Profile", ["All Profiles"] + _tab2_profiles, key="fe_profile_sel")
        _profile_filter_t2 = None if target_profile_t2 == "All Profiles" else target_profile_t2

    with col_model_sel:
        if _perf_tab2 is not None and PERF_MODEL_COL in _perf_tab2.columns:
            if _profile_filter_t2 and PERF_PROFILE_COL in _perf_tab2.columns:
                all_models_flat = sorted(
                    _perf_tab2[_perf_tab2[PERF_PROFILE_COL] == _profile_filter_t2][PERF_MODEL_COL]
                    .dropna().unique().tolist()
                )
            else:
                all_models_flat = sorted(_perf_tab2[PERF_MODEL_COL].dropna().unique().tolist())
        else:
            all_models_flat = [m for models in PROFILE_MODEL_MAP.values() for m in models]
        target_model = st.selectbox("Select Machine Model", all_models_flat, key="fe_model_sel")

    with col_uploader:
        feat_file = st.file_uploader(
            f"Upload Feature File for **{target_model}**",
            type=["csv", "xlsx", "xls"],
            key="feat_uploader",
            help="File should have column names as header row. Can be header-only or with sample data.",
        )

    if feat_file is not None:
        try:
            columns = parse_feature_file(feat_file)
            st.success(f"✅ Parsed **{len(columns)}** feature columns for **{target_model}**.")
            st.write("**Detected columns:**")
            st.code(", ".join(columns))

            # Resolve which profile target_model belongs to (from loaded data)
            _perf_tab2 = st.session_state.performance_df
            _model_profile: str = ""
            _profile_siblings: list[str] = []
            if (
                _perf_tab2 is not None
                and PERF_MODEL_COL in _perf_tab2.columns
                and PERF_PROFILE_COL in _perf_tab2.columns
            ):
                _m2p = dict(zip(_perf_tab2[PERF_MODEL_COL], _perf_tab2[PERF_PROFILE_COL]))
                _model_profile = _m2p.get(target_model, "")
                if _model_profile:
                    _profile_siblings = sorted(
                        _perf_tab2[_perf_tab2[PERF_PROFILE_COL] == _model_profile][PERF_MODEL_COL]
                        .dropna().unique().tolist()
                    )

            reg_col1, reg_col2 = st.columns(2)
            with reg_col1:
                if st.button(f"💾 Register for {target_model}", type="primary", use_container_width=True):
                    st.session_state.feature_files[target_model] = {
                        "columns": columns,
                        "filename": feat_file.name,
                    }
                    st.success(f"Feature schema for **{target_model}** registered.")
                    st.rerun()
            with reg_col2:
                _profile_label = _model_profile or "same profile"
                _siblings_count = len(_profile_siblings)
                if _siblings_count > 1 and st.button(
                    f"💾 Register for all {_profile_label} models ({_siblings_count})",
                    type="secondary",
                    use_container_width=True,
                    help=f"Applies to: {', '.join(_profile_siblings)}",
                ):
                    for _sibling in _profile_siblings:
                        st.session_state.feature_files[_sibling] = {
                            "columns": columns,
                            "filename": feat_file.name,
                        }
                    st.success(
                        f"Feature schema registered for all **{_profile_label}** models: "
                        + ", ".join(f"**{m}**" for m in _profile_siblings)
                    )
                    st.rerun()
        except Exception as e:
            st.error(f"Failed to parse file: {e}")

    st.divider()

    # ── Registered feature files ──────────────────────────────
    st.subheader("Registered Feature Files")

    if not st.session_state.feature_files:
        st.info("No feature files registered yet.")
    else:
        for model_name, info in st.session_state.feature_files.items():
            with st.expander(f"📄 **{model_name}** — {info['filename']} ({len(info['columns'])} features)"):
                cols_df = pd.DataFrame({"Column Name": info["columns"]})
                # Check coverage against master DataFrames
                if st.session_state.performance_df is not None:
                    perf_cols = set(st.session_state.performance_df.columns)
                    alert_cols = set(st.session_state.alerts_df.columns)
                    cols_df["In Performance DF"] = cols_df["Column Name"].apply(
                        lambda c: "✅" if c in perf_cols else "—"
                    )
                    cols_df["In Alerts DF"] = cols_df["Column Name"].apply(
                        lambda c: "✅" if c in alert_cols else "—"
                    )
                st.dataframe(cols_df, use_container_width=True, hide_index=True)

                if st.button(f"🗑️ Remove {model_name}", key=f"rm_{model_name}"):
                    del st.session_state.feature_files[model_name]
                    st.rerun()

    st.divider()

    # ── Quick-create helper ───────────────────────────────────
    st.subheader("Quick Feature File Creator")
    st.caption(
        "Select columns from the master dataset to generate and download a feature file template."
    )

    if st.session_state.performance_df is not None:
        all_master_cols = (
            list(st.session_state.performance_df.columns)
            + [c for c in st.session_state.alerts_df.columns
               if c not in st.session_state.performance_df.columns]
        )
        selected_cols = st.multiselect("Select features to include", all_master_cols)
        qc_model = st.selectbox("For model", all_models_flat, key="qc_model")

        if selected_cols:
            template_df = pd.DataFrame(columns=selected_cols)
            csv_bytes = template_df.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download Feature File Template",
                data=csv_bytes,
                file_name=f"{qc_model}_features.csv",
                mime="text/csv",
            )

            # Resolve profile siblings for quick-creator model
            _qc_profile: str = ""
            _qc_siblings: list[str] = []
            _perf_qc = st.session_state.performance_df
            if (
                _perf_qc is not None
                and PERF_MODEL_COL in _perf_qc.columns
                and PERF_PROFILE_COL in _perf_qc.columns
            ):
                _qc_m2p = dict(zip(_perf_qc[PERF_MODEL_COL], _perf_qc[PERF_PROFILE_COL]))
                _qc_profile = _qc_m2p.get(qc_model, "")
                if _qc_profile:
                    _qc_siblings = sorted(
                        _perf_qc[_perf_qc[PERF_PROFILE_COL] == _qc_profile][PERF_MODEL_COL]
                        .dropna().unique().tolist()
                    )

            qc_col1, qc_col2 = st.columns(2)
            with qc_col1:
                if st.button("✅ Register for Selected Model", type="primary", use_container_width=True):
                    st.session_state.feature_files[qc_model] = {
                        "columns": selected_cols,
                        "filename": f"{qc_model}_features.csv (quick-created)",
                    }
                    st.success(f"Feature schema for **{qc_model}** registered.")
                    st.rerun()
            with qc_col2:
                _qc_label = _qc_profile or "same profile"
                if len(_qc_siblings) > 1 and st.button(
                    f"✅ Register for all {_qc_label} models ({len(_qc_siblings)})",
                    type="secondary",
                    use_container_width=True,
                    help=f"Applies to: {', '.join(_qc_siblings)}",
                ):
                    for _qs in _qc_siblings:
                        st.session_state.feature_files[_qs] = {
                            "columns": selected_cols,
                            "filename": f"{_qs}_features.csv (quick-created)",
                        }
                    st.success(
                        f"Feature schema registered for all **{_qc_label}** models: "
                        + ", ".join(f"**{m}**" for m in _qc_siblings)
                    )
                    st.rerun()
    else:
        st.info("Load sample data first (sidebar) to use the Quick Creator.")


# ══════════════════════════════════════════════════════════════
# TAB 3 — ML Model Registry
# ══════════════════════════════════════════════════════════════

with tab3:
    st.header("ML Model Registry")
    st.markdown(
        """
        Register pre-trained ML models (scikit-learn, custom callables serialised with **joblib**).
        The Orchestrator will automatically route queries that match a model's stated **Purpose**
        to execute that model on the pre-processed telemetry data.
        """
    )
    st.divider()

    # ── Registration form ─────────────────────────────────────
    with st.form("ml_model_form", clear_on_submit=True):
        st.subheader("Register a New ML Model")

        col_name, col_purpose = st.columns(2)
        with col_name:
            ml_name = st.text_input("Model Name", placeholder="e.g. JS220_PredMaint_v1")
        with col_purpose:
            ml_purpose = st.selectbox(
                "Purpose",
                ["Predictive Maintenance", "Anomaly Detection", "Fuel Efficiency", "Load Classification", "Other"],
            )

        col_target, col_desc = st.columns(2)
        with col_target:
            _perf_tab3 = st.session_state.performance_df
            if _perf_tab3 is not None and PERF_MODEL_COL in _perf_tab3.columns:
                all_models_flat2 = ["Any"] + sorted(_perf_tab3[PERF_MODEL_COL].dropna().unique().tolist())
            else:
                all_models_flat2 = ["Any"] + [m for ms in PROFILE_MODEL_MAP.values() for m in ms]
            ml_target = st.selectbox("Target Vehicle Model", all_models_flat2)
        with col_desc:
            ml_desc = st.text_input("Description (optional)", placeholder="Brief description")

        ml_features = st.text_area(
            "Feature Columns (comma-separated, optional)",
            placeholder="Fuel Used(in Liters), Idle Time(Period in hrs), Hydraulic Choke Events, …",
            help="Columns the model expects as input. Leave blank to use all numeric columns.",
        )

        ml_file = st.file_uploader(
            "Upload Model File (.joblib or .pkl)",
            type=["joblib", "pkl", "pickle"],
            help="Serialised sklearn-compatible model or custom callable.",
        )

        submitted = st.form_submit_button("Register Model", type="primary")

        if submitted:
            if not ml_name:
                st.error("Model name is required.")
            elif ml_file is None:
                st.error("Please upload a model file.")
            else:
                try:
                    model_bytes = ml_file.read()
                    model_obj = load_model_from_bytes(model_bytes)
                    feature_cols = (
                        [c.strip() for c in ml_features.split(",") if c.strip()]
                        if ml_features
                        else []
                    )
                    entry = MLModelEntry(
                        name=ml_name,
                        purpose=ml_purpose,
                        target_model=ml_target,
                        model_obj=model_obj,
                        feature_columns=feature_cols,
                        description=ml_desc,
                    )
                    # Replace if same name already exists
                    st.session_state.ml_registry = [
                        e for e in st.session_state.ml_registry if e.name != ml_name
                    ]
                    st.session_state.ml_registry.append(entry)
                    st.success(f"✅ Model **{ml_name}** registered successfully.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load model: {e}")

    st.divider()

    # ── Registry table ────────────────────────────────────────
    st.subheader("Registered ML Models")

    if not st.session_state.ml_registry:
        st.info("No ML models registered yet.")
    else:
        registry_rows = [
            {
                "Name": e.name,
                "Purpose": e.purpose,
                "Target Model": e.target_model,
                "Feature Columns": len(e.feature_columns) or "Auto (all numeric)",
                "Description": e.description or "—",
            }
            for e in st.session_state.ml_registry
        ]
        st.dataframe(pd.DataFrame(registry_rows), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Remove a Model")
        names = [e.name for e in st.session_state.ml_registry]
        to_remove = st.selectbox("Select model to remove", names, key="rm_ml")
        if st.button("🗑️ Remove Selected Model", type="secondary"):
            st.session_state.ml_registry = [
                e for e in st.session_state.ml_registry if e.name != to_remove
            ]
            st.success(f"Removed **{to_remove}**.")
            st.rerun()

    st.divider()

    # ── Test harness ──────────────────────────────────────────
    st.subheader("🧪 Test Registered Model")
    st.caption(
        "Run a registered model against the current filtered data to verify it works before querying."
    )

    if st.session_state.ml_registry and st.session_state.performance_df is not None:
        test_model_name = st.selectbox(
            "Model to test", [e.name for e in st.session_state.ml_registry], key="test_ml"
        )
        _perf_test = st.session_state.performance_df
        if PERF_PROFILE_COL in _perf_test.columns:
            _test_profiles = sorted(_perf_test[PERF_PROFILE_COL].dropna().unique().tolist())
        else:
            _test_profiles = list(PROFILE_MODEL_MAP.keys())
        test_profile = st.selectbox("Profile filter", ["All"] + _test_profiles, key="test_prof")

        if PERF_MODEL_COL in _perf_test.columns:
            if test_profile != "All" and PERF_PROFILE_COL in _perf_test.columns:
                _test_models = sorted(
                    _perf_test[_perf_test[PERF_PROFILE_COL] == test_profile][PERF_MODEL_COL]
                    .dropna().unique().tolist()
                )
            else:
                _test_models = sorted(_perf_test[PERF_MODEL_COL].dropna().unique().tolist())
        else:
            _test_models = [m for ms in PROFILE_MODEL_MAP.values() for m in ms]
        test_model_filter = st.selectbox("Model filter", ["All"] + _test_models, key="test_mod_filter")

        if st.button("▶️ Run Test", type="primary"):
            entry = next(e for e in st.session_state.ml_registry if e.name == test_model_name)
            p_filter = None if test_profile == "All" else test_profile
            m_filter = None if test_model_filter == "All" else test_model_filter
            feat_cols = (
                st.session_state.feature_files.get(m_filter, {}).get("columns")
                if m_filter
                else None
            )
            p_df, a_df = filter_dataframes(
                st.session_state.performance_df,
                st.session_state.alerts_df,
                p_filter,
                m_filter,
                feat_cols,
            )

            from tools.ml_model_tool import MLModelTool
            result = MLModelTool(entry).execute(p_df, a_df, "test run")

            if result["success"]:
                st.success("Model executed successfully.")
                st.markdown(result["result"])
                if result.get("dataframe") is not None:
                    st.dataframe(result["dataframe"].head(20), use_container_width=True)
            else:
                st.error(result["result"])
    else:
        st.info("Register an ML model and load data to run a test.")

    st.divider()

    # ── Orchestration flow diagram ────────────────────────────
    with st.expander("📐 Orchestration Architecture"):
        st.markdown(
            """
```
User Query (Tab 1)
       │
       ▼
 ┌─────────────────────────────────────────────┐
 │          IIoT Orchestrator (Google ADK)      │
 │                                             │
 │  1. Keyword / Gemini intent classification  │
 │  2. Check ML Registry for purpose match     │
 │                                             │
 │  ┌──────────────────┐  ┌──────────────────┐ │
 │  │  execute_ml_model│  │ run_pandas_analysis│ │
 │  │  Tool            │  │ Tool              │ │
 │  └────────┬─────────┘  └────────┬──────────┘ │
 └───────────┼──────────────────────┼────────────┘
             │                      │
             ▼                      ▼
      MLModelTool.execute()   PandasAnalysisAgent.analyze()
      (sklearn / callable)    (PandasAI + Gemini fallback)
             │                      │
             └──────────┬───────────┘
                        ▼
              Structured Response
              (narrative + table + source badge)
```
            """
        )
