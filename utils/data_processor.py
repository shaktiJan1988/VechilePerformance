"""Data generation, filtering, and schema-enforcement utilities."""

from __future__ import annotations

import io
import random
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    ALERT_ASSET_COL, ALERT_MODEL_COL, ALERT_PROFILE_COL,
    PERF_ASSET_COL, PERF_MODEL_COL, PERF_PROFILE_COL,
    PROFILE_MODEL_MAP,
)

# ──────────────────────────────────────────────────────────────
# Sample data generation
# ──────────────────────────────────────────────────────────────

rng = np.random.default_rng(42)


def _random_date(start: datetime, end: datetime) -> str:
    delta = end - start
    return (start + timedelta(seconds=int(delta.total_seconds() * rng.random()))).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def generate_performance_data(n_rows: int = 300) -> pd.DataFrame:
    """Generate realistic JCB machine performance telemetry."""
    profiles_models = [
        (profile, model)
        for profile, models in PROFILE_MODEL_MAP.items()
        for model in models
    ]

    rows = []
    for i in range(n_rows):
        profile, model = profiles_models[i % len(profiles_models)]
        engine_on = round(rng.uniform(2, 10), 2)
        idle_pct = round(rng.uniform(10, 40), 1)
        idle_hrs = round(engine_on * idle_pct / 100, 2)
        working_hrs = round(engine_on - idle_hrs, 2)
        fuel_used = round(rng.uniform(20, 120), 2)

        rows.append({
            "PIN": f"JCB{str(i + 1000).zfill(6)}",
            "Profile": profile,
            "Model": model,
            "Customer": f"Customer_{rng.integers(1, 50)}",
            "Customer Contact": f"+91-{rng.integers(7000000000, 9999999999)}",
            "Dealer": f"Dealer_{rng.integers(1, 20)}",
            "Zone": random.choice(["North", "South", "East", "West", "Central"]),
            "Address": f"City_{rng.integers(1, 100)}, State_{rng.integers(1, 30)}",
            "PeriodHMR": round(rng.uniform(1, 14), 2),
            "Non RPM Value": round(rng.uniform(0, 1), 2),
            "Engine Off(Period in hrs)": round(24 - engine_on, 2),
            "Engine Off(% of period)": round((24 - engine_on) / 24 * 100, 1),
            "Engine On(Period in hrs)": engine_on,
            "Engine On(% of period)": round(engine_on / 24 * 100, 1),
            "WorkingTime": working_hrs,
            "Working Time(% of period)": round(working_hrs / 24 * 100, 1),
            "Idle Time(Period in hrs)": idle_hrs,
            "Idle Time(% of period)": idle_pct,
            "Power Band low(Period in hrs)": round(working_hrs * rng.uniform(0.1, 0.3), 2),
            "Power Band low(% of EngineOn time)": round(rng.uniform(10, 30), 1),
            "Power Band medium(Period in hrs)": round(working_hrs * rng.uniform(0.3, 0.5), 2),
            "Power Band medium(% of EngineOn time)": round(rng.uniform(30, 50), 1),
            "Power Band high(Period in hrs)": round(working_hrs * rng.uniform(0.2, 0.4), 2),
            "Power Band high(% of EngineOn time)": round(rng.uniform(20, 40), 1),
            "Start Engine Run Hrs(value)": round(rng.uniform(500, 5000), 0),
            "End Engine Run Hrs(value)": round(rng.uniform(500, 5000) + engine_on, 0),
            "Fuel Used In Idle(ltr)": round(idle_hrs * rng.uniform(1.5, 3.5), 2),
            "DSCPump Control hrs": round(rng.uniform(0, working_hrs), 2),
            "Accumulated_hrs_in_Engine Idling": round(rng.uniform(100, 2000), 1),
            "Excavation Economy Mode hrs": round(rng.uniform(0, working_hrs * 0.4), 2)
            if profile == "Excavator" else 0.0,
            "Excavation Active Mode hrs": round(rng.uniform(0, working_hrs * 0.4), 2)
            if profile == "Excavator" else 0.0,
            "Excavation Power Mode hrs": round(rng.uniform(0, working_hrs * 0.2), 2)
            if profile == "Excavator" else 0.0,
            "Forward Direction hrs": round(rng.uniform(0, working_hrs * 0.6), 2),
            "Load Job Hrs": round(rng.uniform(0, working_hrs * 0.7), 2),
            "Neutral Position hrs": round(rng.uniform(0.1, 1.0), 2),
            "Reverse Direction hrs": round(rng.uniform(0, working_hrs * 0.2), 2),
            "No AutoIdle Events": int(rng.integers(0, 50)),
            "No Auto Off Events": int(rng.integers(0, 20)),
            "No Kickdowns LoadingMode": int(rng.integers(0, 30)),
            "Pressure Running Band1(Upstream)": round(rng.uniform(50, 150), 1),
            "Pressure Running Band2(Upstream)": round(rng.uniform(150, 250), 1),
            "Pressure Running Band3(Upstream)": round(rng.uniform(250, 350), 1),
            "Pressure Running Band4(Upstream)": round(rng.uniform(350, 450), 1),
            "Pressure Running Band5(Upstream)": round(rng.uniform(0, 50), 1),
            "Pressure Running Band6(Upstream)": round(rng.uniform(0, 20), 1),
            "Pressure Running Band1(Downstream)": round(rng.uniform(40, 120), 1),
            "Pressure Running Band2(Downstream)": round(rng.uniform(120, 220), 1),
            "Pressure Running Band3(Downstream)": round(rng.uniform(220, 320), 1),
            "Pressure Running Band4(Downstream)": round(rng.uniform(320, 420), 1),
            "Pressure Running Band5(Downstream)": round(rng.uniform(0, 40), 1),
            "Pressure Running Band6(Downstream)": round(rng.uniform(0, 15), 1),
            "Pump Displacement Running Band1": round(rng.uniform(10, 40), 1),
            "Pump Displacement Running Band2": round(rng.uniform(40, 70), 1),
            "Pump Displacement Running Band3": round(rng.uniform(70, 100), 1),
            "Pump Displacement Running Band4": round(rng.uniform(0, 30), 1),
            "Pump Displacement Running Band5": round(rng.uniform(0, 10), 1),
            "Pump Displacement Running Band6": round(rng.uniform(0, 5), 1),
            "Fuel Level(%)": round(rng.uniform(10, 100), 1),
            "Fuel Rate": round(rng.uniform(5, 25), 2),
            "Start Fuel(In Liters)": round(rng.uniform(50, 200), 1),
            "Finish Fuel(In Liters)": round(rng.uniform(10, 150), 1),
            "Fuel Used(in Liters)": fuel_used,
            "Fuel Used In LPB(Low Power Band)": round(fuel_used * rng.uniform(0.05, 0.15), 2),
            "Fuel Used In MPB(Medium Power Band)": round(fuel_used * rng.uniform(0.3, 0.5), 2),
            "Fuel Used In HPB(High Power Band)": round(fuel_used * rng.uniform(0.3, 0.5), 2),
            "Average Fuel Consumption": round(fuel_used / max(engine_on, 0.1), 2),
            "Fuel Loss": round(rng.uniform(0, 5), 2),
            "Fuel Used in Working": round(fuel_used * rng.uniform(0.6, 0.9), 2),
            "L Band": round(rng.uniform(0, 20), 1),
            "G Band": round(rng.uniform(0, 20), 1),
            "H band": round(rng.uniform(0, 20), 1),
            "H Plus Band": round(rng.uniform(0, 10), 1),
            "Travelling Time In Hrs": round(rng.uniform(0, 3), 2),
            "SlewTimeInHrs": round(rng.uniform(0, working_hrs * 0.3), 2)
            if profile == "Excavator" else 0.0,
            "Hydraulic Choke Events": int(rng.integers(0, 20)),
            "Hammer Use Time In Hrs": round(rng.uniform(0, 2), 2)
            if profile in ("Excavator", "Backhoe Loader") else 0.0,
            "Hammer Abuse Count": int(rng.integers(0, 10)),
            "Carbon Emission": round(fuel_used * 2.68, 2),
            "Power Boost Time": round(rng.uniform(0, 0.5), 2),
            "Regeneration Time for Tier4 Engine Only": round(rng.uniform(0, 0.3), 2),
            "Operator Out Of Seat Count": int(rng.integers(0, 15)),
            "Engine Manufacturer": random.choice(["JCB", "Cummins", "Kohler"]),
            "Gear1 Forward Utilization": round(rng.uniform(0, 20), 1),
            "Gear2 Forward Utilization": round(rng.uniform(0, 20), 1),
            "Gear3 Forward Utilization": round(rng.uniform(0, 20), 1),
            "Gear4 Forward Utilization": round(rng.uniform(0, 10), 1),
            "Gear1 Backward Utilization": round(rng.uniform(0, 10), 1),
            "Gear2 Backward Utilization": round(rng.uniform(0, 8), 1),
            "Gear1 Lockup Utilization": round(rng.uniform(0, 5), 1),
            "Neutral Gear Utilization": round(rng.uniform(0, 15), 1),
            "EngineOnCount": int(rng.integers(1, 10)),
            "EngineOffCount": int(rng.integers(1, 10)),
            "Accumulated_hrs_in_ExcavationJob": round(rng.uniform(0, 1000), 1)
            if profile == "Excavator" else 0.0,
            "Accumulated_hrs_in_RoadingJob": round(rng.uniform(0, 500), 1),
            "Long Engine Idling Event": int(rng.integers(0, 5)),
            "Hot Shut Down Event": int(rng.integers(0, 3)),
            "Coolant Temperature": round(rng.uniform(70, 105), 1),
            "UsageCategory": random.choice(["Heavy", "Medium", "Light"]),
            "MachineCategory": random.choice(["Construction", "Agriculture", "Industrial"]),
        })

    return pd.DataFrame(rows)


def generate_alerts_data(performance_df: pd.DataFrame, alerts_per_pin: int = 5) -> pd.DataFrame:
    """
    Generate alert events with multiple rows per PIN.

    Every PIN in performance_df gets between 2 and (alerts_per_pin * 2) alert
    rows, guaranteeing at least 2 events per machine while keeping the
    distribution realistic.
    """
    alert_descriptions = [
        "Hydraulic Oil Temperature High",
        "Engine Coolant Temperature High",
        "Hydraulic Filter Restriction",
        "Low Engine Oil Pressure",
        "Battery Voltage Low",
        "DPF Regeneration Required",
        "Fuel Level Critical",
        "Transmission Fault",
        "Brake System Fault",
        "Alternator Fault",
        "Hydraulic Pump Fault",
        "Steering System Warning",
        "Air Filter Restriction",
        "Engine Overload",
        "Glow Plug Fault",
    ]
    severities = ["Critical", "High", "Medium", "Low"]
    severity_weights = [0.1, 0.25, 0.40, 0.25]

    pin_to_profile = dict(zip(performance_df[PERF_ASSET_COL], performance_df[PERF_PROFILE_COL]))
    pin_to_model  = dict(zip(performance_df[PERF_ASSET_COL], performance_df[PERF_MODEL_COL]))

    start = datetime(2024, 1, 1)
    end   = datetime(2024, 12, 31)

    rows = []
    for pin in performance_df[PERF_ASSET_COL]:
        # Each PIN gets 2 … (alerts_per_pin * 2) alert events
        n_alerts = int(rng.integers(2, alerts_per_pin * 2 + 1))
        for _ in range(n_alerts):
            gen_time = _random_date(start, end)
            closure_time = (
                datetime.strptime(gen_time, "%Y-%m-%d %H:%M:%S")
                + timedelta(hours=int(rng.integers(1, 48)))
            ).strftime("%Y-%m-%d %H:%M:%S")

            rows.append({
                "AssetID":                pin,
                "ProfileName":            pin_to_profile[pin],
                "ModelName":              pin_to_model[pin],
                "AGAddress":              f"AG{rng.integers(100, 999)}",
                "AlertDescription":       random.choice(alert_descriptions),
                "AlertSeverity":          random.choices(severities, severity_weights)[0],
                "DtcAlertGenerationTime": gen_time,
                "AGCMH":                  round(rng.uniform(500, 6000), 1),
                "AlertCode":              f"AC{rng.integers(100, 999)}",
                "DTCCode":                f"DTC{rng.integers(1000, 9999)}",
                "DtcAlertClosureTime":    closure_time,
                "ACCMH":                  round(rng.uniform(500, 6000), 1),
                "ErrorCode":              f"EC{rng.integers(10, 99)}",
                "ECUAddress":             random.choice(["ECU_ENGINE", "ECU_TRANS", "ECU_HYD", "ECU_ELEC"]),
            })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# Filtering and schema enforcement
# ──────────────────────────────────────────────────────────────

def filter_dataframes(
    performance_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
    profile: Optional[str],
    model: Optional[str],
    feature_columns: Optional[list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Subset both DataFrames by profile/model and enforce the feature file schema.
    Returns (filtered_performance_df, filtered_alerts_df).
    """
    perf = performance_df.copy()
    alrt = alerts_df.copy()

    if profile:
        if PERF_PROFILE_COL in perf.columns:
            perf = perf[perf[PERF_PROFILE_COL] == profile]
        if ALERT_PROFILE_COL in alrt.columns:
            alrt = alrt[alrt[ALERT_PROFILE_COL] == profile]

    if model:
        if PERF_MODEL_COL in perf.columns:
            perf = perf[perf[PERF_MODEL_COL] == model]
        if ALERT_MODEL_COL in alrt.columns:
            alrt = alrt[alrt[ALERT_MODEL_COL] == model]

    if feature_columns:
        # Always retain the identifier columns so joins remain possible.
        # Use dict.fromkeys to deduplicate while preserving order — feature file
        # may already contain the identity columns, which would create duplicates.
        identity_perf = [PERF_ASSET_COL, PERF_PROFILE_COL, PERF_MODEL_COL]
        identity_alrt = [ALERT_ASSET_COL, ALERT_PROFILE_COL, ALERT_MODEL_COL]

        perf_keep = list(dict.fromkeys(
            identity_perf + [c for c in feature_columns if c in perf.columns]
        ))
        alrt_keep = list(dict.fromkeys(
            identity_alrt + [c for c in feature_columns if c in alrt.columns]
        ))

        perf = perf[perf_keep]
        alrt = alrt[alrt_keep]

    return perf.reset_index(drop=True), alrt.reset_index(drop=True)


def get_valid_models(profile: str, feature_files: dict) -> list[str]:
    """Return models for the given profile that have a validated feature file."""
    all_models = PROFILE_MODEL_MAP.get(profile, [])
    return [m for m in all_models if m in feature_files]


def parse_feature_file(uploaded_file) -> list[str]:
    """
    Parse an uploaded CSV or Excel feature file and return the list of column names.
    The file should have column names in the first row (header-only or with sample data).
    """
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, nrows=5)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file, nrows=5)
    else:
        raise ValueError(f"Unsupported file type: {uploaded_file.name}")
    return list(df.columns)


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    return df.head(max_rows).to_markdown(index=False)
