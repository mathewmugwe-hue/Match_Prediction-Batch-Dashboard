"""Batch UI for the unmodified Match_Prediction inference pipeline.

Only the input/output layer is added here. All per-fixture predictions are
returned by the repository's existing src/predict.py::predict_match function.
"""
from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from predict import predict_match  # noqa: E402


st.set_page_config(
    page_title="Match_Prediction | Batch",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #f5f7fa; }
      .block-container { padding-top: 1.8rem; max-width: 1500px; }
      .hero { background: linear-gradient(115deg,#10243a,#204b68); color:white;
              border-radius:14px; padding:24px 28px; margin-bottom:18px; }
      .hero h1 { color:white; margin:0 0 8px 0; font-size:2rem; }
      .hero p { color:#d7e5f1; margin:0; }
      .note { background:#fff6de; border-left:4px solid #d49a24; padding:12px 14px;
              border-radius:4px; color:#4e401f; margin:12px 0 18px 0; }
      [data-testid="stMetric"] { background:white; border:1px solid #e0e6ed;
                                border-radius:10px; padding:12px 14px; }
    </style>
    <div class="hero">
      <h1>Match_Prediction · Batch predictions</h1>
      <p>Original calibrated Bayesian/MCMC + XGBoost pipeline. Paste a slate and run
      each fixture through the repository's existing predictor.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Original model")
    st.caption("Read-only settings · not modified by this dashboard")
    st.markdown(
        """
        - Outcome classifier: calibrated XGBoost, 5-seed ensemble
        - Outputs: home win / draw / away win
        - Additional outputs: expected goals for each side
        - Elo: K = 20, home advantage = 50, initial rating = 1500
        - Online Poisson: LR = 0.02, μ = 0.2, home advantage = 0.25
        - MCMC ratings: 600 draws, 600 tune, 4 chains, target acceptance 0.9
        - Unspecified tournament defaults to Friendly
        - Neutral venue defaults to False
        """
    )
    st.divider()
    st.markdown("**Model scope**")
    st.write("Men's full international fixtures. This data/model is not a club-football model.")
    st.markdown("[Original repository](https://github.com/Luis-ntonio/Match_Prediction)")
    st.markdown("[Published evaluation](https://github.com/Luis-ntonio/Match_Prediction/blob/main/README.md)")

st.markdown(
    '<div class="note"><b>Scope warning:</b> this repository models men\'s full international teams, '
    'not club teams. Use country-vs-country fixtures represented in its historical data. '
    'Predictions are not generated from club data or live lineup feeds. '
    'Bookmaker odds are not consumed by the original model and do not affect its probabilities.</div>',
    unsafe_allow_html=True,
)

st.subheader("1. Paste fixtures")
st.markdown(
    """
    One fixture per line. Supported formats:

    - Argentina vs Brazil (defaults to tournament Friendly, neutral False)
    - FIFA World Cup, France, England
    - 2026-11-12, FIFA World Cup qualification, Argentina, Brazil
    - 2026-11-12, FIFA World Cup, Argentina, Brazil, neutral

    For comma-separated formats, keep the order date (optional), tournament, home team, away team, neutral (optional).
    The date is shown for reference only: the original predict_match function does not take fixture date as an argument.
    """
)

placeholder = (
    "Argentina vs Brazil\n"
    "FIFA World Cup, France, England\n"
    "2026-11-12, FIFA World Cup qualification, Argentina, Brazil"
)
fixture_text = st.text_area(
    "Fixture list",
    height=210,
    placeholder=placeholder,
    help="The output row order matches this text exactly. Blank lines are ignored.",
)
run = st.button("Run original model on all fixtures", type="primary", use_container_width=True)


def parse_neutral(value: str) -> bool:
    return value.strip().casefold() in {"neutral", "true", "yes", "y", "1", "neutral=true"}


def parse_fixture(line: str, index: int) -> dict[str, Any]:
    raw = line.strip()
    result: dict[str, Any] = {
        "Input order": index,
        "Input line": raw,
        "Fixture date": "",
        "Tournament": "Friendly",
        "Home team": "",
        "Away team": "",
        "Neutral venue": False,
    }
    pieces = [p.strip() for p in next(csv.reader([raw], skipinitialspace=True)) if p.strip()]

    if len(pieces) >= 4 and re.fullmatch(r"\d{4}-\d{2}-\d{2}", pieces[0]):
        result["Fixture date"] = pieces[0]
        result["Tournament"] = pieces[1]
        result["Home team"] = pieces[2]
        result["Away team"] = pieces[3]
        if len(pieces) >= 5:
            result["Neutral venue"] = parse_neutral(pieces[4])
        return result

    if len(pieces) >= 3:
        result["Tournament"] = pieces[0]
        result["Home team"] = pieces[1]
        result["Away team"] = pieces[2]
        if len(pieces) >= 4:
            result["Neutral venue"] = parse_neutral(pieces[3])
        return result

    if len(pieces) == 2:
        result["Home team"], result["Away team"] = pieces
        return result

    match = re.split(r"\s+(?:vs?\.?|v\.?|@)\s+", raw, maxsplit=1, flags=re.IGNORECASE)
    if len(match) == 2 and match[0].strip() and match[1].strip():
        result["Home team"], result["Away team"] = match[0].strip(), match[1].strip()
        return result

    raise ValueError(
        "Use 'Home vs Away', 'Tournament, Home, Away' or "
        "'YYYY-MM-DD, Tournament, Home, Away[, neutral]'."
    )


def outcome_label(probs: dict[str, float]) -> str:
    key = max(("p_home_win", "p_draw", "p_away_win"), key=lambda k: probs[k])
    return {
        "p_home_win": "Home win",
        "p_draw": "Draw",
        "p_away_win": "Away win",
    }[key]


def ensure_original_artifacts() -> None:
    """Build only missing intermediate files with the upstream scripts and defaults."""
    processed = ROOT / "data" / "processed"
    results_path = processed / "results_clean.parquet"
    ratings_path = processed / "mcmc_ratings.parquet"

    required_models = [
        ROOT / "models" / "xgb_1x2_calibrated.joblib",
        ROOT / "models" / "xgb_home_goals.json",
        ROOT / "models" / "xgb_away_goals.json",
        ROOT / "models" / "encoders.joblib",
    ]
    missing_models = [str(path.relative_to(ROOT)) for path in required_models if not path.is_file()]
    if missing_models:
        st.error("Original model artifacts are missing: " + ", ".join(missing_models))
        st.stop()

    if results_path.exists() and ratings_path.exists():
        return

    with st.status("Preparing the original model's required data (first run only)…", expanded=True) as status_box:
        try:
            if not results_path.exists():
                st.write("Step 1/2 · Cleaning the repository's included international results dataset")
                result = subprocess.run(
                    [sys.executable, str(ROOT / "src" / "data_cleaning.py")],
                    cwd=ROOT, capture_output=True, text=True, timeout=600, check=False,
                )
                if result.returncode != 0:
                    raise RuntimeError("data_cleaning.py failed:\\n" + (result.stderr or result.stdout)[-5000:])
            if not ratings_path.exists():
                st.write("Step 2/2 · Fitting the original Bayesian MCMC team ratings (600 draws, 600 tune, 4 chains)")
                result = subprocess.run(
                    [sys.executable, str(ROOT / "src" / "mcmc_model.py")],
                    cwd=ROOT, capture_output=True, text=True, timeout=2400, check=False,
                )
                if result.returncode != 0:
                    raise RuntimeError("mcmc_model.py failed:\\n" + (result.stderr or result.stdout)[-7000:])
            if not results_path.exists() or not ratings_path.exists():
                raise RuntimeError("The original scripts finished without creating all required parquet files.")
            status_box.update(label="Original model data is ready", state="complete", expanded=False)
        except Exception as exc:
            status_box.update(label="Could not prepare original model data", state="error", expanded=True)
            st.error(str(exc))
            st.info("No substitute model or guessed probabilities were used. If this hosting provider terminates the MCMC build for memory or time limits, deploy this same repository on a Python host with more memory; the model settings remain unchanged.")
            st.stop()


if run:
    lines = [line.strip() for line in fixture_text.splitlines() if line.strip()]
    if not lines:
        st.error("Paste at least one fixture before running predictions.")
    else:
        ensure_original_artifacts()

        # Validate against the original historical national-team dataset before calling
        # the upstream predictor. Its own fallbacks can otherwise emit numbers for an
        # unknown club side, which would look like a real model prediction.
        history = pd.read_parquet(
            ROOT / "data" / "processed" / "results_clean.parquet",
            columns=["home_team", "away_team"],
        )
        known_teams = set(history["home_team"].dropna().astype(str)) | set(
            history["away_team"].dropna().astype(str)
        )
        team_lookup = {team.casefold(): team for team in known_teams}

        parsed: list[dict[str, Any]] = []
        for index, line in enumerate(lines, start=1):
            try:
                parsed.append(parse_fixture(line, index))
            except ValueError as exc:
                parsed.append({
                    "Input order": index,
                    "Input line": line,
                    "Fixture date": "",
                    "Tournament": "",
                    "Home team": "",
                    "Away team": "",
                    "Neutral venue": False,
                    "Parse error": str(exc),
                })

        output: list[dict[str, Any]] = []
        progress = st.progress(0, text="Preparing fixture predictions…")
        status = st.empty()
        for row_num, fixture in enumerate(parsed, start=1):
            base = {
                "Input order": fixture["Input order"],
                "Input line": fixture["Input line"],
                "Fixture date": fixture["Fixture date"],
                "Tournament": fixture["Tournament"],
                "Home team": fixture["Home team"],
                "Away team": fixture["Away team"],
                "Neutral venue": fixture["Neutral venue"],
            }
            if fixture.get("Parse error"):
                output.append({**base, "Status": "INPUT ERROR", "Error": fixture["Parse error"]})
            else:
                unknown_teams = [
                    team for team in (fixture["Home team"], fixture["Away team"])
                    if team.casefold() not in team_lookup
                ]
                if unknown_teams:
                    output.append({
                        **base,
                        "Status": "UNSUPPORTED TEAM",
                        "Error": (
                            "Not found in the original international-team dataset: "
                            + ", ".join(unknown_teams)
                            + ". This model does not predict club teams. Use the current national-team names from the source dataset."
                        ),
                    })
                    progress.progress(row_num / len(parsed), text=f"Processed {row_num} of {len(parsed)} fixtures")
                    continue

                model_home = team_lookup[fixture["Home team"].casefold()]
                model_away = team_lookup[fixture["Away team"].casefold()]
                base["Home team"] = model_home
                base["Away team"] = model_away
                try:
                    # Deliberately call the upstream predictor unchanged.
                    prediction = predict_match(
                        model_home,
                        model_away,
                        fixture["Tournament"],
                        fixture["Neutral venue"],
                    )
                    p_home = prediction["p_home_win"]
                    p_draw = prediction["p_draw"]
                    p_away = prediction["p_away_win"]
                    output.append({
                        **base,
                        "Prediction": outcome_label({
                            "p_home_win": p_home,
                            "p_draw": p_draw,
                            "p_away_win": p_away,
                        }),
                        "P(Home win)": p_home,
                        "P(Draw)": p_draw,
                        "P(Away win)": p_away,
                        "Top outcome probability": max(p_home, p_draw, p_away),
                        "Expected home goals": prediction["expected_home_goals"],
                        "Expected away goals": prediction["expected_away_goals"],
                        "Status": "OK",
                        "Error": "",
                    })
                except Exception as exc:
                    output.append({**base, "Status": "PREDICTION ERROR", "Error": f"{type(exc).__name__}: {exc}"})
            progress.progress(row_num / len(parsed), text=f"Processed {row_num} of {len(parsed)} fixtures")
        progress.empty()
        status.empty()

        frame = pd.DataFrame(output).sort_values("Input order", kind="stable")
        st.session_state["last_batch_frame"] = frame
        st.session_state["last_batch_input"] = fixture_text
        ok = int((frame["Status"] == "OK").sum()) if "Status" in frame else 0
        errors = len(frame) - ok
        m1, m2, m3 = st.columns(3)
        m1.metric("Fixtures submitted", len(frame))
        m2.metric("Predictions returned", ok)
        m3.metric("Rows with errors", errors)
        st.subheader("2. Predictions — original input order")
        st.caption("Probabilities are produced by the repository's original predict_match() function. Bookmaker odds, if supplied separately, are not inputs to this model.")
        st.markdown("**Batch diagnostics**")
        status_counts = frame["Status"].value_counts(dropna=False)
        diag_cols = st.columns(max(1, min(4, len(status_counts))))
        for diag_col, (status_name, status_count) in zip(diag_cols, status_counts.items()):
            diag_col.metric(str(status_name), int(status_count))
        failed = frame[frame["Status"] != "OK"]
        if not failed.empty:
            with st.expander("Why rows failed", expanded=True):
                reasons = (
                    failed.assign(Error=failed["Error"].fillna("").astype(str))
                    .groupby(["Status", "Error"], dropna=False)
                    .size().reset_index(name="Rows")
                    .sort_values("Rows", ascending=False)
                )
                st.dataframe(reasons, use_container_width=True, hide_index=True)
        display = frame.copy()
        for col in ("P(Home win)", "P(Draw)", "P(Away win)", "Top outcome probability"):
            if col in display:
                display[col] = display[col].map(lambda x: f"{float(x):.1%}" if pd.notna(x) else "—")
        for col in ("Expected home goals", "Expected away goals"):
            if col in display:
                display[col] = display[col].map(lambda x: f"{float(x):.2f}" if pd.notna(x) else "—")
        st.dataframe(display, use_container_width=True, hide_index=True)

if "last_batch_frame" in st.session_state:
    st.subheader("Export results")
    export_frame = st.session_state["last_batch_frame"]
    csv_bytes = export_frame.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download predictions as CSV",
        data=csv_bytes,
        file_name="match_prediction_batch.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.caption(
    "No training parameters or feature code were changed for this batch interface. "
    "The original repository reports 62.1% 1X2 accuracy on its held-out temporal split; "
    "this app does not claim to have independently reproduced that benchmark."
)
