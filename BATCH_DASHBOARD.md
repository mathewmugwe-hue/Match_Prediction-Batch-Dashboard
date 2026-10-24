# Batch interface for the original Match_Prediction model

This fork adds only a batch input/output surface. The upstream src/predict.py, feature logic, XGBoost models, dependencies pinned by requirements.txt, Elo constants, online Poisson constants, and MCMC defaults are not edited.

## Run locally

1. Install Python 3.12.
2. Install the original requirements plus Streamlit: pip install -r requirements-dashboard.txt.
3. Build the data and ratings using the repository's original defaults:
   - python src/data_cleaning.py
   - python src/mcmc_model.py
4. Start the batch page: streamlit run dashboard.py

The original trained XGBoost artifacts remain tracked in models/. Runtime preprocessing files are generated from the tracked historical-results dataset; the ignored MCMC trace file is removed from the deployment image after the original script has generated the ratings parquet needed for inference.

## Deploy on Render

Create a new Render Web Service from this repository and use the root render.yaml blueprint configuration. The first build runs the original MCMC rating-generation step and may take several minutes. This is deliberately not replaced with a faster approximation because that would change the model pipeline.

## Scope

The upstream source dataset is men's full international football, not club football. Input formats accept a fixture date for display, but the upstream predict_match() function does not take a target match date; this interface deliberately does not alter that behavior.
