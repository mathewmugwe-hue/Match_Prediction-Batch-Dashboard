# Match_Prediction Batch Dashboard

This repository is a fork of [Luis-ntonio/Match_Prediction](https://github.com/Luis-ntonio/Match_Prediction) with a batch fixture input/output page.

## Deploy without an inference API or API key

This app runs the original Python inference function inside the Streamlit app. It does not need `MATCH_PREDICTION_API_URL`, `MATCH_PREDICTION_API_KEY`, or any separate inference service.

1. Open the [Streamlit Community Cloud deployment page](https://share.streamlit.io/deploy?repository=mathewmugwe-hue/Match_Prediction-Batch-Dashboard&branch=main&mainModule=dashboard.py).
2. Sign in with GitHub and authorize Streamlit Community Cloud if prompted.
3. Confirm the repository, branch `main`, and main file `dashboard.py`, then deploy.
4. On the first prediction run, the app automatically creates the missing intermediate data and Bayesian ratings by running the original repository scripts with their original default settings. This can take several minutes.
5. Paste international fixtures, one per line, and run the predictions. Export the result table as CSV.

### Input examples, including odds
- `Argentina vs Brazil`
- `Argentina vs Brazil, 2.10, 3.20, 3.60`
- `FIFA World Cup, Argentina, Brazil, 2.10, 3.20, 3.60`
- `2026-11-12, FIFA World Cup, Argentina, Brazil, 2.10, 3.20, 3.60`

Optional decimal odds are interpreted as home/draw/away 1X2 odds. The app displays a margin-normalized bookmaker baseline for comparison, but **the odds are never passed into the original model**. The batch inference code caches the original model's loaded data, ratings and estimators once per app process to avoid recomputing the entire history for every fixture.

No external API keys are required. The only unavoidable setup is authorizing the hosting platform and pressing Deploy; the host must run Python because the predictor is a Python model.

## Model integrity

The batch page calls the original `src/predict.py::predict_match` function. The original training and inference code, shipped XGBoost artifacts, and pinned dependencies remain unchanged; Streamlit is an additional UI dependency only. Missing intermediate files are built by the original preprocessing and MCMC scripts without changing their parameters. The interface rejects teams absent from the original men's full international dataset instead of returning misleading club-team probabilities.

## Validation
The GitHub Actions workflow compiles the Python source, checks that the original model artifacts and results data are tracked, and runs fixture-parser regression tests. This is code/input validation, not an end-to-end accuracy benchmark or proof of the hosted app's first-run MCMC build.

## Important limits

- This is a men's full international model, not a club football model.
- Fixture date is shown for reference but is not an argument to the upstream `predict_match()` function.
- The upstream reported 62.1% held-out accuracy is a published benchmark, not a guarantee that a new live fixture slate will achieve that accuracy.
- Streamlit Community Cloud resources are limited. The original MCMC step may exceed the free host's memory/time limits; if so, use a Python host with more resources, not a simplified model.
