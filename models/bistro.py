# %% [markdown]
# ### Notebook overview
# This notebook reproduces an **unconditional (baseline)** forecast for a single macroeconomic time series using **BISTRO** (BIS Time-series Regression Oracle), as described in the accompanying paper.
# 
# - **Task**: forecast US CPI inflation (year-over-year), monthly, with no covariates.
# - **Workflow**: choose the forecast horizon (PDT) and history length (CTX), run a rolling-origin backtest, and collect probabilistic forecasts (median and uncertainty bands).
# - **Outputs**: a window-by-window forecast table and an overlay plot of forecasts versus the realised series.
# 
# ### Data
# - Source: BIS CPI statistics (US CPI, YoY; monthly).
# - More CPI series: see `data/`.
# 

# %% [markdown]
# ### Step 1 - Setup
# - Make the project code in `src/` available to the notebook.
# - Import the required libraries and helper functions.
# 

# %% [markdown]
# ### Google Colab users
# - Colab may preinstall **NumPy 2.x**.
# - After the downgrade runs, it triggers an automatic restart by terminating the current process.
# 

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "models"))

from config import BISTRO_ROOT

repo_root = Path(BISTRO_ROOT)
sys.path.insert(0, str(repo_root / "src"))

# %%
import numpy as np
import pandas as pd

from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split

from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

from inference_util import plot_publication_forecast_comparison

from preprocessing_util import (
    aggregate_daily_forecast_to_monthly,
    create_boolean_masks,
    prepare_yoy_monthly_for_daily_inference,
)


from config import (
    COUNTRIES,
    LOOKBACK,
    N_SAMPLES,
    QUANTILES,
    FORECAST_START_DATE as CONFIG_FORECAST_START_DATE,
)

# %% [markdown]
# ### Step 2 - Data and settings
# - Set the forecast horizon (PDT) and how much history the model reads (CTX).
# 

# %%
MODEL_REPO = repo_root / 'bistro-finetuned'

FREQ = 'M'  # monthly data frequency

PDT = 12   # how many months to forecast ahead
CTX = LOOKBACK # how many months of past data the model reads
PSZ = 32  # patch size used by the model (kept as default for this setup)
BSZ = 32  # batch size for faster forecasting
ROLLING_WINDOWS = 1  # how many starting points to evaluate
WINDOW_DISTANCE = 1 # gap (in months) between those starting points

FORECAST_START_DATE = CONFIG_FORECAST_START_DATE

config = {
    "MODEL_REPO": str(MODEL_REPO),
    "PDT": PDT,
    "CTX": CTX,
    "PSZ": PSZ,
    "BSZ": BSZ,
    "ROLLING_WINDOWS": ROLLING_WINDOWS,
    "WINDOW_DISTANCE": WINDOW_DISTANCE,
}


# %% [markdown]
# - Load the CPI series and construct rolling test windows.

# %%
from pathlib import Path
import pandas as pd

panel_path = PROJECT_ROOT / "data" / "OECD_ipi_growth_monthly_panel.csv"

panel = pd.read_csv(panel_path)

# %% [markdown]
# ### Step 3 - Run BISTRO
# - Load the pretrained BISTRO checkpoint once.
# - Generate probabilistic forecasts for each country.
# 

predictor = None
bistro_quantiles_by_country = {}

for target_col in COUNTRIES:

    print(f"\n{'=' * 60}")
    print(f"Processing {target_col}")
    print(f"{'=' * 60}")

    # Check that country exists
    if target_col not in panel.columns:
        print(f"Skipping {target_col}: not found in panel")
        continue

    # ---------------------------------------------------------
    # Prepare data for this country
    # ---------------------------------------------------------

    df = panel[["TIME_PERIOD", target_col]].copy()

    df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"])

    df = df.set_index("TIME_PERIOD")

    df.index = pd.PeriodIndex(pd.to_datetime(df.index), freq=FREQ)

    prep = prepare_yoy_monthly_for_daily_inference(
        df,
        target_col=target_col,
        freq=FREQ,
        forecast_start_date=FORECAST_START_DATE,
        pdt_patches=PDT,
        ctx_patches=CTX,
        steps_per_period=PSZ,
        rolling_windows=ROLLING_WINDOWS,
        window_distance_patches=WINDOW_DISTANCE,
    )

    if prep.windows < 1:
        print(
            f"Skipping {target_col}: not enough test data "
            f"after cutoff {prep.train_end}"
        )
        continue

    # ---------------------------------------------------------
    # Create GluonTS dataset
    # ---------------------------------------------------------

    ds = PandasDataset(
        prep.daily_df,
        target=target_col,
    )

    train, test_template = split(
        ds,
        date=prep.cutoff_period_daily,
    )

    test_data = test_template.generate_instances(
        prediction_length=prep.pdt_steps,
        windows=prep.windows,
        distance=prep.dist_steps,
        max_history=prep.ctx_steps,
    )

    if predictor is None:
        model = MoiraiForecast(
            module=MoiraiModule.from_pretrained(str(MODEL_REPO)),
            prediction_length=int(prep.pdt_steps),
            context_length=int(prep.ctx_steps),
            patch_size=int(PSZ),
            num_samples=int(N_SAMPLES),
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        predictor = model.create_predictor(batch_size=BSZ)

    # ---------------------------------------------------------
    # Forecast
    # ---------------------------------------------------------

    inputs = list(test_data.input)
    labels = list(test_data.label)

    forecasts = list(
        predictor.predict(test_data.input)
    )

    # ---------------------------------------------------------
    # Construct monthly quantiles
    # ---------------------------------------------------------

    bistro_quantiles_by_window = {}

    for w in range(prep.windows):

        samples = np.asarray(
            forecasts[w].samples,
            dtype=float,
        )

        print(
            f"{target_col} | window {w} | "
            f"samples shape: {samples.shape}"
        )

        label_target = np.asarray(labels[w]["target"], dtype=float)
        input_target = np.asarray(inputs[w]["target"], dtype=float)
        last_input = float(input_target[-1])

        monthly_predictions, _, monthly_intervals = (
            aggregate_daily_forecast_to_monthly(
                samples,
                label_target,
                last_input,
                steps_per_period=PSZ,
                expected_periods=PDT,
            )
        )

        masks = create_boolean_masks(label_target, steps_per_period=PSZ)
        if label_target.size > 0 and last_input == float(label_target[0]):
            masks = masks[1:]
        monthly_samples = np.column_stack(
            [samples[:, mask].mean(axis=1) for mask in masks[:PDT] if mask.any()]
        )

        pred_index = pd.period_range(
            start=prep.forecast_start + w * WINDOW_DISTANCE,
            periods=PDT,
            freq=FREQ,
        )

        dfw = pd.DataFrame(index=pred_index)

        for q in QUANTILES:
            column = f"q{int(q * 100):02d}"
            if q == 0.05:
                dfw[column] = monthly_intervals[:, 0]
            elif q == 0.50:
                dfw[column] = monthly_predictions
            elif q == 0.95:
                dfw[column] = monthly_intervals[:, 1]
            else:
                dfw[column] = np.quantile(monthly_samples, q, axis=0)

        bistro_quantiles_by_window[w] = dfw

    # Store this country's results
    bistro_quantiles_by_country[target_col] = (
        bistro_quantiles_by_window
    )

    print(f"\n{target_col} forecast:")
    print(bistro_quantiles_by_window[0])


# %%
from pathlib import Path

# Combine all countries and windows into one DataFrame
results = []

for country, windows in bistro_quantiles_by_country.items():
    for w, dfw in windows.items():
        temp = dfw.copy()
        temp["country"] = country
        temp["window"] = w
        temp["TIME_PERIOD"] = temp.index.astype(str)
        results.append(temp.reset_index(drop=True))

bistro_results = pd.concat(results, ignore_index=True)

# Put identifying columns first
bistro_results = bistro_results[
    ["country", "window", "TIME_PERIOD"]
    + [f"q{int(q * 100):02d}" for q in QUANTILES]
]

# Save one file containing all countries
results_path = PROJECT_ROOT / "results"
results_path.mkdir(parents=True, exist_ok=True)
bistro_results.to_csv(results_path / "bistro.csv", index=False)

print(f"\nSaved BISTRO forecasts to: {results_path / 'bistro.csv'}")
print(bistro_results)


# %%
# %%
import matplotlib.pyplot as plt
import pandas as pd

# Combine all 1-month forecast windows for this country
df_pred = pd.concat(
    bistro_quantiles_by_country[target_col].values()
).sort_index()

df_actual = prep.df_monthly[[target_col]].rename(
    columns={target_col: "actual"}
)

plot_from = prep.forecast_start - min(CTX, 120)
plot_to = df_pred.index.max()

df_plot = df_actual.join(
    df_pred[["q05", "q10", "q50", "q90", "q95"]],
    how="outer"
).sort_index()

df_plot = df_plot.loc[plot_from:plot_to]

fig, ax = plt.subplots(figsize=(10, 5))

dates = df_plot.index.to_timestamp()

# Actual
ax.plot(
    dates,
    df_plot["actual"],
    label="Actual",
)

# 90% quantile band: q05 - q95
ax.fill_between(
    dates,
    df_plot["q05"],
    df_plot["q95"],
    alpha=0.15,
    label="90% interval",
)

# 80% quantile band: q10 - q90
ax.fill_between(
    dates,
    df_plot["q10"],
    df_plot["q90"],
    alpha=0.25,
    label="80% interval",
)

# Median
ax.plot(
    dates,
    df_plot["q50"],
    label="BISTRO (median)",
    linewidth=2,
)

# Forecast start
ax.axvline(
    prep.forecast_start.to_timestamp(),
    linestyle="--",
    linewidth=1,
)

ax.set_title(
    f"{target_col} — rolling forecast"
)

ax.set_ylabel("YoY inflation (%)")

ax.set_xlabel("")

ax.legend()

ax.grid(True, alpha=0.2)

fig.tight_layout()

