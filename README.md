# JJ Forecasting Project

## Overview

This repository builds macroeconomic forecasting panels from the OECD SDMX API, evaluates probabilistic forecasting benchmarks, and compares them against pretrained foundation models such as Chronos-2 and Sundial.

## Required environment

Use the repository environment file, not a separate ad-hoc virtualenv.

To create and activate the project environment:

```bash
micromamba env create -f environment.yml
micromamba activate jj
```

The repository must be run from this micromamba environment. The file [environment.yml](environment.yml) is the authoritative setup for reproducing the project on any machine.

## Data pipeline

- [oecd_api.py](oecd_api.py) downloads and standardises OECD CPI and GDP series into panel CSV files in [data](data)
- the panels are then used by the forecasting backtests and evaluation scripts

## Forecasting and evaluation

- [run_benchmarks.py](run_benchmarks.py) runs the rolling-origin backtest for the benchmark models
- [models](models) contains the forecasting model implementations and scoring logic
- [plots](plots) contains the descriptive and evaluation visualisations

## Typical workflow

```bash
micromamba activate jj
python oecd_api.py
python run_benchmarks.py
```

This project is designed to be reproducible from the micromamba environment file and should not rely on an older local `.venv` setup.
