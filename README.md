# Store Sales Forecasting (LSTM)

Predict daily store sales using an LSTM-based time series forecasting approach. This repository provides a notebook-driven workflow to prepare data, train, validate, and evaluate an LSTM model for sales forecasting.

This project was developed as part of the Advanced Data Analytics course by Hoang Anh Thu, Nguyen Quynh Chi, and Le Hai Anh.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Reference](#reference)
- [Key Improvements](#key-improvements)
- [Artifacts](#artifacts)
- [How to Run](#how-to-run)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

---

## Overview
This project implements a Long Short-Term Memory (LSTM) model for forecasting daily store sales. The task is framed as supervised learning with a sliding window over time and optional exogenous features.

- Problem formulation: fixed-length lookback windows of past sales (and optional exogenous signals such as calendar/holiday indicators and rolling statistics) are used to predict the next step or a short multi-step horizon.
- Model: a single or stacked LSTM with dropout to capture temporal dependencies; trained as a global model across multiple series (e.g., stores/product families) to share seasonal and trend structure.
- Training protocol: strictly time-respecting splits; normalization/encoding fit only on the training period; early stopping and model checkpointing based on validation performance.
- Evaluation: time-aware validation (rolling-origin/backtesting or single chronological holdout) with appropriate error metrics (RMSE, MAE, MAPE).

---

## Dataset
We use the Kaggle competition dataset:
- [Store Sales – Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

Please follow Kaggle’s terms when downloading and using the data.

---

## Reference
- [Darts Forecasting – Deep Learning & Global Models](https://www.kaggle.com/code/ferdinandberr/darts-forecasting-deep-learning-global-models)

---

## Key Improvements
Grounded in the LSTM approach above, the pipeline focuses on:

- Global LSTM training:
  - Train a single LSTM across multiple related series to leverage shared patterns and improve data efficiency.
- Windowing and targets:
  - Consistent sliding windows for inputs (lookback) and clearly defined forecast horizon (1-step or short multi-step).
- Feature pipeline (exogenous + lags):
  - Lag features (e.g., t−1, t−7, t−14) and rolling stats (mean/std) computed from past data only.
  - Calendar/holiday features where applicable to encode seasonality and events.
- Leakage-safe preprocessing:
  - Fit scalers/encoders exclusively on the training split; apply the fitted transforms to validation/test.
  - Generate all lag/rolling features without peeking into the future.
- Regularization and training stability:
  - Dropout/recurrent dropout, early stopping, and model checkpointing.
  - Seed control; optional learning-rate scheduling during training.
- Time-aware evaluation:
  - Rolling-origin backtesting or a chronological holdout.
  - Report RMSE/MAE/MAPE on validation and test segments.

---

## Artifacts
- Final Notebook (English): [Final5.ipynb](https://github.com/AnhThuHoang0518/Deep-Learning-model-for-Sale-Forecasting/blob/main/Final5.ipynb)

---

## How to Run
1. Clone this repository.
2. Open the notebook (e.g., Jupyter or Google Colab): `Final5.ipynb`.
3. Download the Kaggle dataset and place files in a local folder (e.g., `data/`).
4. Update dataset paths in the notebook to your local data location, and set key configs (lookback, horizon, features).
5. Run the notebook cells in order to train and evaluate the LSTM model.

---

## Results
- The notebook demonstrates an LSTM-based training and evaluation workflow with time-aware validation to avoid leakage.
- See the final sections of [Final5.ipynb](https://github.com/AnhThuHoang0518/Deep-Learning-model-for-Sale-Forecasting/blob/main/Final5.ipynb) for evaluation metrics and qualitative analysis.

---

## Acknowledgments
- Course: Advanced Data Analytics
- Authors: Hoang Anh Thu, Nguyen Quynh Chi, and Le Hai Anh
