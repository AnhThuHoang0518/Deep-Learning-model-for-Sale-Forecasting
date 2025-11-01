# Store Sales Forecasting (Deep Learning)

Forecast daily store sales using deep learning for time series. This repository contains a notebook-based workflow to train, validate, and evaluate sequence models for sales forecasting.

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
The goal is to build a robust time series forecasting pipeline to predict store sales. The approach focuses on sequence modeling (e.g., RNN-based models such as LSTM/GRU) and sound evaluation practices tailored for temporal data.

Key objectives:
- Prepare clean, leak-free training/validation/test splits that respect time order
- Engineer informative time-based features (lags, rolling statistics, calendar signals)
- Train deep learning models and compare performance
- Produce reproducible experiments via a single notebook workflow

---

## Dataset
We use the Kaggle competition dataset:
- Store Sales - Time Series Forecasting: https://www.kaggle.com/competitions/store-sales-time-series-forecasting

Please follow Kaggle's terms when downloading and using the data.

---

## Reference
- Kaggle Competition: Store Sales - Time Series Forecasting (problem description and data schema)

---

## Key Improvements
The pipeline emphasizes good time-series modeling hygiene:
- Temporal splits and validation:
  - Use time-aware train/validation/test splits (no random shuffling across time)
  - Optionally leverage rolling-origin or expanding-window validation
- Feature engineering:
  - Lag features (e.g., t-1, t-7, t-14, etc.)
  - Rolling statistics (mean/std/min/max) over recent windows
  - Calendar features (day-of-week, month, holidays) when applicable
- Leakage prevention and preprocessing:
  - Fit scalers/encoders on training data only
  - Generate lagged and rolling features strictly from past data
  - Keep test period fully out-of-sample

---

## Artifacts
- Final notebook (English): Final5.ipynb

---

## How to Run
1. Clone this repository.
2. Open the notebook (e.g., in Jupyter or Google Colab): `Final5.ipynb`.
3. Download the Kaggle dataset and place the files in a local folder (for example, `data/`).
4. Update any dataset paths in the notebook to point to your local data location.
5. Run the notebook cells in order to train and evaluate the model(s).

---

## Results
- The notebook demonstrates end-to-end training and evaluation on the competition dataset with time-aware validation to avoid leakage.
- Review the final sections of `Final5.ipynb` for evaluation metrics and qualitative analysis.

---

## Acknowledgments
- Course: Advanced Data Analytics
- Authors: Hoang Anh Thu, Nguyen Quynh Chi, and Le Hai Anh