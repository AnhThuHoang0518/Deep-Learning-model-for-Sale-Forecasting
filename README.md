# Store Sales Forecasting (Deep Learning)

Predict daily store sales using deep learning for time series forecasting. This repository provides a notebook-based workflow to train, validate, and evaluate sequence models for sales forecasting.

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
The goal of this project is to build a robust time series forecasting pipeline to predict store sales. The approach focuses on sequence modeling (e.g., LSTM/GRU) and time-aware evaluation tailored for temporal data.

Highlights:
- Construct clean, leak-free train/validation/test splits that respect time order
- Engineer informative time-based features (lags, rolling statistics, calendar/holiday signals)
- Train deep learning models and compare their performance
- Ensure reproducibility with a single notebook workflow

---

## Dataset
We use the Kaggle competition dataset:
- [Store Sales – Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

Please follow Kaggle’s terms when downloading and using the data.

---

## Reference
This work is aligned with the Kaggle competition problem setup and data schema:
- [Darts Forecasting - Deep Learning & Global Models] (https://www.kaggle.com/code/ferdinandberr/darts-forecasting-deep-learning-global-models)

---

## Key Improvements
The pipeline emphasizes good time-series modeling hygiene and generalization:

- Temporal splits and validation:
  - Use time-aware train/validation/test splits (no random shuffling across time)
  - Optionally consider rolling-origin or expanding-window validation
- Feature engineering:
  - Lag features (e.g., t−1, t−7, t−14)
  - Rolling statistics (mean/std/min/max) over recent windows
  - Calendar features (day-of-week, month, holidays) when applicable
- Leakage prevention and preprocessing:
  - Fit scalers/encoders on training data only
  - Generate lagged and rolling features strictly from past data
  - Keep the test period fully out-of-sample

---

## Artifacts
- Final Notebook (English): [Final5.ipynb](https://github.com/AnhThuHoang0518/Deep-Learning-model-for-Sale-Forecasting/blob/main/Final5.ipynb)

---

## How to Run
1. Clone this repository.
2. Open the notebook (e.g., Jupyter or Google Colab): `Final5.ipynb`.
3. Download the Kaggle dataset and place files in a local folder (e.g., `data/`).
4. Update any dataset paths in the notebook to your local data location.
5. Run the notebook cells in order to train and evaluate the model(s).

---

## Results
- The notebook demonstrates end-to-end training and evaluation on the competition dataset with time-aware validation to avoid leakage.
- See the final sections of [Final5.ipynb](https://github.com/AnhThuHoang0518/Deep-Learning-model-for-Sale-Forecasting/blob/main/Final5.ipynb) for evaluation metrics and qualitative analysis.

---

## Acknowledgments
- Course: Advanced Data Analytics
- Authors: Hoang Anh Thu, Nguyen Quynh Chi, and Le Hai Anh
