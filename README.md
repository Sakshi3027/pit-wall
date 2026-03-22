# 🏎️ Pit Wall — F1 Intelligence Platform

A full-stack AI-powered Formula 1 intelligence platform built with Python, featuring live 2026 season data, ML predictions, and an AI Race Engineer chatbot.

## Features
- **Live Standings** — Real-time 2026 F1 driver & constructor championship data
- **Race Analysis** — Lap times, tire strategy & pace analysis via FastF1
- **Race Predictor** — XGBoost ML model predicting podium probabilities
- **Season Championship** — Monte Carlo simulator forecasting WDC winner
- **AI Race Engineer** — Gemini-powered chatbot with live F1 context

## Tech Stack
- **Data**: FastF1, OpenF1, Jolpica (Ergast replacement)
- **ML**: XGBoost, scikit-learn, Monte Carlo simulation
- **Frontend**: Streamlit + Plotly
- **AI**: Google Gemini 2.0 Flash

## Setup
```bash
conda create -n pitwall python=3.11 -y
conda activate pitwall
pip install -r requirements.txt
cp .env.example .env  # add GEMINI_API_KEY
streamlit run frontend/app.py
```

## Model Performance
- Race podium predictor: **90.7% accuracy** on 2021-2023 test data
- Season simulator: **10,000 Monte Carlo runs** per simulation
- Training data: 2022-2024 seasons (300+ race entries)
