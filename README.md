# 📈 SEO Organic Traffic Forecaster

Application for forecasting organic traffic using Google Search Console data and Facebook Prophet, with support for custom events (Algorithm Updates, Marketing campaigns).

## 🚀 Quick Start
1.  Double-click `run_app.bat`
2.  Or run manually: `python -m streamlit run app.py`

## 📁 Data Requirements

### 1. GSC CSV
Export from Google Search Console. Must contain:
- `date`
- `clicks`
- At least 60 days of history.

### 2. Regressors (Excel)
Must have two sheets:
- **Template**: Definitions of event types (default impact/duration).
- **Eventi**: Specific instances of events with dates.

## 🛠️ Architecture
- **Frontend**: Streamlit
- **Engine**: Facebook Prophet
- **Visualization**: Plotly

## 📍 File Structure
- `app.py`: Main application UI.
- `tools/`: Logic modules.
    - `ingest_data.py`: Data loading.
    - `run_forecast.py`: Prophet execution.
    - `regressor_logic.py`: Mathematical models (Decay/Window).
- `architecture/`: Technical SOPs.
