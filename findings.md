# Findings & Research

## Discoveries
- **Application Type:** Local Streamlit Web App.
- **Core Library:** Prophet (Facebook/Meta) for forecasting.
- **Data Sources:** 
    - Google Search Console Export (CSV).
    - Custom Regressors File (Excel with 'Eventi' and 'Template' sheets).
- **Logic:**
    - Custom regressor types: `decay` (exponential) and `window` (constant).
    - Inheritance model for parameters (Event overrides Template).
    - Validation rules are strict (minimum days, required sheets).

## Constraints
- **OS:** Windows.
- **Environment:** Local Dev (Python).
- **No External APIs:** Completely local execution.
- **Language:** Italian only interface.

## Tech Stack
- Frontend: `streamlit`
- Logic: `prophet`, `pandas`, `numpy`
- Visualization: `plotly`
- I/O: `openpyxl` (Excel)
