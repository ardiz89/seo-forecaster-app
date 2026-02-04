# SOP: Data Ingestion & Validation

## 1. Goal
Parse, validate, and standardize input data from Google Search Console (GSC) CSVs and User Regressor Excels before passing them to the Forecasting Engine.

## 2. Inputs
- `gsc_file`: CSV file object (Streamlit UploadedFile)
- `regressors_file`: Excel file object (Streamlit UploadedFile)

## 3. Logic & Rules

### 3.1 GSC CSV Parsing
1.  **Load CSV:** Use `pandas.read_csv`.
2.  **Check Columns:** Ensure `date` and `clicks` exist.
    - If `date` missing -> Error "Manca colonna 'date'".
    - If `clicks` missing -> Error "Manca colonna 'clicks'".
3.  **Date Conversion:** Convert `date` to datetime objects. Drop rows with invalid dates.
4.  **Sorting:** Sort by `date` ascending.
5.  **Deduplication:** Group by `date` and sum `clicks` (in case of multiple rows per day due to other dimensions).
6.  **Validation:**
    - Count rows. If < 60 days -> Warning "Dati storici insufficienti (< 60 giorni). L'affidabilità sarà bassa."
    - Check for gaps? (Prophet handles gaps, but good to note).

### 3.2 Regressors Excel Parsing
1.  **Load Excel:** Use `pandas.read_excel` with `sheet_name=None` to get all sheets.
2.  **Sheet Check:** Ensure `Eventi` and `Template` sheets exist.
3.  **Parse Templates:**
    - Load `Template` sheet.
    - Set index to `template_name`.
    - Columns: `event_type`, `default_duration_days`, `default_impact`, `regressor_type`.
    - Validate `regressor_type` is in `['decay', 'window']`.
4.  **Parse Events:**
    - Load `Eventi` sheet.
    - **Validations:**
        - Check `template_type` exists in Templates. If not -> Error "Template 'X' non trovato".
    - **Merge/Fill:**
        - Join with Template data on `template_type`.
        - `duration` = `custom_duration_days` if present else `default_duration_days`.
        - `impact` = `custom_impact` if present else `default_impact`.
    - **Cleaning:** Keep only [`date`, `name`, `duration`, `impact`, `regressor_type`].

## 4. Output Schema
Returns a dictionary:
```python
{
    "history": pd.DataFrame(columns=["ds", "y"]), # validated GSC data
    "regressors": list(dict(
        "name": str,
        "date": datetime,
        "duration": int,
        "impact": float,
        "type": str
    )), # cleaner list of events
    "status": "success" | "error",
    "messages": [str]
}
```
