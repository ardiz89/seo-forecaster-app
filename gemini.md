# Project Constitution: System Pilot (B.L.A.S.T. Protocol)

## 1. Data Schemas

### 1.1 Input: GSC Data (CSV)
Raw export from Google Search Console.
```json
{
  "columns": {
    "date": "YYYY-MM-DD (Required)",
    "clicks": "Integer (Required, Target)",
    "impressions": "Integer (Optional)",
    "ctr": "Float (Optional)",
    "position": "Float (Optional)",
    "searchType": "String (Optional)",
    "dataState": "String (Optional)",
    "aggregationType": "String (Optional)"
  },
  "constraints": {
    "min_rows": 60,
    "required_columns": ["date", "clicks"]
  }
}
```

### 1.2 Input: Regressors (Excel)

**Sheet: Eventi**
```json
{
  "columns": {
    "name": "String (Description)",
    "date": "YYYY-MM-DD (Start Date)",
    "template_type": "String (Foreign Key to Templates)",
    "custom_duration_days": "Integer (Nullable, Override)",
    "custom_impact": "Float (Nullable, Override [-1 to 1])",
    "notes": "String (Optional)"
  }
}
```

**Sheet: Template**
```json
{
  "columns": {
    "template_name": "String (Unique ID)",
    "event_type": "String (Category: algorithm, content, etc.)",
    "default_duration_days": "Integer (Default Duration)",
    "default_impact": "Float (Default Impact [-1 to 1])",
    "regressor_type": "String (Enum: 'decay', 'window')",
    "description": "String (Human Readable)"
  }
}
```

### 1.3 Internal: Forecast Config
```json
{
  "horizon_days": "Integer (Default: 90)",
  "seasonality_mode": "String ('multiplicative' | 'additive')",
  "changepoint_prior_scale": "Float",
  "seasonality_prior_scale": "Float",
  "yearly_seasonality": "Boolean",
  "weekly_seasonality": "Boolean"
}
```

## 2. Behavioral Rules

### 2.1 Regressor Logic (Strict)
1.  **Inheritance:**
    *   If `Event.custom_duration_days` is NULL -> Use `Template.default_duration_days`
    *   If `Event.custom_impact` is NULL -> Use `Template.default_impact`
2.  **Formulas:**
    *   **Window:** `impact` if `t` in `[date, date + duration]`, else `0`
    *   **Decay:** `impact * exp(-t / (duration / 3))` for `t` in `[0, duration]`, else `0`
3.  **Validation:**
    *   `template_type` MUST exist in `Template` sheet.
    *   `impact` values must be likely normalized or handled as coefficients. (User specified -1 to +1 range for impact input, likely meaning percent change or coefficients).

### 2.2 User Interface (Streamlit)
- **Language:** Italian (IT).
- **Error Handling:**
    - Green: Success.
    - Yellow: Warnings (e.g., <90 days data).
    - Red: Blocking Errors (Missing columns, invalid templates).
- **Visuals:** Plotly Line Chart (Historical + Forecast + Confidence Interval).

### 2.3 Do Not Rules
- DO NOT run forecast with < 60 days of data (blocking error).
- DO NOT guess regressor parameters if missing; fail or skip specific event with warning.

## 3. Architectural Invariants
- **Forecasting Engine:** Facebook Prophet.
- **File Parsing:** Pandas for CSV/Excel.
- **State Management:** Streamlit Session State for data persistence across reruns.

## 4. Maintenance Log
| Date | Event | Status |
|------|-------|--------|
| 2026-02-04 | Project Initialized | Active |
| 2026-02-04 | Schema Defined (GSC + Excel) | Active |
