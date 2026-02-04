# SOP: Forecasting Engine (Prophet)

## 1. Goal
Configure and execute the Facebook Prophet model using historical data and calculated regressors.

## 2. Inputs
- `history_df`: DataFrame with `ds` (date) and `y` (clicks).
- `regressor_events`: List of dicts (from Ingestion SOP).
- `config`: Dict with parameters (horizon, seasonality_mode, etc.).

## 3. Logic & Rules

### 3.1 Pre-Processing Regressors
Prophet requires regressors as columns in the historical dataframe (and future dataframe).
1.  **Initialize Regressor Columns:** Add columns for each *unique* event/regressor or aggregate them?
    - *Decision:* Since events can overlap, we create a separate column for each EVENT instance is too much.
    - *Better Approach:* We create aggregate regressor columns based on `event_type` OR specific named events if they are major.
    - *Revised Strategy per User Request:* User defined "Eventi" which have impacts.
    - **Implementation:**
        - We apply the "Logic Regressors" formulas to create numerical vectors.
        - For each event in `regressor_events`:
            - Let `vec` be a zero array matching the timeline (history + future).
            - Apply "Window" or "Decay" formula starting at `event.date`.
            - Add `vec` as a column named `regressor_{i}_{clean_name}` to the dataframe.
            - Register this column with `m.add_regressor()`.

### 3.2 Model Initialization
1.  **Setup:**
    ```python
    m = Prophet(
        seasonality_mode=config.get('mode', 'multiplicative'),
        yearly_seasonality=config.get('yearly', 'auto'),
        weekly_seasonality=config.get('weekly', True),
        daily_seasonality=False
    )
    ```
2.  **Add Regressors:** Iterate through the columns created in 3.1 and call `m.add_regressor(name)`.

### 3.3 Training & Forecasting
1.  **Fit:** `m.fit(history_df)`
2.  **Future Dataframe:** `future = m.make_future_dataframe(periods=config['horizon_days'])`
3.  **Apply Regressors to Future:** Ensure the regressor columns in `future` are populated using the same formulas from 3.1 extending into the future.
4.  **Predict:** `forecast = m.predict(future)`

## 4. Output Schema
Returns a dictionary or object:
```python
{
    "model": Prophet object,
    "forecast": pd.DataFrame, # The full forecast df
    "metrics": {
       "historical_mean": float,
       "forecast_mean": float,
       "delta_abs": float,
       "delta_perc": float
    }
}
```
