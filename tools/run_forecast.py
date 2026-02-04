import pandas as pd
import numpy as np
from prophet import Prophet
from tools.regressor_logic import apply_regressors

from prophet.utilities import regressor_coefficients

def calculate_metrics(y_true, y_pred):
    """Calculates MAPE, RMSE, MAE using numpy."""
    # Remove NaNs
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {"mape": 0, "rmse": 0, "mae": 0}

    # Manual implementations to avoid sklearn dependency
    # MAPE
    # Avoid division by zero
    non_zero_mask = y_true != 0
    mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    
    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    # MAE
    mae = np.mean(np.abs(y_true - y_pred))
    
    return {"mape": mape, "rmse": rmse, "mae": mae}

def execute_forecast(history_df, events, config):
    """
    Runs Prophet forecast.
    
    Args:
        history_df: DF with 'date' and 'clicks'
        events: List of event dicts
        config: Dict of params 
               (horizon_days, seasonality_mode, changepoint_prior_scale, etc.)
               
    Returns:
        Dict: {
            "forecast": df,
            "model": object,
            "metrics": dict
        }
    """
    # 1. Prepare Data for Prophet
    df = history_df.rename(columns={'date': 'ds', 'clicks': 'y'})
    
    # 2. Setup Future Dataframe (we need it early to calculate regressors for both history and future)
    horizon = config.get('horizon_days', 90)
    
    # We can't use make_future_dataframe yet because we don't have the model fitted.
    # But we need the full timeline to calculate regressors accurately if we want one pass.
    # Actually, standard Prophet flow:
    # 1. Add regressors to history
    # 2. Fit
    # 3. Create future
    # 4. Add regressors to future
    # 5. Predict
    
    # 3. Initialize Model
    m = Prophet(
        seasonality_mode=config.get('seasonality_mode', 'multiplicative'),
        yearly_seasonality=config.get('yearly_seasonality', 'auto'),
        weekly_seasonality=config.get('weekly_seasonality', True),
        daily_seasonality=config.get('daily_seasonality', False),
        changepoint_prior_scale=config.get('changepoint_prior_scale', 0.05),
        seasonality_prior_scale=config.get('seasonality_prior_scale', 10.0),
        changepoint_range=config.get('changepoint_range', 0.8)
    )
    
    # 3. Separate Events: Fit (Past) vs Override (Future Only)
    events_to_fit = []
    events_to_override = []
    
    # We need to check if the event overlaps with history
    history_min = pd.to_datetime(history_df['date'].min())
    history_max = pd.to_datetime(history_df['date'].max())
    
    for evt in events:
        evt_start = pd.to_datetime(evt['date'])
        
        # If event starts AFTER history ends -> Override
        # If event starts BEFORE history ends -> Fit (Prophet learns it)
        # Note: Even if it started in history but has 0 duration overlap?
        # Let's keep it simple: If start > history_max, it's future only.
        if evt_start > history_max:
             events_to_override.append(evt)
        else:
             events_to_fit.append(evt)

    # 4. Add Regressors to History (Only Fit events)
    df_with_reg, reg_columns = apply_regressors(df, events_to_fit)
    
    # Register columns with Prophet
    for col in reg_columns:
        m.add_regressor(col)
        
    # 5. Fit
    m.fit(df_with_reg)
    
    # 6. Future
    future = m.make_future_dataframe(periods=horizon)
    
    # 7. Add Regressors to Future (Only Fit events)
    future_with_reg, _ = apply_regressors(future, events_to_fit)
    
    # 8. Predict
    forecast = m.predict(future_with_reg)
    
    # --- MANUAL OVERRIDE LOGIC ---
    # Apply impact of future-only events manually to yhat
    active_overrides = []
    
    # We iterate over future-only events and calculate their theoretical impact curve
    # Then simply ADD it to yhat, yhat_lower, yhat_upper
    
    for evt in events_to_override:
         # Calculate the regressor vector for the whole future timeline
         # We can reuse apply_regressors but passing just this single event
         temp_df, new_cols = apply_regressors(future_with_reg[['ds']].copy(), [evt])
         col_name = new_cols[0]
         
         # The vector contains 0s and values (e.g. 0.5 for impact)
         # We assume the impact is additive to the Baseline (yhat)
         # If the model is multiplicative, this changes things:
         # Multiplicative: y = trend * seasonality * (1 + regressors)
         # Additive: y = trend + seasonality + regressors
         
         impact_vector = temp_df[col_name].values
         
         if config.get('seasonality_mode') == 'multiplicative':
             # Here impact is a % change? 
             # Prophet regressors in multiplicative mode are percentages?
             # Actually Prophet: y(t) * (1 + beta * regressor(t))
             # If we want to force impact, we assume 'custom_impact' from Excel IS the beta * regressor.
             # E.g. Impact 0.2 means +20%.
             # So we multiply yhat by (1 + impact_vector)
             
             # But wait, user input 0.5 or -1.0. 
             # If impact is -0.5 (-50%), we want yhat * (1 - 0.5) = yhat * 0.5
             
             multiplier = 1.0 + impact_vector
             forecast['yhat'] *= multiplier
             forecast['yhat_lower'] *= multiplier
             forecast['yhat_upper'] *= multiplier
             
         else:
             # Additive mode.
             # Impact 1000 means +1000 clicks.
             # But user input is likely small float like 0.5?
             # User said range [-1, 1]. In additive mode, 1 click is nothing.
             # So likely the user THINKS in percentage even if mode is additive?
             # Or maybe they want to shift the baseline?
             
             # SAFE BET: Treat User Impact as PERCENTAGE CHANGE regardless of mode?
             # Or treat as absolute?
             # Given "Migrazione sito" (Site Migration), impact is % drop usually.
             
             # Let's assume User Impact is ALWAYS % change (e.g. -0.2 = -20%).
             multiplier = 1.0 + impact_vector
             forecast['yhat'] *= multiplier
             forecast['yhat_lower'] *= multiplier
             forecast['yhat_upper'] *= multiplier
         
         active_overrides.append(evt['name'])

    # 9. Diagnostics & Debug Info
    debug_info = {
        "regressor_diagnostics": [],
        "data_check": {},
        "overrides": active_overrides
    }
    
    # A. Check Input Data (Are regressors actually non-zero?)
    for col in reg_columns:
        # History check
        h_non_zeros = (df_with_reg[col] != 0).sum()
        h_max = df_with_reg[col].max()
        
        # Future check
        f_non_zeros = (future_with_reg[col] != 0).sum()
        f_max = future_with_reg[col].max()
        
        debug_info["data_check"][col] = {
            "history_non_zeros": int(h_non_zeros),
            "history_max_val": float(h_max),
            "future_non_zeros": int(f_non_zeros),
            "future_max_val": float(f_max)
        }

    # B. Extract Coefficients
    try:
        coefs = regressor_coefficients(m)
        debug_info["coefficients"] = coefs
    except Exception as e:
        debug_info["coeff_error"] = str(e)

    # C. Check Impact on Forecast (The component in the result df)
    # The component column usually maps to the regressor name if added separately, 
    # or inside 'extra_regressors_multiplicative' / 'additive'.
    # If we added 'reg_0_foo', prophet usually creates a column 'reg_0_foo' in forecast df too.
    for col in reg_columns:
        if col in forecast.columns:
            impact_abs = forecast[col].abs().sum()
            impact_max = forecast[col].abs().max()
            debug_info["regressor_diagnostics"].append({
                "name": col,
                "total_abs_impact": float(impact_abs),
                "max_impact": float(impact_max)
            })

    # 10. Metrics
    # Calculate historical fit metrics (Performance on training data)
    historical_forecast = forecast[forecast['ds'].isin(df['ds'])]
    metric_df = pd.merge(df, historical_forecast[['ds', 'yhat']], on='ds')
    perf_metrics = calculate_metrics(metric_df['y'], metric_df['yhat'])

    hist_mean = df['y'].mean()
    # Forecast mean (only future part)
    future_mask = forecast['ds'] > df['ds'].max()
    forecast_mean = forecast.loc[future_mask, 'yhat'].mean()
    
    delta_abs = forecast_mean - hist_mean
    delta_perc = (delta_abs / hist_mean) * 100 if hist_mean != 0 else 0
    
    return {
        "forecast": forecast,
        "model": m,
        "metrics": {
            "historical_mean": hist_mean,
            "forecast_mean": forecast_mean,
            "delta_abs": delta_abs,
            "delta_perc": delta_perc,
            "mape": perf_metrics['mape'],
            "rmse": perf_metrics['rmse'],
            "mae": perf_metrics['mae']
        },
        "debug_info": debug_info
    }
