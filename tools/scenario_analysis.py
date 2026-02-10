import pandas as pd
import numpy as np

def calculate_scenario_comparison(forecast_df, baseline_df=None):
    """
    Compares the current forecast (Scenario) with a Baseline forecast.
    Aggregates data by Month and Quarter for reporting.
    
    Returns:
        dict: {
            'monthly': DataFrame (Month, Scenario, Baseline, Delta, Delta%),
            'quarterly': DataFrame (Quarter, Scenario, Baseline, Delta, Delta%)
        }
    """
    # 1. Prepare Data
    if forecast_df is None or forecast_df.empty:
        return None
    
    # Filter Future Only
    today = pd.to_datetime('today')
    scen = forecast_df[forecast_df['ds'] > today].copy()
    
    # Periods
    scen['month'] = scen['ds'].dt.to_period('M')
    scen['quarter'] = scen['ds'].dt.to_period('Q')
    
    # Aggregation
    s_m = scen.groupby('month')['yhat'].sum().rename('Scenario')
    s_q = scen.groupby('quarter')['yhat'].sum().rename('Scenario')
    
    # Baseline Processing
    b_m = None
    b_q = None
    
    if baseline_df is not None and not baseline_df.empty:
        base = baseline_df[baseline_df['ds'] > today].copy()
        base['month'] = base['ds'].dt.to_period('M')
        base['quarter'] = base['ds'].dt.to_period('Q')
        
        b_m = base.groupby('month')['yhat'].sum().rename('Baseline')
        b_q = base.groupby('quarter')['yhat'].sum().rename('Baseline')
    else:
        # Create Dummy Baseline (0) or matching Scenario (0 delta) if none exists?
        # User wants comparison. If no baseline, we can't compare.
        # But we can still return Scenario stats.
        pass

    # 2. Build Comparisons
    
    # --- MONTHLY ---
    if b_m is not None:
        df_m = pd.concat([s_m, b_m], axis=1).sort_index()
    else:
        df_m = pd.DataFrame(s_m).sort_index()
        df_m['Baseline'] = np.nan

    df_m = df_m.fillna(0)
    df_m['Delta (Click)'] = df_m['Scenario'] - df_m['Baseline']
    # Handle division by zero
    df_m['Delta %'] = df_m.apply(lambda row: (row['Scenario'] - row['Baseline']) / row['Baseline'] if row['Baseline'] > 0 else 0, axis=1)
    
    # Formatting
    df_m_reset = df_m.reset_index()
    df_m_reset['Periodo'] = df_m_reset['month'].astype(str)
    
    # --- QUARTERLY ---
    if b_q is not None:
        df_q = pd.concat([s_q, b_q], axis=1).sort_index()
    else:
        df_q = pd.DataFrame(s_q).sort_index()
        df_q['Baseline'] = np.nan
        
    df_q = df_q.fillna(0)
    df_q['Delta (Click)'] = df_q['Scenario'] - df_q['Baseline']
    df_q['Delta %'] = df_q.apply(lambda row: (row['Scenario'] - row['Baseline']) / row['Baseline'] if row['Baseline'] > 0 else 0, axis=1)
    
    df_q_reset = df_q.reset_index()
    df_q_reset['Periodo'] = df_q_reset['quarter'].astype(str)

    return {
        'monthly': df_m_reset[['Periodo', 'Scenario', 'Baseline', 'Delta (Click)', 'Delta %']],
        'quarterly': df_q_reset[['Periodo', 'Scenario', 'Baseline', 'Delta (Click)', 'Delta %']]
    }

def analyze_regressor_impacts(forecast_df, events, target_period_str=None):
    """
    Analyzes the contribution of each active regressor to the total forecast.
    Focuses on a specific target period (e.g., '2026Q4') if provided, otherwise Total Future.
    
    Args:
        forecast_df (pd.DataFrame): The forecast output.
        events (list): List of event dicts to identify regressor columns.
        target_period_str (str, optional): Quarter string like '2026Q4' to filter specific impact.
        
    Returns:
        pd.DataFrame: Table with per-regressor stats.
    """
    if forecast_df is None or not events:
        return pd.DataFrame()
        
    # Filter Future
    today = pd.to_datetime('today')
    future = forecast_df[forecast_df['ds'] > today].copy()
    
    # Define Time Scope
    scope_mask = pd.Series([True] * len(future), index=future.index)
    scope_label = "Totale Futuro"
    
    if target_period_str:
        # Check if quarter
        if 'Q' in target_period_str:
            try:
                # Convert string to Period, then checks dates
                target_q = pd.Period(target_period_str, freq='Q')
                # Start/End date of Q
                s_date = target_q.start_time
                e_date = target_q.end_time
                scope_mask = (future['ds'] >= s_date) & (future['ds'] <= e_date)
                scope_label = f"Impact in {target_period_str}"
            except:
                pass # Fallback to total
    
    # Filter data for calculation
    target_data = future[scope_mask]
    
    if target_data.empty:
        return pd.DataFrame(columns=["Regresso", "Tipo", "Impatto (Click)", "Note"])
        
    results = []
    
    # Check Regressor Columns
    avail_cols = forecast_df.columns.tolist()
    
    for evt in events:
        name = evt.get('name')
        e_type = evt.get('type')
        
        # Prophet regressor column usually matches name
        if name in avail_cols:
            # Sum impact in target period
            total_impact = target_data[name].sum()
            
            # Determine Peak Date (if ramp/decay) ?
            # This requires looking at the series
            
            # Add to list
            results.append({
                "Regresso": name,
                "Tipo": e_type,
                f"{scope_label} (Click)": int(total_impact),
                "Parametro Impact": evt.get('impact', 0)
            })
            
            
    return pd.DataFrame(results)

def calculate_total_yoy_metrics(forecast_df, history_df):
    """
    Calculates the Year-Over-Year variation between the full forecast period and the corresponding historical period.
    Comparison is based on Daily Mean Clicks to handle slightly different interval lengths (leap years, etc).
    
    Returns:
        dict: {
            "status": "ok" | "insufficient_history" | "no_forecast",
            "delta_abs_mean": float,
            "delta_pct_mean": float,
            "forecast_days": int,
            "matched_history_days": int
        }
    """
    if forecast_df is None or forecast_df.empty:
        return {"status": "no_forecast"}
    
    if history_df is None or history_df.empty:
        return {"status": "insufficient_history"}

    today = pd.to_datetime('today')
    future = forecast_df[forecast_df['ds'] > today].copy()
    
    if future.empty:
        return {"status": "no_forecast"}

    start_f = future['ds'].min()
    end_f = future['ds'].max()
    
    # Calculate target historical range (Shifted back 1 year)
    start_h = start_f - pd.DateOffset(years=1)
    end_h = end_f - pd.DateOffset(years=1)
    
    # Check history coverage
    h_df = history_df.copy()
    d_col = 'date' if 'date' in h_df.columns else 'ds'
    c_col = 'clicks' if 'clicks' in h_df.columns else 'y'
    
    # Convert dates if needed
    if not pd.api.types.is_datetime64_any_dtype(h_df[d_col]):
        h_df[d_col] = pd.to_datetime(h_df[d_col])

    hist_min = h_df[d_col].min()
    hist_max = h_df[d_col].max()
    
    # Coverage check: History must start BEFORE start_h and end AFTER end_h (ideally)
    if hist_min > start_h:
        return {
            "status": "insufficient_history",
            "msg": "Mancano dati storici iniziali"
        }
    
    if hist_max < end_h:
        return {
             "status": "insufficient_history",
             "msg": "Mancano dati storici finali"
        }

    # Filter ranges
    mask_h = (h_df[d_col] >= start_h) & (h_df[d_col] <= end_h)
    matched_hist = h_df[mask_h]
    
    if matched_hist.empty:
         return {"status": "insufficient_history", "msg": "Nessun dato storico nel periodo"}

    # Calculate Means
    mean_f = future['yhat'].mean()
    mean_h = matched_hist[c_col].mean()
    
    if mean_h == 0 or pd.isna(mean_h):
        return {"status": "error", "msg": "Media storica zero"}

    delta_abs = mean_f - mean_h
    delta_pct = (delta_abs / mean_h)
    
    return {
        "status": "ok",
        "delta_abs_mean": delta_abs,
        "delta_pct_mean": delta_pct,
        "forecast_mean": mean_f,
        "history_mean": mean_h,
        "period_label": f"{start_f.date()} - {end_f.date()}"
    }
