
import pandas as pd
import numpy as np

def analyze_gsc_data_heuristics(df):
    """
    Analyzes the GSC dataframe (date, clicks/y) to suggest Prophet parameters.
    Returns a dictionary of suggestions and reasons.
    """
    suggestions = {
        "seasonality_mode": "additive",
        "yearly_seasonality": "auto",
        "weekly_seasonality": True,
        "changepoint_prior_scale": 0.05,
        "changepoint_range": 0.8,
        "reasons": []
    }
    
    if df is None or df.empty:
        return suggestions
        
    # Ensure correct types
    df = df.sort_values('date')
    y = df['clicks'].values
    dates = df['date'].dt.date
    
    # 1. Check Data Length
    days = (df['date'].max() - df['date'].min()).days
    if days < 180:
         suggestions['yearly_seasonality'] = False
         suggestions['reasons'].append("Meno di 6 mesi di dati: Stagionalità annuale disabilitata.")
    elif days > 370:
         suggestions['yearly_seasonality'] = True
         suggestions['reasons'].append("Più di 1 anno di dati: Stagionalità annuale attiva.")
    
    # 2. Check Seasonality Mode (Multiplicative vs Additive)
    # Simple heuristic: Check if variance of residuals increases with trend level?
    # Simpler: Split data in two halves. Compare (std / mean). If coeff of variation is similar, maybe additive. 
    # If std grows with mean, multiplicative.
    
    half = len(df) // 2
    part1 = df.iloc[:half]['clicks']
    part2 = df.iloc[half:]['clicks']
    
    if part1.mean() > 0 and part2.mean() > 0:
        cv1 = part1.std() / part1.mean()
        cv2 = part2.std() / part2.mean()
        
        # If mean doubled but CV stayed similar -> Multiplicative likelihood high?
        # Actually, Prophet doc: "if the magnitude of the seasonality grows with the trend"
        # We can look at this later. For SEO, usually multiplicative is safer if high growth.
        # Let's default to additive unless strong growth + variance growth.
        
        mean_growth = part2.mean() / part1.mean()
        std_growth = part2.std() / part1.std()
        
        if mean_growth > 1.5 and std_growth > 1.3:
            suggestions['seasonality_mode'] = 'multiplicative'
            suggestions['reasons'].append("Trend e varianza in crescita: Suggerita modalità 'multiplicative'.")
            
    # 3. Weekly Seasonality
    # Group by weekday and check variance
    df['weekday'] = df['date'].dt.weekday
    weekly_means = df.groupby('weekday')['clicks'].mean()
    if weekly_means.max() - weekly_means.min() > (y.mean() * 0.2):
         suggestions['weekly_seasonality'] = True
         suggestions['reasons'].append("Rilevata forte variazione settimanale (es. calo weekend).")
    else:
         suggestions['weekly_seasonality'] = False
         suggestions['reasons'].append("Variazione settimanale minima.")

    # 4. Changepoint Prior Scale (Trend flexibility)
    # If data is very noisy or has sharp jumps, increase scale.
    # We can measure 'volatility'.
    # Calculate difference between consecutive points
    diffs = np.abs(np.diff(y))
    volatility = np.mean(diffs) / np.mean(y) if np.mean(y) > 0 else 0
    
    if volatility > 0.5:
        suggestions['changepoint_prior_scale'] = 0.1
        suggestions['reasons'].append("Alta volatilità rilevata: Changepoint Scale aumentato a 0.1 per flessibilità.")
    elif volatility < 0.1:
        suggestions['changepoint_prior_scale'] = 0.01
        suggestions['reasons'].append("Bassa volatilità: Changepoint Scale ridotto a 0.01 per trend più rigido.")

    return suggestions
