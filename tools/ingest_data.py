import pandas as pd
import numpy as np
from datetime import datetime

def validate_gsc_data(df):
    """
    Validates and cleans GSC data.
    """
    required_cols = ['date', 'clicks']
    missing = [c for c in required_cols if c not in df.columns]
    
    if missing:
        return {"status": "error", "message": f"Mancano colonne: {', '.join(missing)}"}
    
    # Clean types
    try:
        df['date'] = pd.to_datetime(df['date'])
    except Exception as e:
        return {"status": "error", "message": f"Errore conversione date: {str(e)}"}
    
    # Aggregate duplicates
    if df['date'].duplicated().any():
        df = df.groupby('date')['clicks'].sum().reset_index()
        
    df = df.sort_values('date')
    
    # Check length
    if len(df) < 60:
        return {
            "status": "warning", 
            "message": f"Dati storici insufficienti ({len(df)} giorni). Minimo 60 raccomandati.",
            "data": df
        }
        
    return {"status": "success", "data": df}

def parse_regressors(excel_file):
    """
    Parses Regressor Excel file (both sheets).
    Returns list of event dicts.
    """
    try:
        xls = pd.read_excel(excel_file, sheet_name=None)
    except Exception as e:
        return {"status": "error", "message": f"Impossibile leggere Excel: {str(e)}"}
    
    if "Eventi" not in xls or "Template" not in xls:
        return {"status": "error", "message": "Fogli mancanti. Richiesti: 'Eventi', 'Template'"}
        
    templates_df = xls['Template']
    events_df = xls['Eventi']
    
    # Validate Templates
    req_tpl_cols = ['template_name', 'event_type', 'default_duration_days', 'default_impact', 'regressor_type']
    missing_tpl = [c for c in req_tpl_cols if c not in templates_df.columns]
    if missing_tpl:
        return {"status": "error", "message": f"Foglio Template: mancano colonne {missing_tpl}"}
        
    # Validate Events
    req_evt_cols = ['name', 'date', 'template_type']
    missing_evt = [c for c in req_evt_cols if c not in events_df.columns]
    if missing_evt:
        return {"status": "error", "message": f"Foglio Eventi: mancano colonne {missing_evt}"}
        
    # Process
    processed_events = []
    
    for _, event in events_df.iterrows():
        tpl_entry = templates_df[templates_df['template_name'] == event['template_type']]
        
        if tpl_entry.empty:
            return {"status": "error", "message": f"Template '{event['template_type']}' non trovato per evento '{event['name']}'"}
            
        tpl = tpl_entry.iloc[0]
        
        # Merge logic
        duration = event.get('custom_duration_days')
        if pd.isna(duration) or duration == '':
            duration = tpl['default_duration_days']
            
        impact = event.get('custom_impact')
        if pd.isna(impact) or impact == '':
            impact = tpl['default_impact']
            
        processed_events.append({
            "name": event['name'],
            "date": pd.to_datetime(event['date']),
            "duration": int(duration),
            "impact": float(impact),
            "type": tpl['regressor_type'],
            "event_type": tpl['event_type'] # Store category for grouping if needed
        })
        
    return {"status": "success", "data": processed_events}
