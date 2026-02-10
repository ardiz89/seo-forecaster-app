import numpy as np
import pandas as pd

def apply_regressors(df, events):
    """
    Applies regressor logic to the DataFrame.
    Adds columns for each valid event.
    Returns: DataFrame with added columns, List of column names added
    """
    df = df.copy()
    added_columns = []
    
    # Ensure dates are datetime
    df['ds'] = pd.to_datetime(df['ds'])
    
    for i, event in enumerate(events):
        col_name = f"reg_{i}_{event['name'].lower().replace(' ', '_')}"
        col_name = "".join(c for c in col_name if c.isalnum() or c == '_') # Sanitize
        
        # Initialize zero column
        df[col_name] = 0.0
        
        event_start = event['date']
        duration = event['duration']
        impact = event['impact']
        reg_type = event['type']
        
        # Mask for the relevant period
        # Note: We need to handle the logic row by row or vectorized.
        # Vectorized is faster.
        
        # Calculate days since event start
        # (This will be negative for dates before event)
        days_since = (df['ds'] - event_start).dt.days
        
        if reg_type == 'window':
            # Impact is constant [0, duration]
            mask = (days_since >= 0) & (days_since <= duration)
            df.loc[mask, col_name] = impact
            
        elif reg_type == 'decay':
            # Formula: impact * exp(-t / (duration / 3))
            # Only valid for t in [0, duration]
            mask = (days_since >= 0) & (days_since <= duration)
            
            # Avoid division by zero if duration is 0 (though unlikely)
            tau = duration / 3.0 if duration > 0 else 1.0
            
            decay_values = impact * np.exp(-days_since[mask] / tau)
            df.loc[mask, col_name] = decay_values

        elif reg_type == 'step':
            # Impact is constant from Start DATE until FOREVER (or end of dataframe)
            # Useful for site migrations, permanent penalties, or structural changes.
            mask = (days_since >= 0)
            df.loc[mask, col_name] = impact

        elif reg_type == 'ramp':
            # Linear growth from 0 to Impact over 'Duration' days.
            # Then stays at Impact level? Or drops?
            # Usually "Ramp up to effect". Let's assume it stays at impact after duration (like a slow Step).
            
            # Period 1: Growth [0, duration]
            mask_ramp = (days_since >= 0) & (days_since < duration)
            if duration > 0:
                # Linear: (t / duration) * impact
                df.loc[mask_ramp, col_name] = (days_since[mask_ramp] / duration) * impact
            
            # Period 2: Plateau [duration, infinity]
            mask_plateau = (days_since >= duration)
            df.loc[mask_plateau, col_name] = impact
            
        added_columns.append(col_name)
        
    return df, added_columns

def parse_regressors(file_obj):
    """
    Parses an uploaded regressor file (CSV or Excel).
    Returns dict: {'status': 'ok'|'error', 'data': list_of_dicts|None, 'message': str}
    """
    try:
        # Reset file pointer if needed
        file_obj.seek(0)
        
        if file_obj.name.endswith('.csv'):
            df = pd.read_csv(file_obj)
        elif file_obj.name.endswith('.xlsx'):
            # Read 'Eventi' sheet if exists, else first sheet
            try:
                xls = pd.ExcelFile(file_obj)
                if 'Eventi' in xls.sheet_names:
                    df = pd.read_excel(file_obj, sheet_name='Eventi')
                else:
                    df = pd.read_excel(file_obj)
            except Exception as e:
                return {'status': 'error', 'message': f"Errore lettura Excel: {e}", 'data': None}
        else:
            return {'status': 'error', 'message': "Formato non supportato. Usa CSV o XLSX.", 'data': None}

        # Normalize Columns
        # Expected: name, date, type, duration, impact
        req_cols = ['name', 'date', 'type', 'duration', 'impact']
        
        # Check missing cols
        missing = [c for c in req_cols if c not in df.columns]
        if missing:
             return {'status': 'error', 'message': f"Colonne mancanti nel file: {missing}. Usa il template.", 'data': None}
        
        # Parse Dates
        try:
             df['date'] = pd.to_datetime(df['date'], errors='coerce')
        except:
             return {'status': 'error', 'message': "Errore parsing date. Formato non valido.", 'data': None}

        if df['date'].isnull().any():
             # Drop rows with invalid dates
             df = df.dropna(subset=['date'])
        
        if df.empty:
             return {'status': 'warning', 'message': "Nessun evento valido trovato (controlla le date).", 'data': []}

        # Fill values / Enforce Types
        df['type'] = df['type'].fillna('step').astype(str)
        df['duration'] = df['duration'].fillna(30).astype(int)
        df['impact'] = df['impact'].fillna(0.0).astype(float)
        df['name'] = df['name'].fillna('Evento').astype(str)
        
        if 'event_type' not in df.columns:
            df['event_type'] = 'manual'
        else:
            df['event_type'] = df['event_type'].fillna('manual').astype(str)
            
        # Convert to list of dicts
        events = df.to_dict('records')
        return {'status': 'ok', 'data': events, 'message': 'Success'}

    except Exception as e:
        return {'status': 'error', 'message': str(e), 'data': None}
