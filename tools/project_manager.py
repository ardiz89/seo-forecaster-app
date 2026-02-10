import os
import json
import pandas as pd
import shutil

PROJECTS_DIR = "user_projects"

def ensure_projects_dir():
    if not os.path.exists(PROJECTS_DIR):
        os.makedirs(PROJECTS_DIR)

def get_all_projects():
    ensure_projects_dir()
    return [d for d in os.listdir(PROJECTS_DIR) if os.path.isdir(os.path.join(PROJECTS_DIR, d))]

def create_new_project(name):
    ensure_projects_dir()
    safe_name = "".join([c for c in name if c.isalnum() or c in (' ', '_', '-')]).strip()
    if not safe_name: return False, "Nome non valido"
    
    path = os.path.join(PROJECTS_DIR, safe_name)
    if os.path.exists(path):
        return False, "Progetto esistente"
    
    os.makedirs(path)
    return True, safe_name

def save_scenario(project_name, scenario_name, forecast_df, events, metrics_dict):
    """Saves a scenario to the project folder."""
    project_path = os.path.join(PROJECTS_DIR, project_name)
    if not os.path.exists(project_path):
        return False, "Progetto non trovato"
        
    # Prepare Metadata
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    safe_scen_name = "".join([c for c in scenario_name if c.isalnum() or c in (' ', '_', '-')]).strip()
    if not safe_scen_name: safe_scen_name = "scenario"
    
    file_id = f"{ts}_{safe_scen_name}"
    csv_filename = f"{file_id}.csv"
    
    # Save Forecast Data (Lite version)
    # We save ds, yhat, and maybe yhat_lower/upper
    save_cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
    # Ensure they exist
    valid_cols = [c for c in save_cols if c in forecast_df.columns]
    
    csv_path = os.path.join(project_path, csv_filename)
    forecast_df[valid_cols].to_csv(csv_path, index=False)
    
    # Load/Update Index
    index_path = os.path.join(project_path, "index.json")
    scenarios = []
    if os.path.exists(index_path):
        try:
            with open(index_path, 'r') as f:
                scenarios = json.load(f)
        except: pass
        
    # Extract total clicks from metrics or dataframe future sum
    total_clicks = 0
    if metrics_dict and 'forecast_total' in metrics_dict:
        total_clicks = metrics_dict['forecast_total']
    else:
        # Fallback calc
        future_mask = forecast_df['ds'] > pd.Timestamp.now()
        total_clicks = forecast_df.loc[future_mask, 'yhat'].sum()

    new_entry = {
        "id": file_id,
        "name": scenario_name,
        "file": csv_filename,
        "created_at": ts,
        "total_clicks": float(total_clicks),
        "events_count": len(events),
        "events_summary": [e['name'] for e in events]
    }
    
    scenarios.append(new_entry)
    
    with open(index_path, 'w') as f:
        json.dump(scenarios, f, indent=2)
        
    return True, "Scenario salvato correttamente"

def load_scenarios(project_name):
    project_path = os.path.join(PROJECTS_DIR, project_name)
    index_path = os.path.join(project_path, "index.json")
    if not os.path.exists(index_path):
        return []
    try:
        with open(index_path, 'r') as f:
            data = json.load(f)
            # Sort by date desc
            return sorted(data, key=lambda x: x['created_at'], reverse=True)
    except:
        return []

def load_scenario_df(project_name, filename):
    path = os.path.join(PROJECTS_DIR, project_name, filename)
    if os.path.exists(path):
        df = pd.read_csv(path)
        if 'ds' in df.columns:
            df['ds'] = pd.to_datetime(df['ds'])
        return df
    return None

def delete_scenario(project_name, scenario_id):
    project_path = os.path.join(PROJECTS_DIR, project_name)
    index_path = os.path.join(project_path, "index.json")
    
    scenarios = load_scenarios(project_name)
    target = next((s for s in scenarios if s['id'] == scenario_id), None)
    
    if target:
        # Remove file
        csv_path = os.path.join(project_path, target['file'])
        if os.path.exists(csv_path):
            try:
                os.remove(csv_path)
            except: pass
            
        # Update Index
        new_list = [s for s in scenarios if s['id'] != scenario_id]
        with open(index_path, 'w') as f:
            json.dump(new_list, f, indent=2)
        return True
    return False
