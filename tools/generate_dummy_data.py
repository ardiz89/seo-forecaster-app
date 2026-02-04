import pandas as pd
import numpy as np
import os

# Paths
TMP_DIR = "C:\\Users\\undrg\\.gemini\\antigravity\\scratch\\system_pilot_blast\\.tmp"
os.makedirs(TMP_DIR, exist_ok=True)

def generate_data():
    print("Generating Dummy Data...")

    # 1. GSC CSV
    dates = pd.date_range(start="2024-01-01", end="2025-01-31", freq="D")
    base_clicks = 1000
    trend = np.linspace(0, 500, len(dates)) # Increasing trend
    seasonality = 100 * np.sin(np.arange(len(dates)) * (2 * np.pi / 7)) # Weekly
    noise = np.random.normal(0, 50, len(dates))
    
    clicks = base_clicks + trend + seasonality + noise
    clicks = clicks.astype(int)
    
    gsc_df = pd.DataFrame({
        "date": dates,
        "clicks": clicks,
        "impressions": clicks * 10,
        "ctr": 0.1,
        "position": 5.5
    })
    
    gsc_path = os.path.join(TMP_DIR, "dummy_gsc.csv")
    gsc_df.to_csv(gsc_path, index=False)
    print(f"Created: {gsc_path}")

    # 2. Regressors Excel
    # Template Sheet
    templates = pd.DataFrame([
        {
            "template_name": "core_update",
            "event_type": "algorithm",
            "default_duration_days": 14,
            "default_impact": -0.2,
            "regressor_type": "decay",
            "description": "Core algo update"
        },
        {
            "template_name": "black_friday",
            "event_type": "marketing",
            "default_duration_days": 5,
            "default_impact": 0.5,
            "regressor_type": "window",
            "description": "Sales event"
        }
    ])
    
    # Eventi Sheet
    events = pd.DataFrame([
        {
            "name": "Nov Core Update",
            "date": "2024-11-15",
            "template_type": "core_update",
            "custom_duration_days": None,
            "custom_impact": None,
            "notes": "Big drop"
        },
        {
            "name": "BF 2024",
            "date": "2024-11-25",
            "template_type": "black_friday",
            "custom_duration_days": 7, # Override
            "custom_impact": None,
            "notes": ""
        },
        # Future event
        {
            "name": "Spring Sale 2025",
            "date": "2025-03-01",
            "template_type": "black_friday",
            "custom_duration_days": None,
            "custom_impact": 0.3,
            "notes": "Future event"
        }
    ])
    
    excel_path = os.path.join(TMP_DIR, "dummy_regressors.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        events.to_excel(writer, sheet_name="Eventi", index=False)
        templates.to_excel(writer, sheet_name="Template", index=False)
    
    print(f"Created: {excel_path}")

if __name__ == "__main__":
    generate_data()
