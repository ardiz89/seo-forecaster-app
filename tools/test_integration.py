import pandas as pd
import os
from tools.ingest_data import validate_gsc_data, parse_regressors
from tools.run_forecast import execute_forecast

# Paths
TMP_DIR = "C:\\Users\\undrg\\.gemini\\antigravity\\scratch\\system_pilot_blast\\.tmp"
GSC_FILE = os.path.join(TMP_DIR, "dummy_gsc.csv")
REG_FILE = os.path.join(TMP_DIR, "dummy_regressors.xlsx")

def run_test():
    print("Test 1: Load GSC Data")
    df = pd.read_csv(GSC_FILE)
    res_gsc = validate_gsc_data(df)
    if res_gsc['status'] != 'success':
        print(f"FAILED GSC: {res_gsc['message']}")
        return
    print(f"GSC OK. Rows: {len(res_gsc['data'])}")

    print("\nTest 2: Load Regressors")
    res_reg = parse_regressors(REG_FILE)
    if res_reg['status'] != 'success':
        print(f"FAILED Regressors: {res_reg['message']}")
        return
    print(f"Regressors OK. Count: {len(res_reg['data'])}")
    for e in res_reg['data']:
        print(f" - {e['name']} ({e['type']})")

    print("\nTest 3: Run Forecast")
    config = {
        "horizon_days": 90,
        "seasonality_mode": "multiplicative"
    }
    
    try:
        results = execute_forecast(res_gsc['data'], res_reg['data'], config)
        metrics = results['metrics']
        print("Forecast OK.")
        print(f"Hist Mean: {metrics['historical_mean']:.2f}")
        print(f"Future Mean: {metrics['forecast_mean']:.2f}")
        print(f"Delta: {metrics['delta_perc']:.2f}%")
        
        print("\n--- DEBUG INFO ---")
        debug = results.get('debug_info', {})
        if 'data_check' in debug:
            print("Data Check (History NZ / Future NZ):")
            for k, v in debug['data_check'].items():
                print(f"  {k}: {v['history_non_zeros']} / {v['future_non_zeros']}")
                
        if 'coefficients' in debug:
            print("\nCoefficients:")
            print(debug['coefficients'])
            
    except Exception as e:
        print(f"FAILED Forecast: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
