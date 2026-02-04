import sys
import importlib

def check_libs():
    libs = ['streamlit', 'prophet', 'pandas', 'openpyxl', 'plotly']
    missing = []
    
    print("Checking libraries...")
    for lib in libs:
        try:
            importlib.import_module(lib)
            print(f"[OK] {lib} found")
        except ImportError:
            print(f"[FAIL] {lib} NOT found")
            missing.append(lib)
            
    if missing:
        print(f"\nMissing libraries: {', '.join(missing)}")
        sys.exit(1)
    else:
        print("\nAll systems go.")
        sys.exit(0)

if __name__ == "__main__":
    check_libs()
