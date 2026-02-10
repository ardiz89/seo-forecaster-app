import streamlit as st
import re
import json
import pandas as pd
import time

def handle_chat_actions(response_text, key_suffix):
    """Parses AI response for JSON actions and renders buttons."""
    
    # Extract JSON block
    match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if "suggested_regressors" in data:
                reg_list = data["suggested_regressors"]
                if reg_list:
                    st.caption(f"💡 L'AI propone {len(reg_list)} modifiche ai regressori.")
                    if st.button("👉 Applica Modifiche", key=f"btn_act_{key_suffix}"):
                        count = 0
                        
                        # Debug info
                        with st.expander("🔍 Debug Dati AI (JSON)", expanded=False):
                            st.json(reg_list)
                            if 'events' in st.session_state:
                                st.write("Eventi Attuali (Nomi):", [e['name'] for e in st.session_state.events])
                        
                        if 'events' not in st.session_state: st.session_state.events = []
                        # Work on deep copy
                        current_events = [e.copy() for e in st.session_state.events]
                        
                        for reg in reg_list:
                            try:
                                action = reg.get("action", "add").lower().strip()
                                nm = str(reg.get("name", "Unknown")).strip()
                                
                                # Helper
                                def is_match(n1, n2):
                                    return str(n1).lower().strip() == str(n2).lower().strip()
                                
                                # 1. REMOVE
                                if action == "remove":
                                    init_len = len(current_events)
                                    current_events = [e for e in current_events if not is_match(e['name'], nm)]
                                    if len(current_events) < init_len:
                                        count += 1
                                    continue
                                
                                # 2. UPDATE (with Upsert logic)
                                if action == "update":
                                    updated = False
                                    for e in current_events:
                                        if is_match(e['name'], nm):
                                            if "date" in reg: e["date"] = pd.to_datetime(reg["date"])
                                            if "type" in reg: e["type"] = str(reg["type"])
                                            if "impact" in reg: e["impact"] = float(reg["impact"])
                                            if "duration" in reg: e["duration"] = int(reg["duration"])
                                            updated = True
                                            count += 1
                                            break
                                    if updated:
                                        continue
                                    # Fallthrough to ADD (Upsert)
                                    action = "add"
                                
                                # 3. ADD (or Upsert Fallthrough)
                                if action == "add" or action == "create":
                                    # Check existence again (in case it was explicitly 'add' but exists)
                                    existing = next((e for e in current_events if is_match(e['name'], nm)), None)
                                    
                                    if existing:
                                        # Force Update
                                        if "date" in reg: existing["date"] = pd.to_datetime(reg["date"])
                                        if "type" in reg: existing["type"] = str(reg["type"])
                                        if "duration" in reg: existing["duration"] = int(reg["duration"])
                                        if "impact" in reg: existing["impact"] = float(reg["impact"])
                                        count += 1
                                    else:
                                        # Create New
                                        if "date" in reg:
                                            new_event = {
                                                "name": nm,
                                                "date": pd.to_datetime(reg["date"]),
                                                "type": str(reg.get("type", "step")),
                                                "duration": int(reg.get("duration", 30)),
                                                "impact": float(reg.get("impact", 0.0))
                                            }
                                            current_events.append(new_event)
                                            count += 1
                                        else:
                                            st.warning(f"⚠️ Impossibile creare '{nm}': manca la data.")

                            except Exception as e:
                                st.warning(f"Errore su {nm}: {e}")
                        
                        if count > 0:
                            st.session_state.events = current_events
                            if 'events_editor_key' not in st.session_state: st.session_state.events_editor_key = 0
                            st.session_state.events_editor_key += 1
                            st.session_state.trigger_forecast_run = True # Auto-run forecast
                            st.success(f"✅ Applicati {count} regressori! Ricalcolo Forecast...")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.warning("Nessuna modifica valida applicata. Controlla il Debug.")
                            
        except Exception:
            pass
