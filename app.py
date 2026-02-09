import streamlit as st
import os
import pandas as pd
import plotly.graph_objects as go
import time
import importlib
import tools.run_forecast
import tools.ingest_data
import tools.regressor_logic
import tools.report_generator
import numpy as np
import copy
import json
from tools.ingest_data import validate_gsc_data, parse_regressors
from tools.run_forecast import execute_forecast
from tools.report_generator import generate_marketing_report, check_openai_credits, analyze_parameters_with_ai, analyze_regressors_with_ai
from tools.param_advisor import analyze_gsc_data_heuristics
from tools.preset_generator import generate_prospecting_events

# Force reload of tools to pick up debug logic changes without restart
importlib.reload(tools.run_forecast)
importlib.reload(tools.ingest_data)
importlib.reload(tools.regressor_logic)
importlib.reload(tools.report_generator)
importlib.reload(tools.param_advisor)
importlib.reload(tools.preset_generator)

# --- Helper Functions ---
def init_session_state():
    if 'events' not in st.session_state:
        st.session_state.events = []
    if 'baseline_forecast' not in st.session_state:
        st.session_state.baseline_forecast = None
    if 'last_run_config' not in st.session_state:
        st.session_state.last_run_config = None
    if 'last_forecast' not in st.session_state:
        st.session_state.last_forecast = None
    if 'last_metrics' not in st.session_state:
        st.session_state.last_metrics = None
    if 'last_debug' not in st.session_state:
        st.session_state.last_debug = None

    if 'generated_report' not in st.session_state:
        st.session_state.generated_report = None
    
    if 'regressor_suggestions' not in st.session_state:
        st.session_state.regressor_suggestions = None

def apply_suggestions_callback():
    """Updates session state widgets with suggested values before rerun."""
    if 'param_suggestions' in st.session_state:
        sugg = st.session_state['param_suggestions']
        # Map suggestions to widget keys
        if 'seasonality_mode' in sugg: st.session_state.seasonality_mode = sugg['seasonality_mode']
        if 'changepoint_prior_scale' in sugg: st.session_state.changepoint_prior_scale = sugg['changepoint_prior_scale']
        if 'changepoint_range' in sugg: st.session_state.changepoint_range = sugg['changepoint_range']
        if 'seasonality_prior_scale' in sugg: st.session_state.seasonality_prior_scale = sugg['seasonality_prior_scale']
        if 'yearly_seasonality' in sugg: st.session_state.yearly_seasonality = sugg['yearly_seasonality']
        if 'weekly_seasonality' in sugg: st.session_state.weekly_seasonality = sugg['weekly_seasonality']
        if 'daily_seasonality' in sugg: st.session_state.daily_seasonality = sugg['daily_seasonality']

def apply_regressor_suggestions():
    """Applies accepted AI suggestions to the events list."""
    if 'regressor_suggestions' in st.session_state and st.session_state.regressor_suggestions:
        suggs = st.session_state.regressor_suggestions.get('suggestions', [])
        current_events = st.session_state.events
        
        updated_count = 0
        for sugg in suggs:
            idx = sugg.get('id')
            new_type = sugg.get('suggested_type')
            
            # Simple validation to ensure index is valid
            if idx is not None and 0 <= idx < len(current_events):
                # Update logic: Only if different? Or force update?
                # Usually user clicks "Apply All". If we want selective apply, we need a form.
                # For this MVP, we apply all suggestions.
                current_events[idx]['type'] = new_type
                updated_count += 1
        
        st.session_state.events = current_events
        st.session_state.regressor_suggestions = None # Clear after applying
        # st.success(f"Aggiornati {updated_count} regressori!") # Cannot verify in callback strictly visual, but state is updated.

init_session_state()

# --- Page Config ---
st.set_page_config(
    page_title="SEO Forecaster",
    page_icon="📈",
    layout="wide"
)

# --- CSS Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("⚙️ Configurazione")


# API Key is now handled via environment variables only for security.
st.session_state['openai_api_key'] = os.getenv("OPENAI_API_KEY", "")

# 1. File Uploads
st.sidebar.header("1. Upload Dati")
gsc_file = st.sidebar.file_uploader("GSC Export (CSV)", type=['csv'])
reg_file = st.sidebar.file_uploader("Regressori (Excel)", type=['xlsx'])

# 2. Configura Forecast
st.sidebar.header("2. Parametri Forecast")

# Aggregation for Chart
agg_mode = st.sidebar.selectbox("Aggregazione Grafico", ["Giornaliero", "Settimanale", "Mensile"], index=0)

horizon = st.sidebar.selectbox(
    "Orizzonte Temporale", 
    [30, 60, 90, 180, 365], 
    index=2
)

with st.sidebar.expander("Avanzate (Prophet)"):
    st.markdown("""
    **Guida ai Parametri:**
    - **Seasonality Mode:** 
        - `multiplicative`: Se la stagionalità cresce col traffico (Standard SEO).
        - `additive`: Se l'ampiezza stagionale è costante.
    - **Changepoint Scale:** (Trend) Flessibilità. ⏫ Alto = reattivo, ⏬ Basso = rigido.
    - **Changepoint Range:** (Trend) % di storia usata per imparare il trend (Default 0.8 = primi 80%).
    - **Seasonality Scale:** (Stagionalità) Quanto è forte la stagionalità.
    - **Singole Stagionalità:** Forza/Disabilita componenti specifiche.
    """)
    
    seasonality = st.selectbox("Seasonality Mode", ["multiplicative", "additive"], index=0, key="seasonality_mode")
    
    col_cp1, col_cp2 = st.columns(2)
    with col_cp1:
        changepoint_scale = st.slider("Changepoint Prior Scale", 0.001, 0.5, 0.05, 0.001, key="changepoint_prior_scale")
    with col_cp2:
        changepoint_range = st.slider("Changepoint Range", 0.1, 1.0, 0.8, 0.1, key="changepoint_range")

    seasonality_scale = st.slider("Seasonality Prior Scale", 0.01, 20.0, 10.0, 0.1, key="seasonality_prior_scale")
    
    st.markdown("---")
    st.caption("Componenti Stagionali")
    # Toggles for specific seasonalities
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        yearly_seas = st.selectbox("Yearly", ["auto", True, False], index=0, key="yearly_seasonality")
    with col_s2:
        weekly_seas = st.selectbox("Weekly", ["auto", True, False], index=1, key="weekly_seasonality") # Default True usually better
    with col_s3:
        daily_seas = st.selectbox("Daily", ["auto", True, False], index=2, key="daily_seasonality") # Default False for SEO usually

config = {
    "horizon_days": horizon,
    "seasonality_mode": seasonality,
    "changepoint_prior_scale": changepoint_scale,
    "changepoint_range": changepoint_range,
    "seasonality_prior_scale": seasonality_scale,
    "yearly_seasonality": yearly_seas,
    "weekly_seasonality": weekly_seas,
    "daily_seasonality": daily_seas
}

# --- Main Interface ---
st.title("📈 SEO Organic Traffic Forecaster")
st.markdown("Genera previsioni di traffico basate su dati storici e regressori personalizzati (Core Updates, Stagionalità, Eventi).")

if not gsc_file:
    st.info("👋 Per iniziare, carica il file CSV di Google Search Console.")
    # Show dummy data option?
    # Keeping it simple for now.
    st.stop()

# --- Execution Logic ---
if gsc_file:
    # 1. Ingest GSC
    with st.spinner("Analisi dati GSC..."):
        try:
            df_gsc = pd.read_csv(gsc_file)
            res_gsc = validate_gsc_data(df_gsc)
            
            if res_gsc['status'] == 'error':
                st.error(f"Errore GSC: {res_gsc['message']}")
                st.stop()
            elif res_gsc['status'] == 'warning':
                st.warning(res_gsc['message'])
            
            history_df = res_gsc['data']
            st.success(f"✅ Dati caricati: {len(history_df)} giorni analizzati ({history_df['date'].min().date()} - {history_df['date'].max().date()})")
            
        except Exception as e:
            st.error(f"Errore lettura CSV: {str(e)}")
            st.stop()

    # 2. Ingest Regressors (Optional but recommended)
    # Only load from file if events list is empty OR if a new file is uploaded
    # We track file upload via uploader key? No, simpler:
    # If reg_file is present, we try to parse.
    
    # We need a robust way to ONLY parse the file when it changes or is first uploaded,
    # and NOT overwrite manual edits unless user requests.
    # Streamlit file_uploader has no "on_change" easy access without cache.
    # Strategy: Use a specific button "Carica/Reset Regressori" to parse the file into session state.
    
    if reg_file:
         if st.sidebar.button("🔄 Ricarica da Excel (Sovrascrive modifiche)"):
            with st.spinner("Lettura Regressori..."):
                res_reg = parse_regressors(reg_file)
                if res_reg['status'] == 'error':
                    st.error(f"Errore Regressori: {res_reg['message']}")
                else:
                    st.session_state.events = res_reg['data']
                    st.sidebar.success(f"Caricati {len(st.session_state.events)} eventi.")
       
    # --- NUOVA SEZIONE: GUIDA E ANALISI PARAMETRI ---
    st.markdown("---")
    st.subheader("🔮 Guida ai Parametri avanzati di Prophet")
    
    with st.expander("📚 Clicca per leggere la guida dettagliata ai parametri", expanded=False):
        st.markdown("""
        ### Come funziona Prophet?
        Prophet scompone la serie temporale in tre componenti principali: **Trend**, **Stagionalità**, **Festività**.
        
        #### 1. Seasonality Mode (Multiplicative vs Additive)
        *   **Additive**: La stagionalità è costante. Esempio: Ogni Natale vendo 100 panettoni in più, sia che ne venda 1000 o 10000 tot.
        *   **Multiplicative**: La stagionalità è in percentuale. Esempio: Ogni Natale vendo il +20%. Se il mio traffico raddoppia, anche il picco natalizio raddoppia. **(Consigliato per SEO in crescita)**.
        
        #### 2. Trend & Changepoints
        *   **Changepoint Scale (0.001 - 0.5)**: Flessibilità del trend.
            *   Basso (0.001-0.05): Trend rigido, ignora brevi fluttuazioni.
            *   Alto (0.1-0.5): Trend reattivo. Attenzione all'overfitting.
        *   **Changepoint Range (0.8 = 80%)**: Quanta storia usare per imparare il trend. Di solito l'80% iniziale, lasciando il 20% finale seguire il trend appreso (per evitare che oscillazioni recenti stravolgano tutto).
        
        #### 3. Stagionalità (Seasonality)
        *   **Seasonality Prior Scale**: "Forza" della stagionalità.
            *   Basso (0.01-1.0): Stagionalità debole o incerta.
            *   Alto (10.0+): Stagionalità forte e rigida (es. il picco di Natale è SEMPRE quello).
        *   **Componenti**:
            *   **Yearly**: Ciclo annuale (richiede > 1 anno di dati).
            *   **Weekly**: Ciclo settimanale (es. calo weekend).
            *   **Daily**: Ciclo giornaliero (mattina vs sera). Per SEO giornaliero è spesso irrilevante (dati aggregati a giornata), lasciare `False` o `Auto`.
        """)

    # SECTION: AI & SCRIPT ADVISOR
    st.markdown("#### 🔬 Analisi Automatica Dati & Impostazioni")
    st.caption("Analizza la struttura del tuo traffico per trovare i parametri ideali.")
    
    col_anal1, col_anal2 = st.columns([1, 2])
    
    with col_anal1:
        if st.button("📉 Analizza Dati e Suggerisci Preset"):
            with st.spinner("Analisi euristiche in corso..."):
                suggestions = analyze_gsc_data_heuristics(history_df)
                st.session_state['param_suggestions'] = suggestions
                st.session_state['ai_param_explanation'] = None # Reset AI expl
                
    if st.session_state.get('param_suggestions'):
        sugg = st.session_state['param_suggestions']
        
        # Display Suggestions
        with col_anal2:
            st.success("✅ Analisi Completata!")
            st.json(sugg)
            
            # Apply Button
            st.button("👉 Applica questi parametri alla Sidebar", on_click=apply_suggestions_callback)

        # AI Analyst
        st.markdown("##### 🧠 Chiedi all'AI (Analisi Approfondita)")
        if st.button("🤖 Spiega e Conferma con AI"):
            api_key = st.session_state.get('openai_api_key', os.getenv("OPENAI_API_KEY"))
            valid, msg = check_openai_credits(api_key)
            if not valid:
                st.error(msg)
            else:
                with st.spinner("L'AI sta studiando i tuoi dati..."):
                    # Prepare data snippet for AI
                    head = history_df.head(3).to_string()
                    tail = history_df.tail(3).to_string()
                    stats = history_df['clicks'].describe().to_string()
                    data_str = f"Head:\n{head}\n\nTail:\n{tail}\n\nStats:\n{stats}"
                    
                    explanation, err = analyze_parameters_with_ai(str(sugg), data_str, api_key)
                    if err:
                        st.error(err)
                    else:
                        st.session_state['ai_param_explanation'] = explanation
        
        if st.session_state.get('ai_param_explanation'):
            st.info("📝 **Risposta dell'Agente AI:**")
            st.markdown(st.session_state['ai_param_explanation'])

    st.markdown("---") 

    # --- REGRESSOR EDITOR ---
    # --- REGRESSOR SELECTION ---
    # --- REGRESSOR SELECTION ---
    st.subheader("🛠️ Gestione Regressori (Scenario)")
    
    # Tabs for Manual vs Generator
    tab_manual, tab_generator = st.tabs(["📝 Editor Manuale", "⚡ Generatore Prospecting"])
    
    with tab_manual:
        # --- AI REGRESSOR AUDITOR ---
        if st.session_state.events:
            with st.expander("🤖 Analisi Regressori con AI", expanded=True):
                st.caption("L'AI controlla se i tipi di regressori scelti (decay, ramp, etc.) sono coerenti con gli eventi descritti.")
                
                if st.button("🔍 Avvia Audit Regressori"):
                    api_key = st.session_state.get('openai_api_key', os.getenv("OPENAI_API_KEY"))
                    valid, msg = check_openai_credits(api_key)
                    if not valid:
                        st.error(msg)
                    else:
                        with st.spinner("Analisi configurazione regressori in corso..."):
                            resp_json, err = analyze_regressors_with_ai(st.session_state.events, api_key)
                            if err:
                                st.error(err)
                            else:
                                try:
                                    parsed = json.loads(resp_json)
                                    st.session_state.regressor_suggestions = parsed
                                except Exception as e:
                                    st.error(f"Errore nel parsing della risposta AI: {e}")
                
                if st.session_state.regressor_suggestions:
                    suggs = st.session_state.regressor_suggestions.get('suggestions', [])
                    if not suggs:
                        st.info("L'AI non ha suggerimenti. La configurazione sembra ottimale!")
                    else:
                        st.write("### 💡 Suggerimenti Trovati:")
                        
                        # Create comparison table
                        comp_data = []
                        for s in suggs:
                            idx = s['id']
                            if idx < len(st.session_state.events):
                                curr = st.session_state.events[idx]
                                comp_data.append({
                                    "Evento": curr['name'],
                                    "Tipo Attuale": curr['type'],
                                    "Tipo Suggerito": s['suggested_type'],
                                    "Motivazione": s['reason'],
                                    "Confidence": s.get('confidence', 'N/A')
                                })
                        
                        if comp_data:
                            st.dataframe(pd.DataFrame(comp_data))
                            
                            st.button("✅ Applica Modifiche Consigliate", on_click=apply_regressor_suggestions)
                        else:
                            st.info("Nessuna modifica sostanziale suggerita.")

        with st.expander("📖 Guida alla compilazione dei Regressori", expanded=False):
            st.markdown("""
            **Come definire i regressori:**
            
            *   **Tipo (`type`):**
                *   `decay`: Effetto improvviso che decresce nel tempo (es. Core Update negativo, Viral News).
                *   `window`: Effetto costante temporaneo (es. Saldi Black Friday, Problemi tecnici server).
                *   `step`: Cambio permanente (es. Migrazione dominio, cambio CMS, Penalizzazione algoritmica non recuperata). Dura per sempre.
                *   `ramp`: Crescita graduale fino al valore target (es. Link Building, Ottimizzazione SEO On-page progressiva). Mantiene il valore dopo la durata.

            *   **Durata (`duration`):**
                *   Numero di giorni in cui l'evento ha effetto (per `window` e `decay`).
                *   Per `ramp`, è il tempo necessario per raggiungere il 100% dell'impatto.
                *   Ignorato per `step` (dura sempre).
            
            *   **Impatto (`impact`):**
                *   Valore tra **-1.0** e **+1.0**.
                *   Esempio: `-0.2` = Perdita stimata del 20%.
                *   Esempio: `0.5` = Guadagno stimato del 50%.
                *   *Nota:* Per eventi futuri, questo valore viene applicato direttamente come moltiplicatore. Per eventi passati, è un input per il training (Prophet proverà ad avvicinarsi).
                
            *   **Event Type:** Categoria descrittiva (es. `algorithm`, `marketing`, `technical`). Utile per raggruppare analisi future.
            """)

        with st.expander("Modifica / Aggiungi / Rimuovi Regressori", expanded=False): 
        # (Default expanded=False for less clutter if generator is used)
            # Convert list of dicts to DF for editing
            if st.session_state.events:
                df_editor = pd.DataFrame(st.session_state.events)
                
                # Ensure columns order
                cols_order = ['name', 'date', 'type', 'duration', 'impact', 'event_type']
                for c in cols_order:
                    if c not in df_editor.columns:
                        df_editor[c] = None
                
                df_editor = df_editor[cols_order]
                
                # STRICT TYPE ENFORCEMENT TO PREVENT EDITOR RELOADS
                if not df_editor.empty:
                    df_editor['date'] = pd.to_datetime(df_editor['date'])
                    df_editor['duration'] = df_editor['duration'].fillna(30).astype(int)
                    df_editor['impact'] = df_editor['impact'].fillna(0.0).astype(float)
            else:
                # Empty template with correct types
                df_editor = pd.DataFrame({
                    'name': pd.Series(dtype='str'),
                    'date': pd.Series(dtype='datetime64[ns]'),
                    'type': pd.Series(dtype='str'),
                    'duration': pd.Series(dtype='int'),
                    'impact': pd.Series(dtype='float'),
                    'event_type': pd.Series(dtype='str')
                })

            # Add "Seleziona" column for deletion if not exists
            if "seleziona" not in df_editor.columns:
                df_editor.insert(0, "seleziona", False)
            df_editor['seleziona'] = df_editor['seleziona'].astype(bool)

            # Data Editor
            edited_df = st.data_editor(
                df_editor,
                num_rows="dynamic",
                column_config={
                    "seleziona": st.column_config.CheckboxColumn("Elimina?", default=False),
                    "date": st.column_config.DateColumn("Data", format="YYYY-MM-DD"),
                    "type": st.column_config.SelectboxColumn("Tipo", options=["decay", "window", "step", "ramp"]),
                    "impact": st.column_config.NumberColumn("Impatto", help="Range -1.0 a 1.0"),
                    "duration": st.column_config.NumberColumn("Durata (gg)", default=30),
                },
                key="regressor_editor",
                use_container_width=True
            )
            
            # Deletion Logic
            rows_to_delete = edited_df[edited_df["seleziona"] == True]
            
            if not rows_to_delete.empty:
                st.warning(f"Hai selezionato {len(rows_to_delete)} regressori per l'eliminazione.")
                if st.button("🗑️ Elimina Regressori Selezionati", type="primary"):
                    # Filter out selected rows
                    kept_df = edited_df[edited_df["seleziona"] == False].copy()
                    
                    # Drop the selection column before saving back to state
                    if "seleziona" in kept_df.columns:
                        kept_df = kept_df.drop(columns=["seleziona"])
                    
                    # Convert logic
                    try:
                        # Clean types
                        kept_df['date'] = pd.to_datetime(kept_df['date'])
                        kept_df['type'] = kept_df['type'].fillna('window')
                        kept_df['impact'] = kept_df['impact'].fillna(0.0)
                        kept_df['duration'] = kept_df['duration'].fillna(1).astype(int)
                        
                        st.session_state.events = kept_df.to_dict('records')
                        st.success("Regressori eliminati!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Errore durante l'eliminazione: {e}")
            
            # Sync back to session state logic (MANUAL SAVE to fix Reload Bug)
            if st.button("💾 Salva Modifiche", type="primary", help="Clicca per confermare le modifiche alla tabella."):
                try:
                    # We want to save EDITS (typing), but not persist the "seleziona" column forever in session state events.
                    state_df = edited_df.copy()
                    if "seleziona" in state_df.columns:
                        state_df = state_df.drop(columns=["seleziona"])
                        
                    state_df['date'] = pd.to_datetime(state_df['date'])
                    state_df['type'] = state_df['type'].fillna('window')
                    state_df['impact'] = state_df['impact'].fillna(0.0)
                    state_df['duration'] = state_df['duration'].fillna(1).astype(int)
                    
                    st.session_state.events = state_df.to_dict('records')
                    st.success("Modifiche salvate con successo!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Errore durante il salvataggio: {e}") 

    # --- GENERATOR TAB ---
    with tab_generator:
        st.markdown("### 🪄 Generatore Automatico Pacchetti")
        st.caption("Crea scenari complessi selezionando i servizi commerciali venduti. Compila i campi per generare automaticamente gli eventi nel grafico.")

        st.markdown("### 🪄 Generatore Automatico Pacchetti")
        st.caption("Crea scenari complessi selezionando i servizi commerciali venduti. Compila i campi per generare automaticamente gli eventi nel grafico.")

        # REMOVED st.form to allow dynamic interactions (enabling/disabling fields)
        
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            contract_name = st.text_input("Nome Contratto/Progetto", placeholder="es. Cliente Alpha 2026")
        with col_c2:
            default_start = pd.Timestamp.today().date().replace(day=1)
            contract_start = st.date_input("Data Inizio Contratto", value=default_start)
        with col_c3:
            contract_months = st.number_input("Durata (Mesi)", min_value=1, max_value=36, value=12)
        
        st.divider()
        
        # Setup
        st.markdown("#### 1. Setup Iniziale")
        st.info("""
        **Scegli il livello di setup iniziale:**
        *   **Lite**: Include solo fix tecnici essenziali (bloccanti).
        *   **Full**: Include audit tecnico completo e riorganizzazione contenuti.
        *   **Strategy**: Include tutto il Full + Definizione piano editoriale strategico (spinta iniziale più forte).
        """)
        setup_mode = st.radio("Pacchetto Setup", ["nessuno", "lite", "full", "strategy"], 
                              format_func=lambda x: {
                                  "nessuno": "Nessuno",
                                  "lite": "Setup SEO Lite (Tech Fix)",
                                  "full": "Setup SEO Full (Tech + Audit)",
                                  "strategy": "Setup + Content Strategy"
                              }[x], horizontal=True)
        
        st.divider()
        
        # Monthly Packages
        st.markdown("#### 2. Servizi Ricorrenti")
        
        # CONTENT
        with st.expander("📝 Content Marketing", expanded=True):
            st.markdown("**Configura il servizio Content Marketing:**")
            col_cnt1, col_cnt2 = st.columns([1, 2])
            with col_cnt1:
                content_active = st.checkbox("Attiva Content Marketing", value=True)
            with col_cnt2:
                 content_impact = st.number_input("Efficacia Stimata (% Incremento Traffico)", min_value=0.0, max_value=500.0, value=15.0, step=0.5, disabled=not content_active, help="Stima percentuale dell'impatto positivo totale generato dai contenuti alla fine del periodo.")
            
            c_start, c_end = st.slider("Periodo Attivazione Content (Mesi)", 1, 36, (1, 12), key="range_content", disabled=not content_active)
        
        # LINK BUILDING
        with st.expander("🔗 Link Building", expanded=False):
            st.markdown("**Configura il servizio Link Building:**")
            col_lnk1, col_lnk2 = st.columns([1, 2])
            with col_lnk1:
                link_active = st.checkbox("Attiva Link Building", value=False)
            with col_lnk2:
                link_impact = st.number_input("Efficacia Stimata (% Incremento Traffico)", min_value=0.0, max_value=500.0, value=20.0, step=0.5, disabled=not link_active, help="Stima percentuale dell'impatto positivo totale generato dai link alla fine del periodo.")
            
            l_start, l_end = st.slider("Periodo Attivazione Link (Mesi)", 1, 36, (1, 12), key="range_link", disabled=not link_active)

        # TECH & ONPAGE
        with st.expander("⚙️ Technical & On-Page", expanded=False):
            st.markdown("**Manutenzione Tecnica e Ottimizzazione:**")
            tech_mode = st.selectbox("Manutenzione Tecnica", ["none", "care"], 
                                   format_func=lambda x: {"none": "Nessuna", "care": "Technical Care (Interventi Trimestrali)"}[x],
                                   help="Se attivo, aggiunge un evento di fix tecnico ogni 3 mesi.")
            
            col_op1, col_op2 = st.columns(2)
            with col_op1:
                onpage_active = st.checkbox("On-Page Optimization (Mensile)", help="Ottimizzazione continua dei metadati e struttura.")
                onpage_range = st.slider("Mesi On-Page", 1, 36, (1, 12), disabled=not onpage_active)
            with col_op2:
                local_active = st.checkbox("Local / Listing SEO (Mensile)", help="Gestione schede GMB e directory locali.")
                local_range = st.slider("Mesi Local", 1, 36, (1, 12), disabled=not local_active)

        st.divider()
        st.markdown("#### 3. Eventi Speciali")
        st.caption("Seleziona eventi una tantum e definisci il loro impatto specifico.")
        
        # MIGRATION
        migr_active = st.checkbox("Migrazione / Replatform")
        if migr_active:
            st.markdown("--- *Configurazione Migrazione* ---")
            col_mig1, col_mig2, col_mig3 = st.columns(3)
            with col_mig1:
                migr_month = st.number_input("Mese Migrazione", 1, 36, 6)
            with col_mig2:
                migr_drop = st.number_input("Calo Iniziale (%)", min_value=0.0, max_value=100.0, value=10.0, help="Perdita di traffico prevista nei mesi successivi al go-live.")
            with col_mig3:
                migr_growth = st.number_input("Recupero/Crescita (%)", min_value=-100.0, max_value=500.0, value=15.0, help="Incremento del traffico previsto a regime (entro fine contratto).")
        
        # REVAMP
        revamp_active = st.checkbox("Mega Content Revamp")
        if revamp_active:
            st.markdown("--- *Configurazione Revamp* ---")
            col_rev1, col_rev2 = st.columns(2)
            with col_rev1:
                revamp_month = st.number_input("Mese Revamp", 1, 36, 3)
            with col_rev2:
                revamp_impact = st.number_input("Boost Totale (%)", min_value=0.0, max_value=500.0, value=20.0, help="Incremento incrementale del traffico generato dal revamp.")
            
        # ADV
        adv_active = st.checkbox("Campagna Brand/ADV")
        if adv_active:
             adv_month = st.number_input("Mese Inizio Campagna", 1, 36, 1)

        st.divider()

        # Submit Button (Outside Form)
        if st.button("🚀 Genera Scenario Regressori", type="primary"):
            # Build form data dict
            form_data = {
                "contract_start_date": contract_start,
                "contract_months": contract_months,
                "setup_mode": setup_mode,
                
                # Content
                "content_enabled": content_active,
                "content_months": (c_start, c_end) if content_active else (1,1),
                "content_impact_total": content_impact if content_active else 0,
                
                # Link
                "link_enabled": link_active,
                "link_months": (l_start, l_end) if link_active else (1,1),
                "link_impact_total": link_impact if link_active else 0,
                
                # Tech
                "tech_mode": tech_mode,
                "tech_months": (1, contract_months), 
                "onpage_enabled": onpage_active,
                "onpage_months": onpage_range if onpage_active else (1,1),
                "local_enabled": local_active,
                "local_months": local_range if local_active else (1,1),
                
                # Extras
                "extra_events": []
            }
            
            if migr_active:
                form_data['extra_events'].append({
                    'type': 'migration', 
                    'month': migr_month,
                    'drop_pct': migr_drop,
                    'growth_pct': migr_growth
                })
            if revamp_active:
                form_data['extra_events'].append({
                    'type': 'revamp', 
                    'month': revamp_month,
                    'growth_pct': revamp_impact
                })
            if adv_active:
                form_data['extra_events'].append({'type': 'campaign', 'month': adv_month, 'name': 'Campagna Brand'})
            
            # Generate
            new_events = generate_prospecting_events(form_data)
            
            # Replace events
            st.session_state.events = new_events
            st.success(f"Generati {len(new_events)} eventi! Verifica nel tab 'Editor Manuale'.")
            st.rerun()

    st.info(f"ℹ️ Regressori Attivi: {len(st.session_state.events)}")
    
    # 3. Running Forecast
    col_btn_1, col_btn_2 = st.columns([1, 1])
    
    run_forecast_btn = col_btn_1.button("🚀 Genera Forecast", type="primary")
    set_baseline_btn = col_btn_2.button("💾 Imposta come Baseline")
    
    if set_baseline_btn and 'last_forecast' in st.session_state:
        st.session_state.baseline_forecast = copy.deepcopy(st.session_state.last_forecast)
        st.success("Forecast attuale salvato come Baseline per confronti futuri.")

    if run_forecast_btn:
        with st.spinner("Addestramento modello Prophet in corso..."):
            try:
                # Use events from session state
                results = execute_forecast(history_df, st.session_state.events, config)
                
                # Save as last forecast
                st.session_state.last_forecast = results['forecast']
                st.session_state.last_metrics = results['metrics']
                st.session_state.last_debug = results['debug_info'] # Store specific debug for current run
                
            except Exception as e:
                st.error(f"Errore durante il forecast: {str(e)}")
    
    # --- Results Display (Persists across reruns) ---
    if st.session_state.last_forecast is not None:
        try:
            forecast = st.session_state.last_forecast
            metrics = st.session_state.last_metrics
            debug = st.session_state.last_debug
            
            # --- Results Display ---
            st.divider()
            st.subheader("📊 Risultati Previsione")
                
            # Metrics Row
            # Check comparison
            delta_baseline_abs = None
            delta_baseline_perc = None
            
            if st.session_state.baseline_forecast is not None:
                # Calculate difference from baseline
                # Baseline mean vs Current mean logic?
                # Or pure total clicks difference in the overlap period?
                
                # Let's simple compare the metrics mean vs baseline metrics mean
                # But we need baseline metrics stored OR reformat logic needed.
                pass 

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Media Storica", f"{int(metrics['historical_mean'])}")
            col2.metric("Media Forecast (Scenario)", f"{int(metrics['forecast_mean'])}")
            col3.metric("Variazione vs Storico", f"{int(metrics['delta_abs'])}", delta=f"{metrics['delta_perc']:.2f}%")
                
            if st.session_state.baseline_forecast is not None:
                 # Calculate delta vs Baseline
                 # We need to compute baseline mean again or store it.
                 # Let's just compare the forecast column of baseline against current.
                 base_future = st.session_state.baseline_forecast[st.session_state.baseline_forecast['ds'] > history_df['date'].max()]
                 curr_future = forecast[forecast['ds'] > history_df['date'].max()]
                 
                 base_mean = base_future['yhat'].mean()
                 curr_mean = curr_future['yhat'].mean()
                 diff = curr_mean - base_mean
                 pct = (diff / base_mean) * 100 if base_mean != 0 else 0
                 
                 col4.metric("Variazione vs BASELINE", f"{int(diff)}", delta=f"{pct:.2f}%", help="Differenza rispetto al forecast salvato come Baseline.")
            else:
                 col4.metric("Baseline", "Non impostata", help="Clicca 'Imposta come Baseline' per confrontare scenari.")
            
            # Chart
            st.subheader("Trend Temporale: Scenario vs Baseline")
            
            # Filter forecast to avoid showing too much history if not needed, 
            # but usually user wants context.
            
            fig = go.Figure()
            
            # Metrics Performance & Overview
            st.markdown("#### 📉 Performance Modello (Fit Storico)")
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("MAPE (Errore %)", f"{metrics['mape']:.2f}%", help="Mean Absolute Percentage Error. Più basso è meglio.")
            m_col2.metric("RMSE", f"{int(metrics['rmse'])}", help="Root Mean Squared Error.")
            m_col3.metric("MAE", f"{int(metrics['mae'])}", help="Mean Absolute Error.")
            st.divider()

            # Chart
            st.subheader(f"Trend Temporale ({agg_mode})")
            
            # Chart Aggregation Logic
            plot_df_hist = history_df.copy()
            plot_df_forecast = forecast.copy()
            plot_base = None
            
            if st.session_state.baseline_forecast is not None:
                plot_base = st.session_state.baseline_forecast.copy()
            
            if agg_mode == "Settimanale":
                plot_df_hist = plot_df_hist.resample('W', on='date').sum().reset_index()
                plot_df_forecast = plot_df_forecast.resample('W', on='ds').sum().reset_index()
                if plot_base is not None: plot_base = plot_base.resample('W', on='ds').sum().reset_index()
            elif agg_mode == "Mensile":
                plot_df_hist = plot_df_hist.resample('ME', on='date').sum().reset_index()
                plot_df_forecast = plot_df_forecast.resample('ME', on='ds').sum().reset_index()
                if plot_base is not None: plot_base = plot_base.resample('ME', on='ds').sum().reset_index()

            fig = go.Figure()
            
            # Historical
            fig.add_trace(go.Scatter(
                x=plot_df_hist['date'], 
                y=plot_df_hist['clicks'],
                mode='lines',
                name='Storico',
                line=dict(color='#333333', width=1)
            ))
            
            # Forecast
            future_only = plot_df_forecast[plot_df_forecast['ds'] > plot_df_hist['date'].max()]
            fig.add_trace(go.Scatter(
                x=future_only['ds'],
                y=future_only['yhat'],
                mode='lines',
                name='Previsione',
                line=dict(color='#4CAF50', width=2)
            ))
            
            # Baseline Line (if exists)
            if plot_base is not None:
                base_future = plot_base[plot_base['ds'] > plot_df_hist['date'].max()]
                fig.add_trace(go.Scatter(
                    x=base_future['ds'],
                    y=base_future['yhat'],
                    mode='lines',
                    name='BASELINE (Precedente)',
                    line=dict(color='#999999', width=2, dash='dot')
                ))
                
            # Events Markers setup
            if st.session_state.events:
                 # Filter events in range? Plotly handles zooming.
                 pass 

            # Confidence Interval (Only valid for Daily usually, summing upper/lower is mathematically approx but acceptable for viz)
            # We skip CI for aggressive aggregation to avoid confusion or just sum it? 
            # Summing variance is complex. Let's hide CI for Monthly/Weekly for simplicity or just keep it daily.
            # User requested aggregation.
            
            # Events Markers
            if st.session_state.events:
                # Group events by date
                events_by_date = {}
                for evt in st.session_state.events:
                    d = evt['date'].strftime("%Y-%m-%d")
                    if d not in events_by_date:
                        events_by_date[d] = []
                    events_by_date[d].append(evt)

                # Arrays for the "Tokens" trace
                evt_x = []
                evt_y = [] # We place markers at the top or on the forecast line?
                evt_text = []

                # Iterate uniquely
                for date_str, evts in events_by_date.items():
                    count = len(evts)
                    names = [e['name'] for e in evts]
                    
                    # Label text
                    label_text = names[0] if count == 1 else f"{count} eventi"
                    
                    # Hover text (HTML allowed in plotly)
                    hover_content = f"<b>{date_str}</b><br>" + "<br>".join([f"• {n}" for n in names])
                    
                    # Add VLine
                    # Helper to parse TS
                    ts = pd.Timestamp(date_str).timestamp() * 1000
                    
                    fig.add_vline(
                        x=ts, 
                        line_width=1, 
                        line_dash="dot", 
                        line_color="red",
                        annotation_text=label_text,
                        annotation_textangle=-90,
                        annotation_position="top left",
                        opacity=0.5
                    )
                    
                    # Add Point for Hover
                    # Finding Y height for the marker?
                    # If date is in history, use history Y. If future, use forecast Y.
                    # For simplicity, we can let it be somewhat arbitrary or max of entire chart.
                    # Better: Use the max y of the chart to place markers at top?
                    # Or just rely on "x unified" hover.
                    
                    # If we just add a scatter trace with correct x, the hover will pick it up.
                    # Set y to 0 (or some valid value) but make marker invisible.
                    # Actually if y is 0 it might distort auto-range if data is 5000+.
                    # Let's try to match y-value from data.
                    
                    check_date = pd.to_datetime(date_str)
                    current_y = 0
                    
                    # Look in history
                    hist_row = history_df[history_df['date'] == check_date]
                    if not hist_row.empty:
                        current_y = hist_row['clicks'].values[0]
                    else:
                        # Look in forecast
                        fut_row = forecast[forecast['ds'] == check_date]
                        if not fut_row.empty:
                            current_y = fut_row['yhat'].values[0]
                    
                    evt_x.append(date_str)
                    evt_y.append(current_y)
                    evt_text.append(hover_content)

                # Add the invisible "Events" trace for Hover
                fig.add_trace(go.Scatter(
                    x=evt_x,
                    y=evt_y,
                    mode='markers',
                    name='Eventi',
                    marker=dict(size=10, color='red', symbol='diamond'),
                    text=evt_text,
                    hovertemplate="%{text}<extra></extra>"
                ))
            
            # ... (rest of layout)
            
            fig.update_layout(
                template="simple_white",
                hovermode="x unified",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # --- DEBUG INFO ---
            # Force expander open to ensure user sees it
            with st.expander("🛠️ Debug & Diagnostica Regressori", expanded=True):
                # debug is already extracted above
                
                if not debug:
                    st.warning("Nessuna info di debug restituita dal backend.")
                else:
                    st.markdown("### 1. Controllo Dati Input")
                    st.caption("Verifica se i vettori dei regressori sono diversi da zero nella timeline (History vs Future).")
                    
                    if 'data_check' in debug and debug['data_check']:
                        # Convert dict to clean DF
                        check_df = pd.DataFrame.from_dict(debug['data_check'], orient='index')
                        st.dataframe(check_df)
                        
                        # Warning for Future-Only events
                        zero_hist = check_df[check_df['history_non_zeros'] == 0]
                        if not zero_hist.empty:
                            st.warning(f"⚠️ Attenzione: I seguenti eventi non hanno dati storici e Prophet assegnerà loro coefficiente 0 (Nessun impatto): {', '.join(zero_hist.index.tolist())}")

                    else:
                        st.write("Nessun dato di check disponibile.")
                    
                    st.markdown("### 2. Coefficienti Calcolati (Beta)")
                    st.caption("Quanto il modello ha 'imparato' da ogni regressor. Se vicino a 0, il regressor è ignorato.")
                    
                    if 'coefficients' in debug:
                        coefs = debug['coefficients']
                        # Clean up the view
                        cols_show = ['regressor', 'regressor_mode', 'coef', 'coef_lower', 'coef_upper']
                        # Filter for columns that actually exist
                        cols_show = [c for c in cols_show if c in coefs.columns]
                        st.dataframe(coefs[cols_show])

                    st.markdown("### 3. Impatto Reale sul Forecast")
                    st.caption("Contributo totale assoluto al valore finale predetto.")
                    if 'regressor_diagnostics' in debug and debug['regressor_diagnostics']:
                        st.dataframe(pd.DataFrame(debug['regressor_diagnostics']))

                    st.markdown("### 4. Eventi Futuri (Override Manuale)")
                    st.caption("I seguenti eventi sono stati applicati **dopo** il forecast come moltiplicatori diretti (es. -0.5 = -50%), perché non esistono nel passato.")
                    if 'overrides' in debug and debug['overrides']:
                         st.write(debug['overrides'])
                    else:
                         st.write("Nessun override manuale applicato.")

            # Export
            st.subheader("📥 Export Dati")
            csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Scarica CSV Forecast",
                data=csv,
                file_name='seo_forecast.csv',
                mime='text/csv',
            )
            
            # --- AI REPORTING ---
            st.divider()
            st.subheader("🤖 AI Executive Report (GPT-5.1)")
            st.caption("Genera un'analisi strategica proattiva per il cliente.")
            
            col_ai1, col_ai2 = st.columns([1, 2])
            
            with col_ai1:
                # Credit Check
                if st.button("🔑 Verifica API OpenAI"):
                    api_key_to_use = st.session_state.get('openai_api_key', os.getenv("OPENAI_API_KEY"))
                    valid, msg = check_openai_credits(api_key=api_key_to_use)
                    if valid:
                        st.success(msg)
                    else:
                        st.error(msg)
                
                st.write("") # Spacer
                
                if st.button("✨ Genera Report Prospect", type="primary"):
                    with st.spinner("L'AI sta analizzando i dati... (GPT-5.1)"):
                        api_key_to_use = st.session_state.get('openai_api_key', os.getenv("OPENAI_API_KEY"))
                        report_text, err = generate_marketing_report(
                            metrics, 
                            st.session_state.events, 
                            config['horizon_days'], 
                            forecast,
                            api_key=api_key_to_use
                        )
                        
                        if err:
                            st.error(err)
                        else:
                            st.session_state.generated_report = report_text

            with col_ai2:
                if st.session_state.generated_report:
                    st.markdown(st.session_state.generated_report)
                    st.download_button(
                        "Scarica Report .md",
                        st.session_state.generated_report,
                        "smart_report_seo.md"
                    )
                else:
                    st.info("Clicca su 'Genera Report' per creare un'analisi automatica.")

        except Exception as e:
            st.error(f"Errore nella visualizzazione dei risultati: {str(e)}")

