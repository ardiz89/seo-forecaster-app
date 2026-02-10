import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import copy
import os
import time
import json
import re
import base64
import importlib

# Imports for Reloading
import tools.run_forecast
import tools.ingest_data
import tools.regressor_logic
import tools.report_generator
from tools import scenario_analysis

# Imports for Usage
from tools.ingest_data import validate_gsc_data
from tools.regressor_logic import apply_regressors, parse_regressors
from tools.run_forecast import execute_forecast
from tools.report_generator import generate_marketing_report, check_openai_credits, analyze_parameters_with_ai, analyze_regressors_with_ai
from tools.chat_actions import handle_chat_actions
from tools.param_advisor import analyze_gsc_data_heuristics
from tools.preset_generator import generate_prospecting_events
from tools.chatbot import chat_with_assistant, prepare_context_data

importlib.reload(tools.run_forecast)
import tools.project_manager
importlib.reload(tools.project_manager)


# --- HELPER FUNCTIONS ---
def render_smart_report(report_text):
    """Parses text and replaces {{TAGS}} with components."""
    parts = re.split(r'\{\{(\w+)\}\}', report_text)
    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip(): st.markdown(part)
        else:
            tag = part.strip().upper()
            if tag == "KPI_SUMMARY":
                m = st.session_state.get('last_metrics', {})
                if m:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Media Prevista", f"{int(m.get('forecast_mean', 0))}")
                    c2.metric("Variazione", f"{m.get('delta_perc', 0):.1f}%", f"{m.get('delta_abs', 0):+.0f}")
                    c3.metric("MAPE", f"{m.get('mape', 0):.1f}%")
                    
                    # Monthly Breakdown
                    monthly = m.get('monthly_data', [])
                    if monthly:
                        st.caption("Proiezione Mensile")
                        # Show first 4 months
                        num_m = min(len(monthly), 4)
                        cols = st.columns(num_m)
                        for i in range(num_m):
                            d = monthly[i]
                            # Format month YYYY-MM
                            cols[i].metric(d['month'], f"{int(d['mean'])} avg", f"Tot: {int(d['sum']/1000)}k")
                        
                        if len(monthly) > 4:
                            with st.expander("Vedi tutti i mesi"):
                                st.dataframe(pd.DataFrame(monthly)[['month', 'mean', 'sum']])
            elif tag == "CHART_TREND":
                fc = st.session_state.get('last_forecast')
                if fc is not None:
                     try:
                         # Filter > Today - 30 days
                         today = pd.to_datetime('today')
                         f_sub = fc[fc['ds'] > (today - pd.Timedelta(days=30))].copy()
                         st.caption("Trend (Ultimi 30gg + Futuro)")
                         st.line_chart(f_sub.set_index('ds')['yhat'], color="#FF4B4B")
                     except: st.error("Errore grafico.")
            elif tag == "EVENTS_TABLE":
                ev = st.session_state.get('events', [])
                if ev: st.dataframe(pd.DataFrame(ev)[['name', 'date', 'type', 'impact']], use_container_width=True)
            else: st.markdown(f"**{{ {tag} }}**")
importlib.reload(tools.ingest_data)
importlib.reload(tools.regressor_logic)
importlib.reload(tools.report_generator)
importlib.reload(tools.param_advisor)
importlib.reload(tools.preset_generator)
importlib.reload(tools.chatbot)
importlib.reload(tools.chat_actions)
importlib.reload(scenario_analysis)

# --- DIALOG DEFINITIONS ---
# Try to import dialog (st.dialog is new in 1.34+, st.experimental_dialog in 1.23+)
DialogDecorator = None
if hasattr(st, "dialog"):
    DialogDecorator = st.dialog
elif hasattr(st, "experimental_dialog"):
    DialogDecorator = st.experimental_dialog

if DialogDecorator:
    @DialogDecorator("✏️ Modifica System Prompt (Fullscreen)")
    def edit_sys_prompt_dialog():
        st.caption("Modifica il prompt di sistema. Usa i tab per vedere l'anteprima Markdown.")
        
        # Load current prompt from session
        if 'chat_config' not in st.session_state:
            st.session_state.chat_config = {}
        curr = st.session_state.chat_config.get("system_prompt", "")
        
        tab_edit, tab_view = st.tabs(["📝 Editor (Testo)", "👁️ Anteprima (Markdown)"])
        
        with tab_edit:
            new_val = st.text_area("System Prompt Source", value=curr, height=500, key="sys_prompt_editor_area")
        
        with tab_view:
            st.markdown("### Anteprima Rendering")
            st.markdown(new_val)
            st.divider()
        
        col_d1, col_d2 = st.columns(2)
        if col_d1.button("💾 Salva e Chiudi", type="primary"):
            st.session_state.chat_config["system_prompt"] = new_val
            st.rerun()
        if col_d2.button("❌ Chiudi senza salvare"):
            st.rerun()

    @DialogDecorator("✏️ Modifica Report Prompt (Fullscreen)")
    def edit_report_prompt_dialog():
        st.caption("Modifica le istruzioni per l'AI Executive Report.")
        curr = st.session_state.get("report_system_prompt", "")
        
        tab_edit, tab_view = st.tabs(["📝 Editor", "👁️ Anteprima"])
        with tab_edit:
            new_val = st.text_area("System Prompt Source", value=curr, height=500, key="rep_prompt_editor_area")
        with tab_view:
            st.markdown(new_val)
            
        c1, c2 = st.columns(2)
        if c1.button("💾 Salva", type="primary"):
            st.session_state.report_system_prompt = new_val
            st.rerun()
        if c2.button("❌ Annulla"):
            st.rerun()

else:
    # Fallback function if dialogs not supported
    def edit_sys_prompt_dialog():
        st.warning("Feature 'Dialog' non supportata da questa versione di Streamlit. Usa l'editor nella sidebar.")
    def edit_report_prompt_dialog():
        st.warning("Feature 'Dialog' non supportata.")

import tools.export_utils
importlib.reload(tools.export_utils)


DEFAULT_SYSTEM_PROMPT = """# SEO Forecasting Assistant - System Prompt (v2.0 - App Optimized)

Sei un assistente esperto in forecasting SEO e analisi predittiva, integrato in una Dashboard proprietaria basata su Prophet.
Il tuo obiettivo è supportare il SEO Manager trasformando i grafici tecnici in strategie di business azionabili.

## 🧠 CONOSCENZA DEL SISTEMA (Specifiche App)

L'utente sta guardando un grafico generato da Facebook Prophet con le seguenti specificità che devi saper interpretare:

### 1. Tipi di Regressori (Eventi)
Il modello usa 4 logiche di impatto specifiche. Quando spieghi i trend, usa questa terminologia:
- **`step` (Salto)**: Cambio strutturale permanente (es. Migrazione, Penalizzazione, Rebranding, Cambio CMS). Il traffico cambia livello e rimane lì.
- **`ramp` (Crescita/Decrescita Progressiva)**: Effetto cumulativo che aumenta nel tempo fino a saturazione (es. Link Building, Content Marketing continuativo, Ottimizzazione On-page). La curva si "impenna" gradualmente.
- **`decay` (Decadimento)**: Effetto shock immediato che svanisce nel tempo (es. Viral News, Core Update temporaneo, Buzz sui social).
- **`window` (Finestra)**: Evento temporaneo costante (es. Black Friday, Saldi, Stagionalità spot, Problemi tecnici temporanei). Inizia e finisce bruscamente.

### 2. Scenari vs Baseline
L'app permette di confrontare uno **Scenario Attivo** con una **Baseline** (scenario precedente/inerziale).
- Se nel contesto vedi riferimenti a "Baseline", **CONFRONTA SEMPRE** i due valori.
- Esempio: "Lo scenario 'Aggressivo' porta un +15% rispetto alla Baseline inerziale, grazie all'attivazione del pacchetto Link Building."

## 💼 MODALITÀ DI RISPOSTA

### Fase 1: Analisi Tecnica & Causale
Spiega il "perché" matematico del trend usando i regressori attivi:
- *"Il picco di Luglio non è stagionale, ma è causato dal regressore `window` 'Saldi Estivi'..."*
- *"La crescita costante da Settembre è l'effetto `ramp` del pacchetto Content Marketing..."*

### Fase 2: Business Translation (Cruciale)
Traduci il traffico in valore. Poiché l'app fornisce solo dati di Traffico (Click/Impression):
- **CHIEDI SEMPRE** all'utente i tassi di conversione (CVR) o il valore medio ordine (AOV) se vuoi fare stime economiche e non sono presenti nel contesto.
- **NON INVENTARE** valori monetari. Usa stime ipotetiche dichiarandole esplicitamente: *"Se assumiamo un CVR del 2%, questi 10k click extra valgono 200 lead."*

### Fase 3: Risk & Confidence
- Valuta l'ampiezza dell'intervallo di confidenza (area ombreggiata).
- Se l'intervallo è ampio, suggerisci cautela negli investimenti e test incrementali.

### Fase 4: Interpretazione Valori Impatto (Regressor Impact)
Quando leggi il valore `impact` di un regressore:
- **0.0**: Neutro (nessun effetto).
- **Positivo (es. 0.2)**: Uplift stimato del +20% circa. Usalo per giustificare crescite.
- **Negativo (es. -0.3)**: Drop stimato del -30% circa. Usalo per spiegare cali.
- **Magnitudo**: Valori > 0.5 sono impatti molto forti (trasformativi). Valori < 0.1 sono marginali.

## 🛠️ GUIDA PROATTIVA E TROUBLESHOOTING

Se l'utente esprime dubbi ("perché scende?", "mi aspettavo di più"), **NON limitarti a spiegare**. **PROPONI SOLUZIONI**:

1.  **Previsione Piatta/Conservativa?**
    *   *Analisi*: Prophet tende al mean-reversion. Senza stimoli, il futuro è uguale al passato.
    *   *Proposta*: "Il modello è conservativo. Se prevedi attività SEO, ti suggerisco di aggiungere un regressore **`ramp`** (es. 'Content Plan 2026') con impatto positivo progressivo (es. +20% a 12 mesi)."

2.  **Calo Inaspettato?**
    *   *Analisi*: Verifica se ci sono regressori negativi (`step` migration, `decay`) o stagionalità negativa in quel periodo.
    *   *Proposta*: "Il calo a Marzo è stagionale. Per contrastarlo, potresti pianificare una campagna di Link Building (evento **`window`** o **`ramp`**) in quel trimestre."

3.  **Picchi Anomali?**
    *   *Analisi*: Potrebbero esserci outlier storici che Prophet sta replicando.
    *   *Proposta*: "Vedo un picco anomalo l'anno scorso. Era un evento reale? Se no, possiamo considerarlo outlier."

## TONE OF VOICE

- **Analitico ma Strategico**: Non limitarti a leggere i numeri ("il traffico sale"), spiega le implicazioni ("il traffico sale, permettendo di scalare le vendite").
- **Proattivo**: Suggerisci sempre setup di regressori specifici (Tipo, Durata, Impatto stimato) per modellare gli scenari desiderati dall'utente.
- **Onesto**: Ammetti i limiti del modello (es. "Prophet non sa che hai rifatto il sito se non glielo dici con un regressore").

## EXAMPLE DI RISPOSTA PROATTIVA

**Utente**: "Perché la previsione scende a Marzo? Mi aspettavo una crescita per il lancio del nuovo blog."

**AI**: "Il calo del 10% a Marzo che vedi nel grafico è dovuto alla **stagionalità storica**: negli anni passati il traffico è sempre sceso in Q1. Il modello non 'sa' del lancio del blog.
**Soluzione**: Per modellare correttamente il lancio, ti suggerisco di aggiungere un regressore:
*   **Nome**: Lancio Blog
*   **Tipo**: `ramp` (la crescita organica è graduale)
*   **Data**: 1 Marzo
*   **Durata**: 6-12 mesi (tempo per andare a regime)
*   **Impatto**: +15% (o il target che ti aspetti)

Vuoi che ti aiuti a stimare l'impatto corretto basandoci sui competitor?"
"""

# --- Helper Functions ---
def init_session_state():
    if 'events' not in st.session_state:
        st.session_state.events = []
    if 'events_editor_key' not in st.session_state:
        st.session_state.events_editor_key = 0
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

    # Chatbot State
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'chat_config' not in st.session_state:
        st.session_state.chat_config = {
            "model": "gpt-5.2",
            "temperature": 0.7,
            "system_prompt": DEFAULT_SYSTEM_PROMPT
        }

def OLD_handle_chat_actions(response_text, key_suffix):
    """Parses AI response for JSON actions and renders buttons."""
    import re
    import json
    import pandas as pd
    import time
    
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
                        
                        for reg in reg_list:
                            try:
                                action = reg.get("action", "add").lower()
                                nm = str(reg.get("name", "Unknown"))
                                
                                # --- 1. REMOVE ---
                                if action == "remove":
                                    prev_len = len(st.session_state.events)
                                    st.session_state.events = [e for e in st.session_state.events if e['name'] != nm]
                                    if len(st.session_state.events) < prev_len:
                                        count += 1
                                    continue
                                
                                # --- 2. UPDATE ---
                                if action == "update":
                                    updated = False
                                    for e in st.session_state.events:
                                        if e['name'] == nm:
                                            if "date" in reg: e["date"] = pd.to_datetime(reg["date"])
                                            if "type" in reg: e["type"] = str(reg["type"])
                                            if "impact" in reg: e["impact"] = float(reg["impact"])
                                            if "duration" in reg: e["duration"] = int(reg["duration"])
                                            updated = True
                                            count += 1
                                    if updated: continue
                                    # If not found, fallthrough to Add? Or stop? Let's stop.
                                    pass
                                
                                # --- 3. ADD (Default) ---
                                if (action == "add" or action == "create") and "date" in reg:
                                    # Check duplicates? User might want multiple events.
                                    new_event = {
                                        "name": nm,
                                        "date": pd.to_datetime(reg["date"]),
                                        "type": str(reg.get("type", "step")),
                                        "duration": int(reg.get("duration", 28)),
                                        "impact": float(reg.get("impact", 0.0))
                                    }
                                    st.session_state.events.append(new_event)
                                    count += 1

                            except Exception as e:
                                st.warning(f"Errore su {nm}: {e}")
                        
                        if count > 0:
                            st.success(f"✅ Applicati {count} regressori!")
                            st.session_state.events_editor_key += 1
                            time.sleep(1)
                            st.rerun()
        except Exception as e:
            # st.warning(f"Debug: JSON parsing error {e}") 
            pass

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

# --- PROJECT MANAGEMENT ---
if 'current_project' not in st.session_state: st.session_state.current_project = None

st.sidebar.markdown("### 📂 Gestione Progetto")
projects = tools.project_manager.get_all_projects()
proj_options = ["+ Nuovo Progetto"] + projects

# Determine index
idx_p = 0
if st.session_state.current_project in projects:
    idx_p = proj_options.index(st.session_state.current_project)

selected_proj = st.sidebar.selectbox("Seleziona Progetto", proj_options, index=idx_p, key="sb_project_selector")

if selected_proj == "+ Nuovo Progetto":
    st.sidebar.markdown("---")
    new_proj_name = st.sidebar.text_input("Nome Nuovo Progetto", placeholder="es. Cliente Alpha")
    if st.sidebar.button("Crea Progetto", key="btn_create_proj"):
        ok, msg = tools.project_manager.create_new_project(new_proj_name)
        if ok:
            st.sidebar.success(f"Creato: {msg}")
            st.session_state.current_project = msg
            time.sleep(1)
            st.rerun()
        else:
            st.sidebar.error(msg)
else:
    st.session_state.current_project = selected_proj
    st.sidebar.caption(f"Progetto attivo: **{selected_proj}**")

st.sidebar.markdown("---")



# API Key Configuration
st.sidebar.markdown("### 🔑 Credenziali AI")
user_api_key = st.sidebar.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password", help="Inserisci la tua chiave API OpenAI per abilitare le funzioni di Assistant e Report.")
if user_api_key:
    st.session_state['openai_api_key'] = user_api_key
else:
    st.session_state['openai_api_key'] = ""


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
    
    # Seasonality Mode
    seas_opts = ["multiplicative", "additive"]
    curr_seas = st.session_state.get("seasonality_mode", "multiplicative")
    seas_idx = seas_opts.index(curr_seas) if curr_seas in seas_opts else 0
    seasonality = st.selectbox("Seasonality Mode", seas_opts, index=seas_idx, key="seasonality_mode")
    
    col_cp1, col_cp2 = st.columns(2)
    with col_cp1:
        cp_scale_val = st.session_state.get("changepoint_prior_scale", 0.05)
        changepoint_scale = st.number_input(
            "Changepoint Prior Scale", 
            min_value=0.001, max_value=0.5, value=float(cp_scale_val), step=0.001, format="%.3f",
            help="Flessibilità del trend. Alto=Reattivo, Basso=Rigido.",
            key="changepoint_prior_scale"
        )
    with col_cp2:
        cp_range_val = st.session_state.get("changepoint_range", 0.8)
        changepoint_range = st.number_input(
            "Changepoint Range", 
            min_value=0.1, max_value=0.95, value=float(cp_range_val), step=0.05, format="%.2f",
            help="% Storia usata per il training (0.8 = 80%)",
            key="changepoint_range"
        )

    seas_scale_val = st.session_state.get("seasonality_prior_scale", 10.0)
    seasonality_scale = st.number_input(
        "Seasonality Prior Scale", 
        min_value=0.01, max_value=50.0, value=float(seas_scale_val), step=1.0, format="%.1f",
        help="Forza della stagionalità.",
        key="seasonality_prior_scale"
    )
    
    st.markdown("---")
    st.caption("Componenti Stagionali")
    # Toggles for specific seasonalities
    col_s1, col_s2, col_s3 = st.columns(3)
    
    seas_toggles = ["auto", True, False]
    
    with col_s1:
        curr_y = st.session_state.get("yearly_seasonality", "auto")
        idx_y = seas_toggles.index(curr_y) if curr_y in seas_toggles else 0
        yearly_seas = st.selectbox("Yearly", seas_toggles, index=idx_y, key="yearly_seasonality")
        
    with col_s2:
        curr_w = st.session_state.get("weekly_seasonality", True) # Default True
        idx_w = seas_toggles.index(curr_w) if curr_w in seas_toggles else 1
        weekly_seas = st.selectbox("Weekly", seas_toggles, index=idx_w, key="weekly_seasonality") 
        
    with col_s3:
        curr_d = st.session_state.get("daily_seasonality", False) # Default False
        idx_d = seas_toggles.index(curr_d) if curr_d in seas_toggles else 2
        daily_seas = st.selectbox("Daily", seas_toggles, index=idx_d, key="daily_seasonality")

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

    # --- TABS DEFINITION (Global) ---
    tab_forecast, tab_compare, tab_regressors, tab_report, tab_chat = st.tabs([
        "📊 Analisi & Forecast", 
        "⚔️ Confronto Scenari", 
        "📝 Gestione Regressori", 
        "📋 Report", 
        "🤖 Assistant"
    ])

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

    # --- TABS DEFINED ABOVE ---


    # --- REGRESSOR EDITOR ---
    # --- REGRESSOR SELECTION ---
    # --- REGRESSOR SELECTION ---
    with tab_regressors:
        st.subheader("🛠️ Gestione Regressori (Scenario)")
        
        # Wrapper to maintain indentation of validation logic
        if True:
            st.caption("Gestisci manualmente gli eventi e i regressori.")
            # AI Analysis removed


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
                # Bulk Selection Buttons
                col_sel1, col_sel2, _ = st.columns([1, 1, 3])
                if col_sel1.button("✅ Seleziona Tutto", key="btn_sel_all"):
                    for e in st.session_state.events: e['seleziona'] = True
                    st.session_state.events_editor_key += 1
                    st.rerun()
                if col_sel2.button("❌ Deseleziona Tutto", key="btn_desel_all"):
                    for e in st.session_state.events: e['seleziona'] = False
                    st.session_state.events_editor_key += 1
                    st.rerun()

                df_editor = pd.DataFrame(st.session_state.events)
                
                # Ensure columns order
                cols_order = ['name', 'date', 'type', 'duration', 'impact', 'event_type']
                for c in cols_order:
                    if c not in df_editor.columns:
                        df_editor[c] = None
                
                # Reorder columns (keeping 'seleziona' if present from Bulk Action)
                if 'seleziona' in df_editor.columns:
                    df_editor = df_editor[['seleziona'] + cols_order]
                else:
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
                key=f"regressor_editor_{st.session_state.events_editor_key}",
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
            
            # Export Excel with Template
            if not edited_df.empty:
                import io
                # Template Data Structure
                template_data = [
                    {"template_name": "google_core_update", "event_type": "algorithm", "default_duration_days": 14, "default_impact": -1.0, "regressor_type": "decay", "description": "Google Core Algorithm Update (broad impact, negative)"},
                    {"template_name": "google_spam_update", "event_type": "algorithm", "default_duration_days": 7, "default_impact": -1.0, "regressor_type": "window", "description": "Google Spam Update (focused, negative)"},
                    {"template_name": "content_publication", "event_type": "content", "default_duration_days": 60, "default_impact": 1.0, "regressor_type": "ramp", "description": "New content publication (long-tail positive)"},
                    {"template_name": "content_update", "event_type": "content", "default_duration_days": 30, "default_impact": 1.0, "regressor_type": "ramp", "description": "Content refresh/update (medium-term positive)"},
                    {"template_name": "ppc_campaign", "event_type": "marketing", "default_duration_days": 30, "default_impact": 1.0, "regressor_type": "window", "description": "PPC campaign (awareness boost)"},
                    {"template_name": "email_campaign", "event_type": "marketing", "default_duration_days": 7, "default_impact": 1.0, "regressor_type": "window", "description": "Email marketing campaign (short-term spike)"},
                    {"template_name": "social_campaign", "event_type": "marketing", "default_duration_days": 14, "default_impact": 1.0, "regressor_type": "decay", "description": "Social media campaign (viral decay)"},
                    {"template_name": "site_migration", "event_type": "technical", "default_duration_days": 90, "default_impact": -1.0, "regressor_type": "decay", "description": "Site migration (long recovery, negative)"},
                    {"template_name": "technical_fix", "event_type": "technical", "default_duration_days": 14, "default_impact": 1.0, "regressor_type": "decay", "description": "Technical SEO fix (gradual positive impact)"},
                    {"template_name": "link_building", "event_type": "marketing", "default_duration_days": 7, "default_impact": 0.5, "regressor_type": "decay", "description": "Digital PR campaign"}
                ]
                
                buffer = io.BytesIO()
                # Writing to Excel buffer
                # Note: Requires openpyxl or xlsxwriter installed
                try:
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        # Sheet 1: Eventi (Current Regressors)
                        save_df = edited_df.copy()
                        if 'seleziona' in save_df.columns: 
                            save_df = save_df.drop(columns=['seleziona'])
                        save_df.to_excel(writer, sheet_name='Eventi', index=False)
                        
                        # Sheet 2: Template (Defaults)
                        pd.DataFrame(template_data).to_excel(writer, sheet_name='Template', index=False)
                except Exception as e:
                    # Fallback if xlsxwriter missing, try default (often openpyxl)
                    buffer = io.BytesIO() # reset
                    with pd.ExcelWriter(buffer) as writer:
                         # Sheet 1
                        save_df = edited_df.copy()
                        if 'seleziona' in save_df.columns: 
                            save_df = save_df.drop(columns=['seleziona'])
                        save_df.to_excel(writer, sheet_name='Eventi', index=False)
                        # Sheet 2
                        pd.DataFrame(template_data).to_excel(writer, sheet_name='Template', index=False)

                st.download_button(
                    label="📥 Scarica Regressori (Excel)",
                    data=buffer.getvalue(),
                    file_name="regressori_seo.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Scarica il file Excel con due fogli: 'Eventi' (configurazione attuale) e 'Template' (riferimento)."
                )

    # --- GENERATOR TAB ---


    # --- FORECAST TAB ---
    with tab_forecast:
        st.info(f"ℹ️ Regressori Attivi: {len(st.session_state.events)}")
        
        # 3. Running Forecast
        col_btn_1, col_btn_2 = st.columns([1, 1])
        
        run_forecast_btn = col_btn_1.button("🚀 Genera Forecast", type="primary")
        
        # Save Scenario UI
        with col_btn_2:
            if 'last_forecast' in st.session_state and st.session_state.get('current_project'):
                with st.expander("💾 Salva in Scenari", expanded=False):
                    scen_name = st.text_input("Nome Scenario", value=f"Scenario {pd.Timestamp.now().strftime('%d/%m %H:%M')}")
                    if st.button("Salva Scenario"):
                        ok, msg = tools.project_manager.save_scenario(
                            st.session_state.current_project,
                            scen_name,
                            st.session_state.last_forecast,
                            st.session_state.events,
                            st.session_state.last_metrics
                        )
                        if ok:
                            st.success("Scenario Salvato!")
                        else:
                            st.error(msg)
            elif not st.session_state.get('current_project'):
                st.info("Seleziona un progetto nella sidebar per salvare gli scenari.")


        # Check if triggered by button OR Chat Action
        if run_forecast_btn or st.session_state.get('trigger_forecast_run'):
            # Clear trigger immediately to avoid loops
            st.session_state.trigger_forecast_run = False
            
            with st.spinner("Addestramento modello Prophet in corso..."):
                try:
                    # Use events from session state
                    results = execute_forecast(history_df, st.session_state.events, config)
                    
                    # Save as last forecast
                    st.session_state.last_forecast = results['forecast']
                    st.session_state.last_metrics = results['metrics']
                    st.session_state.last_debug = results['debug_info']
                    st.session_state.last_run_config = config
                    st.rerun()
                    
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

            # --- DASHBOARD STRATEGICA (Decision-Ready) ---
            st.markdown("### 🧭 Cruscotto Decisionale")
            
            # --- RIGA 0: ANALISI GIORNALIERA ---
            # --- CALCOLI PRELIMINARI ---
            target_days = config.get('horizon_days', 90)
            
            # Legacy Baseline Removed

            # A. Future & Scenario Total
            ft_future = forecast[forecast['ds'] > pd.to_datetime('today')]
            tot_scen = ft_future['yhat'].sum()
            
            # C. YoY Metrics
            yoy_metrics = scenario_analysis.calculate_total_yoy_metrics(forecast, history_df)
            
            # D. Pre-Forecast Averages (History Last N Days)
            avg_m_pre = 0
            avg_d_pre = 0
            if 'date' in history_df.columns:
                 last_hist_date = history_df['date'].max()
                 start_hist_window = last_hist_date - pd.Timedelta(days=target_days)
                 pre_df = history_df[history_df['date'] > start_hist_window]
                 if not pre_df.empty:
                     tot_pre = pre_df['clicks'].sum()
                     days_pre = (pre_df['date'].max() - pre_df['date'].min()).days + 1
                     if days_pre > 0: 
                         avg_d_pre = tot_pre / days_pre
                         avg_m_pre = avg_d_pre * 30.44

            # E. Post-Forecast Monthly Avg (Future)
            avg_m_post = 0
            if not ft_future.empty:
                tot_fut = ft_future['yhat'].sum()
                days_fut = (ft_future['ds'].max() - ft_future['ds'].min()).days + 1
                if days_fut > 0: avg_m_post = (tot_fut / days_fut) * 30.44
            
            # --- LAYOUT VISUALIZZAZIONE ---
            
            # RIGA 1: Analisi Giornaliera
            c_d1, c_d2 = st.columns(2)
            c_d1.metric("Media Giornaliera (Storico)", f"{int(avg_d_pre):,}", help=f"Media storico recente ({target_days}gg).")
            
            # Forecast Daily
            curr_daily_fc = metrics.get('forecast_mean', 0)
            d_delta = curr_daily_fc - avg_d_pre
            d_pct = (d_delta / avg_d_pre) if avg_d_pre > 0 else 0
            
            c_d2.metric("Media Giornaliera (Forecast)", f"{int(curr_daily_fc):,}", delta=f"{d_pct:+.1%} vs Storico")
            
            st.divider()
            
            # RIGA 2: Analisi Mensile
            c_m1, c_m2 = st.columns(2)
            c_m1.metric("Media Mensile (Storico)", f"{int(avg_m_pre):,}")
            
            m_delta = avg_m_post - avg_m_pre
            m_pct = (m_delta / avg_m_pre) if avg_m_pre > 0 else 0
            
            c_m2.metric("Media Mensile (Forecast)", f"{int(avg_m_post):,}", delta=f"{m_pct:+.1%} vs Storico")
            
            st.divider()
            
            # RIGA 3: Macro & YoY
            c_t1, c_t2 = st.columns(2)
            c_t1.metric("Totale Traffico Stimato", f"{int(tot_scen):,}", help=f"Totale click previsti ({target_days}gg).")
            
            if yoy_metrics.get('status') == 'ok':
                yoy_d = yoy_metrics.get('delta_pct_mean', 0)
                yoy_abs = yoy_metrics.get('delta_abs_mean', 0)
                c_t2.metric("Trend YoY (vs Anno Scorso)", f"{yoy_d:+.1%}", delta=f"{yoy_abs:+.0f} click/gg")
            else:
                c_t2.metric("Trend YoY", "-", help="Dati storici insufficienti.")
            
            st.divider()

            # --- 2. DETTAGLIO ANALISI ---
            # --- 2. DETTAGLIO ANALISI ---
            st.divider()
            st.markdown("### ⚙️ Metriche Modello")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("MAPE", f"{metrics['mape']:.2f}%")
            if 'rmse' in metrics: c2.metric("RMSE", f"{int(metrics['rmse'])}")
            if 'mae' in metrics: c3.metric("MAE", f"{int(metrics.get('mae', 0))}")

            
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
            
            # Baseline Trace Removed

                
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

            # --- Analisi YoY Futura ---
            st.markdown("### 📅 Variazione YoY (Forecast vs Anno Precedente)")
            
            # Prepare data
            future_daily = forecast[forecast['ds'] > history_df['date'].max()].copy()
            
            if not future_daily.empty:
                # 1. Monthly Analysis
                future_daily['Month'] = future_daily['ds'].dt.to_period('M')
                
                # Group by Month
                f_cancel = future_daily.groupby('Month')['yhat'].sum().reset_index().rename(columns={'yhat': 'Forecast'})
                f_cancel['Days'] = future_daily.groupby('Month')['ds'].count().values
                
                # History Monthly
                h_daily = history_df.copy()
                d_col = 'date' if 'date' in h_daily.columns else 'ds'
                clicks_col = 'clicks' if 'clicks' in h_daily.columns else 'y'
                # Ensure datetime
                if not pd.api.types.is_datetime64_any_dtype(h_daily[d_col]):
                    h_daily[d_col] = pd.to_datetime(h_daily[d_col])

                h_daily['Month'] = h_daily[d_col].dt.to_period('M')
                h_monthly = h_daily.groupby('Month')[clicks_col].sum().reset_index().rename(columns={clicks_col: 'History'})
                
                # Merge logic: Forecast Month M vs History Month M-12
                yoy_rows = []
                
                for idx, row in f_cancel.iterrows():
                    m = row['Month']
                    prev_m = m - 12
                    
                    # Check completeness
                    days_in_m = m.days_in_month
                    is_partial = row['Days'] < days_in_m
                    # Also check if historical data was partial? (Assume history is complete usually)
                    note = "⚠️ Parziale" if is_partial else ""
                    
                    hist_val = 0
                    yoy_abs = 0
                    yoy_pct = 0
                    
                    match = h_monthly[h_monthly['Month'] == prev_m]
                    if not match.empty:
                        hist_val = match.iloc[0]['History']
                        yoy_abs = row['Forecast'] - hist_val
                        yoy_pct = (yoy_abs / hist_val) if hist_val > 0 else 0
                    
                    yoy_rows.append({
                        "Mese": str(m),
                        "Forecast": row['Forecast'],
                        "Anno Prec": hist_val if hist_val > 0 else None,
                        "Δ Assoluto": yoy_abs,
                        "Δ %": yoy_pct,
                        "Note": note
                    })
                    
                st.markdown("#### Variazione Mensile")
                st.dataframe(pd.DataFrame(yoy_rows).style.format({
                    "Forecast": "{:,.0f}",
                    "Anno Prec": "{:,.0f}",
                    "Δ Assoluto": "{:+,.0f}",
                    "Δ %": "{:+.1%}"
                }).applymap(lambda v: f'color: {"#28a745" if v > 0 else "#dc3545" if v < 0 else "inherit"}', subset=['Δ Assoluto', 'Δ %']), use_container_width=True)


                # 2. Quarterly Analysis
                future_daily['Quarter'] = future_daily['ds'].dt.to_period('Q')
                f_q = future_daily.groupby('Quarter')['yhat'].sum().reset_index().rename(columns={'yhat': 'Forecast'})
                
                max_date = future_daily['ds'].max()
                
                q_rows = []
                h_daily['Quarter'] = h_daily[d_col].dt.to_period('Q')
                h_q = h_daily.groupby('Quarter')[clicks_col].sum().reset_index().rename(columns={clicks_col: 'History'})
                
                for idx, row in f_q.iterrows():
                    q = row['Quarter']
                    prev_q = q - 4
                    
                    # Check partial
                    # If q.end_time > max_date, then it's partial
                    is_partial = q.end_time.date() > max_date.date()
                    note = "⚠️ Parziale" if is_partial else ""

                    hist_val = 0
                    yoy_abs = 0
                    yoy_pct = 0
                    
                    match = h_q[h_q['Quarter'] == prev_q]
                    if not match.empty:
                        hist_val = match.iloc[0]['History']
                        yoy_abs = row['Forecast'] - hist_val
                        yoy_pct = (yoy_abs / hist_val) if hist_val > 0 else 0
                        
                    q_rows.append({
                        "Quarter": str(q),
                        "Forecast": row['Forecast'],
                        "Anno Prec": hist_val if hist_val > 0 else None,
                        "Δ Assoluto": yoy_abs,
                        "Δ %": yoy_pct,
                        "Note": note
                    })
                
                st.markdown("#### Variazione Trimestrale")
                st.dataframe(pd.DataFrame(q_rows).style.format({
                    "Forecast": "{:,.0f}",
                    "Anno Prec": "{:,.0f}",
                    "Δ Assoluto": "{:+,.0f}",
                    "Δ %": "{:+.1%}"
                }).applymap(lambda v: f'color: {"#28a745" if v > 0 else "#dc3545" if v < 0 else "inherit"}', subset=['Δ Assoluto', 'Δ %']), use_container_width=True)

                
            else:
                st.info("Nessun dato forecast futuro disponibile per analisi YoY.")
            
            st.divider()
            
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

        except Exception as e:
             st.error(f"Errore visualizzazione risultati: {e}")

    # --- TAB CONFRONTO SCENARI ---
    with tab_compare:
        st.subheader("⚔️ Confronto Scenari")
        st.caption("Analizza e confronta i diversi scenari salvati per il progetto corrente.")
        
        curr_p = st.session_state.get('current_project')
        if not curr_p:
            st.info("👈 Seleziona un progetto dalla sidebar per accedere al confronto scenari.")
        else:
            scenarios = tools.project_manager.load_scenarios(curr_p)
            if len(scenarios) < 2:
                st.warning(f"⚠️ Hai salvato solo {len(scenarios)} scenario/i. Salvane almeno 2 per attivare il confronto.")
                if len(scenarios) == 1:
                     st.write(f"Scenario attuale: **{scenarios[0]['name']}** ({scenarios[0]['created_at']})")
            else:
                # 1. Summary Table
                st.markdown("### 📋 Riepilogo Scenari")
                comp_data = []
                for s in scenarios:
                    comp_data.append({
                        "Scenario": s['name'],
                        "Data Creazione": s['created_at'],
                        "Click Totali (Forecast)": s['total_clicks'],
                        "Eventi Attivi": s['events_count']
                    })
                
                df_comp_table = pd.DataFrame(comp_data)
                st.dataframe(
                    df_comp_table.style.format({'Click Totali (Forecast)': '{:,.0f}'}),
                    use_container_width=True
                )
                
                # 2. Comparative Chart
                st.divider()
                st.markdown("### 📈 Trend a Confronto")
                
                fig_comp = go.Figure()
                palette = ['#2E7D32', '#1565C0', '#D84315', '#6A1B9A', '#00838F', '#AD1457']
                
                loaded_count = 0
                for i, s in enumerate(scenarios):
                    df_s = tools.project_manager.load_scenario_df(curr_p, s['file'])
                    if df_s is not None and not df_s.empty:
                         # Filter Future Only for clarity
                         future_s = df_s[df_s['ds'] > pd.Timestamp.now()]
                         if not future_s.empty:
                             fig_comp.add_trace(go.Scatter(
                                 x=future_s['ds'],
                                 y=future_s['yhat'],
                                 mode='lines',
                                 name=s['name'],
                                 line=dict(width=2, color=palette[i % len(palette)])
                             ))
                             loaded_count += 1
                
                if loaded_count > 0:
                    fig_comp.update_layout(
                        title="Confronto Curve di Traffico (Forecast)",
                        hovermode="x unified",
                        template="plotly_white",
                        yaxis_title="Click Stimati",
                        legend=dict(orientation="h", y=1.1)
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)
                else:
                    st.warning("Impossibile caricare i dati dei grafici per gli scenari selezionati.")
                
                # 3. Delta Analysis
                st.divider()
                st.markdown("### ⚖️ Analisi Differenziale (A vs B)")
                
                c_sel1, c_sel2 = st.columns(2)
                with c_sel1:
                    s_a_name = st.selectbox("Scenario Base (A)", [s['name'] for s in scenarios], index=len(scenarios)-1, key="scen_a")
                with c_sel2:
                    s_b_name = st.selectbox("Scenario Target (B)", [s['name'] for s in scenarios], index=0, key="scen_b")
                    
                if s_a_name and s_b_name:
                    s_a = next(s for s in scenarios if s['name'] == s_a_name)
                    s_b = next(s for s in scenarios if s['name'] == s_b_name)
                    
                    # Calculate Metrics
                    delta_clicks = s_b['total_clicks'] - s_a['total_clicks']
                    delta_pct = (delta_clicks / s_a['total_clicks']) if s_a['total_clicks'] > 0 else 0
                    
                    c_res1, c_res2, c_res3 = st.columns(3)
                    c_res1.metric("Differenza Totale Click", f"{int(delta_clicks):+,.0f}", delta=f"{delta_pct:+.1%}")
                    
                    # More deep analysis if feasible (needs loading both DFs)
                    # For now, simplistic total is enough for executive view.
                    
                    st.info(f"Confrontando **{s_b_name}** rispetto a **{s_a_name}**: lo scenario Target porta {int(delta_clicks):+,.0f} click in totale.")




    try:
        # --- AI REPORTING ---
        with tab_report:
            st.divider()
            st.subheader("🤖 AI Executive Report")
            st.caption("Genera un'analisi strategica proattiva basata sugli ultimi dati.")
            
            if 'last_forecast' not in st.session_state or st.session_state.last_forecast is None:
                st.warning("⚠️ Esegui prima un Forecast per generare il report.")
            else:
                col_rep_cfg, col_rep_view = st.columns([1, 2])
                
                with col_rep_cfg:
                    with st.expander("⚙️ Configurazione Report", expanded=True):
                        # Model
                        rep_models = ["gpt-5.1", "gpt-5-turbo", "gpt-4o", "gpt-4-turbo"]
                        sel_rep_model = st.selectbox("Modello AI", rep_models, index=0, key="rep_model_sel")
                        
                        # Default Prompt Value
                        default_rep_prompt = """Sei un Senior SEO Strategist.
Analizza i dati di forecast forniti e scrivi un Executive Report per il cliente (Prospect).

PUOI INCLUDERE ELEMENTI INTERATTIVI NEL TESTO USANDO QUESTI TAG:
- {{KPI_SUMMARY}}: Mostra card con metriche chiave.
- {{CHART_TREND}}: Inserisce il grafico del forecast.
- {{EVENTS_TABLE}}: Inserisce la tabella degli eventi.

STRUTTURA REPORT:
1. **Executive Summary**: Sintesi estrema (Crescita/Calo e Medie Mensili previste). Usa {{KPI_SUMMARY}} qui sotto.
2. **Analisi Scenario**: Spiega l'impatto degli eventi. Inserisci {{EVENTS_TABLE}} per riferimento.
3. **Trend Futuro**: Descrivi cosa accadrà. Inserisci {{CHART_TREND}}.

Tono: Professionale, Diretto, Persuasivo."""

                        if 'report_system_prompt' not in st.session_state:
                            st.session_state.report_system_prompt = default_rep_prompt
                        
                        st.caption("Istruzioni Generazione (System Prompt)")
                        if st.button("⛶ Modifica Istruzioni", help="Apri Editor Fullscreen"):
                            edit_report_prompt_dialog()
                        
                        txt_sys_rep = st.session_state.report_system_prompt
                        with st.expander("Anteprima Prompt", expanded=False):
                            st.code(txt_sys_rep, language="markdown")
                        
                        st.divider()
                        
                        # Trigger Button
                        if st.button("✨ Genera Report Prospect", type="primary", key="btn_gen_rep"):
                            st.session_state._report_gen_trigger = True
                        
                        if st.session_state.get('_report_gen_trigger'):
                            status_ph = st.empty()
                            status_ph.info("⏳ Avvio generazione report...")
                            
                            try:
                                # Fallback spinner logic
                                with st.spinner(f"Analisi in corso con {sel_rep_model}..."):
                                    # Retrieve latest data SAFELY
                                    curr_metrics = st.session_state.get('last_metrics')
                                    if curr_metrics is None: curr_metrics = {}
                                    
                                    curr_events = st.session_state.get('events', [])
                                    if curr_events is None: curr_events = []
                                    
                                    curr_forecast = st.session_state.get('last_forecast')
                                    
                                    r_conf = st.session_state.get('last_run_config')
                                    if r_conf is None: r_conf = {}
                                    curr_horizon = r_conf.get('horizon_days', 90)
                                    
                                    api_key_use = st.session_state.get('openai_api_key', os.getenv("OPENAI_API_KEY"))
                                    
                                    rep_txt, err = generate_marketing_report(
                                        metrics=curr_metrics,
                                        events=curr_events,
                                        horizon=curr_horizon,
                                        forecast_df=curr_forecast,
                                        api_key=api_key_use,
                                        model=sel_rep_model,
                                        system_instruction=txt_sys_rep
                                    )
                                    
                                    if err:
                                        status_ph.error(err)
                                    else:
                                        status_ph.success("✅ Report completato!")
                                        st.session_state.generated_report = rep_txt
                                        # Clear cached files
                                        if 'rep_pdf_bytes' in st.session_state: del st.session_state.rep_pdf_bytes
                                        if 'rep_ppt_bytes' in st.session_state: del st.session_state.rep_ppt_bytes
                                        
                                        time.sleep(0.5)
                                        st.rerun()
                                        
                            except Exception as e:
                                status_ph.error(f"Errore critico: {e}")
                            finally:
                                st.session_state._report_gen_trigger = False

                with col_rep_view:
                    if st.session_state.get('generated_report'):
                        st.success("✅ Report Generato con Successo!")
                        # Use Custom Renderer
                        render_smart_report(st.session_state.generated_report)
                        st.divider()
                        st.subheader("📥 Export")
                        
                        c_ex1, c_ex2, c_ex3 = st.columns(3)
                        
                        with c_ex1:
                            st.download_button(
                                "📄 Markdown",
                                st.session_state.generated_report,
                                "report.md",
                                "text/markdown"
                            )
                        
                        with c_ex2:
                            if 'rep_pdf_bytes' not in st.session_state:
                                if st.button("Genera PDF"):
                                    ok, b, err = tools.export_utils.create_pdf(st.session_state.generated_report)
                                    if ok: 
                                        st.session_state.rep_pdf_bytes = b
                                        st.rerun()
                                    else: st.error(err)
                            else:
                                st.download_button(
                                    "📄 Scarica PDF", 
                                    st.session_state.rep_pdf_bytes, 
                                    "report.pdf", 
                                    "application/pdf"
                                )
                                if st.button("Rigenera PDF", key="regen_pdf"):
                                    del st.session_state.rep_pdf_bytes
                                    st.rerun()

                        with c_ex3:
                            if 'rep_ppt_bytes' not in st.session_state:
                                if st.button("📊 Genera PPT"):
                                    ok, b, err = tools.export_utils.create_ppt_bytes(st.session_state.generated_report)
                                    if ok:
                                        st.session_state.rep_ppt_bytes = b
                                        st.rerun()
                                    else: st.error(err)
                            else:
                                st.download_button(
                                    "📊 Scarica PPT",
                                    st.session_state.rep_ppt_bytes,
                                    "report.pptx",
                                    "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                                )
                                if st.button("Rigenera PPT", key="regen_ppt"):
                                    del st.session_state.rep_ppt_bytes
                                    st.rerun()
                                    
                    else:
                        st.info("👈 Configura le istruzioni e clicca 'Genera Report'.")
                        st.markdown("---")
                        st.caption("Puoi usare i tag {{KPI_SUMMARY}}, {{CHART_TREND}} nel prompt per includere componenti interattivi.")

    except Exception as e:
        st.error(f"Errore nella visualizzazione dei risultati: {str(e)}")

# --- CHATBOT ASSISTANT ---
with tab_chat:
    st.subheader("🤖 AI SEO Assistant")
    
    col_chat_main, col_chat_side = st.columns([3, 1])

    with col_chat_side:
        with st.expander("⚙️ Configurazione", expanded=True):
            # Model Selection
            model_options = ["gpt-5.2", "gpt-5-mini", "gpt-5-nano", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
            if 'chat_config' not in st.session_state: st.session_state.chat_config = {}
            curr_model = st.session_state.chat_config.get("model", "gpt-5.2")
            sel_model = st.selectbox("Modello", model_options, index=model_options.index(curr_model) if curr_model in model_options else 0)
            
            # Temperature
            curr_temp = st.session_state.chat_config.get("temperature", 0.7)
            sel_temp = st.slider("Creatività", 0.0, 1.0, curr_temp, 0.1)
            
            st.divider()
            
            # System Prompt Fullscreen
            if st.button("⛶ Modifica System Prompt", help="Apri l'editor a tutto schermo"):
                if DialogDecorator:
                    edit_sys_prompt_dialog()
                else:
                    st.error("Funzione Fullscreen non supportata.")
            
            if st.button("💾 Salva Config"):
                st.session_state.chat_config["model"] = sel_model
                st.session_state.chat_config["temperature"] = sel_temp
                st.success("Salvataggi aggiornati!")
            
            st.divider()
            
            # Export Chat
            if st.session_state.chat_history:
                md_chat = "# Chat History\n\n"
                for m in st.session_state.chat_history:
                    icon = "👤" if m['role'] == "user" else "🤖"
                    md_chat += f"### {icon} {m['role'].title()}\n{m['content']}\n\n---\n\n"
                
                st.download_button("📥 Export Chat (.md)", md_chat, "chat_history.md", mime="text/markdown")

            if st.button("🗑️ Reset Chat", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()

    with col_chat_main:
        with st.expander("📎 Area Allegati (Drag & Drop)", expanded=False):
             # Dynamic Key to clear after send
            if 'uploader_key' not in st.session_state: st.session_state.uploader_key = 0
            up_key = f"chat_up_main_{st.session_state.uploader_key}"
            st.caption("I file caricati vengono inviati SOLO al prossimo messaggio e poi rimossi.")
            st.file_uploader("Trascina qui i file", type=['txt', 'csv', 'xlsx', 'pdf', 'png', 'jpg', 'webp'], key=up_key, accept_multiple_files=True)

        chat_container = st.container(height=600)
        
        with chat_container:
            if not st.session_state.chat_history:
                st.info("👋 Ciao! Sono il tuo assistente SEO. Analizzo i tuoi dati di forecast. Chiedimi pure!")
            
            for i, msg in enumerate(st.session_state.chat_history):
                if msg["role"] == "user":
                    # --- USER MESSAGE (RIGHT ALIGNED) ---
                    col_u1, col_u2 = st.columns([1, 3]) 
                    with col_u2:
                        st.info(msg["content"], icon="👤")
                else:
                    # --- ASSISTANT MESSAGE (LEFT ALIGNED) ---
                    with st.chat_message("assistant"):
                        st.markdown(msg["content"])
                        handle_chat_actions(msg["content"], key_suffix=f"hist_{i}")

        if prompt := st.chat_input("Scrivi una domanda sui dati..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Collect files
            up_key = f"chat_up_main_{st.session_state.uploader_key}"
            files = st.session_state.get(up_key, [])
            if not files: files = []
            
            file_text_content = ""
            images_list = []
            
            for uploaded_file in files:
                uploaded_file.seek(0)
                try:
                    if uploaded_file.type.startswith('image'):
                        img_b64 = base64.b64encode(uploaded_file.read()).decode('utf-8')
                        images_list.append(img_b64)
                    elif uploaded_file.type == 'text/plain':
                        txt = uploaded_file.getvalue().decode("utf-8")
                        file_text_content += f"\nFILE {uploaded_file.name}:\n{txt}\n"
                    elif uploaded_file.type == 'text/csv':
                        df_up = pd.read_csv(uploaded_file)
                        ctx = df_up.head(300).to_csv(index=False)
                        file_text_content += f"\nFILE {uploaded_file.name} (CSV):\n{ctx}\n"
                    elif 'spreadsheet' in uploaded_file.type or uploaded_file.name.endswith('.xlsx'):
                        df_up = pd.read_excel(uploaded_file)
                        ctx = df_up.head(300).to_csv(index=False)
                        file_text_content += f"\nFILE {uploaded_file.name} (XLSX):\n{ctx}\n"
                except Exception as e:
                    st.warning(f"Errore file {uploaded_file.name}: {e}")
            
            # Render User Msg immediately
            with chat_container:
                col_u1, col_u2 = st.columns([1, 3])
                with col_u2:
                    st.info(prompt, icon="👤")
                    if files: st.caption(f"📎 {len(files)} file allegati.")

            # Prepare Context
            context_data = prepare_context_data(
                history_df, 
                st.session_state.events, 
                st.session_state.last_forecast, 
                st.session_state.last_metrics,
                config,
                None # st.session_state.baseline_forecast (Legacy)
            )
            
            # Call AI
            with chat_container:
               with st.chat_message("assistant"):
                   with st.spinner("Ragionando..."):
                       sys_p = st.session_state.chat_config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
                       mod = st.session_state.chat_config.get("model", "gpt-5.2")
                       temp = st.session_state.chat_config.get("temperature", 0.7)
                       
                       resp = chat_with_assistant(
                           prompt, 
                           st.session_state.chat_history[:-1], 
                           context_data, 
                           st.session_state.get('openai_api_key'),
                           model=mod,
                           system_prompt=sys_p,
                           temperature=temp,
                           images=images_list,
                           file_text=file_text_content
                       )
                       
                       st.markdown(resp)
                       handle_chat_actions(resp, key_suffix="stream")
            
            st.session_state.chat_history.append({"role": "assistant", "content": resp})
            
            # Clear Files
            st.session_state.uploader_key += 1
            
            # JS Scroll
            js = """
            <script>
                var chatContainer = window.parent.document.querySelector('.stChatInput');
                if (chatContainer) {
                    chatContainer.scrollIntoView({behavior: "smooth"});
                }
            </script>
            """
            st.components.v1.html(js, height=0)
            st.rerun()


