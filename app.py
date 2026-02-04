import streamlit as st
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
from tools.ingest_data import validate_gsc_data, parse_regressors
from tools.run_forecast import execute_forecast
from tools.report_generator import generate_marketing_report, check_openai_credits

# Force reload of tools to pick up debug logic changes without restart
importlib.reload(tools.run_forecast)
importlib.reload(tools.ingest_data)
importlib.reload(tools.regressor_logic)
importlib.reload(tools.report_generator)

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
    
    seasonality = st.selectbox("Seasonality Mode", ["multiplicative", "additive"], index=0)
    
    col_cp1, col_cp2 = st.columns(2)
    with col_cp1:
        changepoint_scale = st.slider("Changepoint Prior Scale", 0.001, 0.5, 0.05, 0.001)
    with col_cp2:
        changepoint_range = st.slider("Changepoint Range", 0.1, 1.0, 0.8, 0.1)

    seasonality_scale = st.slider("Seasonality Prior Scale", 0.01, 20.0, 10.0, 0.1)
    
    st.markdown("---")
    st.caption("Componenti Stagionali")
    # Toggles for specific seasonalities
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        yearly_seas = st.selectbox("Yearly", ["auto", True, False], index=0)
    with col_s2:
        weekly_seas = st.selectbox("Weekly", ["auto", True, False], index=1) # Default True usually better
    with col_s3:
        daily_seas = st.selectbox("Daily", ["auto", True, False], index=2) # Default False for SEO usually

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
       
    # AUTO-TUNING SUGGESTION logic
    if st.session_state.get('auto_tuned') is not True and gsc_file:
         # Simple heuristic based on data length/variance could go here
         # For now, we just suggest defaults, but we can make it look "smart"
         pass 

    # --- REGRESSOR EDITOR ---
    st.subheader("🛠️ Gestione Regressori (Scenario)")
    
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
        # Convert list of dicts to DF for editing
        if st.session_state.events:
            df_editor = pd.DataFrame(st.session_state.events)
            
            # Ensure columns order
            cols_order = ['name', 'date', 'type', 'duration', 'impact', 'event_type']
            for c in cols_order:
                if c not in df_editor.columns:
                    df_editor[c] = None
            
            df_editor = df_editor[cols_order]
        else:
            # Empty template
            df_editor = pd.DataFrame(columns=['name', 'date', 'type', 'duration', 'impact', 'event_type'])

        # Add "Seleziona" column for deletion if not exists
        if "seleziona" not in df_editor.columns:
            df_editor.insert(0, "seleziona", False)

        # Data Editor
        edited_df = st.data_editor(
            df_editor,
            num_rows="dynamic",
            column_config={
                "seleziona": st.column_config.CheckboxColumn("Elimina?", default=False),
                "date": st.column_config.DateColumn("Data", format="YYYY-MM-DD"),
                "type": st.column_config.SelectboxColumn("Tipo", options=["decay", "window", "step", "ramp"]),
                "impact": st.column_config.NumberColumn("Impatto", help="Range -1.0 a 1.0"),
                "duration": st.column_config.NumberColumn("Durata (gg)"),
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
        
        # Sync back to session state logic (Automatic Save for Edits)
        # We only sync rows that are NOT marked for deletion to avoid confusion,
        # OR we just sync everything sans the checkbox. 
        # Standard approach: Sync everything. Deletion is a separate action initiated by the button.
        # But if we sync everything, we save the "seleziona" state? 
        # No, because we rebuild the DF from st.session_state.events which doesn't have 'seleziona'.
        
        # NOTE: Auto-save logic must ignore the 'seleziona' column and only run if NO delete button was pressed 
        # (though streamilt order creates button after editor).
        # To strictly follow "auto-save edits", we should update state here too.
        
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
        except Exception as e:
            pass # Transient error during typing possible

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
                    valid, msg = check_openai_credits()
                    if valid:
                        st.success(msg)
                    else:
                        st.error(msg)
                
                st.write("") # Spacer
                
                if st.button("✨ Genera Report Prospect", type="primary"):
                    with st.spinner("L'AI sta analizzando i dati... (GPT-5.1)"):
                        report_text, err = generate_marketing_report(
                            metrics, 
                            st.session_state.events, 
                            config['horizon_days'], 
                            forecast
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

