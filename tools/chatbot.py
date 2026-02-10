
import streamlit as st
from openai import OpenAI
import pandas as pd
import json

def get_system_prompt(custom_prompt=None, context_data=None):
    base_prompt = """Sei un assistente esperto in SEO e Forecasting.
Il tuo compito è aiutare l'utente a comprendere i dati di traffico, le previsioni generate da Prophet e l'impatto dei regressori.
Hai accesso DIRETTO e TEMPO REALE a tutti i dati del progetto: storico GSC, configurazione, eventi attivi e Forecast aggiornato.

⚠️ REGOLA AUREA SUI DATI:
NON chiedere MAI all'utente di incollare numeri, CSV o screenshot dei grafici.
Tutte le informazioni necessarie (Click totali, Breakdown trimestrale, Impatto regressori, Confronto Baseline) sono contenute nel blocco "DATI CONTESTO ATTUALE" che ricevi in ogni messaggio.
Se l'utente ha modificato i regressori, i dati che leggi sono già quelli ricalcolati. Fidati del contesto.

Quando rispondi:
1. Sii preciso e basati sui dati forniti nel contesto.
2. Se l'utente chiede il motivo di un picco o un calo, analizza i regressori attivi in quel periodo o la stagionalità.
3. Giustifica sempre le tue affermazioni citando i numeri (es. "Il traffico cresce del 10%...").
4. Sii proattivo: suggerisci se vedi regressori mancanti o parametri strani.

Struttura le risposte in modo chiaro, usando elenchi puntati o grassetto per i concetti chiave.
"""
    if custom_prompt:
        base_prompt = custom_prompt

    technical_instructions = """

--- PROTOCOLLO COMUNICAZIONE JSON (Regressori) ---
Se suggerisci modifiche ai regressori (aggiunte, rimozioni o update), includi un blocco JSON alla fine.
Usa il campo "action" per specificare l'operazione (`add`, `remove`, `update`).

Esempi:
```json
{
  "suggested_regressors": [
    { "action": "add", "name": "Nuova Campagna", "type": "ramp", "date": "2026-03-01", "duration": 28, "impact": 0.3 },
    { "action": "remove", "name": "Evento Cancellato" },
    { "action": "update", "name": "Evento Esistente", "impact": 0.5, "duration": 45 }
  ]
}
```
Regole:
1. Controlla sempre i "Regressori Attivi" nel contesto prima di suggerire duplicati.
2. Se un evento esiste già ma vuoi cambiarlo, usa `update`.
3. Se un evento è sbagliato/obsoleto, usa `remove`.
"""
    
    if context_data:
        context_str = f"\n\n--- DATI CONTESTO ATTUALE ---\n{context_data}\n-----------------------------"
        return base_prompt + technical_instructions + context_str
    
    return base_prompt + technical_instructions

def prepare_context_data(gsc_data, events, forecast_data, metrics, config, baseline_data=None):
    """
    Condenses the application state into a text summary for the LLM.
    """
    context = []
    
    # Force awareness of live state
    now_str = pd.Timestamp.now().strftime('%H:%M:%S')
    context.append(f"!!! STATUS DASHBOARD: DATI AGGIORNATI (LIVE {now_str}) !!!")
    context.append("AVVISO AI: I dati numerici qui sotto (Forecast, Quarters, Componenti) sono CALCOLATI IN TEMPO REALE dopo le ultime modifiche.")
    context.append("NON CHIEDERE SCREENSHOT. Rispondi basandoti su questi numeri.")
    
    # 1. Config
    context.append(f"Configurazione Attuale: {json.dumps(config, indent=2, default=str)}")
    
    # 2. Input Data Stats
    if gsc_data is not None and not gsc_data.empty:
        # Normalize columns for analysis (handle raw vs prophet format)
        df = gsc_data.copy()
        if 'ds' not in df.columns and 'date' in df.columns:
            df = df.rename(columns={'date': 'ds'})
        if 'y' not in df.columns and 'clicks' in df.columns:
            df = df.rename(columns={'clicks': 'y'})
            
        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(df['ds']):
             df['ds'] = pd.to_datetime(df['ds'])

        last_date = df['ds'].max()
        first_date = df['ds'].min()
        total_clicks = df['y'].sum()
        avg_clicks = df['y'].mean()
        context.append(f"Dati Storici (GSC): Dal {first_date} al {last_date}. Totale Click: {int(total_clicks)}. Media/Giorno: {int(avg_clicks)}.")
        
        # Last 30 days trend
        last_30 = df.tail(30)
        trend_30 = "Crescente" if last_30['y'].is_monotonic_increasing else "Variabile"
        context.append(f"Trend ultimi 30gg input: {trend_30}. Media ultimi 30gg: {int(last_30['y'].mean())}.")

    # 3. Regressors
    if events:
        events_summary = [f"{e['name']} ({e['type']}): {e['date']} impact={e['impact']}" for e in events]
        context.append(f"Regressori Attivi ({len(events)}): " + ", ".join(events_summary))
    else:
        context.append("Regressori Attivi: Nessuno.")

    # 4. Forecast Results
    if forecast_data is not None and not forecast_data.empty:
        # Forecast summary (future only)
        future_mask = forecast_data['ds'] > pd.to_datetime('today')
        future_forecast = forecast_data[future_mask]
        
        if not future_forecast.empty:
            tot_future = future_forecast['yhat'].sum()
            avg_future = future_forecast['yhat'].mean()
            context.append(f"Previsione Futura (Scenario Attuale): Totale stimato {int(tot_future)}, Media {int(avg_future)}.")
            
            # Trend info
            start_val = future_forecast.iloc[0]['yhat']
            end_val = future_forecast.iloc[-1]['yhat']
            diff_pct = ((end_val - start_val) / start_val) * 100 if start_val != 0 else 0
            context.append(f"Trend Scenario: da {int(start_val)} a {int(end_val)} ({diff_pct:.1f}%).")

            # Baseline Comparison
            if baseline_data is not None and not baseline_data.empty:
                # Ensure date alignment logic matches (simple sum for now)
                future_baseline = baseline_data[baseline_data['ds'] > pd.to_datetime('today')]
                
                if not future_baseline.empty:
                    tot_baseline = future_baseline['yhat'].sum()
                    delta = tot_future - tot_baseline
                    delta_pct = (delta / tot_baseline) * 100 if tot_baseline != 0 else 0
                    context.append(f"Confronto con Baseline: {int(delta):+} click ({delta_pct:+.1f}%) rispetto allo scenario salvato.")

            # 5. Quarterly Breakdown & YoY
            try:
                context.append("\n=== REPORT TRIMESTRALE (VALIDAZIONE TARGET - LIVE DATA) ===")
                context.append("Questo report mostra l'impatto REALE dei nuovi regressori:")
                
                # Setup
                df_curr = future_forecast.copy()
                df_curr['quarter'] = df_curr['ds'].dt.to_period('Q')
                q_curr = df_curr.groupby('quarter')['yhat'].sum()
                
                # History (for YoY)
                q_hist = pd.Series(dtype='float64')
                if gsc_data is not None and not gsc_data.empty:
                    df_h = gsc_data.copy()
                    # Normalize
                    if 'ds' not in df_h.columns and 'date' in df_h.columns: df_h = df_h.rename(columns={'date': 'ds'})
                    if 'y' not in df_h.columns and 'clicks' in df_h.columns: df_h = df_h.rename(columns={'clicks': 'y'})
                    if 'ds' in df_h.columns: # Verify column exists
                        if not pd.api.types.is_datetime64_any_dtype(df_h['ds']):
                             df_h['ds'] = pd.to_datetime(df_h['ds'])
                        df_h['quarter'] = df_h['ds'].dt.to_period('Q')
                        q_hist = df_h.groupby('quarter')['y'].sum()
                
                # Baseline (for Scenario Gap)
                q_base = pd.Series(dtype='float64')
                if baseline_data is not None and not baseline_data.empty:
                    df_b = baseline_data[baseline_data['ds'] > pd.to_datetime('today')].copy()
                    df_b['quarter'] = df_b['ds'].dt.to_period('Q')
                    q_base = df_b.groupby('quarter')['yhat'].sum()

                # Loop Forecast Quarters
                cols_avail = df_curr.columns.tolist()
                
                for q in q_curr.index:
                    val_scen = q_curr[q]
                    line = f"• {q}: {int(val_scen):,}"
                    
                    # Components Breakdown (Trend vs Regressors)
                    mask_q = df_curr['quarter'] == q
                    rows_q = df_curr[mask_q]
                    
                    if not rows_q.empty and 'trend' in cols_avail:
                        trend_val = rows_q['trend'].sum()
                        
                        # Sum Active Regressors
                        regs_val = 0
                        if events:
                            for evt in events:
                                ename = evt['name']
                                if ename in cols_avail:
                                    regs_val += rows_q[ename].sum()
                        
                        # Append component info
                        line += f" [Trend: {int(trend_val):,+}, Regs: {int(regs_val):,+}]"

                    # vs Baseline
                    if q in q_base.index:
                        val_b = q_base[q]
                        diff_b = val_scen - val_b
                        pct_b = (diff_b / val_b * 100) if val_b else 0
                        line += f" | vs Base: {pct_b:+.1f}%"
                    
                    # vs History (YoY)
                    prev_q = q - 4
                    if prev_q in q_hist.index:
                        val_h = q_hist[prev_q]
                        diff_h = val_scen - val_h
                        pct_h = (diff_h / val_h * 100) if val_h else 0
                        line += f" | YoY: {pct_h:+.1f}% (vs {int(val_h):,})"
                    
                    context.append(line)
            except Exception as e:
                context.append(f"(Errore nel calcolo trimestrale: {str(e)})")
    
    # 5. Metrics
    if metrics:
        context.append(f"Metriche Accuratezza (Cross-Validation): {json.dumps(metrics, indent=2)}")

    return "\n".join(context)

def chat_with_assistant(user_input, history, context_data, api_key, model="gpt-4o", system_prompt=None, temperature=0.7, images=None, file_text=None):
    """
    Sends message to OpenAI and returns response.
    Supports Multiple Images input (base64) and File Text content.
    """
    if not api_key:
        return "⚠️ Errore: API Key mancante. Aggiungila nelle impostazioni o nel .env."

    client = OpenAI(api_key=api_key)
    
    # Append File Text to Context if present
    full_context = context_data
    if file_text:
        full_context += f"\n\n--- CONTENUTO FILE ALLEGATO ---\n{file_text}\n------------------------------"
    
    # Build messages
    full_system_prompt = get_system_prompt(system_prompt, full_context)
    
    messages = [{"role": "system", "content": full_system_prompt}]
    
    # Add history
    for msg in history:
        # Check if msg content is list (multimodal history not fully supported yet in simple history, sticking to text for history for now)
        # If we really want multimodal history, we need to store the structured object.
        # For now, let's assume history is text-only or simplified. 
        # If we pushed a complex object to history, this might break. 
        # But app.py appends {"role": "user", "content": prompt} (text).
        # We handle the CURRENT image only.
        messages.append({"role": msg["role"], "content": msg["content"]})
        
    # Current User Message
    if images:
        # Vision Request with Multiple Images
        content_list = [{"type": "text", "text": user_input}]
        for img_b64 in images:
             content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})
        user_msg = {"role": "user", "content": content_list}
    else:
        user_msg = {"role": "user", "content": user_input}

    messages.append(user_msg)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Errore API OpenAI: {str(e)}"
