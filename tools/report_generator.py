import os
from openai import OpenAI
from dotenv import load_dotenv

# Load env immediately
load_dotenv()


def get_openai_client(api_key=None):
    # Priority: Passed key > Env key
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        return None, "OpenAI API Key mancante. Inseriscila nelle impostazioni."
        
    try:
        client = OpenAI(api_key=api_key)
        return client, None
    except Exception as e:
        return None, str(e)

def generate_marketing_report(metrics, events, horizon, forecast_df=None, api_key=None):
    """
    Generates a marketing report using GPT-5.1.
    """
    client, error = get_openai_client(api_key)
    if not client:
        return None, f"Errore configurazione OpenAI: {error}"

    # Prepare context
    # Summarize events
    events_summary = "\n".join([f"- {e['name']} ({e['date']}): {e['type']}, Impact {e['impact']}" for e in events]) if events else "Nessun evento significativo."
    
    # Calculate simple trends if DF provided
    trend_txt = ""
    if forecast_df is not None:
         # Get last 30 days of forecast
         future = forecast_df.tail(30)
         trend_dir = "In crescita" if future['yhat'].iloc[-1] > future['yhat'].iloc[0] else "In calo"
         trend_txt = f"Trend ultimi 30gg: {trend_dir}."

    # Prompt
    prompt = f"""
    Sei un Senior SEO Strategist e Data Analyst.
    Il tuo compito è analizzare i dati di un forecast di traffico organico e generare un report sintetico e persuasivo per un PROSPECT (potenziale cliente).
    
    DATI FORECAST (Orizzonte {horizon} giorni):
    - Media Storica Giornaliera: {metrics.get('historical_mean', 0):.0f} click
    - Media Forecast Giornaliera: {metrics.get('forecast_mean', 0):.0f} click
    - Variazione Assoluta: {metrics.get('delta_abs', 0):.0f} click
    - Variazione Percentuale: {metrics.get('delta_perc', 0):.2f}%
    - Performance Modello (MAPE): {metrics.get('mape', 0):.2f}%
    
    {trend_txt}
    
    EVENTI/SCENARI INCLUSI NEL CALCOLO:
    {events_summary}
    
    ISTRUZIONI:
    1. Scrivi un report in Markdown.
    2. Tono: Professionale, Proattivo, Orientato al Business (non troppo tecnico).
    3. Struttura:
       - **Executive Summary**: Il dato chiave (crescita o calo previsto).
       - **Analisi dello Scenario**: Come gli eventi inseriti stanno influenzando il futuro.
       - **Affidabilità**: Spiega brevemente se possiamo fidarci del dato (basandosi sul MAPE).
       - **Next Steps**: Consigli strategici (inventa basandoti sui dati: se cala, proponi audit; se cresce, proponi content strategy).
    
    IMPORTANTE:
    - Usa il modello 'gpt-5.1'.
    - Sii sintetico ma convincente.
    """

    try:
        response = client.chat.completions.create(
            # model="gpt-4o",  # Fallback (Commented out to avoid syntax error)
            model="gpt-5.1",  # User requested specifically gpt-5.1
            messages=[
                {"role": "system", "content": "Sei un assistente SEO esperto."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content, None
    except Exception as e:
        return None, f"Errore generazione report: {str(e)}"

def check_openai_credits(api_key=None):
    """
    Checks if API key works by making a minimal 'hello' call.
    OpenAI API doesn't expose a 'credits balance' endpoint directly for standard keys easily,
    so we test validity.
    """
    client, error = get_openai_client(api_key)
    if not client:
        return False, error
    
    try:
        # Minimal inexpensive call
        client.chat.completions.create(
            model="gpt-3.5-turbo", # Use cheapest for check
            messages=[{"role": "user", "content": "Ping"}],
            max_tokens=1
        )
        return True, "API Key valida e operativa."
    except Exception as e:
        return False, f"API Key non valida o credito esaurito: {e}"

def analyze_parameters_with_ai(metrics_heuristics, df_head_tail_str, api_key=None):
    """
    Asks the AI to explain the suggested parameters based on the data heuristics.
    """
    client, error = get_openai_client(api_key)
    if not client:
        return None, f"Errore configurazione OpenAI: {error}"

    prompt = f'''
    Sei un Data Scientist esperto in Time Series Forecasting (Prophet).
    Ho analizzato un dataset SEO (Google Search Console) e calcolato delle euristiche.
    
    DATI ANALIZZATI:
    {df_head_tail_str}
    
    PARAMETRI SUGGERITI DALL'ALGORITMO:
    {metrics_heuristics}
    
    COMPITO:
    Spiega all'utente PERCHÉ questi parametri sono ideali per il suo caso specifico.
    Conferma o correggi (se ritieni assurdo) il suggerimento.
    
    FORMATO RISPOSTA:
    Markdown. Breve, elenco puntato. 
    Usa icone. Spiega concetti complessi come 'Seasonality Mode' o 'Changepoint' in termini semplici per un utente business.
    '''

    try:
        response = client.chat.completions.create(
            model="gpt-4", # Using gpt-4 or 5.1 if available
            messages=[
                {"role": "system", "content": "Sei un esperto di analisi dati e time series."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content, None
    except Exception as e:
        return None, f"Errore analisi AI: {str(e)}"

def analyze_regressors_with_ai(events, api_key=None):
    """
    Analyzes the user's regressors and suggests improvements.
    Expects a list of dicts: [{'name':..., 'type':..., 'impact':..., 'event_type':...}, ...]
    Returns a list of suggested changes/confirmations in JSON format or similar.
    """
    client, error = get_openai_client(api_key)
    if not client:
        return None, f"Errore configurazione OpenAI: {error}"
    
    # Format events for prompt
    events_str = ""
    for i, e in enumerate(events):
        events_str += f"ID {i}: Name='{e.get('name')}', Type='{e.get('type')}', Impact={e.get('impact')}, Duration={e.get('duration')}, EventType='{e.get('event_type')}'\n"

    prompt = f"""
    Sei un Esperto SEO e Tencico di Prophet.
    Hai il compito di revisionare la configurazione dei "Regressori" (Eventi significativi) impostati dall'utente per il forecast.
    
    TIPO DI REGRESSORI DISPONIBILI:
    - decay: Effetto improvviso che decresce (es. google update negativo).
    - window: Effetto costante per N giorni (es. promozione, server down).
    - step: Cambio permanente (es. migrazione).
    - ramp: Crescita graduale (es. link building).
    
    REGRESSORI UTENTE:
    {events_str}
    
    COMPITO:
    Per ogni regresso, analizza se il 'Type' scelto è coerente con il 'Name' o 'EventType'.
    Se trovi un'incongruenza (es. "Core Update" segnato come "window" invece che "decay" o "step"), suggerisci la correzione.
    Se è tutto ok, conferma.
    
    OUTPUT RICHIESTO (JSON only):
    Un array di oggetti JSON all'interno di un oggetto root "suggestions".
    Esempio: {{ "suggestions": [ {{ "id": 0, "name": "...", "suggested_type": "...", "reason": "...", "confidence": "Alto" }} ] }}
    
    Rispondi SOLO con il JSON valido.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-5.1", # Or gpt-4 if 5.1 not avail
            messages=[
                {"role": "system", "content": "Sei un analista tecnico SEO. Rispondi in JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content, None
    except Exception as e:
        return None, f"Errore AI Regressori: {str(e)}"



