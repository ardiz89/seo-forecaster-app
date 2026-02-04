import os
from openai import OpenAI
from dotenv import load_dotenv

# Load env immediately
load_dotenv()

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, "OpenAI API Key mancante nel file .env."
    try:
        client = OpenAI(api_key=api_key)
        return client, None
    except Exception as e:
        return None, str(e)

def generate_marketing_report(metrics, events, horizon, forecast_df=None):
    """
    Generates a marketing report using GPT-5.1.
    """
    client, error = get_openai_client()
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

def check_openai_credits():
    """
    Checks if API key works by making a minimal 'hello' call.
    OpenAI API doesn't expose a 'credits balance' endpoint directly for standard keys easily,
    so we test validity.
    """
    client, error = get_openai_client()
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
