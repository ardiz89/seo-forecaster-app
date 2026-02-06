
import pandas as pd
from datetime import timedelta
from dateutil.relativedelta import relativedelta

# --- DEFINIZIONE PRESET (Valori Default) ---
# Base impacts per unit (approximate lift)
BASE_IMPACT_ARTICLE = 0.015  # ~1.5% lift per optimized article
BASE_IMPACT_LINK = 0.03      # ~3.0% lift per quality link
BASE_IMPACT_TECH = 0.05      # ~5% for distinct tech fix
BASE_IMPACT_UPDATE = 0.08    # ~8% for content update/revamp

PRESET_TEMPLATES = {
    "content_publication": {
        "type": "ramp",         
        "duration": 60,         
        "impact": BASE_IMPACT_ARTICLE, 
        "event_type": "content"
    },
    "content_update": {
        "type": "step",         
        "duration": 1,          
        "impact": BASE_IMPACT_UPDATE,         
        "event_type": "content"
    },
    "technical_fix": {
        "type": "step",
        "duration": 1,
        "impact": BASE_IMPACT_TECH,       
        "event_type": "technical"
    },
    "link_building": {
        "type": "ramp",
        "duration": 90,         
        "impact": BASE_IMPACT_LINK,       
        "event_type": "offpage"
    },
    "site_migration": {
        "type": "step",
        "duration": 1,
        "impact": -0.10,        
        "event_type": "technical"
    },
    "ppc_campaign": {
        "type": "window",       
        "duration": 30,         
        "impact": 0.15,         
        "event_type": "marketing"
    }
}

def get_template_data(template_key, multiplier=1.0):
    """Retrieves template data scaled by a multiplier (quantity or intensity)."""
    tpl = PRESET_TEMPLATES.get(template_key, PRESET_TEMPLATES['content_publication']).copy()
    tpl['impact'] = round(tpl['impact'] * multiplier, 3)
    return tpl

def generate_prospecting_events(form_data):
    """
    Generates a list of event dictionaries based on the form data.
    """
    events = []
    start_date = pd.to_datetime(form_data['contract_start_date'])
    months = form_data['contract_months']
    
    # helper to get date from month index (1-based)
    def get_date_month(m_idx):
        return start_date + relativedelta(months=m_idx-1)

    # --- 1. SETUP ---
    s_mode = form_data.get('setup_mode', 'none')
    if s_mode != 'none':
        date_setup = start_date # Month 1
        
        if s_mode == 'lite':
            # Setup SEO Lite → technical_fix
            tpl = get_template_data('technical_fix')
            events.append({
                "name": "Setup SEO Lite (Tech Fix)",
                "date": date_setup,
                "type": tpl['type'],
                "duration": tpl['duration'],
                "impact": tpl['impact'],
                "event_type": tpl['event_type'],
                "notes": "Risoluzione errori tecnici di base (404, redirect, robots.txt)"
            })
            
        elif s_mode == 'full':
            # Setup SEO Full → technical_fix + content_update
            tpl_t = get_template_data('technical_fix')
            events.append({
                "name": "Setup SEO Full (Tech)",
                "date": date_setup,
                "type": tpl_t['type'],
                "duration": tpl_t['duration'],
                "impact": tpl_t['impact'],
                "event_type": tpl_t['event_type'],
                "notes": "Audit tecnico approfondito e fix strutturali"
            })
            tpl_c = get_template_data('content_update')
            events.append({
                "name": "Setup SEO Full (Content Audit)",
                "date": date_setup,
                "type": tpl_c['type'],
                "duration": tpl_c['duration'],
                "impact": tpl_c['impact'],
                "event_type": tpl_c['event_type'],
                "notes": "Analisi semantica e potatura contenuti obsoleti"
            })
            
        elif s_mode == 'strategy':
            # Setup + Content Strategy
            tpl_t = get_template_data('technical_fix')
            events.append({"name": "Setup Strategy (Tech)", "date": date_setup, "type": tpl_t['type'], "duration": tpl_t['duration'], "impact": tpl_t['impact'], "event_type": tpl_t['event_type']})
            tpl_c = get_template_data('content_update')
            events.append({"name": "Setup Strategy (Audit)", "date": date_setup, "type": tpl_c['type'], "duration": tpl_c['duration'], "impact": tpl_c['impact'], "event_type": tpl_c['event_type']})
            # Add Publication (Strategy Initial Push)
            tpl_p = get_template_data('content_publication', multiplier=3.0) # Approx 3 articles worth of strategy value
            events.append({"name": "Setup Strategy (Initial Plan)", "date": date_setup, "type": tpl_p['type'], "duration": tpl_p['duration'], "impact": tpl_p['impact'], "event_type": tpl_p['event_type'], "notes": "Definizione piano editoriale strategico"})

    # --- 2. MONTHLY PACKAGES ---
    
    def process_monthly_package(mode, range_tuple, quantity, handler_func):
        if mode and mode != 'none':
            start_m, end_m = range_tuple
            end_m = min(end_m, months)
            
            for m in range(start_m, end_m + 1):
                d = get_date_month(m)
                handler_func(mode, m, d, quantity)

    # CONTENT
    def handle_content(mode, month_idx, date_obj, quantity):
        # mode: basic, plus, authority (determines update frequency mostly)
        # quantity: number of articles per month
        
        # Publication Event (Ramp)
        # Impact = quantity * BASE_IMPACT_ARTICLE
        if quantity > 0:
            tpl_pub = get_template_data('content_publication', multiplier=quantity)
            events.append({
                "name": f"Content Pub ({int(quantity)} art.) M{month_idx}", 
                "date": date_obj, 
                "type": tpl_pub['type'], 
                "duration": tpl_pub['duration'], 
                "impact": tpl_pub['impact'], 
                "event_type": "content"
            })
            
        # Optimization Updates (Maintenance)
        # Frequency based on package mode
        tpl_upd = get_template_data('content_update') 
        
        is_update_month = False
        if mode == 'basic':
            if month_idx % 3 == 0: is_update_month = True
        elif mode in ['plus', 'authority']:
            is_update_month = True
            
        if is_update_month:
            events.append({
                "name": f"Content Optimization ({mode}) M{month_idx}", 
                "date": date_obj, 
                "type": tpl_upd['type'], 
                "duration": tpl_upd['duration'], 
                "impact": tpl_upd['impact'], 
                "event_type": "content"
            })

    process_monthly_package(
        form_data.get('content_mode'), 
        form_data.get('content_months', (1, months)), 
        form_data.get('content_quantity', 0), 
        handle_content
    )

    # LINK BUILDING
    def handle_links(mode, month_idx, date_obj, quantity):
        # quantity: number of links per month
        
        if quantity > 0:
            tpl = get_template_data('link_building', multiplier=quantity)
            events.append({
                "name": f"Link Building ({int(quantity)} links) M{month_idx}", 
                "date": date_obj, 
                "type": tpl['type'], 
                "duration": tpl['duration'], # 90 days ramp
                "impact": tpl['impact'], 
                "event_type": "offpage"
            })

    process_monthly_package(
        form_data.get('link_mode'), 
        form_data.get('link_months', (1, months)), 
        form_data.get('link_quantity', 0), 
        handle_links
    )
    
    # TECHNICAL CARE
    def handle_tech(mode, month_idx, date_obj, _):
        if mode == 'care':
            if month_idx % 3 == 0:
                 tpl = get_template_data('technical_fix')
                 events.append({"name": f"Tech Care (Quarterly) M{month_idx}", "date": date_obj, "type": tpl['type'], "duration": tpl['duration'], "impact": tpl['impact'], "event_type": "technical"})

    process_monthly_package(
        form_data.get('tech_mode'), 
        form_data.get('tech_months', (1, months)), 
        0, 
        handle_tech
    )

    # ON-PAGE OPTIMIZATION
    if form_data.get('onpage_enabled'):
        def handle_onpage(mode, month_idx, date_obj, _):
             tpl = get_template_data('content_update')
             events.append({"name": f"On-Page Opt M{month_idx}", "date": date_obj, "type": tpl['type'], "duration": tpl['duration'], "impact": tpl['impact'], "event_type": "content"})
             
        process_monthly_package(
            'active', 
            form_data.get('onpage_months', (1, months)), 
            0,
            handle_onpage
        )
        
    # LOCAL SEO
    if form_data.get('local_enabled'):
        def handle_local(mode, month_idx, date_obj, _):
             tpl = get_template_data('content_update')
             # Slightly less impact than full content update usually? Or targeted.
             # Let's use 0.05 impact
             tpl['impact'] = 0.05
             events.append({"name": f"Local SEO M{month_idx}", "date": date_obj, "type": tpl['type'], "duration": tpl['duration'], "impact": tpl['impact'], "event_type": "local"})

        process_monthly_package(
            'active', 
            form_data.get('local_months', (1, months)), 
            0, 
            handle_local
        )

    # --- 3. EVENTI SPECIALI ---
    extras = form_data.get('extra_events', [])
    for ext in extras:
        e_type = ext.get('type')
        m_idx = ext.get('month', 1)
        date_obj = get_date_month(m_idx)
        
        if e_type == 'migration':
            tpl = get_template_data('site_migration')
            events.append({"name": "Site Migration / Replatform", "date": date_obj, "type": tpl['type'], "duration": tpl['duration'], "impact": tpl['impact'], "event_type": "technical"})
        
        elif e_type == 'revamp':
            tpl = get_template_data('content_update', multiplier=3.0) # High impact
            events.append({"name": "Mega Content Revamp", "date": date_obj, "type": tpl['type'], "duration": tpl['duration'], "impact": tpl['impact'], "event_type": "content"})
            
        elif e_type == 'campaign':
            tpl = get_template_data('ppc_campaign')
            name = ext.get('name', 'Brand Campaign')
            events.append({"name": name, "date": date_obj, "type": tpl['type'], "duration": tpl['duration'], "impact": tpl['impact'], "event_type": "marketing"})

    return events
