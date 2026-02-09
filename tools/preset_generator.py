
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
    New Logic: Uses explicit total impact values provided by the commercial user.
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
            tpl = get_template_data('technical_fix')
            events.append({
                "name": "Setup SEO Lite (Tech Fix)",
                "date": date_setup,
                "type": tpl['type'],
                "duration": tpl['duration'],
                "impact": tpl['impact'],
                "event_type": tpl['event_type']
            })
            
        elif s_mode == 'full':
            tpl_t = get_template_data('technical_fix')
            events.append({
                "name": "Setup SEO Full (Tech)",
                "date": date_setup,
                "type": tpl_t['type'],
                "duration": tpl_t['duration'],
                "impact": tpl_t['impact'],
                "event_type": tpl_t['event_type']
            })
            tpl_c = get_template_data('content_update')
            events.append({
                "name": "Setup SEO Full (Content Audit)",
                "date": date_setup,
                "type": tpl_c['type'],
                "duration": tpl_c['duration'],
                "impact": tpl_c['impact'],
                "event_type": tpl_c['event_type']
            })
            
        elif s_mode == 'strategy':
            tpl_t = get_template_data('technical_fix')
            events.append({"name": "Setup Strategy (Tech)", "date": date_setup, "type": tpl_t['type'], "duration": tpl_t['duration'], "impact": tpl_t['impact'], "event_type": tpl_t['event_type']})
            tpl_c = get_template_data('content_update')
            events.append({"name": "Setup Strategy (Audit)", "date": date_setup, "type": tpl_c['type'], "duration": tpl_c['duration'], "impact": tpl_c['impact'], "event_type": tpl_c['event_type']})
            # Strategy Plan
            events.append({"name": "Setup Strategy (Initial Plan)", "date": date_setup, "type": "step", "duration": 1, "impact": 0.05, "event_type": "content"})

    # --- 2. MONTHLY PACKAGES (Aggregated Impact) ---
    
    # CONTENT
    if form_data.get('content_enabled'):
        c_range = form_data.get('content_months', (1, 12))
        total_impact = form_data.get('content_impact_total', 0) / 100.0
        
        start_m, end_m = c_range
        end_m = min(end_m, months)
        duration_months = max(1, end_m - start_m + 1)
        duration_days = duration_months * 30
        
        if total_impact > 0:
            events.append({
                "name": f"Content Marketing Strategy (Target +{int(total_impact*100)}%)",
                "date": get_date_month(start_m),
                "type": "ramp",
                "duration": duration_days,
                "impact": total_impact,
                "event_type": "content",
                "notes": f"Attività continuativa M{start_m}-M{end_m}"
            })

    # LINK BUILDING
    if form_data.get('link_enabled'):
        l_range = form_data.get('link_months', (1, 12))
        total_impact = form_data.get('link_impact_total', 0) / 100.0
        
        start_m, end_m = l_range
        end_m = min(end_m, months)
        duration_months = max(1, end_m - start_m + 1)
        duration_days = duration_months * 30

        if total_impact > 0:
             events.append({
                "name": f"Link Building Strategy (Target +{int(total_impact*100)}%)",
                "date": get_date_month(start_m),
                "type": "ramp",
                "duration": duration_days,
                "impact": total_impact,
                "event_type": "offpage",
                "notes": f"Attività continuativa M{start_m}-M{end_m}"
            })
            
    # TECH & ONPAGE (Fixed logic per checkbox)
    # TECH
    t_mode = form_data.get('tech_mode')
    if t_mode == 'care':
        # Quarterly maintenance
        for m in range(1, months + 1):
            if m % 3 == 0:
                events.append({
                    "name": f"Tech Care M{m}",
                    "date": get_date_month(m),
                    "type": "step",
                    "duration": 1,
                    "impact": 0.02, # Small incremental fix
                    "event_type": "technical"
                })
                
    # ONPAGE
    if form_data.get('onpage_enabled'):
        # Modeled as a slow steady ramp over the whole active period
        op_range = form_data.get('onpage_months', (1, 12))
        start_m, end_m = op_range
        duration_days = (end_m - start_m + 1) * 30
        
        events.append({
            "name": "On-Page Optimization Cycle",
            "date": get_date_month(start_m),
            "type": "ramp",
            "duration": duration_days,
            "impact": 0.10, # default 10% lift for on-page
            "event_type": "content"
        })

    # LOCAL
    if form_data.get('local_enabled'):
        l_range = form_data.get('local_months', (1, 12))
        start_m, end_m = l_range
        duration_days = (end_m - start_m + 1) * 30
        
        events.append({
            "name": "Local SEO Optimization",
            "date": get_date_month(start_m),
            "type": "ramp",
            "duration": duration_days,
            "impact": 0.08, # default 8% lift
            "event_type": "local"
        })

    # --- 3. EVENTI SPECIALI ---
    extras = form_data.get('extra_events', [])
    for ext in extras:
        e_type = ext.get('type')
        m_idx = ext.get('month', 1)
        date_obj = get_date_month(m_idx)
        
        if e_type == 'migration':
            # Migration has Drop and Recovery
            drop_pct = ext.get('drop_pct', 10.0) / 100.0
            growth_pct = ext.get('growth_pct', 15.0) / 100.0
            
            # 1. Initial Drop (Step down)
            events.append({
                "name": "Migration Drop",
                "date": date_obj,
                "type": "step",
                "duration": 1,
                "impact": -drop_pct,
                "event_type": "technical",
                "notes": f"Calo fisiologico post-migrazione (-{int(drop_pct*100)}%)"
            })
            
            # 2. Recovery & Growth (Ramp up)
            # To achieve net +Growth, we need to recover the Drop AND add Growth.
            # Ramp Impact = Drop + Growth
            ramp_impact = drop_pct + growth_pct
            
            events.append({
                "name": "Migration Recovery & Growth",
                "date": date_obj + timedelta(days=7), # Start shortly after
                "type": "ramp",
                "duration": 180, # 6 months to fully realize?
                "impact": ramp_impact,
                "event_type": "technical",
                "notes": f"Recupero tecnico e crescita (+{int(growth_pct*100)}% net)"
            })
        
        elif e_type == 'revamp':
            growth_pct = ext.get('growth_pct', 20.0) / 100.0
            events.append({
                "name": "Mega Content Revamp",
                "date": date_obj,
                "type": "ramp",
                "duration": 90, # 3 months rollout
                "impact": growth_pct,
                "event_type": "content"
            })
            
        elif e_type == 'campaign':
            tpl = get_template_data('ppc_campaign')
            name = ext.get('name', 'Brand Campaign')
            events.append({"name": name, "date": date_obj, "type": tpl['type'], "duration": tpl['duration'], "impact": tpl['impact'], "event_type": "marketing"})

    return events
