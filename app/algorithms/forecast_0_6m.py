"""
================================================================================
0-6 Month Forecast Algorithm
================================================================================
Version: 1.0.0 | Status: LOCKED - DO NOT MODIFY
Uses peak-based forecasting with seasonality elasticity (^0.65)
================================================================================
"""
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple
from ..database import execute_query

# LOCKED CONSTANTS - DO NOT MODIFY
ELASTICITY = 0.65
DOI_GOAL = 93
INBOUND_LEAD_TIME = 30
MFG_LEAD_TIME = 7
PLANNING_HORIZON = DOI_GOAL + INBOUND_LEAD_TIME + MFG_LEAD_TIME
AGE_CUTOFF_MONTHS = 6
DAYS_PER_MONTH = 30.44


def calculate_seasonality_engine(sv: List[int]) -> tuple:
    """
    Seasonality Engine - matches Excel spec exactly.
    
    Columns:
    - C: sv_peak_env = MAX(B[t-2:t]) - 3-week trailing peak
    - D: sv_peak_env_offset = (C[t] + C[t+1]) / 2
    - E: sv_smooth_env = AVERAGE(D[t-1:t+1]) - 3-week centered
    - F: sv_final_curve = AVERAGE(B, D, E)
    - G: sv_smooth = moving average with edge handling
    - H: sv_smooth_env_final = (G[t] + G[t+1]) / 2
    - I: sv_smooth_env_96 = H * 0.96 (scaled for CVR calc in 6-18m)
    - J: seasonality_index = H / MAX(H) (normalized 0-1 for forecast)
    
    Returns: (sv_smooth_env_96, seasonality_index) as dicts keyed by week 1-52
    """
    if not sv or all(v == 0 for v in sv):
        return {}, {}
    
    n = len(sv)
    
    # C: sv_peak_env - trailing 3-week max (t-2, t-1, t)
    C = [max(sv[max(0, t-2):t+1]) for t in range(n)]
    
    # D: sv_peak_env_offset - forward average (C[t] + C[t+1]) / 2
    D = [(C[t] + C[t+1]) / 2 if t + 1 < n else C[t] for t in range(n)]
    
    # E: sv_smooth_env - 3-week centered average of D
    E = [sum(D[max(0, t-1):min(n, t+2)]) / len(D[max(0, t-1):min(n, t+2)]) for t in range(n)]
    
    # F: sv_final_curve - blend of raw (B), offset peak (D), smoothed env (E)
    F = [(sv[t] + D[t] + E[t]) / 3 for t in range(n)]
    
    # G: sv_smooth - moving average with edge handling
    G = []
    for t in range(n):
        if t == 0:
            window = F[0:2]      # Week 1: F1-F2
        elif t == 1:
            window = F[0:3]      # Week 2: F1-F3
        else:
            window = F[t-1:t+2]  # Week 3+: centered (t-1, t, t+1)
        G.append(sum(window) / len(window) if window else 0)
    
    # H: sv_smooth_env_final - forward 2-week average (G[t] + G[t+1]) / 2
    H = [(G[t] + G[t+1]) / 2 if t + 1 < n else G[t] for t in range(n)]
    
    # I: sv_smooth_env_96 - scaled by 0.96 (used in 6-18m CVR calc)
    I = {t + 1: H[t] * 0.96 for t in range(min(52, n))}
    
    # J: seasonality_index - normalized 0-1 (used in 0-6m and 6-18m forecast)
    max_H = max(H[:52]) if H else 1
    J = {t + 1: H[t] / max_H if max_H > 0 else 0 for t in range(min(52, n))}
    
    return I, J


def calculate_seasonality_fast(sv: List[int]) -> Dict[int, float]:
    """Legacy wrapper - returns seasonality_index only (for 0-6m)."""
    _, seasonality_index = calculate_seasonality_engine(sv)
    return seasonality_index


def calculate_total_inventory(inv: dict) -> int:
    return sum([
        inv.get('fba_available', 0) or 0,
        inv.get('fba_reserved', 0) or 0,
        inv.get('fba_inbound', 0) or 0,
        inv.get('awd_available', 0) or 0,
        inv.get('awd_reserved', 0) or 0,
        inv.get('awd_inbound', 0) or 0,
    ])


def calculate_forecast(peak: int, idx: float, idx_now: float) -> float:
    """Column H: forecast = MAX(0, peak * (G[t] / idxNow)^0.65)"""
    if idx_now <= 0: idx_now = 0.5
    return max(0, peak * (idx / idx_now) ** ELASTICITY)


def calculate_doi(inventory: float, forecasts: List[Tuple[date, float]], today: date) -> int:
    """Calculate Days of Inventory"""
    remaining = float(inventory)
    for week_end, forecast in forecasts:
        if forecast <= 0: continue
        start_inv = remaining
        remaining -= forecast
        if remaining <= 0:
            fraction = start_inv / forecast
            runout = week_end - timedelta(days=7 - fraction * 7)
            return max(0, (runout - today).days)
    return 365


def calculate_units_to_make(forecasts: List[Tuple[date, float]], total_inv: int, 
                            today: date, horizon_end: date) -> int:
    """Calculate units needed to manufacture"""
    total_needed = 0
    for week_end, forecast in forecasts:
        week_start = week_end - timedelta(days=7)
        overlap = max(0, (min(horizon_end, week_end) - max(today, week_start)).days)
        if overlap > 0:
            total_needed += forecast * (overlap / 7)
    return max(0, int(round(total_needed - total_inv)))


def forecast_0_6m(asin: str, today_override: date = None, 
                  total_inventory: int = None, fba_inventory: int = None,
                  amazon_doi_goal: Optional[int] = None,
                  inbound_lead_time: Optional[int] = None,
                  manufacture_lead_time: Optional[int] = None) -> dict:
    """
    Main forecast function for 0-6 month products.
    
    Args:
        asin: Product ASIN
        today_override: Override today's date for testing
        total_inventory: Override total inventory (if None, fetches from DB)
        fba_inventory: Override FBA available inventory (if None, fetches from DB)
        amazon_doi_goal: Amazon DOI goal in days (defaults to DOI_GOAL constant if None)
        inbound_lead_time: Inbound lead time in days (defaults to INBOUND_LEAD_TIME constant if None)
        manufacture_lead_time: Manufacture lead time in days (defaults to MFG_LEAD_TIME constant if None)
    """
    today = today_override or date.today()
    # Use provided DOI settings or fall back to constants
    doi_goal = amazon_doi_goal if amazon_doi_goal is not None else DOI_GOAL
    inbound_lt = inbound_lead_time if inbound_lead_time is not None else INBOUND_LEAD_TIME
    mfg_lt = manufacture_lead_time if manufacture_lead_time is not None else MFG_LEAD_TIME
    planning_horizon = doi_goal + inbound_lt + mfg_lt
    horizon_end = today + timedelta(days=planning_horizon)
    
    # Get product info
    product = execute_query(
        "SELECT asin, parent_asin, release_date FROM products WHERE asin = %s",
        (asin,), fetch_one=True
    )
    if not product:
        return {'error': f'Product not found: {asin}', 'asin': asin, 'algorithm': '0-6m'}
    
    # Calculate age
    release_date = product.get('release_date')
    if release_date:
        age_months = max(0, (today - release_date).days / DAYS_PER_MONTH)
    else:
        age_months = 999
    
    if age_months >= AGE_CUTOFF_MONTHS:
        return {'error': f'Product age ({age_months:.1f}m) >= 6m cutoff', 
                'asin': asin, 'age_months': round(age_months, 1), 'algorithm': '0-6m'}
    
    # Get inventory (use overrides if provided)
    if total_inventory is not None and fba_inventory is not None:
        total_inv = total_inventory
        fba_available = fba_inventory
    else:
        inv = execute_query(
            "SELECT * FROM inventory WHERE asin = %s", (asin,), fetch_one=True
        ) or {}
        fba_available = fba_inventory if fba_inventory is not None else (inv.get('fba_available', 0) or 0)
        total_inv = total_inventory if total_inventory is not None else calculate_total_inventory(inv)
    
    # Get sales and vine claims
    sales = execute_query(
        "SELECT week_end, units_sold FROM weekly_sales WHERE asin = %s AND week_end < %s",
        (asin, today)
    )
    vine = execute_query(
        """SELECT DATE_TRUNC('week', claim_date)::date + 5 as week_end, SUM(units_claimed) as units
           FROM vine_claims WHERE asin = %s GROUP BY DATE_TRUNC('week', claim_date)""",
        (asin,)
    )
    vine_map = {v['week_end']: v['units'] for v in vine}
    
    # Calculate peak (adjusted for vine)
    historical = [max(0, (s['units_sold'] or 0) - vine_map.get(s['week_end'], 0)) for s in sales]
    peak = max(historical) if historical else 0
    
    if peak == 0:
        return {'asin': asin, 'algorithm': '0-6m', 'age_months': round(age_months, 1),
                'doi_total': 365, 'doi_fba': 365, 'units_to_make': 0, 'peak': 0,
                'total_inventory': total_inv, 'fba_available': fba_available, 'status': 'good'}
    
    # Get seasonality
    parent = product.get('parent_asin') or asin
    sv_rows = execute_query(
        "SELECT week_of_year, search_volume FROM asin_search_volume WHERE asin = %s",
        (parent,)
    )
    sv_dict = {r['week_of_year']: r['search_volume'] for r in sv_rows}
    sv_list = [sv_dict.get(w, 0) for w in range(1, 54)]
    seasonality = calculate_seasonality_fast(sv_list) or {w: 1.0 for w in range(1, 53)}
    
    # idxNow = seasonality index of the LAST historical row
    last_historical_week = max((s['week_end'] for s in sales if s['week_end'] < today), default=None)
    if last_historical_week:
        idx_now = seasonality.get(last_historical_week.isocalendar()[1], 0.5)
    else:
        idx_now = seasonality.get(today.isocalendar()[1], 0.5)
    if idx_now <= 0: idx_now = 0.5
    
    # Generate forecasts
    days_until_sat = (5 - today.weekday()) % 7 or 7
    current = today + timedelta(days=days_until_sat)
    forecasts = []
    while current <= today + timedelta(days=365):
        idx = seasonality.get(current.isocalendar()[1], 0.5)
        forecasts.append((current, calculate_forecast(peak, idx, idx_now)))
        current += timedelta(days=7)
    
    # Calculate metrics
    doi_total = calculate_doi(total_inv, forecasts, today)
    doi_fba = calculate_doi(fba_available, forecasts, today)
    units_to_make = calculate_units_to_make(forecasts, total_inv, today, horizon_end)
    
    # Calculate cumulative forecast at key intervals (for instant DOI recalculation)
    cumulative_intervals = [30, 60, 90, 100, 110, 120, 130, 140, 150, 180, 200, 250, 300, 365]
    cumulative_forecast = {}
    
    for interval in cumulative_intervals:
        interval_end = today + timedelta(days=interval)
        interval_total = 0
        for week_end, forecast in forecasts:
            week_start = week_end - timedelta(days=7)
            overlap = max(0, (min(interval_end, week_end) - max(today, week_start)).days)
            if overlap > 0:
                interval_total += forecast * (overlap / 7)
        cumulative_forecast[str(interval)] = round(interval_total, 2)
    
    # Calculate daily forecast rate (average over 180 days for DOI calculation)
    daily_forecast_rate = cumulative_forecast.get('180', 0) / 180 if cumulative_forecast.get('180', 0) > 0 else 0
    
    status = 'critical' if doi_total <= 14 else 'low' if doi_total <= 30 else 'good'
    
    return {
        'asin': asin, 'algorithm': '0-6m', 'age_months': round(age_months, 1),
        'doi_total': doi_total, 'doi_fba': doi_fba, 'units_to_make': units_to_make,
        'peak': peak, 'idx_now': round(idx_now, 4),
        'total_inventory': total_inv, 'fba_available': fba_available, 'status': status,
        'daily_forecast_rate': round(daily_forecast_rate, 4),
        'cumulative_forecast': cumulative_forecast
    }
