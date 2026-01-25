"""
================================================================================
6-18 Month Forecast Algorithm
================================================================================
Version: 1.1.0 | Status: LOCKED - DO NOT MODIFY
Uses CVR-based forecasting with seasonality weighting (25%)
================================================================================
"""
from datetime import date, timedelta
from typing import Dict, List, Tuple
from ..database import execute_query

# LOCKED CONSTANTS - DO NOT MODIFY
CVR_SEASONALITY_WEIGHT = 0.25
SV_SCALE_FACTOR = 0.96
DOI_GOAL = 93
INBOUND_LEAD_TIME = 30
MFG_LEAD_TIME = 7
PLANNING_HORIZON = DOI_GOAL + INBOUND_LEAD_TIME + MFG_LEAD_TIME
AGE_MIN_MONTHS = 6
AGE_MAX_MONTHS = 18
DAYS_PER_MONTH = 30.44


def calculate_seasonality_curve(sv: List[int]) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Calculate sv_smooth_env_0.96 and seasonality_index."""
    if not sv or len(sv) < 2 or all(v == 0 for v in sv[1:]):
        return {}, {}
    
    n = len(sv)
    sv_peak_env = [0] * n
    for t in range(1, n):
        sv_peak_env[t] = max(sv[max(1, t-2):t+1])
    
    sv_peak_env_offset = [0] * n
    for t in range(1, n):
        sv_peak_env_offset[t] = (sv_peak_env[t] + sv_peak_env[t+1]) / 2 if t + 1 < n else sv_peak_env[t]
    
    sv_smooth_env = [0] * n
    for t in range(1, n):
        window = sv_peak_env_offset[max(1, t-1):min(n, t+2)]
        sv_smooth_env[t] = sum(window) / len(window) if window else 0
    
    sv_final = [0] * n
    for t in range(1, n):
        sv_final[t] = (sv[t] + sv_peak_env_offset[t] + sv_smooth_env[t]) / 3
    
    sv_smooth = [0] * n
    for t in range(1, n):
        if t == 1: window = sv_final[1:3]
        elif t == 2: window = sv_final[1:4]
        else: window = sv_final[t-1:t+2]
        sv_smooth[t] = sum(window) / len(window) if window else 0
    
    sv_smooth_env_H = [0] * n
    for t in range(1, n):
        sv_smooth_env_H[t] = (sv_smooth[t] + sv_smooth[t+1]) / 2 if t + 1 < n else sv_smooth[t]
    
    sv_smooth_scaled = {t: sv_smooth_env_H[t] * SV_SCALE_FACTOR for t in range(1, min(53, n))}
    max_val = max(sv_smooth_env_H[1:min(53, n)]) or 1
    seasonality_idx = {t: sv_smooth_env_H[t] / max_val for t in range(1, min(53, n))}
    
    return sv_smooth_scaled, seasonality_idx


def calculate_avg_peak_cvr(sales_sv_list: List[Tuple[date, float]]) -> float:
    """Calculate avg peak CVR using 5-row window around max."""
    if not sales_sv_list:
        return 0.0
    max_val = max(val for _, val in sales_sv_list)
    if max_val <= 0:
        return 0.0
    
    threshold = max_val * 0.99
    candidates = [i for i, (_, val) in enumerate(sales_sv_list) if val >= threshold]
    
    def get_window_avg(peak_idx):
        vals = []
        for offset in range(-2, 3):
            idx = peak_idx + offset
            vals.append(sales_sv_list[idx][1] if 0 <= idx < len(sales_sv_list) else 0.0)
        return sum(vals) / 5
    
    return max(get_window_avg(c) for c in candidates) if candidates else 0.0


def calculate_total_inventory(inv: dict) -> int:
    return sum([
        inv.get('fba_available', 0) or 0,
        inv.get('fba_reserved', 0) or 0,
        inv.get('fba_inbound', 0) or 0,
        inv.get('awd_available', 0) or 0,
        inv.get('awd_reserved', 0) or 0,
        inv.get('awd_inbound', 0) or 0,
    ])


def calculate_doi(inventory: float, forecasts: List[Tuple[date, float]], today: date) -> int:
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
    total_needed = 0
    for week_end, forecast in forecasts:
        week_start = week_end - timedelta(days=7)
        overlap = max(0, (min(horizon_end, week_end) - max(today, week_start)).days)
        if overlap > 0:
            total_needed += forecast * (overlap / 7)
    return max(0, int(round(total_needed - total_inv)))


def forecast_6_18m(asin: str, today_override: date = None,
                   total_inventory: int = None, fba_inventory: int = None) -> dict:
    """
    Main forecast function for 6-18 month products.
    
    Args:
        asin: Product ASIN
        today_override: Override today's date for testing
        total_inventory: Override total inventory (if None, fetches from DB)
        fba_inventory: Override FBA available inventory (if None, fetches from DB)
    """
    today = today_override or date.today()
    horizon_end = today + timedelta(days=PLANNING_HORIZON)
    one_year_ago = today - timedelta(days=365)
    
    # Get product info
    product = execute_query(
        "SELECT asin, parent_asin, release_date FROM products WHERE asin = %s",
        (asin,), fetch_one=True
    )
    if not product:
        return {'error': f'Product not found: {asin}', 'asin': asin, 'algorithm': '6-18m'}
    
    # Calculate age
    release_date = product.get('release_date')
    age_months = max(0, (today - release_date).days / DAYS_PER_MONTH) if release_date else 999
    
    if age_months < AGE_MIN_MONTHS:
        return {'error': f'Product age ({age_months:.1f}m) < 6m', 'asin': asin, 'algorithm': '6-18m'}
    if age_months >= AGE_MAX_MONTHS:
        return {'error': f'Product age ({age_months:.1f}m) >= 18m', 'asin': asin, 'algorithm': '6-18m'}
    
    # Get inventory (use overrides if provided)
    if total_inventory is not None and fba_inventory is not None:
        total_inv = total_inventory
        fba_available = fba_inventory
    else:
        inv = execute_query("SELECT * FROM inventory WHERE asin = %s", (asin,), fetch_one=True) or {}
        fba_available = fba_inventory if fba_inventory is not None else (inv.get('fba_available', 0) or 0)
        total_inv = total_inventory if total_inventory is not None else calculate_total_inventory(inv)
    
    # Get search volume by DATE
    parent = product.get('parent_asin') or asin
    
    sv_by_date = execute_query(
        "SELECT week_end, search_volume FROM sv_weekly WHERE parent_asin = %s ORDER BY week_end",
        (parent,)
    )
    sv_date_dict = {r['week_end']: r['search_volume'] for r in sv_by_date}
    
    # Calculate seasonality curves from date-based SV
    if sv_by_date:
        sv_dates = sorted(sv_date_dict.keys())
        sv_raw = [sv_date_dict[d] for d in sv_dates]
        sv_scaled_list, seasonality_idx_list = calculate_seasonality_curve([0] + sv_raw)
        
        sv_scaled_by_date = {sv_dates[i]: sv_scaled_list.get(i+1, 1000) for i in range(len(sv_dates))}
        seasonality_by_date = {sv_dates[i]: seasonality_idx_list.get(i+1, 1.0) for i in range(len(sv_dates))}
        
        sv_scaled = {}
        seasonality_idx = {}
        for d in sv_dates:
            woy = d.isocalendar()[1]
            sv_scaled[woy] = sv_scaled_list.get(sv_dates.index(d) + 1, 1000) if d in sv_dates else 1000
            seasonality_idx[woy] = seasonality_idx_list.get(sv_dates.index(d) + 1, 1.0) if d in sv_dates else 1.0
    else:
        sv_rows = execute_query(
            "SELECT week_of_year, search_volume FROM asin_search_volume WHERE asin = %s", (parent,)
        )
        sv_dict = {r['week_of_year']: r['search_volume'] for r in sv_rows}
        sv_list = [0] + [sv_dict.get(w, 0) for w in range(1, 53)]
        sv_scaled, seasonality_idx = calculate_seasonality_curve(sv_list)
        sv_date_dict = {}
        sv_scaled_by_date = {}
        seasonality_by_date = {}
    
    if not sv_scaled:
        sv_scaled = {w: 1000.0 for w in range(1, 53)}
        seasonality_idx = {w: 1.0 for w in range(1, 53)}
    
    # Get sales and vine claims
    sales = execute_query(
        """SELECT week_end, units_sold FROM weekly_sales 
           WHERE asin = %s AND week_end < %s AND week_end >= %s ORDER BY week_end""",
        (asin, today, one_year_ago)
    )
    vine = execute_query(
        """SELECT DATE_TRUNC('week', claim_date)::date + 5 as week_end, SUM(units_claimed) as units
           FROM vine_claims WHERE asin = %s GROUP BY DATE_TRUNC('week', claim_date)""",
        (asin,)
    )
    vine_map = {v['week_end']: v['units'] for v in vine}
    
    # Calculate Sales/SV list
    sales_sv_list = []
    for s in sales:
        units = max(0, (s['units_sold'] or 0) - vine_map.get(s['week_end'], 0))
        sv_val = sv_scaled_by_date.get(s['week_end']) if sv_scaled_by_date else None
        if sv_val is None:
            week_num = s['week_end'].isocalendar()[1]
            sv_val = sv_scaled.get(week_num, 1)
        sales_sv = units / sv_val if sv_val and sv_val > 0 else 0.0
        sales_sv_list.append((s['week_end'], sales_sv))
    
    # Calculate avg peak CVR
    avg_peak_cvr = calculate_avg_peak_cvr(sales_sv_list)
    
    if avg_peak_cvr <= 0:
        return {'asin': asin, 'algorithm': '6-18m', 'age_months': round(age_months, 1),
                'doi_total': 365, 'doi_fba': 365, 'units_to_make': 0, 'avg_peak_cvr': 0,
                'total_inventory': total_inv, 'fba_available': fba_available, 'status': 'good'}
    
    # CVR adjustment: boost for high-volume products
    adjusted_cvr = avg_peak_cvr
    recent_12wk_sales = 0
    weekly_avg = 0
    if len(sales) >= 12:
        recent_12wk_sales = sum(s['units_sold'] or 0 for s in sales[-12:])
        weekly_avg = recent_12wk_sales / 12
        if weekly_avg > 50:
            volume_boost = min(0.10, (weekly_avg - 50) / 500)
            adjusted_cvr = avg_peak_cvr * (1 + volume_boost)
    
    # Seasonality + Growth adjustment
    # Calculate amplitude and growth rate for all products
    if seasonality_idx:
        min_idx = min(seasonality_idx.values())
        max_idx = max(seasonality_idx.values())
        amplitude = max_idx / min_idx if min_idx > 0 else 1
        
        # Calculate growth rate (recent 12wk vs yearly avg)
        yearly_sales = sum(s['units_sold'] or 0 for s in sales)
        yearly_avg = yearly_sales / len(sales) if sales else 0
        growth_rate = weekly_avg / yearly_avg if yearly_avg > 0 else 1.0
        
        # HIGH SEASONALITY adjustments (only with volume boost)
        if weekly_avg > 50 and amplitude > 20:  # Very high seasonality (25x+)
            if growth_rate > 2.5:
                adjusted_cvr *= 1.05
            elif growth_rate < 1.3:
                adjusted_cvr *= 0.70
            else:
                adjusted_cvr *= 0.77
        elif weekly_avg > 50 and amplitude > 10:  # High seasonality (10-20x)
            if growth_rate > 2.5:
                adjusted_cvr *= 1.05
            elif growth_rate < 1.5:
                adjusted_cvr *= 0.85
            else:
                adjusted_cvr *= 0.95
        
        # LOW SEASONALITY adjustments (amplitude <= 10)
        elif amplitude <= 10:
            if weekly_avg > 50 and growth_rate > 2.0:
                # Low seasonality + high growth + volume = +3% boost
                adjusted_cvr *= 1.03
            elif weekly_avg <= 50 and growth_rate >= 1.0:
                # Low seasonality + no volume boost + stable/growing = +10.4% base boost
                # Don't boost declining products (growth_rate < 1.0)
                adjusted_cvr *= 1.104
    
    # Generate forecasts using week_of_year for seasonality lookup
    # (sv_weekly contains historical dates, we need to map future dates by week number)
    days_until_sat = (5 - today.weekday()) % 7 or 7
    current = today + timedelta(days=days_until_sat)
    forecasts = []
    while current <= today + timedelta(days=365):
        week_num = current.isocalendar()[1]
        # Use week_of_year for lookup since sv_weekly has historical dates only
        sv_val = sv_scaled.get(week_num, 1000)
        idx = seasonality_idx.get(week_num, 1.0)
        weighted_cvr = adjusted_cvr * (1 + CVR_SEASONALITY_WEIGHT * (idx - 1))
        forecasts.append((current, max(0, sv_val * weighted_cvr)))
        current += timedelta(days=7)
    
    # Calculate metrics
    doi_total = calculate_doi(total_inv, forecasts, today)
    doi_fba = calculate_doi(fba_available, forecasts, today)
    units_to_make = calculate_units_to_make(forecasts, total_inv, today, horizon_end)
    peak = int(max(f[1] for f in forecasts)) if forecasts else 0
    
    status = 'critical' if doi_total <= 14 else 'low' if doi_total <= 30 else 'good'
    
    return {
        'asin': asin, 'algorithm': '6-18m', 'age_months': round(age_months, 1),
        'doi_total': doi_total, 'doi_fba': doi_fba, 'units_to_make': units_to_make,
        'peak': peak, 'avg_peak_cvr': round(avg_peak_cvr, 6),
        'total_inventory': total_inv, 'fba_available': fba_available, 'status': status
    }
