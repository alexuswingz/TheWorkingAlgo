"""
================================================================================
18m+ Forecast Algorithm
================================================================================
Version: 1.0.0 | Status: LOCKED - DO NOT MODIFY
Uses weighted smoothing with prior year data
================================================================================
"""
from datetime import date, timedelta
from typing import Dict, List, Optional
from ..database import execute_query

# LOCKED CONSTANTS - DO NOT MODIFY
L_CORRECTION = 0.9970
DOI_GOAL = 93
INBOUND_LEAD_TIME = 30
MFG_LEAD_TIME = 7
PLANNING_HORIZON = DOI_GOAL + INBOUND_LEAD_TIME + MFG_LEAD_TIME
AGE_MIN_MONTHS = 18
DAYS_PER_MONTH = 30.44
MARKET_ADJUSTMENT = 0.05
VELOCITY_WEIGHT = 0.15
H_WEIGHTS = [1, 2, 4, 7, 11, 13, 11, 7, 4, 2, 1]
L_WEIGHTS = [1, 3, 5, 7, 5, 3, 1]


def col_D_units_peak_env(units: List[int]) -> List[float]:
    result = []
    for i in range(len(units)):
        if units[i] == 0:
            result.append(0)
            continue
        vals = [units[i+offset] for offset in [-2, -1, 0, 1] if 0 <= i+offset < len(units)]
        result.append(max(vals) if vals else units[i])
    return result


def col_E_units_peak_env_offset(D: List[float]) -> List[float]:
    return [(D[i] + D[i+1]) / 2 if i+1 < len(D) else D[i] for i in range(len(D))]


def col_F_units_smooth_env(E: List[float]) -> List[float]:
    return [sum(E[max(0, i-1):min(len(E), i+2)]) / len(E[max(0, i-1):min(len(E), i+2)]) for i in range(len(E))]


def col_G_units_final_curve(units: List[int], E: List[float], F: List[float]) -> List[float]:
    return [max(units[i], E[i], F[i]) for i in range(len(units))]


def col_H_units_final_smooth(G: List[float], week_dates: List[date], today: date) -> List[Optional[float]]:
    result = []
    for i in range(len(G)):
        if week_dates[i] > today:
            result.append(None)
            continue
        vals, wgts = [], []
        for j, w in enumerate(H_WEIGHTS):
            idx = i + j - 5
            if 0 <= idx < len(G):
                vals.append(G[idx] if G[idx] else 0)
                wgts.append(w if G[idx] and G[idx] > 0 else 0)
        total = sum(wgts)
        result.append(sum(v * w for v, w in zip(vals, wgts)) / total if total > 0 else None)
    return result


def col_I(H: List[Optional[float]]) -> List[Optional[float]]:
    return [v * 0.85 if v is not None else None for v in H]


def col_J(I: List[Optional[float]]) -> List[Optional[float]]:
    return [None] * 52 + I[:-52] if len(I) > 52 else [None] * len(I)


def col_K(J: List[Optional[float]]) -> List[Optional[float]]:
    result = []
    for i in range(len(J)):
        if J[i] is None:
            result.append(None)
            continue
        vals = [J[i+o] for o in [-2, -1] if 0 <= i+o < len(J) and J[i+o] is not None]
        result.append(max(vals) if vals else J[i])
    return result


def col_L(K: List[Optional[float]]) -> List[Optional[float]]:
    result = []
    for i in range(len(K)):
        if K[i] is None:
            result.append(None)
            continue
        vals, wgts = [], []
        for j, w in enumerate(L_WEIGHTS):
            idx = i + j - 3
            if 0 <= idx < len(K):
                vals.append(K[idx] if K[idx] is not None else 0)
                wgts.append(w if K[idx] is not None and K[idx] > 0 else 0)
        total = sum(wgts)
        result.append(sum(v * w for v, w in zip(vals, wgts)) / total if total > 0 else None)
    return result


def calc_velocity(I: List[Optional[float]], L: List[Optional[float]], 
                  week_dates: List[date], today: date) -> float:
    """Calculate sales velocity adjustment."""
    idx = None
    for i in range(len(week_dates) - 1, -1, -1):
        if week_dates[i] <= today and I[i] is not None and L[i] is not None:
            idx = i
            break
    if idx is None or idx < 6:
        return 0.0
    
    windows = [(1, 7), (2, 14), (4, 28), (6, 42)]
    i_sum = l_sum = 0.0
    valid = 0
    for n, d in windows:
        i_tot = l_tot = 0.0
        ok = True
        for j in range(n):
            if idx - j < 0 or I[idx-j] is None or L[idx-j] is None or L[idx-j] <= 0:
                ok = False
                break
            i_tot += I[idx-j]
            l_tot += L[idx-j]
        if ok and l_tot > 0:
            i_sum += 0.25 * (i_tot / d)
            l_sum += 0.25 * (l_tot / d)
            valid += 1
    return (i_sum / l_sum) - 1 if valid == 4 and l_sum > 0 else 0.0


def calculate_total_inventory(inv: dict) -> int:
    return sum([inv.get(k, 0) or 0 for k in 
                ['fba_available', 'fba_reserved', 'fba_inbound', 
                 'awd_available', 'awd_reserved', 'awd_inbound']])


def calculate_doi(inventory: float, forecasts: List[tuple], today: date) -> int:
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


def calculate_units_to_make(forecasts: List[tuple], total_inv: int,
                            today: date, horizon_end: date) -> int:
    total_needed = 0
    for week_end, forecast in forecasts:
        week_start = week_end - timedelta(days=7)
        overlap = max(0, (min(horizon_end, week_end) - max(today, week_start)).days)
        if overlap > 0:
            total_needed += forecast * (overlap / 7)
    return max(0, int(round(total_needed - total_inv)))


def forecast_18m_plus(asin: str, today_override: date = None,
                       total_inventory: int = None, fba_inventory: int = None,
                       amazon_doi_goal: Optional[int] = None,
                       inbound_lead_time: Optional[int] = None,
                       manufacture_lead_time: Optional[int] = None) -> dict:
    """Main forecast function for 18m+ products.
    
    Args:
        asin: Product ASIN
        today_override: Override for today's date
        total_inventory: Override for total inventory (if None, fetched from DB)
        fba_inventory: Override for FBA available inventory (if None, fetched from DB)
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
        "SELECT asin, release_date FROM products WHERE asin = %s", (asin,), fetch_one=True
    )
    if not product:
        return {'error': f'Product not found: {asin}', 'asin': asin, 'algorithm': '18m+'}
    
    release_date = product.get('release_date')
    age_months = max(0, (today - release_date).days / DAYS_PER_MONTH) if release_date else 999
    
    if age_months < AGE_MIN_MONTHS:
        return {'error': f'Product age ({age_months:.1f}m) < 18m', 'asin': asin, 'algorithm': '18m+'}
    
    # Get inventory (use overrides if provided)
    inv = execute_query("SELECT * FROM inventory WHERE asin = %s", (asin,), fetch_one=True) or {}
    if fba_inventory is not None:
        fba_available = fba_inventory
    else:
        fba_available = inv.get('fba_available', 0) or 0
    if total_inventory is not None:
        total_inv = total_inventory
    else:
        total_inv = calculate_total_inventory(inv)
    
    # Get sales
    sales = execute_query(
        "SELECT week_end, units_sold FROM weekly_sales WHERE asin = %s ORDER BY week_end", (asin,)
    )
    if not sales:
        return {'asin': asin, 'algorithm': '18m+', 'age_months': round(age_months, 1),
                'doi_total_days': 365, 'doi_fba_days': 365, 'units_to_make': 0,
                'total_inventory': total_inv, 'fba_available': fba_available, 'status': 'good'}
    
    # Build weekly data and extend to future
    week_dates = [s['week_end'] for s in sales]
    units = [s['units_sold'] or 0 for s in sales]
    
    last = max(week_dates)
    current = last + timedelta(days=7)
    while current <= today + timedelta(days=365):
        week_dates.append(current)
        units.append(0)
        current += timedelta(days=7)
    
    # Calculate columns
    D = col_D_units_peak_env(units)
    E = col_E_units_peak_env_offset(D)
    F = col_F_units_smooth_env(E)
    G = col_G_units_final_curve(units, E, F)
    H = col_H_units_final_smooth(G, week_dates, today)
    I = col_I(H)
    J = col_J(I)
    K = col_K(J)
    L = col_L(K)
    
    velocity = calc_velocity(I, L, week_dates, today)
    multiplier = 1 + (velocity * VELOCITY_WEIGHT) + MARKET_ADJUSTMENT
    
    # High velocity boost: when velocity is extreme (>200%), prior year data is sparse
    if velocity > 2.0:
        high_velocity_boost = min(0.06, (velocity - 2.0) * 0.01)
        multiplier *= (1 + high_velocity_boost)
    
    # Generate forecasts
    forecasts = []
    for i, wd in enumerate(week_dates):
        if wd >= today and wd <= today + timedelta(days=365) and i < len(L) and L[i] is not None:
            O = L[i] * L_CORRECTION * multiplier
            next_L = L[i+1] if i+1 < len(L) and L[i+1] is not None else L[i]
            O_next = next_L * L_CORRECTION * multiplier
            P = (O + O_next) / 2
            forecasts.append((wd, P))
    
    doi_total = calculate_doi(total_inv, forecasts, today)
    doi_fba = calculate_doi(fba_available, forecasts, today)
    units_to_make = calculate_units_to_make(forecasts, total_inv, today, horizon_end)
    
    # Calculate daily forecast rate for fast recalculation
    forecast_180d_end = today + timedelta(days=180)
    total_forecast_180d = 0
    for week_end, forecast in forecasts:
        week_start = week_end - timedelta(days=7)
        overlap = max(0, (min(forecast_180d_end, week_end) - max(today, week_start)).days)
        if overlap > 0:
            total_forecast_180d += forecast * (overlap / 7)
    daily_forecast_rate = total_forecast_180d / 180 if total_forecast_180d > 0 else 0
    
    status = 'critical' if doi_total <= 14 else 'low' if doi_total <= 30 else 'good'
    
    return {
        'asin': asin, 'algorithm': '18m+', 'age_months': round(age_months, 1),
        'doi_total_days': doi_total, 'doi_fba_days': doi_fba, 
        'units_to_make': round(units_to_make, 2),
        'sales_velocity_adj': round(velocity, 4),
        'total_inventory': total_inv, 'fba_available': fba_available, 'status': status,
        'daily_forecast_rate': round(daily_forecast_rate, 4)
    }
