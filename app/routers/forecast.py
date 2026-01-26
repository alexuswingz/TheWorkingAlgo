"""
Forecast API endpoints with database-backed pre-calculated data for instant response
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from datetime import date, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..models import ForecastResult, AllForecastsResponse
from ..database import execute_query
from ..algorithms import forecast_0_6m, forecast_6_18m, forecast_18m_plus

router = APIRouter(prefix="/forecast", tags=["Forecast"])

# Constants for age classification
DAYS_PER_MONTH = 30.44


def get_appropriate_algorithm(asin: str, today: date = None) -> str:
    """Determine which algorithm to use based on product age"""
    today = today or date.today()
    
    product = execute_query(
        "SELECT release_date FROM products WHERE asin = %s",
        (asin,), fetch_one=True
    )
    
    if not product or not product.get('release_date'):
        return '18m+'  # Default to 18m+ for unknown products
    
    age_months = (today - product['release_date']).days / DAYS_PER_MONTH
    
    if age_months < 6:
        return '0-6m'
    elif age_months < 18:
        return '6-18m'
    else:
        return '18m+'


def get_doi_settings(amazon_doi_goal: Optional[int] = None,
                     inbound_lead_time: Optional[int] = None,
                     manufacture_lead_time: Optional[int] = None) -> dict:
    """Get DOI settings from database or use provided overrides"""
    # If all parameters provided, use them directly
    if all(x is not None for x in [amazon_doi_goal, inbound_lead_time, manufacture_lead_time]):
        return {
            'amazon_doi_goal': amazon_doi_goal,
            'inbound_lead_time': inbound_lead_time,
            'manufacture_lead_time': manufacture_lead_time
        }
    
    # Try to get from database
    settings = execute_query(
        "SELECT amazon_doi_goal, inbound_lead_time, manufacture_lead_time FROM doi_settings WHERE is_default = true ORDER BY updated_at DESC LIMIT 1",
        fetch_one=True
    )
    
    if settings:
        return {
            'amazon_doi_goal': amazon_doi_goal if amazon_doi_goal is not None else settings.get('amazon_doi_goal', 130),
            'inbound_lead_time': inbound_lead_time if inbound_lead_time is not None else settings.get('inbound_lead_time', 30),
            'manufacture_lead_time': manufacture_lead_time if manufacture_lead_time is not None else settings.get('manufacture_lead_time', 7)
        }
    
    # Return defaults
    return {
        'amazon_doi_goal': amazon_doi_goal if amazon_doi_goal is not None else 130,
        'inbound_lead_time': inbound_lead_time if inbound_lead_time is not None else 30,
        'manufacture_lead_time': manufacture_lead_time if manufacture_lead_time is not None else 7
    }


def run_forecast(asin: str, algorithm: str = None,
                 amazon_doi_goal: Optional[int] = None,
                 inbound_lead_time: Optional[int] = None,
                 manufacture_lead_time: Optional[int] = None) -> dict:
    """Run forecast for a single ASIN using appropriate algorithm"""
    if algorithm is None:
        algorithm = get_appropriate_algorithm(asin)
    
    # Get DOI settings (will use database defaults if not provided)
    doi_settings = get_doi_settings(amazon_doi_goal, inbound_lead_time, manufacture_lead_time)
    
    if algorithm == '0-6m':
        return forecast_0_6m(
            asin,
            amazon_doi_goal=doi_settings['amazon_doi_goal'],
            inbound_lead_time=doi_settings['inbound_lead_time'],
            manufacture_lead_time=doi_settings['manufacture_lead_time']
        )
    elif algorithm == '6-18m':
        return forecast_6_18m(
            asin,
            amazon_doi_goal=doi_settings['amazon_doi_goal'],
            inbound_lead_time=doi_settings['inbound_lead_time'],
            manufacture_lead_time=doi_settings['manufacture_lead_time']
        )
    else:
        return forecast_18m_plus(
            asin,
            amazon_doi_goal=doi_settings['amazon_doi_goal'],
            inbound_lead_time=doi_settings['inbound_lead_time'],
            manufacture_lead_time=doi_settings['manufacture_lead_time']
        )


@router.get("/{asin}", response_model=ForecastResult)
async def get_forecast(
    asin: str,
    algorithm: Optional[str] = Query(
        None, 
        description="Force specific algorithm (0-6m, 6-18m, 18m+). Auto-detected if not specified."
    ),
    amazon_doi_goal: Optional[int] = Query(
        None,
        description="Amazon DOI goal in days (overrides default from database)"
    ),
    inbound_lead_time: Optional[int] = Query(
        None,
        description="Inbound lead time in days (overrides default from database)"
    ),
    manufacture_lead_time: Optional[int] = Query(
        None,
        description="Manufacture lead time in days (overrides default from database)"
    )
):
    """
    Get forecast for a single ASIN.
    
    The algorithm is automatically selected based on product age:
    - 0-6m: Products less than 6 months old
    - 6-18m: Products 6-18 months old
    - 18m+: Products older than 18 months
    
    You can override the algorithm selection using the `algorithm` query parameter.
    DOI settings (amazon_doi_goal, inbound_lead_time, manufacture_lead_time) can be provided
    as query parameters to override the default values from the database.
    """
    # Validate algorithm if provided
    if algorithm and algorithm not in ['0-6m', '6-18m', '18m+']:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid algorithm: {algorithm}. Must be one of: 0-6m, 6-18m, 18m+"
        )
    
    result = run_forecast(
        asin, 
        algorithm,
        amazon_doi_goal=amazon_doi_goal,
        inbound_lead_time=inbound_lead_time,
        manufacture_lead_time=manufacture_lead_time
    )
    
    # Normalize response keys for 18m+ algorithm
    if 'doi_total_days' in result:
        result['doi_total'] = result.pop('doi_total_days')
    if 'doi_fba_days' in result:
        result['doi_fba'] = result.pop('doi_fba_days')
    
    return ForecastResult(**result)


@router.get("/", response_model=AllForecastsResponse)
async def get_all_forecasts(
    status_filter: Optional[str] = Query(
        None,
        description="Filter by status: critical, low, good"
    ),
    algorithm_filter: Optional[str] = Query(
        None,
        description="Filter by algorithm: 0-6m, 6-18m, 18m+"
    ),
    limit: int = Query(1000, ge=1, le=10000, description="Maximum number of results"),
    min_units: int = Query(0, ge=0, description="Minimum units to make"),
    refresh: bool = Query(False, description="Force recalculation (slow)")
):
    """
    Get pre-calculated forecasts for all products (INSTANT from database).
    
    Returns forecasts from the pre-calculated `forecast_cache` table.
    Use ?refresh=true to force recalculation (takes ~2-3 minutes).
    """
    
    if refresh:
        # Force recalculation - this is slow but updates the cache
        return await _recalculate_all_forecasts(status_filter, algorithm_filter, limit, min_units)
    
    # Read from pre-calculated table with inventory details (FAST!)
    query = """
        SELECT 
            fc.asin, fc.product_name, fc.algorithm, fc.age_months,
            fc.doi_total, fc.doi_fba, fc.units_to_make, fc.peak,
            fc.total_inventory, fc.fba_available, fc.status,
            fc.calculated_at,
            COALESCE(i.fba_reserved, 0) as fba_reserved,
            COALESCE(i.fba_inbound, 0) as fba_inbound,
            COALESCE(i.awd_available, 0) as awd_available,
            COALESCE(i.awd_reserved, 0) as awd_reserved,
            COALESCE(i.awd_inbound, 0) as awd_inbound
        FROM forecast_cache fc
        LEFT JOIN inventory i ON fc.asin = i.asin
        WHERE 1=1
    """
    params = []
    
    if status_filter:
        query += " AND fc.status = %s"
        params.append(status_filter)
    
    if algorithm_filter:
        query += " AND fc.algorithm = %s"
        params.append(algorithm_filter)
    
    if min_units > 0:
        query += " AND fc.units_to_make >= %s"
        params.append(min_units)
    
    query += " ORDER BY fc.units_to_make DESC LIMIT %s"
    params.append(limit)
    
    results = execute_query(query, tuple(params) if params else None)
    
    if not results:
        # Table empty or doesn't exist - return empty response
        return AllForecastsResponse(
            total_products=0,
            forecasts=[],
            critical_count=0,
            low_count=0,
            good_count=0,
            error_count=0,
            total_units_to_make=0
        )
    
    # Get totals from database
    totals = execute_query("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN status = 'critical' THEN 1 ELSE 0 END) as critical_count,
            SUM(CASE WHEN status = 'low' THEN 1 ELSE 0 END) as low_count,
            SUM(CASE WHEN status = 'good' THEN 1 ELSE 0 END) as good_count,
            SUM(units_to_make) as total_units
        FROM forecast_cache
    """, fetch_one=True)
    
    # Convert to ForecastResult objects
    forecast_objects = []
    for r in results:
        forecast_objects.append(ForecastResult(
            asin=r['asin'],
            product_name=r.get('product_name'),
            algorithm=r['algorithm'],
            age_months=float(r['age_months']) if r.get('age_months') else None,
            doi_total=r['doi_total'],
            doi_fba=r['doi_fba'],
            units_to_make=r['units_to_make'],
            peak=r.get('peak'),
            total_inventory=r.get('total_inventory'),
            fba_available=r.get('fba_available'),
            fba_reserved=r.get('fba_reserved'),
            fba_inbound=r.get('fba_inbound'),
            awd_available=r.get('awd_available'),
            awd_reserved=r.get('awd_reserved'),
            awd_inbound=r.get('awd_inbound'),
            status=r['status']
        ))
    
    return AllForecastsResponse(
        total_products=totals['total'] if totals else 0,
        forecasts=forecast_objects,
        critical_count=totals['critical_count'] if totals else 0,
        low_count=totals['low_count'] if totals else 0,
        good_count=totals['good_count'] if totals else 0,
        error_count=0,
        total_units_to_make=int(totals['total_units'] or 0) if totals else 0
    )


async def _recalculate_all_forecasts(status_filter, algorithm_filter, limit, min_units):
    """Recalculate all forecasts and update the cache table"""
    
    # Get products based on algorithm requirements:
    # - 18m+ products: only need sales data (no seasonality required)
    # - 0-6m and 6-18m: need seasonality data AND inventory
    today = date.today()
    products = execute_query("""
        SELECT DISTINCT p.asin,
            CASE 
                WHEN p.release_date IS NULL THEN '18m+'
                WHEN (CURRENT_DATE - p.release_date) / 30.44 < 6 THEN '0-6m'
                WHEN (CURRENT_DATE - p.release_date) / 30.44 < 18 THEN '6-18m'
                ELSE '18m+'
            END as algorithm
        FROM products p
        WHERE (
            -- 18m+ products: only need sales data
            (
                (p.release_date IS NULL OR (CURRENT_DATE - p.release_date) / 30.44 >= 18)
                AND EXISTS (
                    SELECT 1 FROM weekly_sales ws 
                    WHERE ws.asin = p.asin AND ws.units_sold > 0
                )
            )
            OR
            -- 0-6m and 6-18m: need seasonality AND inventory
            (
                (CURRENT_DATE - p.release_date) / 30.44 < 18
                AND EXISTS (
                    SELECT 1 FROM asin_search_volume sv 
                    WHERE (p.parent_asin = sv.asin OR p.asin = sv.asin) 
                    AND sv.search_volume > 0
                )
                AND EXISTS (
                    SELECT 1 FROM inventory i 
                    WHERE i.asin = p.asin 
                    AND (i.fba_available > 0 OR i.awd_available > 0 OR i.fba_inbound > 0 OR i.awd_inbound > 0)
                )
            )
        )
        ORDER BY p.asin
    """)
    
    if not products:
        return AllForecastsResponse(
            total_products=0, forecasts=[], critical_count=0,
            low_count=0, good_count=0, error_count=0, total_units_to_make=0
        )
    
    asins = [p['asin'] for p in products]
    
    # Get product names
    product_names_result = execute_query("""
        SELECT asin, product_name FROM products WHERE product_name IS NOT NULL
    """)
    product_names = {p['asin']: p['product_name'] for p in product_names_result} if product_names_result else {}
    
    # Get DOI settings from database for cache refresh
    doi_settings = get_doi_settings()
    
    # Calculate forecasts in parallel
    results = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_asin = {
            executor.submit(
                run_forecast, 
                asin,
                amazon_doi_goal=doi_settings['amazon_doi_goal'],
                inbound_lead_time=doi_settings['inbound_lead_time'],
                manufacture_lead_time=doi_settings['manufacture_lead_time']
            ): asin 
            for asin in asins
        }
        
        for future in as_completed(future_to_asin):
            asin = future_to_asin[future]
            try:
                result = future.result()
                if result.get('error') or result.get('units_to_make') is None:
                    continue
                
                result['product_name'] = product_names.get(asin)
                
                if 'doi_total_days' in result:
                    result['doi_total'] = result.pop('doi_total_days')
                if 'doi_fba_days' in result:
                    result['doi_fba'] = result.pop('doi_fba_days')
                
                results.append(result)
            except Exception:
                continue
    
    # Save to database
    _save_forecasts_to_db(results)
    
    # Apply filters and return
    filtered = results
    if status_filter:
        filtered = [r for r in filtered if r.get('status') == status_filter]
    if algorithm_filter:
        filtered = [r for r in filtered if r.get('algorithm') == algorithm_filter]
    if min_units > 0:
        filtered = [r for r in filtered if (r.get('units_to_make') or 0) >= min_units]
    
    filtered.sort(key=lambda x: x.get('units_to_make') or 0, reverse=True)
    filtered = filtered[:limit]
    
    critical_count = sum(1 for r in results if r.get('status') == 'critical')
    low_count = sum(1 for r in results if r.get('status') == 'low')
    good_count = sum(1 for r in results if r.get('status') == 'good')
    total_units = sum(r.get('units_to_make') or 0 for r in results)
    
    forecast_objects = [ForecastResult(**r) for r in filtered]
    
    return AllForecastsResponse(
        total_products=len(results),
        forecasts=forecast_objects,
        critical_count=critical_count,
        low_count=low_count,
        good_count=good_count,
        error_count=0,
        total_units_to_make=total_units
    )


def _save_forecasts_to_db(results: list):
    """Save forecast results to the forecast_cache table"""
    from ..database import get_connection
    
    conn = get_connection()
    cur = conn.cursor()
    
    try:
        # Clear existing data
        cur.execute("DELETE FROM forecast_cache")
        
        # Insert new data
        for r in results:
            cur.execute("""
                INSERT INTO forecast_cache 
                (asin, product_name, algorithm, age_months, doi_total, doi_fba, 
                 units_to_make, peak, total_inventory, fba_available, status, calculated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (
                r.get('asin'),
                r.get('product_name'),
                r.get('algorithm'),
                r.get('age_months'),
                r.get('doi_total'),
                r.get('doi_fba'),
                r.get('units_to_make'),
                r.get('peak'),
                r.get('total_inventory'),
                r.get('fba_available'),
                r.get('status')
            ))
        
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()


@router.get("/{asin}/sales")
async def get_sales_data(asin: str):
    """
    Get weekly sales data for a specific ASIN for charting.
    Returns historical sales, smoothed data, and forecast matching Excel chart structure.
    Uses the appropriate algorithm based on product age.
    """
    from datetime import date, timedelta
    
    # Determine which algorithm to use
    algorithm = get_appropriate_algorithm(asin)
    
    # Get weekly sales data
    sales = execute_query(
        "SELECT week_end, units_sold FROM weekly_sales WHERE asin = %s ORDER BY week_end",
        (asin,)
    )
    
    if not sales:
        return {
            "asin": asin,
            "data": []
        }
    
    today = date.today()
    
    # Build weekly data arrays
    week_dates = [s['week_end'] for s in sales]
    units = [s['units_sold'] or 0 for s in sales]
    
    # Extend to future weeks
    last = max(week_dates)
    current = last + timedelta(days=7)
    while current <= today + timedelta(days=365):
        week_dates.append(current)
        units.append(0)
        current += timedelta(days=7)
    
    # Use 18m+ algorithm columns for all (they work for smoothing any historical data)
    from ..algorithms.forecast_18m_plus import (
        col_D_units_peak_env, col_E_units_peak_env_offset, col_F_units_smooth_env,
        col_G_units_final_curve, col_H_units_final_smooth, col_I, col_J, col_K, col_L,
        L_CORRECTION, VELOCITY_WEIGHT, MARKET_ADJUSTMENT, calc_velocity
    )
    
    # Calculate algorithm columns (matching Excel)
    D = col_D_units_peak_env(units)
    E = col_E_units_peak_env_offset(D)
    F = col_F_units_smooth_env(E)
    G = col_G_units_final_curve(units, E, F)
    H = col_H_units_final_smooth(G, week_dates, today)  # Historical smoothed
    I = col_I(H)  # Adjusted (H * 0.85)
    J = col_J(I)  # Prior year shifted
    K = col_K(J)
    L = col_L(K)  # Prior year smoothed
    
    # Calculate forecast (columns O and P) - use 18m+ calculation for all
    velocity = calc_velocity(I, L, week_dates, today)
    multiplier = 1 + (velocity * VELOCITY_WEIGHT) + MARKET_ADJUSTMENT
    
    if velocity > 2.0:
        high_velocity_boost = min(0.06, (velocity - 2.0) * 0.01)
        multiplier *= (1 + high_velocity_boost)
    
    # Build chart data
    chart_data = []
    for i, wd in enumerate(week_dates):
        units_sold = units[i] if i < len(units) else 0
        h_val = H[i] if i < len(H) and H[i] is not None else None
        l_val = L[i] if i < len(L) and L[i] is not None else None
        
        # Calculate forecast (P column) - use 18m+ formula
        forecast_val = None
        if wd >= today and wd <= today + timedelta(days=365) and l_val is not None:
            O = l_val * L_CORRECTION * multiplier
            next_L = L[i+1] if i+1 < len(L) and L[i+1] is not None else l_val
            O_next = next_L * L_CORRECTION * multiplier
            forecast_val = (O + O_next) / 2
        
        chart_data.append({
            "date": str(wd),
            "units_sold": units_sold,
            "smoothed": round(h_val) if h_val is not None else None,
            "prior_year": round(l_val) if l_val is not None else None,
            "forecast": round(forecast_val) if forecast_val is not None else None,
            "is_past": wd <= today,
            "is_future": wd > today
        })
    
    return {
        "asin": asin,
        "algorithm": algorithm,
        "data": chart_data
    }


@router.post("/refresh-cache")
async def refresh_forecast_cache():
    """
    Manually refresh the forecast cache table.
    This recalculates all forecasts and saves them to the database.
    Takes 2-3 minutes to complete.
    """
    products = execute_query("""
        SELECT DISTINCT p.asin 
        FROM products p
        JOIN asin_search_volume sv ON (p.parent_asin = sv.asin OR p.asin = sv.asin)
        WHERE sv.search_volume > 0
    """)
    
    if not products:
        return {"status": "no_products", "count": 0}
    
    asins = [p['asin'] for p in products]
    
    product_names_result = execute_query("""
        SELECT asin, product_name FROM products WHERE product_name IS NOT NULL
    """)
    product_names = {p['asin']: p['product_name'] for p in product_names_result} if product_names_result else {}
    
    results = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_asin = {executor.submit(run_forecast, asin): asin for asin in asins}
        
        for future in as_completed(future_to_asin):
            asin = future_to_asin[future]
            try:
                result = future.result()
                if result.get('error') or result.get('units_to_make') is None:
                    continue
                
                result['product_name'] = product_names.get(asin)
                
                if 'doi_total_days' in result:
                    result['doi_total'] = result.pop('doi_total_days')
                if 'doi_fba_days' in result:
                    result['doi_fba'] = result.pop('doi_fba_days')
                
                results.append(result)
            except Exception:
                continue
    
    _save_forecasts_to_db(results)
    
    return {
        "status": "success",
        "count": len(results),
        "message": f"Refreshed {len(results)} forecasts"
    }
