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


def run_forecast(asin: str, algorithm: str = None) -> dict:
    """Run forecast for a single ASIN using appropriate algorithm"""
    if algorithm is None:
        algorithm = get_appropriate_algorithm(asin)
    
    if algorithm == '0-6m':
        return forecast_0_6m(asin)
    elif algorithm == '6-18m':
        return forecast_6_18m(asin)
    else:
        return forecast_18m_plus(asin)


@router.get("/{asin}", response_model=ForecastResult)
async def get_forecast(
    asin: str,
    algorithm: Optional[str] = Query(
        None, 
        description="Force specific algorithm (0-6m, 6-18m, 18m+). Auto-detected if not specified."
    )
):
    """
    Get forecast for a single ASIN.
    
    The algorithm is automatically selected based on product age:
    - 0-6m: Products less than 6 months old
    - 6-18m: Products 6-18 months old
    - 18m+: Products older than 18 months
    
    You can override the algorithm selection using the `algorithm` query parameter.
    """
    # Validate algorithm if provided
    if algorithm and algorithm not in ['0-6m', '6-18m', '18m+']:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid algorithm: {algorithm}. Must be one of: 0-6m, 6-18m, 18m+"
        )
    
    result = run_forecast(asin, algorithm)
    
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
    
    # Get products with seasonality data AND inventory
    products = execute_query("""
        SELECT DISTINCT p.asin 
        FROM products p
        JOIN asin_search_volume sv ON (p.parent_asin = sv.asin OR p.asin = sv.asin)
        JOIN inventory i ON p.asin = i.asin
        WHERE sv.search_volume > 0
        AND (i.fba_available > 0 OR i.awd_available > 0 OR i.fba_inbound > 0 OR i.awd_inbound > 0)
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
    
    # Calculate forecasts in parallel
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
    Returns historical sales and forecast data.
    """
    # Get weekly sales data
    sales = execute_query(
        "SELECT week_end, units_sold FROM weekly_sales WHERE asin = %s ORDER BY week_end",
        (asin,)
    )
    
    if not sales:
        return {
            "asin": asin,
            "historical": [],
            "forecast": []
        }
    
    # Get forecast data to calculate future projections
    forecast_result = run_forecast(asin)
    
    # Format historical sales
    historical = [
        {
            "date": str(sale['week_end']),
            "units": sale['units_sold'] or 0
        }
        for sale in sales
    ]
    
    # Calculate smoothed values (simple moving average)
    smoothed = []
    for i, sale in enumerate(sales):
        window = sales[max(0, i-2):min(len(sales), i+3)]
        avg = sum(s['units_sold'] or 0 for s in window) / len(window)
        smoothed.append({
            "date": str(sale['week_end']),
            "units": round(avg)
        })
    
    # For forecast, we'd need to calculate from the forecast algorithm
    # For now, return empty and let frontend calculate from peak/velocity
    return {
        "asin": asin,
        "historical": historical,
        "smoothed": smoothed,
        "peak": forecast_result.get('peak'),
        "weekly_velocity": forecast_result.get('peak') or 0
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
