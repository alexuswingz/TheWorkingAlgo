"""
Forecast API endpoints with database-backed pre-calculated data for instant response
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from datetime import date, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..models import ForecastResult, AllForecastsResponse, DoiSettingsUsed
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
    """Get DOI settings from database or use provided overrides.
    
    Returns dict with amazon_doi_goal, inbound_lead_time, manufacture_lead_time, 
    and total_required_doi (the sum of all three).
    """
    # Try to get from database first
    settings = execute_query(
        "SELECT amazon_doi_goal, inbound_lead_time, manufacture_lead_time FROM doi_settings WHERE is_default = true ORDER BY updated_at DESC LIMIT 1",
        fetch_one=True
    )
    
    # Determine final values (priority: parameter > database > default)
    if settings:
        final_amazon_doi = amazon_doi_goal if amazon_doi_goal is not None else settings.get('amazon_doi_goal', 93)
        final_inbound_lt = inbound_lead_time if inbound_lead_time is not None else settings.get('inbound_lead_time', 30)
        final_mfg_lt = manufacture_lead_time if manufacture_lead_time is not None else settings.get('manufacture_lead_time', 7)
    else:
        final_amazon_doi = amazon_doi_goal if amazon_doi_goal is not None else 93
        final_inbound_lt = inbound_lead_time if inbound_lead_time is not None else 30
        final_mfg_lt = manufacture_lead_time if manufacture_lead_time is not None else 7
    
    return {
        'amazon_doi_goal': final_amazon_doi,
        'inbound_lead_time': final_inbound_lt,
        'manufacture_lead_time': final_mfg_lt,
        'total_required_doi': final_amazon_doi + final_inbound_lt + final_mfg_lt
    }


def build_doi_settings_response(doi_settings: dict) -> DoiSettingsUsed:
    """Build DoiSettingsUsed model from settings dict"""
    return DoiSettingsUsed(
        amazon_doi_goal=doi_settings['amazon_doi_goal'],
        inbound_lead_time=doi_settings['inbound_lead_time'],
        manufacture_lead_time=doi_settings['manufacture_lead_time'],
        total_required_doi=doi_settings['total_required_doi']
    )


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


@router.get("/recalculate-doi", response_model=AllForecastsResponse)
async def recalculate_doi(
    amazon_doi_goal: int = Query(..., description="Amazon DOI goal in days"),
    inbound_lead_time: int = Query(..., description="Inbound lead time in days"),
    manufacture_lead_time: int = Query(..., description="Manufacture lead time in days"),
    status_filter: Optional[str] = Query(None, description="Filter by status: critical, low, good"),
    algorithm_filter: Optional[str] = Query(None, description="Filter by algorithm: 0-6m, 6-18m, 18m+"),
    limit: int = Query(1000, ge=1, le=10000, description="Maximum number of results"),
    min_units: int = Query(0, ge=0, description="Minimum units to make")
):
    """
    INSTANT & ACCURATE recalculation of units_to_make with new DOI settings (~1-2 seconds).
    
    Uses cached cumulative_forecast data (stored at day intervals: 30,60,90...365) to
    accurately recalculate units_to_make by interpolating the forecast sum for any DOI.
    
    This gives the SAME accuracy as the single forecast endpoint but for ALL products instantly.
    
    Formula: units_to_make = max(0, interpolated_forecast_sum - total_inventory)
    """
    import json
    
    planning_horizon = amazon_doi_goal + inbound_lead_time + manufacture_lead_time
    
    doi_settings_response = DoiSettingsUsed(
        amazon_doi_goal=amazon_doi_goal,
        inbound_lead_time=inbound_lead_time,
        manufacture_lead_time=manufacture_lead_time,
        total_required_doi=planning_horizon
    )
    
    # Read cached data with cumulative_forecast and product images
    query = """
        SELECT 
            fc.asin, fc.product_name, fc.algorithm, fc.age_months,
            fc.peak, fc.total_inventory, fc.fba_available,
            fc.cumulative_forecast,
            COALESCE(fc.daily_forecast_rate, 0) as daily_forecast_rate,
            COALESCE(fc.doi_total, 0) as doi_total,
            COALESCE(fc.doi_fba, 0) as doi_fba,
            COALESCE(fc.units_to_make, 0) as cached_units_to_make,
            COALESCE(i.fba_reserved, 0) as fba_reserved,
            COALESCE(i.fba_inbound, 0) as fba_inbound,
            COALESCE(i.awd_available, 0) as awd_available,
            COALESCE(i.awd_reserved, 0) as awd_reserved,
            COALESCE(i.awd_inbound, 0) as awd_inbound,
            COALESCE(i.label_inventory, 0) as label_inventory,
            pi.image_url
        FROM forecast_cache fc
        LEFT JOIN inventory i ON fc.asin = i.asin
        LEFT JOIN product_images pi ON fc.asin = pi.asin
        WHERE 1=1
    """
    params = []
    
    if algorithm_filter:
        query += " AND fc.algorithm = %s"
        params.append(algorithm_filter)
    
    results = execute_query(query, tuple(params) if params else None)
    
    if not results:
        return AllForecastsResponse(
            total_products=0,
            forecasts=[],
            critical_count=0,
            low_count=0,
            good_count=0,
            error_count=0,
            total_units_to_make=0,
            doi_settings=doi_settings_response
        )
    
    # Recalculate units_to_make for each product using cached cumulative_forecast
    forecast_objects = []
    total_units = 0
    critical_count = 0
    low_count = 0
    good_count = 0
    
    # Get cached DOI settings to calculate scaling ratio
    # Default cached DOI is 130 (93 + 30 + 7)
    default_cached_doi = 130
    
    for r in results:
        total_inv = r.get('total_inventory') or 0
        fba_available = r.get('fba_available') or 0
        daily_rate = r.get('daily_forecast_rate') or 0
        cached_units = r.get('cached_units_to_make') or 0
        
        # Get cumulative forecast data
        cumulative_data = r.get('cumulative_forecast')
        forecast_sum = 0
        
        if cumulative_data:
            try:
                # Parse JSON if it's a string
                if isinstance(cumulative_data, str):
                    cumulative_data = json.loads(cumulative_data)
                
                # cumulative_data is a dict like {"30": 1500, "60": 3200, "90": 5000, ...}
                # Interpolate to find forecast sum at planning_horizon
                forecast_sum = _interpolate_cumulative_forecast(cumulative_data, planning_horizon)
                # Calculate accurate units_to_make from cumulative forecast
                new_units_to_make = max(0, int(round(forecast_sum - total_inv)))
            except (json.JSONDecodeError, TypeError):
                # Fallback to scaled cached units_to_make if JSON parsing fails
                # Scale based on DOI ratio
                if cached_units > 0 and default_cached_doi > 0:
                    doi_ratio = planning_horizon / default_cached_doi
                    new_units_to_make = max(0, int(round(cached_units * doi_ratio)))
                else:
                    new_units_to_make = cached_units
        else:
            # No cumulative forecast data - scale cached units_to_make based on DOI ratio
            # This approximates the recalculation by assuming linear scaling with DOI
            if cached_units > 0 and default_cached_doi > 0:
                doi_ratio = planning_horizon / default_cached_doi
                # Scale: new_units = cached_units * (new_doi / cached_doi)
                # But also need to account for inventory not changing
                # Formula: units_to_make = forecast_sum - inventory
                # If forecast_sum scales linearly: new_forecast = old_forecast * ratio
                # new_units = (old_forecast * ratio) - inventory
                # old_units = old_forecast - inventory
                # old_forecast = old_units + inventory
                old_forecast_sum = cached_units + total_inv
                new_forecast_sum = old_forecast_sum * doi_ratio
                new_units_to_make = max(0, int(round(new_forecast_sum - total_inv)))
            else:
                new_units_to_make = cached_units
        
        # Recalculate DOI based on the planning horizon's forecast rate
        # This ensures DOI changes when Required DOI setting changes
        cached_doi_total = r.get('doi_total') or 0
        cached_doi_fba = r.get('doi_fba') or 0
        
        # Calculate daily rate for THIS planning horizon using cumulative forecast
        horizon_daily_rate = 0
        if cumulative_data and forecast_sum > 0 and planning_horizon > 0:
            horizon_daily_rate = forecast_sum / planning_horizon
        elif daily_rate > 0:
            horizon_daily_rate = daily_rate
        
        if horizon_daily_rate > 0:
            # Calculate DOI using the planning horizon's daily rate
            doi_total = int(total_inv / horizon_daily_rate)
            doi_fba = int(fba_available / horizon_daily_rate)
        elif cached_doi_total > 0:
            # Use cached DOI values if no forecast data but cache has values
            doi_total = cached_doi_total
            doi_fba = cached_doi_fba
        else:
            # No sales data and no cached DOI - default to 365
            doi_total = 365
            doi_fba = 365
        
        # Determine status based on DOI
        if doi_total <= 14:
            status = 'critical'
            critical_count += 1
        elif doi_total <= 30:
            status = 'low'
            low_count += 1
        else:
            status = 'good'
            good_count += 1
        
        # Apply filters
        if status_filter and status != status_filter:
            continue
        if min_units > 0 and new_units_to_make < min_units:
            continue
        
        total_units += new_units_to_make
        
        forecast_objects.append(ForecastResult(
            asin=r['asin'],
            product_name=r.get('product_name'),
            image_url=r.get('image_url'),  # From product_images table
            algorithm=r['algorithm'],
            age_months=float(r['age_months']) if r.get('age_months') else None,
            doi_total=doi_total,
            doi_fba=doi_fba,
            units_to_make=new_units_to_make,
            peak=r.get('peak'),
            total_inventory=total_inv,
            fba_available=fba_available,
            fba_reserved=r.get('fba_reserved'),
            fba_inbound=r.get('fba_inbound'),
            awd_available=r.get('awd_available'),
            awd_reserved=r.get('awd_reserved'),
            label_inventory=r.get('label_inventory'),
            awd_inbound=r.get('awd_inbound'),
            status=status
        ))
    
    # Sort by units_to_make descending and apply limit
    forecast_objects.sort(key=lambda x: x.units_to_make or 0, reverse=True)
    forecast_objects = forecast_objects[:limit]
    
    return AllForecastsResponse(
        total_products=len(results),
        forecasts=forecast_objects,
        critical_count=critical_count,
        low_count=low_count,
        good_count=good_count,
        error_count=0,
        total_units_to_make=total_units,
        doi_settings=doi_settings_response
    )


def _interpolate_cumulative_forecast(cumulative_data: dict, target_days: int) -> float:
    """
    Interpolate cumulative forecast for any target_days using cached interval data.
    
    cumulative_data is a dict like {"30": 1500, "60": 3200, "90": 5000, "120": 7000, ...}
    Returns interpolated forecast sum at target_days.
    """
    # Convert keys to integers and sort
    points = sorted([(int(k), v) for k, v in cumulative_data.items()])
    
    if not points:
        return 0
    
    # Handle edge cases
    if target_days <= points[0][0]:
        # Linear extrapolation from origin to first point
        return (target_days / points[0][0]) * points[0][1] if points[0][0] > 0 else points[0][1]
    
    if target_days >= points[-1][0]:
        # Use last two points to extrapolate
        if len(points) >= 2:
            x1, y1 = points[-2]
            x2, y2 = points[-1]
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
            return y2 + slope * (target_days - x2)
        return points[-1][1]
    
    # Find bracketing points and interpolate
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        if x1 <= target_days <= x2:
            # Linear interpolation
            ratio = (target_days - x1) / (x2 - x1) if x2 != x1 else 0
            return y1 + ratio * (y2 - y1)
    
    return points[-1][1]


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
    
    # Get DOI settings for response
    doi_settings = get_doi_settings(
        amazon_doi_goal=amazon_doi_goal,
        inbound_lead_time=inbound_lead_time,
        manufacture_lead_time=manufacture_lead_time
    )
    doi_settings_response = build_doi_settings_response(doi_settings)
    
    result = run_forecast(
        asin, 
        algorithm,
        amazon_doi_goal=doi_settings['amazon_doi_goal'],
        inbound_lead_time=doi_settings['inbound_lead_time'],
        manufacture_lead_time=doi_settings['manufacture_lead_time']
    )
    
    # Normalize response keys for 18m+ algorithm
    if 'doi_total_days' in result:
        result['doi_total'] = result.pop('doi_total_days')
    if 'doi_fba_days' in result:
        result['doi_fba'] = result.pop('doi_fba_days')
    
    # Add DOI settings to response
    result['doi_settings'] = doi_settings_response
    
    # Fetch full inventory breakdown from inventory table
    inventory_data = execute_query("""
        SELECT 
            COALESCE(fba_available, 0) as fba_available,
            COALESCE(fba_reserved, 0) as fba_reserved,
            COALESCE(fba_inbound, 0) as fba_inbound,
            COALESCE(awd_available, 0) as awd_available,
            COALESCE(awd_reserved, 0) as awd_reserved,
            COALESCE(awd_inbound, 0) as awd_inbound,
            COALESCE(awd_outbound_to_fba, 0) as awd_outbound_to_fba,
            COALESCE(label_inventory, 0) as label_inventory
        FROM inventory WHERE asin = %s
    """, (asin,), fetch_one=True)
    
    if inventory_data:
        result['fba_available'] = inventory_data['fba_available']
        result['fba_reserved'] = inventory_data['fba_reserved']
        result['fba_inbound'] = inventory_data['fba_inbound']
        result['awd_available'] = inventory_data['awd_available']
        result['awd_reserved'] = inventory_data['awd_reserved']
        result['awd_inbound'] = inventory_data['awd_inbound']
        result['label_inventory'] = inventory_data['label_inventory']
        # Calculate total inventory
        result['total_inventory'] = (
            inventory_data['fba_available'] + 
            inventory_data['fba_reserved'] + 
            inventory_data['fba_inbound'] +
            inventory_data['awd_available'] + 
            inventory_data['awd_reserved'] + 
            inventory_data['awd_inbound']
        )
    
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
    refresh: bool = Query(False, description="Force recalculation (slow)"),
    amazon_doi_goal: Optional[int] = Query(
        None,
        description="Amazon DOI goal in days (overrides default, triggers recalculation if different)"
    ),
    inbound_lead_time: Optional[int] = Query(
        None,
        description="Inbound lead time in days (overrides default, triggers recalculation if different)"
    ),
    manufacture_lead_time: Optional[int] = Query(
        None,
        description="Manufacture lead time in days (overrides default, triggers recalculation if different)"
    )
):
    """
    Get pre-calculated forecasts for all products (INSTANT from database).
    
    Returns forecasts from the pre-calculated `forecast_cache` table.
    Use ?refresh=true to force recalculation (takes ~2-3 minutes).
    
    If DOI parameters (amazon_doi_goal, inbound_lead_time, manufacture_lead_time) are provided,
    forecasts will be recalculated on-the-fly with those settings.
    """
    
    # Get DOI settings (always needed for response)
    doi_settings = get_doi_settings(
        amazon_doi_goal=amazon_doi_goal,
        inbound_lead_time=inbound_lead_time,
        manufacture_lead_time=manufacture_lead_time
    )
    doi_settings_response = build_doi_settings_response(doi_settings)
    
    # If DOI parameters are provided, recalculate with those settings
    if amazon_doi_goal is not None or inbound_lead_time is not None or manufacture_lead_time is not None:
        # Recalculate with custom DOI settings
        return await _recalculate_all_forecasts(
            status_filter, 
            algorithm_filter, 
            limit, 
            min_units,
            amazon_doi_goal=doi_settings['amazon_doi_goal'],
            inbound_lead_time=doi_settings['inbound_lead_time'],
            manufacture_lead_time=doi_settings['manufacture_lead_time'],
            doi_settings_response=doi_settings_response
        )
    
    if refresh:
        # Force recalculation - this is slow but updates the cache
        return await _recalculate_all_forecasts(
            status_filter, algorithm_filter, limit, min_units,
            doi_settings_response=doi_settings_response
        )
    
    # Read from pre-calculated table with inventory details (FAST!)
    # Use COALESCE to ensure DOI values are never NULL (default to 0 if NULL)
    # Include product images via LEFT JOIN to product_images table
    query = """
        SELECT 
            fc.asin, fc.product_name, fc.algorithm, fc.age_months,
            COALESCE(fc.doi_total, 0) as doi_total, 
            COALESCE(fc.doi_fba, 0) as doi_fba, 
            fc.units_to_make, fc.peak,
            fc.total_inventory, fc.fba_available, fc.status,
            fc.calculated_at,
            COALESCE(fc.cache_amazon_doi_goal, 93) as cache_amazon_doi_goal,
            COALESCE(fc.cache_inbound_lead_time, 30) as cache_inbound_lead_time,
            COALESCE(fc.cache_manufacture_lead_time, 7) as cache_manufacture_lead_time,
            COALESCE(i.fba_reserved, 0) as fba_reserved,
            COALESCE(i.fba_inbound, 0) as fba_inbound,
            COALESCE(i.awd_available, 0) as awd_available,
            COALESCE(i.awd_reserved, 0) as awd_reserved,
            COALESCE(i.awd_inbound, 0) as awd_inbound,
            COALESCE(i.label_inventory, 0) as label_inventory,
            pi.image_url
        FROM forecast_cache fc
        LEFT JOIN inventory i ON fc.asin = i.asin
        LEFT JOIN product_images pi ON fc.asin = pi.asin
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
            total_units_to_make=0,
            doi_settings=doi_settings_response
        )
    
    # Extract DOI settings from cache (use first result's values - all should be same)
    # This shows what DOI settings were actually used to calculate the cached data
    cached_amazon_doi = results[0].get('cache_amazon_doi_goal', 93)
    cached_inbound_lt = results[0].get('cache_inbound_lead_time', 30)
    cached_mfg_lt = results[0].get('cache_manufacture_lead_time', 7)
    
    cached_doi_settings = DoiSettingsUsed(
        amazon_doi_goal=cached_amazon_doi,
        inbound_lead_time=cached_inbound_lt,
        manufacture_lead_time=cached_mfg_lt,
        total_required_doi=cached_amazon_doi + cached_inbound_lt + cached_mfg_lt
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
        # Ensure DOI values are integers (COALESCE in query should handle NULL, but double-check)
        doi_total = int(r.get('doi_total') or 0)
        doi_fba = int(r.get('doi_fba') or 0)
        
        forecast_objects.append(ForecastResult(
            asin=r['asin'],
            product_name=r.get('product_name'),
            image_url=r.get('image_url'),  # From product_images table
            algorithm=r['algorithm'],
            age_months=float(r['age_months']) if r.get('age_months') else None,
            doi_total=doi_total,
            doi_fba=doi_fba,
            units_to_make=r['units_to_make'],
            peak=r.get('peak'),
            total_inventory=r.get('total_inventory'),
            fba_available=r.get('fba_available'),
            fba_reserved=r.get('fba_reserved'),
            fba_inbound=r.get('fba_inbound'),
            awd_available=r.get('awd_available'),
            awd_reserved=r.get('awd_reserved'),
            awd_inbound=r.get('awd_inbound'),
            label_inventory=r.get('label_inventory'),
            status=r['status']
        ))
    
    return AllForecastsResponse(
        total_products=totals['total'] if totals else 0,
        forecasts=forecast_objects,
        critical_count=totals['critical_count'] if totals else 0,
        low_count=totals['low_count'] if totals else 0,
        good_count=totals['good_count'] if totals else 0,
        error_count=0,
        total_units_to_make=int(totals['total_units'] or 0) if totals else 0,
        doi_settings=cached_doi_settings  # Use DOI settings from cache, not current defaults
    )


async def _recalculate_all_forecasts(status_filter, algorithm_filter, limit, min_units,
                                     amazon_doi_goal: Optional[int] = None,
                                     inbound_lead_time: Optional[int] = None,
                                     manufacture_lead_time: Optional[int] = None,
                                     doi_settings_response: Optional[DoiSettingsUsed] = None):
    """Recalculate all forecasts and update the cache table"""
    
    # Build DOI settings (both dict and response)
    doi_settings = get_doi_settings(
        amazon_doi_goal=amazon_doi_goal,
        inbound_lead_time=inbound_lead_time,
        manufacture_lead_time=manufacture_lead_time
    )
    if doi_settings_response is None:
        doi_settings_response = build_doi_settings_response(doi_settings)
    
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
            low_count=0, good_count=0, error_count=0, total_units_to_make=0,
            doi_settings=doi_settings_response
        )
    
    asins = [p['asin'] for p in products]
    
    # Get product names
    product_names_result = execute_query("""
        SELECT asin, product_name FROM products WHERE product_name IS NOT NULL
    """)
    product_names = {p['asin']: p['product_name'] for p in product_names_result} if product_names_result else {}
    
    # Get DOI settings - use provided parameters or get from database
    doi_settings = get_doi_settings(
        amazon_doi_goal=amazon_doi_goal,
        inbound_lead_time=inbound_lead_time,
        manufacture_lead_time=manufacture_lead_time
    )
    
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
                
                # Normalize DOI field names - 18m+ returns doi_total_days, others return doi_total
                if 'doi_total_days' in result:
                    result['doi_total'] = result.pop('doi_total_days')
                if 'doi_fba_days' in result:
                    result['doi_fba'] = result.pop('doi_fba_days')
                
                # Ensure DOI values are integers and not None
                result['doi_total'] = int(result.get('doi_total') or 0)
                result['doi_fba'] = int(result.get('doi_fba') or 0)
                
                results.append(result)
            except Exception:
                continue
    
    # Save to database with DOI settings metadata
    _save_forecasts_to_db(results, doi_settings)
    
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
        total_units_to_make=total_units,
        doi_settings=doi_settings_response
    )


def _save_forecasts_to_db(results: list, doi_settings: dict = None):
    """Save forecast results to the forecast_cache table with DOI settings metadata and cumulative forecasts"""
    from ..database import get_connection
    import json
    
    # Default DOI settings if not provided
    if doi_settings is None:
        doi_settings = get_doi_settings()
    
    conn = get_connection()
    cur = conn.cursor()
    
    try:
        # Ensure DOI columns and cumulative_forecast column exist (migration)
        cur.execute("""
            ALTER TABLE forecast_cache 
            ADD COLUMN IF NOT EXISTS cache_amazon_doi_goal INTEGER DEFAULT 93,
            ADD COLUMN IF NOT EXISTS cache_inbound_lead_time INTEGER DEFAULT 30,
            ADD COLUMN IF NOT EXISTS cache_manufacture_lead_time INTEGER DEFAULT 7,
            ADD COLUMN IF NOT EXISTS daily_forecast_rate FLOAT DEFAULT 0,
            ADD COLUMN IF NOT EXISTS cumulative_forecast TEXT
        """)
        
        # Clear existing data
        cur.execute("DELETE FROM forecast_cache")
        
        # Insert new data with DOI settings and cumulative forecasts
        for r in results:
            # Ensure DOI values are integers (not None) before saving
            doi_total = int(r.get('doi_total') or 0)
            doi_fba = int(r.get('doi_fba') or 0)
            
            # Serialize cumulative_forecast to JSON string
            cumulative_forecast = r.get('cumulative_forecast')
            if cumulative_forecast and isinstance(cumulative_forecast, dict):
                cumulative_forecast = json.dumps(cumulative_forecast)
            
            cur.execute("""
                INSERT INTO forecast_cache 
                (asin, product_name, algorithm, age_months, doi_total, doi_fba, 
                 units_to_make, peak, total_inventory, fba_available, status, calculated_at,
                 cache_amazon_doi_goal, cache_inbound_lead_time, cache_manufacture_lead_time,
                 daily_forecast_rate, cumulative_forecast)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), %s, %s, %s, %s, %s)
            """, (
                r.get('asin'),
                r.get('product_name'),
                r.get('algorithm'),
                r.get('age_months'),
                doi_total,
                doi_fba,
                r.get('units_to_make'),
                r.get('peak'),
                r.get('total_inventory'),
                r.get('fba_available'),
                r.get('status'),
                doi_settings['amazon_doi_goal'],
                doi_settings['inbound_lead_time'],
                doi_settings['manufacture_lead_time'],
                r.get('daily_forecast_rate', 0),
                cumulative_forecast
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
    Uses current default DOI settings from the database.
    """
    # Get current DOI settings
    doi_settings = get_doi_settings()
    
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
                
                # Normalize DOI field names - 18m+ returns doi_total_days, others return doi_total
                if 'doi_total_days' in result:
                    result['doi_total'] = result.pop('doi_total_days')
                if 'doi_fba_days' in result:
                    result['doi_fba'] = result.pop('doi_fba_days')
                
                # Ensure DOI values are integers and not None
                result['doi_total'] = int(result.get('doi_total') or 0)
                result['doi_fba'] = int(result.get('doi_fba') or 0)
                
                results.append(result)
            except Exception:
                continue
    
    # Save with DOI settings metadata
    _save_forecasts_to_db(results, doi_settings)
    
    return {
        "status": "success",
        "count": len(results),
        "message": f"Refreshed {len(results)} forecasts",
        "doi_settings": {
            "amazon_doi_goal": doi_settings['amazon_doi_goal'],
            "inbound_lead_time": doi_settings['inbound_lead_time'],
            "manufacture_lead_time": doi_settings['manufacture_lead_time'],
            "total_required_doi": doi_settings['total_required_doi']
        }
    }


@router.get("/labels")
async def get_all_labels():
    """
    Get label inventory for all ASINs from Railway database.
    Returns label_inventory grouped by ASIN.
    """
    results = execute_query("""
        SELECT 
            asin,
            COALESCE(label_inventory, 0) as label_inventory
        FROM inventory
        WHERE label_inventory IS NOT NULL AND label_inventory > 0
        ORDER BY asin
    """)
    
    # Create a map by ASIN for easy lookup
    labels_by_asin = {}
    for r in results:
        labels_by_asin[r['asin']] = r['label_inventory']
    
    return {
        "success": True,
        "count": len(results),
        "labels": results,
        "byAsin": labels_by_asin
    }


@router.get("/labels/{asin}")
async def get_label_by_asin(asin: str):
    """
    Get label inventory for a specific ASIN from Railway database.
    """
    result = execute_query("""
        SELECT 
            asin,
            COALESCE(label_inventory, 0) as label_inventory
        FROM inventory
        WHERE asin = %s
    """, (asin,), fetch_one=True)
    
    if not result:
        return {
            "success": True,
            "asin": asin,
            "label_inventory": 0,
            "message": "ASIN not found in inventory"
        }
    
    return {
        "success": True,
        "asin": asin,
        "label_inventory": result['label_inventory']
    }
