"""
Forecast API endpoints
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from datetime import date, timedelta
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
    min_units: int = Query(0, ge=0, description="Minimum units to make")
):
    """
    Get pre-aggregated forecasts for all products.
    
    Returns forecasts for all products with summary statistics.
    Results are sorted by units_to_make (descending).
    """
    # Get all products
    products = execute_query("SELECT asin FROM products ORDER BY asin")
    
    if not products:
        return AllForecastsResponse(
            total_products=0,
            forecasts=[],
            critical_count=0,
            low_count=0,
            good_count=0,
            error_count=0,
            total_units_to_make=0
        )
    
    asins = [p['asin'] for p in products]
    
    # Run forecasts in parallel
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_asin = {executor.submit(run_forecast, asin): asin for asin in asins}
        
        for future in as_completed(future_to_asin):
            try:
                result = future.result()
                
                # Normalize response keys
                if 'doi_total_days' in result:
                    result['doi_total'] = result.pop('doi_total_days')
                if 'doi_fba_days' in result:
                    result['doi_fba'] = result.pop('doi_fba_days')
                
                results.append(result)
            except Exception as e:
                asin = future_to_asin[future]
                results.append({
                    'asin': asin,
                    'algorithm': 'unknown',
                    'error': str(e),
                    'status': 'good'
                })
    
    # Apply filters
    filtered_results = results
    
    if status_filter:
        filtered_results = [r for r in filtered_results if r.get('status') == status_filter]
    
    if algorithm_filter:
        filtered_results = [r for r in filtered_results if r.get('algorithm') == algorithm_filter]
    
    if min_units > 0:
        filtered_results = [r for r in filtered_results if (r.get('units_to_make') or 0) >= min_units]
    
    # Sort by units_to_make descending
    filtered_results.sort(key=lambda x: x.get('units_to_make') or 0, reverse=True)
    
    # Apply limit
    filtered_results = filtered_results[:limit]
    
    # Calculate summary statistics
    critical_count = sum(1 for r in results if r.get('status') == 'critical')
    low_count = sum(1 for r in results if r.get('status') == 'low')
    good_count = sum(1 for r in results if r.get('status') == 'good')
    error_count = sum(1 for r in results if r.get('error'))
    total_units = sum(r.get('units_to_make') or 0 for r in results if not r.get('error'))
    
    # Convert to ForecastResult objects
    forecast_objects = [ForecastResult(**r) for r in filtered_results]
    
    return AllForecastsResponse(
        total_products=len(products),
        forecasts=forecast_objects,
        critical_count=critical_count,
        low_count=low_count,
        good_count=good_count,
        error_count=error_count,
        total_units_to_make=total_units
    )
