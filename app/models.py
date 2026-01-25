"""
Pydantic models for API request/response schemas
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import date


class ForecastResult(BaseModel):
    """Individual ASIN forecast result"""
    asin: str = Field(..., description="Product ASIN")
    product_name: Optional[str] = Field(None, description="Product name/title")
    algorithm: str = Field(..., description="Algorithm used (0-6m, 6-18m, 18m+)")
    age_months: Optional[float] = Field(None, description="Product age in months")
    
    # Inventory metrics
    total_inventory: Optional[int] = Field(None, description="Total inventory across all locations")
    fba_available: Optional[int] = Field(None, description="FBA available inventory")
    fba_reserved: Optional[int] = Field(None, description="FBA reserved inventory")
    fba_inbound: Optional[int] = Field(None, description="FBA inbound inventory")
    awd_available: Optional[int] = Field(None, description="AWD available inventory")
    awd_reserved: Optional[int] = Field(None, description="AWD reserved inventory")
    awd_inbound: Optional[int] = Field(None, description="AWD inbound inventory")
    
    # Forecast metrics
    units_to_make: Optional[int] = Field(None, description="Units needed to manufacture")
    doi_total: Optional[int] = Field(None, alias="doi_total_days", description="Days of inventory (total)")
    doi_fba: Optional[int] = Field(None, alias="doi_fba_days", description="Days of inventory (FBA)")
    
    # Algorithm-specific metrics
    peak: Optional[int] = Field(None, description="Peak sales (0-6m, 6-18m)")
    idx_now: Optional[float] = Field(None, description="Current seasonality index (0-6m)")
    avg_peak_cvr: Optional[float] = Field(None, description="Average peak CVR (6-18m)")
    sales_velocity_adj: Optional[float] = Field(None, description="Sales velocity adjustment (18m+)")
    
    # Status
    status: Literal["critical", "low", "good"] = Field("good", description="Inventory status")
    error: Optional[str] = Field(None, description="Error message if forecast failed")
    
    class Config:
        populate_by_name = True


class ForecastRequest(BaseModel):
    """Request body for forecast endpoint"""
    asin: str = Field(..., description="Product ASIN to forecast")
    total_inventory: Optional[int] = Field(None, description="Override total inventory")
    fba_inventory: Optional[int] = Field(None, description="Override FBA inventory")


class AllForecastsResponse(BaseModel):
    """Response for /all endpoint with pre-aggregated forecasts"""
    total_products: int = Field(..., description="Total number of products")
    forecasts: list[ForecastResult] = Field(..., description="All forecast results")
    
    # Summary statistics
    critical_count: int = Field(0, description="Products with critical inventory")
    low_count: int = Field(0, description="Products with low inventory")
    good_count: int = Field(0, description="Products with good inventory")
    error_count: int = Field(0, description="Products with forecast errors")
    
    # Totals
    total_units_to_make: int = Field(0, description="Sum of all units to make")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    database: bool = Field(..., description="Database connection status")
    version: str = Field(..., description="API version")
