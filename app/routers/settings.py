"""
Settings API endpoints for DOI configuration
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from ..database import execute_query, get_connection

router = APIRouter(prefix="/settings", tags=["Settings"])


class DoiSettings(BaseModel):
    amazon_doi_goal: int
    inbound_lead_time: int
    manufacture_lead_time: int
    save_as_default: Optional[bool] = False


class DoiSettingsResponse(BaseModel):
    amazon_doi_goal: int
    inbound_lead_time: int
    manufacture_lead_time: int
    total_required_doi: int


@router.get("/doi", response_model=DoiSettingsResponse)
async def get_doi_settings():
    """
    Get current DOI settings (defaults if not set).
    
    Returns:
    - amazon_doi_goal: Amazon DOI goal in days
    - inbound_lead_time: Inbound lead time in days
    - manufacture_lead_time: Manufacture lead time in days  
    - total_required_doi: Sum of all three (planning horizon)
    """
    # Try to get from database first
    settings = execute_query(
        "SELECT amazon_doi_goal, inbound_lead_time, manufacture_lead_time FROM doi_settings WHERE is_default = true ORDER BY updated_at DESC LIMIT 1",
        fetch_one=True
    )
    
    if settings:
        amazon_doi = settings.get('amazon_doi_goal', 93)
        inbound_lt = settings.get('inbound_lead_time', 30)
        mfg_lt = settings.get('manufacture_lead_time', 7)
        return {
            "amazon_doi_goal": amazon_doi,
            "inbound_lead_time": inbound_lt,
            "manufacture_lead_time": mfg_lt,
            "total_required_doi": amazon_doi + inbound_lt + mfg_lt
        }
    
    # Return defaults if no settings found
    return {
        "amazon_doi_goal": 93,
        "inbound_lead_time": 30,
        "manufacture_lead_time": 7,
        "total_required_doi": 130  # 93 + 30 + 7
    }


class DoiSettingsSaveResponse(BaseModel):
    success: bool
    message: str
    settings: DoiSettingsResponse


@router.post("/doi", response_model=DoiSettingsSaveResponse)
async def save_doi_settings(settings: DoiSettings):
    """
    Save DOI settings (optionally as default).
    
    Set save_as_default=true to persist as the default settings for all future forecasts.
    Otherwise, settings are saved as session-specific (for tracking/audit purposes).
    
    Returns the saved settings including calculated total_required_doi.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            if settings.save_as_default:
                # Check if default exists
                cur.execute("SELECT id FROM doi_settings WHERE is_default = true LIMIT 1")
                existing = cur.fetchone()
                
                if existing:
                    # Update existing default
                    cur.execute("""
                        UPDATE doi_settings 
                        SET amazon_doi_goal = %s, 
                            inbound_lead_time = %s, 
                            manufacture_lead_time = %s,
                            updated_at = NOW()
                        WHERE is_default = true
                    """, (settings.amazon_doi_goal, settings.inbound_lead_time, settings.manufacture_lead_time))
                else:
                    # Insert new default
                    cur.execute("""
                        INSERT INTO doi_settings (amazon_doi_goal, inbound_lead_time, manufacture_lead_time, is_default, updated_at)
                        VALUES (%s, %s, %s, true, NOW())
                    """, (settings.amazon_doi_goal, settings.inbound_lead_time, settings.manufacture_lead_time))
            else:
                # Insert as session-specific settings (not default)
                cur.execute("""
                    INSERT INTO doi_settings (amazon_doi_goal, inbound_lead_time, manufacture_lead_time, is_default, updated_at)
                    VALUES (%s, %s, %s, false, NOW())
                """, (settings.amazon_doi_goal, settings.inbound_lead_time, settings.manufacture_lead_time))
            
            conn.commit()
        
        total_required = settings.amazon_doi_goal + settings.inbound_lead_time + settings.manufacture_lead_time
        
        return {
            "success": True, 
            "message": "Settings saved as default" if settings.save_as_default else "Settings saved for session",
            "settings": {
                "amazon_doi_goal": settings.amazon_doi_goal,
                "inbound_lead_time": settings.inbound_lead_time,
                "manufacture_lead_time": settings.manufacture_lead_time,
                "total_required_doi": total_required
            }
        }
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save settings: {str(e)}")
    finally:
        conn.close()
