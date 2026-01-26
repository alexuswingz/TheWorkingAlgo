"""
Create doi_settings table for storing DOI configuration
"""
import psycopg2
from app.config import get_settings

def create_doi_settings_table():
    """Create the doi_settings table"""
    settings = get_settings()
    conn = psycopg2.connect(settings.database_url)
    
    try:
        with conn.cursor() as cur:
            # Create table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS doi_settings (
                    id SERIAL PRIMARY KEY,
                    amazon_doi_goal INTEGER NOT NULL DEFAULT 130,
                    inbound_lead_time INTEGER NOT NULL DEFAULT 30,
                    manufacture_lead_time INTEGER NOT NULL DEFAULT 7,
                    is_default BOOLEAN NOT NULL DEFAULT false,
                    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
                )
            """)
            
            # Create partial unique index to ensure only one default setting
            cur.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_doi_settings_unique_default 
                ON doi_settings (is_default) WHERE is_default = true
            """)
            
            # Create index for faster lookups
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_doi_settings_default 
                ON doi_settings (is_default) WHERE is_default = true
            """)
            
            conn.commit()
            print("doi_settings table created successfully")
    except Exception as e:
        conn.rollback()
        print(f"Error creating table: {e}")
        raise
    finally:
        conn.close()


def add_doi_columns_to_forecast_cache():
    """Add DOI settings columns to forecast_cache table"""
    settings = get_settings()
    conn = psycopg2.connect(settings.database_url)
    
    try:
        with conn.cursor() as cur:
            # Add DOI settings columns to forecast_cache if they don't exist
            cur.execute("""
                ALTER TABLE forecast_cache 
                ADD COLUMN IF NOT EXISTS cache_amazon_doi_goal INTEGER DEFAULT 130,
                ADD COLUMN IF NOT EXISTS cache_inbound_lead_time INTEGER DEFAULT 30,
                ADD COLUMN IF NOT EXISTS cache_manufacture_lead_time INTEGER DEFAULT 7
            """)
            
            conn.commit()
            print("DOI columns added to forecast_cache successfully")
    except Exception as e:
        conn.rollback()
        print(f"Error adding columns: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    create_doi_settings_table()
    add_doi_columns_to_forecast_cache()
