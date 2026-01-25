"""
Database connection module for PostgreSQL
"""
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
from .config import get_settings


def get_connection():
    """Get a PostgreSQL connection"""
    settings = get_settings()
    return psycopg2.connect(settings.database_url)


@contextmanager
def get_db_cursor(commit: bool = False):
    """Context manager for database cursor"""
    conn = get_connection()
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        yield cursor
        if commit:
            conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()


def execute_query(
    query: str, 
    params: tuple = None, 
    fetch_one: bool = False
) -> Optional[List[Dict[str, Any]] | Dict[str, Any]]:
    """Execute a query and return results as list of dicts"""
    with get_db_cursor() as cursor:
        cursor.execute(query, params)
        if fetch_one:
            result = cursor.fetchone()
            return dict(result) if result else None
        results = cursor.fetchall()
        return [dict(row) for row in results]


def test_connection() -> bool:
    """Test the database connection"""
    try:
        with get_db_cursor() as cursor:
            cursor.execute("SELECT 1")
            return True
    except Exception:
        return False
