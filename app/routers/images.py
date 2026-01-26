"""
Product Images Router
Endpoints for fetching product images from Shopify
"""
from fastapi import APIRouter, Query, HTTPException
from typing import Optional

from ..services.shopify import (
    get_product_image,
    fetch_all_products_with_images,
    build_image_cache,
    fetch_product_image_by_title
)

router = APIRouter(prefix="/images", tags=["images"])


@router.get("/product")
async def get_image(
    asin: Optional[str] = Query(None, description="Product ASIN"),
    product_name: Optional[str] = Query(None, description="Product name to search"),
    size: Optional[str] = Query(None, description="Size variant (e.g., 8oz, 16oz)")
):
    """
    Get product image URL by ASIN or product name.
    
    - **asin**: Amazon ASIN (optional)
    - **product_name**: Product name to search in Shopify (optional)
    - **size**: Size variant to filter by (optional)
    
    At least one of asin or product_name must be provided.
    """
    if not asin and not product_name:
        raise HTTPException(
            status_code=400,
            detail="Either 'asin' or 'product_name' must be provided"
        )
    
    result = await get_product_image(asin=asin, product_name=product_name, size=size)
    return result


@router.get("/search")
async def search_product_image(
    query: str = Query(..., description="Search query (product name)"),
    size: Optional[str] = Query(None, description="Size variant (e.g., 8oz, 16oz)")
):
    """
    Search for product images by name.
    
    - **query**: Product name to search
    - **size**: Optional size to filter results
    """
    result = await fetch_product_image_by_title(query, size)
    return result


@router.get("/all")
async def get_all_product_images(
    limit: int = Query(50, ge=1, le=250, description="Number of products to fetch")
):
    """
    Get all products with their images from Shopify.
    Useful for building a local cache or displaying a product catalog.
    
    - **limit**: Maximum number of products to return (1-250)
    """
    result = await fetch_all_products_with_images(first=limit)
    return result


@router.post("/cache/build")
async def rebuild_image_cache():
    """
    Rebuild the in-memory image cache from Shopify.
    This fetches all products and caches their images for faster lookups.
    """
    result = await build_image_cache()
    return result
