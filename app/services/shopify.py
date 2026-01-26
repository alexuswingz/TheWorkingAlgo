"""
Shopify Storefront API Service
Fetches product images from Shopify using GraphQL
"""
import httpx
from typing import Optional
import os
from functools import lru_cache


# Shopify Configuration - Set these in Railway environment variables
SHOPIFY_STORE_DOMAIN = os.getenv("SHOPIFY_STORE_DOMAIN", "n3mpgz-ny.myshopify.com")
SHOPIFY_STOREFRONT_TOKEN = os.getenv("SHOPIFY_STOREFRONT_TOKEN", "")
SHOPIFY_API_VERSION = os.getenv("SHOPIFY_API_VERSION", "2025-01")

GRAPHQL_ENDPOINT = f"https://{SHOPIFY_STORE_DOMAIN}/api/{SHOPIFY_API_VERSION}/graphql.json"


def _check_token():
    """Verify Shopify token is configured"""
    if not SHOPIFY_STOREFRONT_TOKEN:
        return {
            "success": False,
            "error": "SHOPIFY_STOREFRONT_TOKEN environment variable is not set",
            "image_url": None
        }
    return None


async def fetch_product_image_by_title(product_title: str, size: Optional[str] = None) -> dict:
    """
    Fetch product image from Shopify by product title (and optionally size).
    
    Args:
        product_title: The product name/title to search for
        size: Optional size variant (e.g., "8oz", "16oz", "32oz")
    
    Returns:
        dict with image_url, product_title, and other metadata
    """
    # Check if token is configured
    token_error = _check_token()
    if token_error:
        return token_error
    
    # Build search query - search by title
    search_query = f'title:*{product_title}*'
    if size:
        search_query = f'title:*{product_title}* AND title:*{size}*'
    
    graphql_query = """
    query searchProducts($query: String!, $first: Int!) {
        products(first: $first, query: $query) {
            nodes {
                id
                title
                handle
                featuredImage {
                    url
                    altText
                    width
                    height
                }
                images(first: 5) {
                    nodes {
                        url
                        altText
                        width
                        height
                    }
                }
                variants(first: 10) {
                    nodes {
                        id
                        title
                        sku
                        image {
                            url
                            altText
                        }
                    }
                }
            }
        }
    }
    """
    
    variables = {
        "query": search_query,
        "first": 5
    }
    
    headers = {
        "Content-Type": "application/json",
        "X-Shopify-Storefront-Access-Token": SHOPIFY_STOREFRONT_TOKEN
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            GRAPHQL_ENDPOINT,
            json={"query": graphql_query, "variables": variables},
            headers=headers
        )
        
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"Shopify API error: {response.status_code}",
                "image_url": None
            }
        
        data = response.json()
        
        if "errors" in data:
            return {
                "success": False,
                "error": str(data["errors"]),
                "image_url": None
            }
        
        products = data.get("data", {}).get("products", {}).get("nodes", [])
        
        if not products:
            return {
                "success": False,
                "error": "No products found matching the search",
                "image_url": None,
                "search_query": search_query
            }
        
        # Get the first matching product
        product = products[0]
        
        # Get the featured image or first available image
        image_url = None
        if product.get("featuredImage"):
            image_url = product["featuredImage"]["url"]
        elif product.get("images", {}).get("nodes"):
            image_url = product["images"]["nodes"][0]["url"]
        
        return {
            "success": True,
            "image_url": image_url,
            "product_title": product.get("title"),
            "product_handle": product.get("handle"),
            "all_images": [img["url"] for img in product.get("images", {}).get("nodes", [])],
            "variants": [
                {
                    "title": v.get("title"),
                    "sku": v.get("sku"),
                    "image_url": v.get("image", {}).get("url") if v.get("image") else None
                }
                for v in product.get("variants", {}).get("nodes", [])
            ]
        }


async def fetch_all_products_with_images(first: int = 50) -> dict:
    """
    Fetch all products with their images from Shopify.
    Used to build a cache/mapping of product images.
    
    Args:
        first: Number of products to fetch (max 250)
    
    Returns:
        dict with products list containing title, handle, and images
    """
    # Check if token is configured
    token_error = _check_token()
    if token_error:
        token_error["products"] = []
        return token_error
    
    graphql_query = """
    query getAllProducts($first: Int!) {
        products(first: $first) {
            nodes {
                id
                title
                handle
                featuredImage {
                    url
                    altText
                }
                images(first: 3) {
                    nodes {
                        url
                        altText
                    }
                }
                variants(first: 20) {
                    nodes {
                        title
                        sku
                        image {
                            url
                        }
                    }
                }
            }
            pageInfo {
                hasNextPage
                endCursor
            }
        }
    }
    """
    
    variables = {"first": min(first, 250)}
    
    headers = {
        "Content-Type": "application/json",
        "X-Shopify-Storefront-Access-Token": SHOPIFY_STOREFRONT_TOKEN
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            GRAPHQL_ENDPOINT,
            json={"query": graphql_query, "variables": variables},
            headers=headers
        )
        
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"Shopify API error: {response.status_code}",
                "products": []
            }
        
        data = response.json()
        
        if "errors" in data:
            return {
                "success": False,
                "error": str(data["errors"]),
                "products": []
            }
        
        products_data = data.get("data", {}).get("products", {})
        products = products_data.get("nodes", [])
        
        result_products = []
        for product in products:
            image_url = None
            if product.get("featuredImage"):
                image_url = product["featuredImage"]["url"]
            elif product.get("images", {}).get("nodes"):
                image_url = product["images"]["nodes"][0]["url"]
            
            # Build SKU to image mapping from variants
            sku_images = {}
            for variant in product.get("variants", {}).get("nodes", []):
                sku = variant.get("sku")
                if sku:
                    variant_image = variant.get("image", {}).get("url") if variant.get("image") else None
                    sku_images[sku] = variant_image or image_url
            
            result_products.append({
                "title": product.get("title"),
                "handle": product.get("handle"),
                "featured_image": image_url,
                "all_images": [img["url"] for img in product.get("images", {}).get("nodes", [])],
                "sku_images": sku_images
            })
        
        return {
            "success": True,
            "products": result_products,
            "total": len(result_products),
            "has_next_page": products_data.get("pageInfo", {}).get("hasNextPage", False)
        }


# In-memory cache for product images (ASIN -> image URL mapping)
_image_cache: dict = {}


async def get_product_image(asin: str = None, product_name: str = None, size: str = None) -> dict:
    """
    Get product image by ASIN or product name.
    First checks cache, then queries Shopify.
    
    Args:
        asin: Amazon ASIN (used as SKU match or cache key)
        product_name: Product name to search
        size: Optional size variant
    
    Returns:
        dict with image_url and metadata
    """
    # Check cache first
    cache_key = asin or f"{product_name}_{size}"
    if cache_key in _image_cache:
        return {
            "success": True,
            "image_url": _image_cache[cache_key],
            "source": "cache"
        }
    
    # If we have product_name, search by title
    if product_name:
        result = await fetch_product_image_by_title(product_name, size)
        if result.get("success") and result.get("image_url"):
            _image_cache[cache_key] = result["image_url"]
        return result
    
    # If only ASIN, we need to search all products for SKU match
    # or return a default/placeholder
    return {
        "success": False,
        "error": "Please provide product_name to search for images",
        "image_url": None
    }


async def build_image_cache() -> dict:
    """
    Build a cache of all product images from Shopify.
    Maps product titles and SKUs to image URLs.
    Also saves to database for persistence across deployments.
    """
    global _image_cache
    
    result = await fetch_all_products_with_images(first=250)
    
    if not result.get("success"):
        return result
    
    # Build cache from products
    for product in result.get("products", []):
        title = product.get("title", "").lower()
        image_url = product.get("featured_image")
        
        if title and image_url:
            _image_cache[title] = image_url
        
        # Also cache by SKU
        for sku, sku_image in product.get("sku_images", {}).items():
            if sku and sku_image:
                _image_cache[sku] = sku_image
    
    # Also save to database for persistence
    db_result = await save_images_to_database(result.get("products", []))
    
    return {
        "success": True,
        "cached_items": len(_image_cache),
        "database_items": db_result.get("count", 0),
        "message": "Image cache built and saved to database"
    }


async def save_images_to_database(products: list) -> dict:
    """
    Save product images to database for persistence.
    Maps product names and SKUs to ASINs in our products table.
    """
    from ..database import execute_query, get_connection
    
    # First ensure the product_images table exists
    conn = get_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS product_images (
                id SERIAL PRIMARY KEY,
                asin VARCHAR(20) UNIQUE,
                product_name TEXT,
                image_url TEXT,
                shopify_handle TEXT,
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        conn.commit()
    except Exception as e:
        conn.rollback()
        return {"success": False, "error": str(e)}
    finally:
        cur.close()
        conn.close()
    
    # Get all products from our database to match
    db_products = execute_query("SELECT asin, product_name FROM products WHERE product_name IS NOT NULL")
    
    if not db_products:
        return {"success": False, "error": "No products in database"}
    
    # Build a mapping of product name keywords to ASIN
    matched = 0
    conn = get_connection()
    cur = conn.cursor()
    
    # Common stop words to ignore in matching
    stop_words = {'for', 'and', 'the', 'a', 'an', 'of', 'to', 'in', 'with', 'plant', 'plants', 
                  'liquid', 'food', 'complete', 'grow', 'growing', 'healthy', 'premium', 'organic'}
    
    try:
        for shopify_product in products:
            shopify_title = shopify_product.get("title", "").lower()
            image_url = shopify_product.get("featured_image")
            handle = shopify_product.get("handle", "").lower().replace("-", " ")
            
            if not image_url:
                continue
            
            # Try to match with our products by name similarity
            for db_product in db_products:
                db_name = (db_product.get("product_name") or "").lower()
                asin = db_product.get("asin")
                
                if not db_name or not asin:
                    continue
                
                # Extract meaningful words (exclude stop words and short words)
                shopify_words = set(w for w in shopify_title.replace("-", " ").replace(",", " ").split() 
                                   if len(w) > 2 and w not in stop_words)
                db_words = set(w for w in db_name.replace("-", " ").replace(",", " ").split() 
                              if len(w) > 2 and w not in stop_words)
                handle_words = set(w for w in handle.split() if len(w) > 2 and w not in stop_words)
                
                # Match conditions (more flexible):
                common_words = shopify_words & db_words
                handle_match = len(handle_words & db_words) >= 2
                
                # Match if:
                # 1. At least 2 meaningful words match
                # 2. OR exact title/name containment
                # 3. OR handle matches well
                # 4. OR key product identifier matches (e.g., specific plant name + "fertilizer")
                is_match = (
                    len(common_words) >= 2 or 
                    shopify_title in db_name or 
                    db_name in shopify_title or
                    handle_match or
                    (len(common_words) >= 1 and ('fertilizer' in shopify_words or 'fertilizer' in db_words))
                )
                
                if is_match:
                    cur.execute("""
                        INSERT INTO product_images (asin, product_name, image_url, shopify_handle, updated_at)
                        VALUES (%s, %s, %s, %s, NOW())
                        ON CONFLICT (asin) DO UPDATE SET 
                            image_url = EXCLUDED.image_url,
                            shopify_handle = EXCLUDED.shopify_handle,
                            updated_at = NOW()
                    """, (asin, db_product.get("product_name"), image_url, shopify_product.get("handle")))
                    matched += 1
                    break
            
            # Also try matching by SKU in variants
            for sku, sku_image in shopify_product.get("sku_images", {}).items():
                if sku and sku_image:
                    # Check if SKU matches any ASIN
                    for db_product in db_products:
                        asin = db_product.get("asin")
                        if asin and (sku == asin or sku.upper() == asin.upper()):
                            cur.execute("""
                                INSERT INTO product_images (asin, product_name, image_url, shopify_handle, updated_at)
                                VALUES (%s, %s, %s, %s, NOW())
                                ON CONFLICT (asin) DO UPDATE SET 
                                    image_url = EXCLUDED.image_url,
                                    shopify_handle = EXCLUDED.shopify_handle,
                                    updated_at = NOW()
                            """, (asin, db_product.get("product_name"), sku_image, handle))
                            matched += 1
                            break
        
        conn.commit()
        return {"success": True, "count": matched}
    except Exception as e:
        conn.rollback()
        return {"success": False, "error": str(e)}
    finally:
        cur.close()
        conn.close()


def get_image_for_asin(asin: str) -> str:
    """
    Get cached image URL for an ASIN from database.
    Returns None if not found.
    """
    from ..database import execute_query
    
    result = execute_query(
        "SELECT image_url FROM product_images WHERE asin = %s",
        (asin,),
        fetch_one=True
    )
    
    return result.get("image_url") if result else None
