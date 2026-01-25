# Forecast API

Production-ready FastAPI application for inventory forecast calculations.

## Features

- **REST API** with automatic OpenAPI documentation
- **Three algorithms** (0-6m, 6-18m, 18m+) - auto-selected based on product age
- **Pre-aggregated endpoint** for all products with filtering
- **Health check** endpoint for monitoring
- **Parallel processing** for bulk forecasts

## Quick Start

### 1. Install dependencies

```bash
cd forecast_api
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` and set your database URL:

```bash
cp .env.example .env
# Edit .env with your PostgreSQL connection string
```

### 3. Run the server

```bash
# Development
uvicorn app.main:app --reload --port 8000

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4. Access the API

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## API Endpoints

### Health Check
```
GET /health
```
Returns service status and database connection status.

### Single ASIN Forecast
```
GET /forecast/{asin}
GET /forecast/{asin}?algorithm=6-18m
```
Returns forecast for a specific ASIN. Algorithm is auto-detected based on product age, or can be forced with the `algorithm` query parameter.

**Response:**
```json
{
  "asin": "B0F2MZQL9K",
  "algorithm": "0-6m",
  "age_months": 5.8,
  "total_inventory": 1618,
  "fba_available": 1618,
  "units_to_make": 2857,
  "doi_total": 59,
  "doi_fba": 59,
  "peak": 184,
  "status": "good"
}
```

### All Products Forecast
```
GET /forecast/
GET /forecast/?status_filter=critical
GET /forecast/?algorithm_filter=18m+
GET /forecast/?min_units=100&limit=50
```
Returns pre-aggregated forecasts for all products with summary statistics.

**Query Parameters:**
- `status_filter`: Filter by status (critical, low, good)
- `algorithm_filter`: Filter by algorithm (0-6m, 6-18m, 18m+)
- `min_units`: Minimum units to make (default: 0)
- `limit`: Maximum results (default: 1000, max: 10000)

**Response:**
```json
{
  "total_products": 1112,
  "forecasts": [...],
  "critical_count": 5,
  "low_count": 23,
  "good_count": 1084,
  "error_count": 0,
  "total_units_to_make": 125430
}
```

## Algorithm Details

### 0-6 Month (LOCKED)
- Peak-based forecasting
- Seasonality elasticity: 0.65
- Planning horizon: 130 days

### 6-18 Month (LOCKED)
- CVR-based forecasting
- Seasonality weight: 25%
- Volume boost for high-volume products

### 18m+ (LOCKED)
- Prior year comparison with smoothing
- Velocity adjustment: 15%
- Market adjustment: 5%

## Project Structure

```
forecast_api/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   ├── config.py         # Settings
│   ├── database.py       # PostgreSQL connection
│   ├── models.py         # Pydantic models
│   ├── routers/
│   │   ├── forecast.py   # Forecast endpoints
│   │   └── health.py     # Health check
│   └── algorithms/       # LOCKED - DO NOT MODIFY
│       ├── forecast_0_6m.py
│       ├── forecast_6_18m.py
│       └── forecast_18m_plus.py
├── requirements.txt
├── .env.example
└── README.md
```

## Deployment

### Docker

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Railway/Render

Set the following environment variables:
- `DATABASE_URL` or `DATABASE_PUBLIC_URL`: PostgreSQL connection string

Start command:
```
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```
