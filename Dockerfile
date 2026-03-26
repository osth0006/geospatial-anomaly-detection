FROM python:3.11-slim

# System dependencies for geopandas, GDAL, and spatial libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    gdal-bin \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY pyproject.toml .
COPY run_pipeline.py .
COPY src/ src/
COPY notebooks/ notebooks/
COPY tests/ tests/

# Install package in editable mode
RUN pip install --no-cache-dir -e .

# Create directories for data and output volumes
RUN mkdir -p data output

CMD ["python", "run_pipeline.py"]
