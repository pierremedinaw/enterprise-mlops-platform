FROM python:3.11-slim

# Dependencia del runtime numérico de Scikit-learn.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalación reproducible por intervalos de compatibilidad.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN python -m pip install --no-cache-dir .

# Nunca se incluye un modelo de terceros en la imagen.
RUN mkdir -p /app/artifacts /app/mlruns
EXPOSE 8000
CMD ["uvicorn", "mlops_platform.api:app", "--host", "0.0.0.0", "--port", "8000"]
