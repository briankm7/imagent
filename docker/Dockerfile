# syntax=docker/dockerfile:1
#
# Multi-stage: la etapa de construccion trae compiladores y cabeceras que la
# imagen final no necesita. Lo que se despliega solo tiene el interprete, las
# dependencias instaladas y el codigo.

# --------------------------------------------------------------------------
# 1. Construccion
# --------------------------------------------------------------------------
FROM python:3.12-slim AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Se instala en un venv propio y no en el sistema para poder copiarlo entero a
# la etapa final con un solo COPY, sin arrastrar nada de la de construccion.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Primero solo los metadatos: mientras pyproject.toml no cambie, esta capa se
# reutiliza y no se vuelven a bajar las dependencias en cada build.
COPY pyproject.toml README.md ./
COPY src ./src

# Los extras reales van aqui: la imagen de produccion SI habla con Gemini y con
# Qdrant. El que no los instala es el CI, que corre solo con lo base.
RUN pip install --no-cache-dir ".[gemini,qdrant]"

# --------------------------------------------------------------------------
# 2. Ejecucion
# --------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Usuario sin privilegios. Si alguien consigue ejecutar algo dentro del
# contenedor, no lo hace como root.
RUN useradd --create-home --uid 10001 imagent

COPY --from=build /opt/venv /opt/venv

WORKDIR /app
COPY --chown=imagent:imagent src ./src

# Los datos van a un directorio propio del usuario, que es lo que se monta como
# volumen: las imagenes y el sqlite del checkpointer tienen que sobrevivir a que
# se recree el contenedor.
RUN mkdir -p /data && chown -R imagent:imagent /data
ENV IMAGENT_STORAGE_DIR=/data/images \
    IMAGENT_SQLITE_PATH=/data/imagent.sqlite \
    PYTHONPATH=/app/src

USER imagent
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "imagent.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
