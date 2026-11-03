# The PhysioNet pipeline as an image: no credentials and no data inside it.
#
#   docker build -f deploy/physionet/Dockerfile -t aegis-physionet .            # the full run
#   docker build -f deploy/physionet/Dockerfile --build-arg TORCH=0 -t aegis-physionet-lake .
#
# Python 3.9 and these pins match the environment every result so far was produced under.
FROM python:3.9-slim-bookworm

ARG TORCH=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 git ca-certificates && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/requirements.txt
COPY backend/ml/physionet/requirements.txt /app/physionet-requirements.txt
RUN pip install --no-cache-dir --disable-pip-version-check numpy==2.0.2 scipy==1.13.1 pandas==2.3.3 scikit-learn==1.6.1 joblib==1.5.3 lightgbm==4.6.0 certifi \
      -r /app/physionet-requirements.txt \
 && if [ "$TORCH" = "1" ]; then pip install --no-cache-dir --disable-pip-version-check torch==2.8.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu; fi

COPY backend /app/backend
COPY verify_physionet.py /app/verify_physionet.py
WORKDIR /app
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1 PHYSIONET_LAKE=/lake MIMIC_CACHE_DIR=/lake/cache MIMIC_REPORT_DIR=/lake/reports
RUN useradd -m -u 1000 aegis && mkdir -p /lake /app/artifacts && chown -R aegis /lake /app/artifacts
USER aegis
VOLUME /lake
ENTRYPOINT ["python", "-W", "ignore", "-m", "backend.ml.physionet"]
CMD ["status"]
