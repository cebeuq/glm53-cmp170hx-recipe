# CPU-only image for the two offline conversion steps (no GPU, no network at run time).
FROM python:3.12.7-slim-bookworm
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 \
 && pip install --no-cache-dir safetensors==0.4.5 numpy==2.1.3
WORKDIR /work
COPY requant_full.py hybrid_nvfp4e_int8r.py /work/
ENTRYPOINT ["python3", "-u"]
