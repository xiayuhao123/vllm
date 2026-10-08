ARG BASE_IMAGE=vllm/vllm-openai:v0.31.0
FROM ${BASE_IMAGE}
USER root

COPY vllm /tmp/l40-v031/vllm
COPY apply_overlay.py BASE_SHA256SUMS PAYLOAD_SHA256SUMS /tmp/l40-v031/
RUN python3 /tmp/l40-v031/apply_overlay.py

LABEL l40.base="v0.31.0" l40.mtp.decode-graph="experimental-v1" \
      l40.sm89.extra-warmup="opt-in"
