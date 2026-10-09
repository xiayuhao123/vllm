ARG BASE_IMAGE=vllm/vllm-openai:v0.31.0
FROM ${BASE_IMAGE}
USER root

COPY vllm /tmp/l40-v031/vllm
COPY apply_overlay.py BASE_SHA256SUMS PAYLOAD_SHA256SUMS /tmp/l40-v031/
RUN python3 /tmp/l40-v031/apply_overlay.py

ENV VLLM_ENABLE_SM89_FP8_TUNING=1

LABEL l40.base="v0.31.0" l40.dflash.prefix-cache="opt-in-v1" l40.fp8.tuning="measured-exact-m-v1"
