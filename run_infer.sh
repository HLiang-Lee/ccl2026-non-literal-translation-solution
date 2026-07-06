#!/bin/bash
set -ex

# ------- 清理残留进程，避免端口/显存冲突 -------
pkill -9 sglang 2>/dev/null || true
sleep 1
ray stop --force 2>/dev/null || true
pkill -9 ray    2>/dev/null || true
sleep 1

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
WORK_DIR="${SCRIPT_DIR}"

HF_OUT_DIR=${HF_OUT_DIR:-${WORK_DIR}/Qwen3.5-9B_final_hf}

TEST1_FILE=${TEST1_FILE:-${WORK_DIR}/DatasetB/DatasetB_test_task1_public.csv}
TEST2_FILE=${TEST2_FILE:-${WORK_DIR}/DatasetB/DatasetB_test_task2_public.csv}
OUTPUT_FILE=${OUTPUT_FILE:-${WORK_DIR}/d_submit.csv}

BACKEND=${BACKEND:-sglang}
TP_SIZE=${TP_SIZE:-2}



if [ ! -d "${HF_OUT_DIR}" ]; then
    echo "[infer] ERROR: HF model dir not found: ${HF_OUT_DIR}" >&2
    exit 1
fi

# ------- Step 2 & 3: 推理 + 校验 -------
echo "[infer] model   : ${HF_OUT_DIR}"
echo "[infer] backend : ${BACKEND}  tp=${TP_SIZE}"
echo "[infer] output  : ${OUTPUT_FILE}"

cd "${WORK_DIR}"

FINAL_ARGS=()
if [ -n "${FINAL_FILE}" ]; then
    FINAL_ARGS+=(--final "${FINAL_FILE}")
fi

python3 infer.py \
    --model    "${HF_OUT_DIR}" \
    --test1    "${TEST1_FILE}" \
    --test2    "${TEST2_FILE}" \
    --output   "${OUTPUT_FILE}" \
    --backend  "${BACKEND}" \
    --tp-size  "${TP_SIZE}" 

echo "[infer] done -> ${OUTPUT_FILE}"
