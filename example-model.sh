#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
mkdir -p models

VARIANT="${1:-0.5b}"

if [ "$VARIANT" = "1.5b" ] || [ "$VARIANT" = "--1.5b" ]; then
    MODEL_NAME="qwen2.5-1.5b-instruct-q4_0.gguf"
    MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_0.gguf"
    MODEL_DESC="Qwen 2.5 1.5B Instruct (Q4_0 GGUF, ~940 MB)"
else
    MODEL_NAME="qwen2.5-0.5b-instruct-q4_0.gguf"
    MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf"
    MODEL_DESC="Qwen 2.5 0.5B Instruct (Q4_0 GGUF, ~409 MB)"
fi

TARGET_PATH="models/${MODEL_NAME}"

echo "========================================================================"
echo " Cluster GPU — Official Chat Model Downloader"
echo "========================================================================"
echo " Model:   ${MODEL_DESC}"
echo " Source:  ${MODEL_URL}"
echo " Target:  ${TARGET_PATH}"
echo "========================================================================"

if [ -f "$TARGET_PATH" ] && [ -s "$TARGET_PATH" ]; then
    echo "✔ Model already downloaded at: ${TARGET_PATH}"
else
    if command -v curl >/dev/null 2>&1; then
        curl -L --fail --progress-bar -C - -o "${TARGET_PATH}.part" "${MODEL_URL}"
        mv "${TARGET_PATH}.part" "${TARGET_PATH}"
    elif command -v wget >/dev/null 2>&1; then
        wget -c -O "${TARGET_PATH}.part" "${MODEL_URL}"
        mv "${TARGET_PATH}.part" "${TARGET_PATH}"
    else
        echo "Error: curl or wget is required to download models."
        exit 1
    fi
    echo "✔ Downloaded successfully: ${TARGET_PATH}"
fi

echo ""
echo "Next steps:"
echo "  1. Start the GPU Dashboard:  ./cl gpu-ui --port 8888"
echo "  2. Open in browser:          http://127.0.0.1:8888"
echo "  3. Click the Upload area and select: ${TARGET_PATH}"
echo "  4. Ask any question to test live GPU inference, and Offload when finished!"
echo ""
echo "Tip: Pass '1.5b' to fetch the larger 1.5B chat model: ./example-model.sh 1.5b"
