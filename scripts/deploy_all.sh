#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

services=(
  "services/modal_streaming_whisper.py"
  "services/gpt2_service.py"
  "services/tts_service.py"
  "services/feedback_service.py"
)

echo "Deploying Modal services from: $ROOT_DIR"

for service in "${services[@]}"; do
  echo ""
  echo "==> modal deploy $service"
  modal deploy "$service"
done

echo ""
echo "All services deployed successfully."
