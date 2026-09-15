#!/usr/bin/env bash
# run_all.sh — drive a full capacity run from the RHEL VM.
#
# Starts the sampler on the GPU box over SSH, runs the load generator locally,
# stops the sampler, pulls its CSVs back, and builds the report.
#
# If you cannot SSH to the GPU box, run the two halves by hand — see README.md
# section "Running it manually".
#
#   ./run_all.sh
#   GPU_SSH=root@10.66.98.137 RATES="0.67 10 20 30" ./run_all.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_SSH="${GPU_SSH:-root@10.66.98.137}"
GPU_DIR="${GPU_DIR:-/tmp/vlm-capacity}"
PROXY="${PROXY:-http://127.0.0.1:8071}"
API_KEY="${API_KEY:-secret-bench}"
MODEL="${MODEL:-qwen3-vl}"
CORPUS="${CORPUS:-$HERE/corpus}"
RESULTS="${RESULTS:-$HERE/results}"
REPORT="${REPORT:-$HERE/../report}"
RATES="${RATES:-0.67 10 20 30}"
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
DAILY_IMAGES="${DAILY_IMAGES:-24000}"
WINDOW_HOURS="${WINDOW_HOURS:-10}"
SLA_P95="${SLA_P95:-30}"

echo "=== 0. Preflight ==========================================="
if [[ ! -d "$CORPUS" ]] || [[ -z "$(ls -A "$CORPUS" 2>/dev/null)" ]]; then
  echo "No corpus at $CORPUS — generating 300 synthetic pages."
  echo "Replace these with REAL production images before trusting the numbers."
  python3 "$HERE/make_images.py" --out "$CORPUS" --count 300
fi

echo "Proxy health:"
curl -fsS --max-time 10 "$PROXY/v1/health" || {
  echo "!! Proxy not answering at $PROXY. Start llm_proxy_v3 first."; exit 1; }
echo
echo "GPU server health (through the proxy):"
curl -fsS --max-time 20 "$PROXY/v1/gpu-health" || {
  echo "!! GPU server unreachable through the proxy."; exit 1; }
echo

# Clock skew between the two hosts is the one thing that silently corrupts the
# join between GPU samples and request records. Measure it, don't assume it.
VM_NOW=$(date +%s)
GPU_NOW=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$GPU_SSH" 'date +%s') || {
  echo "!! Cannot SSH to $GPU_SSH. Run the two halves manually (see README)."; exit 1; }
SKEW=$(( VM_NOW - GPU_NOW ))
echo "Clock skew (VM - GPU): ${SKEW}s"
if (( SKEW > 2 || SKEW < -2 )); then
  echo "   NOTE: passing --clock-offset-s ${SKEW} to analyze.py to align the two clocks."
  echo "   Fix NTP on both hosts for a cleaner run."
fi

echo
echo "=== 1. Start the sampler on the GPU box ===================="
ssh "$GPU_SSH" "mkdir -p $GPU_DIR"
scp -q "$HERE/gpu_monitor.py" "$GPU_SSH:$GPU_DIR/"
ssh "$GPU_SSH" "cd $GPU_DIR && nohup python3 gpu_monitor.py --out $GPU_DIR/results \
    --interval 1.0 > $GPU_DIR/monitor.log 2>&1 & echo \$! > $GPU_DIR/monitor.pid"
echo "sampler pid $(ssh "$GPU_SSH" "cat $GPU_DIR/monitor.pid")"
sleep 3

cleanup() {
  echo
  echo "=== Stopping sampler ======================================="
  ssh "$GPU_SSH" "kill \$(cat $GPU_DIR/monitor.pid) 2>/dev/null || true"
}
trap cleanup EXIT

echo
echo "=== 2. Load generator ======================================"
mkdir -p "$RESULTS"
# shellcheck disable=SC2086
python3 "$HERE/loadgen.py" \
    --corpus "$CORPUS" --out "$RESULTS" \
    --proxy "$PROXY" --api-key "$API_KEY" --model "$MODEL" \
    --rates $RATES

echo
echo "=== 3. Collect GPU samples ================================="
sleep 3
cleanup
trap - EXIT
sleep 2
scp -q "$GPU_SSH:$GPU_DIR/results/*.csv" "$RESULTS/"
ls -la "$RESULTS"

echo
echo "=== 4. Build the report ===================================="
python3 "$HERE/analyze.py" \
    --results "$RESULTS" --out "$REPORT" \
    --daily-images "$DAILY_IMAGES" --window-hours "$WINDOW_HOURS" \
    --sla-p95 "$SLA_P95" --max-concurrent "$MAX_CONCURRENT" \
    --clock-offset-s "$SKEW"

echo
echo "Done. Report: $REPORT/REPORT.md"
