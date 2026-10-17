#!/usr/bin/env bash
# Keepalive & health monitoring script for Datathon26 (Issue #12)
# Prevents macOS from sleeping and continuously monitors /health endpoint.

set -euo pipefail

URL="${1:-https://usgs-transcripts-tactics-scott.trycloudflare.com/health}"
LOCAL_URL="http://127.0.0.1:8000/health"
INTERVAL_SEC="${2:-300}"
LOG_FILE="${3:-var/keepalive_health.log}"
RESP_TMP="var/.health_response.tmp"

mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$(dirname "$RESP_TMP")"

echo "=== Datathon26 Keepalive & Healthcheck Monitor ==="
echo "Target URL: $URL"
echo "Local fallback URL: $LOCAL_URL"
echo "Interval: ${INTERVAL_SEC}s"
echo "Log file: $LOG_FILE"

# Start caffeinate in background to prevent macOS sleep
if command -v caffeinate >/dev/null 2>&1; then
    echo "[+] Starting caffeinate to prevent system sleep..."
    caffeinate -dimsu &
    CAFFEINATE_PID=$!
    trap 'echo "Stopping caffeinate ($CAFFEINATE_PID)..."; kill "$CAFFEINATE_PID" 2>/dev/null || true' EXIT
fi

echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Monitor active. Pinging every ${INTERVAL_SEC}s..." | tee -a "$LOG_FILE"

while true; do
    TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    HTTP_CODE=$(curl -s --connect-timeout 5 --max-time 10 -o "$RESP_TMP" -w "%{http_code}" "$URL" 2>/dev/null || true)
    HTTP_CODE="${HTTP_CODE:-000}"

    if [ "$HTTP_CODE" = "200" ]; then
        STATUS=$(grep -o '"status":"[^"]*"' "$RESP_TMP" 2>/dev/null || echo 'status: ok')
        echo "[$TIMESTAMP] OK ($HTTP_CODE, public tunnel): $STATUS" | tee -a "$LOG_FILE"
    else
        # Try local fallback to distinguish tunnel vs server issue
        LOCAL_CODE=$(curl -s --connect-timeout 3 --max-time 5 -o /dev/null -w "%{http_code}" "$LOCAL_URL" 2>/dev/null || true)
        LOCAL_CODE="${LOCAL_CODE:-000}"
        echo "[$TIMESTAMP] WARNING: Public healthcheck returned HTTP $HTTP_CODE (local: $LOCAL_CODE)!" | tee -a "$LOG_FILE"
    fi

    sleep "$INTERVAL_SEC"
done
