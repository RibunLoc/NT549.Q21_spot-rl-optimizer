#!/bin/bash
# trigger_workload.sh — kịch bản trigger Jenkins builds để test DQN controller
#
# Cách dùng:
#   chmod +x scripts/trigger_workload.sh
#   ./scripts/trigger_workload.sh <phase>
#
# Phases:
#   baseline  — 1 build nhỏ, kiểm tra connectivity
#   spike     — 6 builds cùng lúc, agent phải PROVISION
#   sustained — 3 builds mỗi 5 phút trong 30 phút
#   drain     — không trigger, đợi queue rỗng

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
JENKINS_URL="${JENKINS_URL:-https://jenkins.holoc.id.vn}"
JENKINS_USER="${JENKINS_USER:-hothanhloc}"
JENKINS_TOKEN="${JENKINS_TOKEN:-}"
JOB_NAME="${JOB_NAME:-Spot-rl-demo}"   # tên job trên Jenkins

PHASE="${1:-baseline}"

# ── Helper ────────────────────────────────────────────────────────────────────
trigger_build() {
    local size="${1:-small}"
    local sleep_s="${2:-120}"
    # Thêm BUILD_LABEL unique để Jenkins không deduplicate các build cùng params
    # Dùng $RANDOM (shell built-in, luôn có) + epoch seconds để đảm bảo unique
    local label="wave-$(date +%s)-${RANDOM}"

    http_code=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST \
        "${JENKINS_URL}/job/${JOB_NAME}/buildWithParameters" \
        --user "${JENKINS_USER}:${JENKINS_TOKEN}" \
        -L \
        --data-urlencode "WORKLOAD_SIZE=${size}" \
        --data-urlencode "SLEEP_SECONDS=${sleep_s}" \
        --data-urlencode "BUILD_LABEL=${label}")

    if [ "$http_code" -eq 200 ] || [ "$http_code" -eq 201 ] || [ "$http_code" -eq 303 ]; then
        echo "  → queued build size=${size} sleep=${sleep_s}s label=${label} (HTTP ${http_code})"
    else
        echo "  ✗ FAILED size=${size} (HTTP ${http_code}) — check JENKINS_TOKEN và job name '${JOB_NAME}'"
    fi
}

queue_depth() {
    local response
    response=$(curl -s -w "\n%{http_code}" \
        "${JENKINS_URL}/queue/api/json?tree=items[id,blocked,buildable,why,task[name]]" \
        --user "${JENKINS_USER}:${JENKINS_TOKEN}")
    local http_code body
    http_code=$(echo "$response" | tail -1)
    body=$(echo "$response" | head -n -1)

    if [ "$http_code" != "200" ]; then
        echo "  Queue: cannot fetch (HTTP ${http_code}) — check JENKINS_TOKEN"
        return
    fi
    echo "$body" | python3 -c "
import json, sys
d = json.load(sys.stdin)
items = d.get('items', [])
print(f'  Queue depth: {len(items)} item(s)')
for it in items:
    name = it.get('task', {}).get('name', '?')
    why  = it.get('why', '')
    print(f'    - {name}: {why}')
" 2>/dev/null || echo "  Queue: parse error (response was not JSON)"
}

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Phases ────────────────────────────────────────────────────────────────────
case "$PHASE" in

  baseline)
    log "=== PHASE: baseline — 1 small build ==="
    log "Purpose: verify connectivity + agent sees pending=1"
    trigger_build "small" "90"
    log "Triggered. Wait 1 controller step (~15min) and check:"
    log "  spot_rl_pending_jobs should momentarily be >= 1"
    log "  agent likely HOLD (1 job không đủ trigger provision)"
    ;;

  spike)
    log "=== PHASE: spike — 6 builds cùng lúc ==="
    log "Purpose: flood queue, agent phải ra PROVISION_SPOT"
    log "Triggering 6 medium builds..."
    for i in $(seq 1 6); do
        trigger_build "medium" "180"
        sleep 8
    done
    log ""
    log "Done. Queue status:"
    queue_depth
    log ""
    log "Chờ tối đa 15 phút (1 controller step). Kỳ vọng:"
    log "  spot_rl_pending_jobs >= 6"
    log "  action_total{op=PROVISION_SPOT} tăng"
    ;;

  medium_wave)
    log "=== PHASE: medium_wave — 3 builds mỗi 5 phút × 3 lần ==="
    log "Purpose: sustained load, agent duy trì fleet"
    for wave in 1 2 3; do
        log "Wave ${wave}/3 — triggering 3 builds..."
        for i in $(seq 1 3); do
            trigger_build "medium" "150"
            sleep 8   # > Jenkins Quiet Period (default 5s) để tránh deduplicate
        done
        queue_depth
        if [ "$wave" -lt 3 ]; then
            log "Đợi 5 phút trước wave tiếp theo..."
            sleep 300
        fi
    done
    log "Tổng 9 builds đã trigger. Theo dõi sla_health + spot_ratio trên Grafana."
    ;;

  large_spike)
    log "=== PHASE: large_spike — 3 large builds (6 tasks mỗi build) ==="
    log "Purpose: heavy workload, kiểm tra agent scale up"
    for i in $(seq 1 3); do
        trigger_build "large" "240"
        sleep 8
    done
    queue_depth
    log "Kỳ vọng: agent provision 2-3 spot instances, cost tăng nhưng SLA maintained"
    ;;

  drain)
    log "=== PHASE: drain — không trigger, đợi queue rỗng ==="
    log "Purpose: agent tự RELEASE khi idle"
    log "Current queue:"
    queue_depth
    log ""
    log "Không trigger build mới. Đợi 2-3 controller steps (30-45 phút)."
    log "Kỳ vọng:"
    log "  spot_rl_pending_jobs → 0"
    log "  action_total{op=RELEASE_SPOT} tăng"
    log "  spot_rl_spot_instances giảm"
    ;;

  status)
    log "=== Jenkins queue status ==="
    queue_depth
    ;;

  *)
    echo "Usage: $0 <phase>"
    echo ""
    echo "Phases:"
    echo "  baseline     — 1 small build, kiểm tra connectivity"
    echo "  spike        — 6 medium builds cùng lúc → trigger PROVISION"
    echo "  medium_wave  — 9 builds theo sóng → test sustained load"
    echo "  large_spike  — 3 large builds → test heavy workload"
    echo "  drain        — không trigger, test RELEASE khi idle"
    echo "  status       — xem queue depth hiện tại"
    echo ""
    echo "Env vars:"
    echo "  JENKINS_URL   (default: https://jenkins.holoc.id.vn)"
    echo "  JENKINS_USER  (default: hothanhloc)"
    echo "  JENKINS_TOKEN (default: từ configs/.env)"
    echo "  JOB_NAME      (default: spot-rl-demo)"
    exit 1
    ;;
esac
