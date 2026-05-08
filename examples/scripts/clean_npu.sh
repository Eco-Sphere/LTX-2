#!/usr/bin/env bash
set -euo pipefail

# Clean only residual processes that belong to this repository. If other users'
# processes are using the cards, wait instead of killing them.

CURRENT_PID=$$
PARENT_PID=$PPID
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "[clean_npu] Checking NPU processes..."

device_count=$(npu-smi info -l 2>/dev/null | grep -c "NPU ID" || true)
if [ "${device_count:-0}" -eq 0 ]; then
    echo "[clean_npu] No NPU devices found, skipping."
    exit 0
fi

all_pids=""
for dev_id in $(seq 0 $((device_count - 1))); do
    pids=$(npu-smi info -t proc-mem -i "$dev_id" -c 0 2>/dev/null \
        | grep -oE 'PID[[:space:]]*:[[:space:]]*[0-9]+' | grep -oE '[0-9]+' || true)
    all_pids="$all_pids $pids"
done
all_pids=$(printf '%s\n' $all_pids | sort -u | tr '\n' ' ')

foreign_count=0
our_count=0

for pid in $all_pids; do
    [ "$pid" = "$CURRENT_PID" ] && continue
    [ "$pid" = "$PARENT_PID" ] && continue
    [ "$pid" = "1" ] && continue
    [ -z "$pid" ] && continue

    cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
    cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null || true)

    if [[ "$cmdline" == *"$PROJECT_ROOT"* || "$cwd" == "$PROJECT_ROOT"* ]]; then
        echo "[clean_npu] Killing residual PID $pid on NPU"
        kill -9 "$pid" 2>/dev/null || true
        our_count=$((our_count + 1))
    else
        foreign_count=$((foreign_count + 1))
    fi
done

if [ "$our_count" -gt 0 ]; then
    echo "[clean_npu] Killed $our_count residual process(es)"
fi

if [ "$foreign_count" -gt 0 ]; then
    echo "[clean_npu] $foreign_count other process(es) on NPU; waiting..."
    while true; do
        still_foreign=0
        for pid in $all_pids; do
            [ "$pid" = "$CURRENT_PID" ] && continue
            [ "$pid" = "$PARENT_PID" ] && continue
            [ "$pid" = "1" ] && continue
            [ -z "$pid" ] && continue
            [ ! -d "/proc/$pid" ] && continue

            cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
            cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null || true)
            if [[ "$cmdline" != *"$PROJECT_ROOT"* && "$cwd" != "$PROJECT_ROOT"* ]]; then
                still_foreign=$((still_foreign + 1))
            fi
        done
        [ "$still_foreign" -eq 0 ] && break
        sleep 10
    done
    echo "[clean_npu] Other processes finished; devices are free."
else
    echo "[clean_npu] No residual processes found. Devices are clean."
fi
