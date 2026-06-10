# Get the container ID and monitor memory directly
POD_NAME="near-real-time-autopilot"

while true; do
  clear
  echo "=== $(date) ==="
  echo "Pod: $POD_NAME"

  # Try different cgroup paths (K8s uses different paths depending on version)
  kubectl exec -n argo $POD_NAME -- sh -c 'cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || cat /sys/fs/cgroup/memory.current 2>/dev/null' 2>/dev/null | awk '{printf "Memory: %.2f MB\n", $1/1024/1024}'

  # Also try to get memory limit
  kubectl exec -n argo $POD_NAME -- sh -c 'cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || cat /sys/fs/cgroup/memory.max 2>/dev/null' 2>/dev/null | awk '{printf "Limit: %.2f MB\n", $1/1024/1024}'

  sleep 2
done