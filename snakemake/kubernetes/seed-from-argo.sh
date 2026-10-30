#!/usr/bin/env bash
# Seed the snakemake-share volume with the pipeline's inputs from the argo
# volume (argo-workflows-share), pod to pod over the cluster network - PVCs
# can't be mounted across namespaces, and streaming ~14GB through your
# machine via `kubectl exec` is slow and tends to time out.
#
# The argo volume is only read from. Copies:
#   input/Nitze_etal_Lakes_filtered_full_set_V2d.parquet
#   region_lake_polygons/
#   test/dynamic_world_data/dynamic_world_historical_2026-06.nc  (test baseline)
#   dynamic_world_data/lakes_dw_V2d_compressed.nc -> test/dynamic_world_data/
#
# Needs both inspector pods running (see docs/10-...md, step 7):
#   pvc-inspector-python in argo (storage_setup/python-inspector.yaml)
#   snakemake-inspector in snakemake (snakemake/kubernetes/inspector.yaml)
#
# Usage (from the repo root):
#   snakemake/kubernetes/seed-from-argo.sh
set -euo pipefail

SRC_POD=${SRC_POD:-pvc-inspector-python}
DST_POD=${DST_POD:-snakemake-inspector}
PORT=${PORT:-9001}
BASELINE=${BASELINE:-dynamic_world_historical_2026-06.nc}
W=/data/water_timeseries

DST_IP=$(kubectl -n snakemake get pod "$DST_POD" -o jsonpath='{.status.podIP}')
echo "Copying argo/$SRC_POD -> snakemake/$DST_POD ($DST_IP:$PORT)"

# Receiver: unpack whatever arrives on the port into the snakemake volume.
# Detached (nohup), so a dropped kubectl connection doesn't stop it.
kubectl -n snakemake exec "$DST_POD" -- sh -c "rm -f /tmp/seed.rc; mkdir -p $W/test/dynamic_world_data; nohup sh -c 'nc -l -p $PORT | tar xf - -C $W; echo \$? > /tmp/seed.rc' > /tmp/seed.log 2>&1 &"
sleep 2

# Sender: one tar stream of everything, with the two .nc files moved under
# test/dynamic_world_data/. tar keeps the modification times, which
# create_new_historical_file.py uses to pick its baseline.
kubectl -n argo exec -i "$SRC_POD" -- sh -c 'cat > /tmp/seed_send.py' <<'EOF'
import socket, subprocess, sys

host, port, baseline = sys.argv[1], int(sys.argv[2]), sys.argv[3]
W = "/data/water_timeseries"
cmd = ["tar", "cf", "-",
       "--transform", rf"s,^\({baseline}\|lakes_dw_V2d_compressed.nc\)$,test/dynamic_world_data/\1,",
       "-C", W, "region_lake_polygons", "input/Nitze_etal_Lakes_filtered_full_set_V2d.parquet",
       "-C", f"{W}/test/dynamic_world_data", baseline,
       "-C", f"{W}/dynamic_world_data", "lakes_dw_V2d_compressed.nc"]
s = socket.create_connection((host, port), timeout=60)
p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
sent = 0
for chunk in iter(lambda: p.stdout.read(1 << 20), b""):
    s.sendall(chunk)
    sent += len(chunk)
s.shutdown(socket.SHUT_WR)
s.close()
rc = p.wait()
print(f"sent {sent} bytes, tar exit code {rc}")
open("/tmp/seed_send.rc", "w").write(str(rc))
EOF
kubectl -n argo exec "$SRC_POD" -- sh -c "rm -f /tmp/seed_send.rc; nohup python3 /tmp/seed_send.py $DST_IP $PORT $BASELINE > /tmp/seed_send.log 2>&1 &"

# Wait for both ends (~14GB; a few minutes inside the cluster).
while true; do
    recv=$(kubectl -n snakemake exec "$DST_POD" -- cat /tmp/seed.rc 2>/dev/null || true)
    send=$(kubectl -n argo exec "$SRC_POD" -- cat /tmp/seed_send.rc 2>/dev/null || true)
    [ -n "$recv" ] && [ -n "$send" ] && break
    mb=$(kubectl -n snakemake exec "$DST_POD" -- du -sm $W/region_lake_polygons $W/input $W/test/dynamic_world_data 2>/dev/null | awk '{s+=$1} END {print s}' || true)
    echo "$(date +%T)  ${mb:-?} MB so far"
    sleep 15
done

kubectl -n argo exec "$SRC_POD" -- cat /tmp/seed_send.log
kubectl -n snakemake exec "$DST_POD" -- cat /tmp/seed.log
kubectl -n argo exec "$SRC_POD" -- rm -f /tmp/seed_send.py /tmp/seed_send.log /tmp/seed_send.rc
kubectl -n snakemake exec "$DST_POD" -- rm -f /tmp/seed.rc /tmp/seed.log

if [ "$recv" != 0 ] || [ "$send" != 0 ]; then
    echo "Copy failed (sender exit $send, receiver exit $recv)" >&2
    exit 1
fi
echo "Done."
