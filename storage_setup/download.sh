#!/bin/bash

until rsync -az --partial --progress --timeout=30 rsync://localhost:$LOCAL_PORT/data/ "$DEST"; do
  echo "rsync failed, restarting port-forward and retrying in 5s..."
  kill $PF_PID 2>/dev/null
  sleep 5
  kubectl -n $NAMESPACE port-forward $POD $LOCAL_PORT:873 &
  PF_PID=$!
  sleep 3
done

echo "Transfer complete."
kill $PF_PID 2>/dev/null
