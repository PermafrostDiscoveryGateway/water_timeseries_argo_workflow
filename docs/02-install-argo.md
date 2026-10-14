# 2. Install Argo

With the `argo` namespace created, install Argo Workflows into it.

```bash
kubectl get pods -n argo
```

If you see no pods, install Argo Workflows (this pipeline has been run against v3.6.5):

```bash
kubectl apply -n argo -f "https://github.com/argoproj/argo-workflows/releases/download/v3.6.5/quick-start-minimal.yaml"
```

Confirm the Argo server and controller pods come up:

```bash
kubectl get pods -n argo
```

Continue to [3. Whitelisting (Autopilot only) and port forwarding](03-whitelist-and-port-forward.md).
