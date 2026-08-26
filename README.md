# water_timeseries_argo_workflow

Argo Workflows for running the water time series lake drainage pipeline on a Google Kubernetes Engine cluster.

Full documentation — cluster setup, storage/Filestore setup, secrets, and the NRT pipeline — is published here:

**https://permafrostdiscoverygateway.github.io/water_timeseries_argo_workflow/**

## Building the docs locally

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

Then open `http://localhost:8000`.
