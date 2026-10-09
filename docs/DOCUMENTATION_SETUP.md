# Documentation Setup Guide

This project uses **mkdocs** with the Material theme for documentation, set up the same way as
[water-timeseries-v2](https://github.com/PermafrostDiscoveryGateway/water-timeseries-v2).

## Building Documentation Locally

### Prerequisites

Install documentation dependencies:

```bash
pip install -r docs/requirements.txt
```

### Building HTML Documentation

From the project root, build the documentation:

```bash
mkdocs build
```

This generates the static site in the `site/` directory (excluded from git).

### Viewing Documentation Locally

Serve the documentation locally with live reload during development:

```bash
mkdocs serve
```

Then open your browser to: `http://localhost:8000`

The documentation will automatically rebuild when you change files.

## Documentation Structure

```
docs/
├── index.md                              # Documentation homepage
├── 01-create-namespace.md
├── 02-install-argo.md
├── 03-whitelist-and-port-forward.md
├── 04-setting-up-secrets.md
├── 05-filestore-setup-and-testing.md
├── 06-how-the-pipeline-works.md
├── 07-running-the-testing-pipeline.md
├── 08-running-the-full-pipeline.md
└── requirements.txt                      # Documentation dependencies

site/                                      # Build output (excluded from git)
mkdocs.yml                                 # mkdocs configuration (project root)
```

## Adding Documentation

Add a new markdown file under `docs/`, then add it to the `nav` section of `mkdocs.yml` so it
shows up in the site navigation and table of contents.

Unlike `water-timeseries-v2`, this repo has no Python package to auto-document, so there is no
`mkdocstrings`/API reference section here — these docs are purely narrative setup/operations
guides for the Argo/Kubernetes pipeline.

## Automatic Deployment

Documentation is automatically built and deployed to GitHub Pages on every push to `main` via the
GitHub Actions workflow in `.github/workflows/docs.yml`.

The workflow:

1. Triggers on push to `main` branch (and builds, without deploying, on pull requests)
2. Installs dependencies from `docs/requirements.txt` and builds with `mkdocs build`
3. Deploys to GitHub Pages
4. Makes documentation available at: `https://PermafrostDiscoveryGateway.github.io/water-timeseries-argo-workflow/`

For this to work, GitHub Pages must be enabled for the repository with the source set to
**GitHub Actions** (Settings → Pages → Build and deployment → Source).

## Troubleshooting

### Build failures

If the build fails:

1. Check for markdown/YAML syntax errors reported by `mkdocs build`
2. Verify every file listed in `mkdocs.yml`'s `nav` section actually exists under `docs/`
3. Run `mkdocs build --strict` locally to catch broken internal links before pushing
