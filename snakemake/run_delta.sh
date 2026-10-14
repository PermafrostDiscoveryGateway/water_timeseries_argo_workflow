#!/usr/bin/env bash
# Run the test pipeline on Delta (see docs/10-running-on-delta.md).
#
# Usage, from anywhere, on a Delta login node:
#   snakemake/run_delta.sh                 # target_date=2025-08
#   snakemake/run_delta.sh 2025-07         # another month
#   snakemake/run_delta.sh 2025-08 -n      # dry run; anything after the date goes to snakemake
#   snakemake/run_delta.sh 2025-08 --overwrite
#       # replace the month's results: reprocess complete regions, resume partial
#       # ones, replace the month in the combined archive, rebuild its PMTiles.
#       # Downloads and merges are not redone. Same as
#       # --config overwrite=true --forcerun process (see config.yaml).
#
# A real run is started inside a tmux session (water_timeseries-<date>), since
# snakemake has to keep running on the login node until every Slurm job is
# done. Detach with Ctrl-b d; reattach from the same login node with
#   tmux attach -t water_timeseries-<date>
#
# Nothing needs to be activated first: snakemake comes from `uv tool install`
# (~/.local/bin), and every pipeline step runs with the venv the build_environment rule
# builds at environment.path in config.delta.yaml.

set -euo pipefail

TARGET_DATE="${1:-2025-08}"
shift || true
if [[ ! "$TARGET_DATE" =~ ^[0-9]{4}-[0-9]{2}$ ]]; then
    echo "First argument must be the target month as YYYY-MM, got: $TARGET_DATE" >&2
    exit 1
fi

# Repo root = the directory above this script, wherever it's checked out.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Shared biyc directories stay group-writable (see config.delta.yaml).
umask 002

# A virtualenv or conda env activated in this shell would only get in the way.
unset VIRTUAL_ENV CONDA_PREFIX PYTHONPATH PYTHONHOME

export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null; then
    echo "Installing uv into ~/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
if ! command -v snakemake >/dev/null; then
    echo "Installing snakemake + the Slurm executor plugin with uv"
    uv tool install snakemake --with snakemake-executor-plugin-slurm
fi

for f in ~/.config/water_timeseries/application_default_credentials.json \
         ~/.config/water_timeseries/earthengine_credentials; do
    if [[ ! -s "$f" ]]; then
        echo "Missing credentials file: $f" >&2
        echo "Copy it from the cluster first - see 'Copying the cluster's credentials to Delta'" >&2
        echo "in docs/07-running-the-testing-pipeline.md." >&2
        exit 1
    fi
done

CONFIG=("target_date=$TARGET_DATE")
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--overwrite" ]]; then
        CONFIG+=("overwrite=true")
        ARGS+=(--forcerun process)
    else
        ARGS+=("$arg")
    fi
done

CMD=(snakemake -s snakemake/Snakefile
     --configfile snakemake/config.delta.yaml
     --profile snakemake/profiles/delta
     --config "${CONFIG[@]}"
     "${ARGS[@]}")

DRY_RUN=false
for arg in "$@"; do
    [[ "$arg" == "-n" || "$arg" == "--dry-run" || "$arg" == "--dryrun" ]] && DRY_RUN=true
done

# Already in tmux, or just a dry run: run here.
if [[ -n "${TMUX:-}" || "$DRY_RUN" == true ]]; then
    echo "Running in $REPO_ROOT on $(hostname):"
    echo "  ${CMD[*]}"
    exec "${CMD[@]}"
fi

SESSION="water_timeseries-$TARGET_DATE"
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session $SESSION already exists on $(hostname); attaching to it."
    exec tmux attach -t "$SESSION"
fi

echo "Starting tmux session $SESSION on $(hostname)."
echo "Detach with Ctrl-b d; reattach later with: ssh $(hostname) then tmux attach -t $SESSION"
# Re-run this script inside tmux ($TMUX is set there, so it runs snakemake
# directly), then keep a shell open so the output stays readable afterwards.
tmux new-session -d -s "$SESSION" -c "$REPO_ROOT" \
    "$(printf '%q ' "$0" "$TARGET_DATE" "$@"); exec bash"
exec tmux attach -t "$SESSION"
