"""Tables of how each step of the snakemake pipeline went: benchmarks, Slurm accounting, per-stage log.

Run it after a run, from the repository folder, with the run's work directory
(<paths.work_dir>/<target_date>, e.g. data/snakemake_work/2025-08; the Snakefile
puts its benchmarks, logs and stage log there):

    python3 snakemake/benchmark_table.py --work data/snakemake_work/2025-08
    python3 snakemake/benchmark_table.py --work <work_dir>/2025-08 --rules process --completed

It prints up to two tables and writes them next to the benchmarks
(<work>/benchmarks/summary.tsv and stages.tsv; --out changes the first). --history
adds a third, the benchmark of every run so far (see below).

1. Jobs: one row per rule job (a region's download, merge_and_backfill, force_merge or
   process; create_nrt_pmtiles, ...). The benchmark columns are from Snakemake's benchmark
   file of that rule and label (<work>/benchmarks/<rule>[/<label>].tsv): wall time, CPU time,
   peak memory and mean CPU load of the whole job. On Delta, Slurm's accounting (`sacct`)
   adds the job's state, partition, elapsed time, peak host memory and file I/O. The Slurm
   job IDs come from the file names under .snakemake/slurm_logs/rule_<rule>/<label>/<id>.log
   and from the "SLURM jobid <id> (log: <path>)" lines of Snakemake's run logs under
   .snakemake/log/ (the Slurm plugin deletes the log of a job that succeeded). Every attempt
   is listed, failed ones too (an out-of-memory kill shows as OUT_OF_MEMORY); --completed
   lists only the jobs that finished. Benchmark files with no Slurm job (a local run, or the
   rules that run on the login node) are listed with their benchmark columns alone.

2. Stages: one row per stage process, from <work>/nrt_pipeline_log.jsonl. A rule's benchmark
   file adds up everything the rule ran (download + its missing-ID fallback; merge +
   backfill + re-merge), so this is where to see which of them took the time. Wall time, CPU
   time and peak memory (ru_maxrss of the stage's process tree) of every attempt.

Notes on the numbers:
  * Snakemake can only sample memory, CPU load and I/O where psutil may read them: on macOS
    those benchmark columns are NA (only wall time is filled in), on Linux they are not.
    The stages table has its own memory and CPU figures everywhere.
  * Snakemake's max_rss samples the process and can miss short peaks. To size a rule's mem_mb
    in profiles/delta/config.yaml or the Snakefile, use host_mem_gib (sacct MaxRSS) or the
    stages' peak_rss_mb.
  * `sacct` reports MaxRSS in KiB; the table shows GiB. Delta charges CPU jobs by cores or
    memory share of the node, whichever is larger.
  * Snakemake overwrites a rule's benchmark file when the job runs again, so the files
    describe the newest run only, and bench_* is filled in on the newest completed job's row.
    To keep every run, each benchmark file is also appended to <benchmarks folder>/history.tsv
    (one row per run of a rule and label, with its target_date, run_name and the time the
    file was written). The Snakefile does it when snakemake starts (before anything is
    overwritten) and when it ends; this script does it too, so a run is never lost, and
    nothing is added twice. --history prints it. Stage rows are never overwritten.

Uses only the standard library.
"""

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SACCT_FIELDS = "JobID,Partition,State,ElapsedRaw,MaxRSS,TRESUsageInMax,ExitCode,Start"
JOB_COLUMNS = ["rule", "label", "job_id", "state", "partition", "elapsed_s", "host_mem_gib", "cpu_s",
               "disk_gib", "bench_wall_s", "bench_cpu_s", "bench_max_rss_mb", "bench_mean_load"]
STAGE_COLUMNS = ["stage", "region", "target_date", "exit_code", "elapsed_s", "cpu_s", "peak_rss_mb", "timestamp"]
HISTORY_NAME = "history.tsv"
HISTORY_KEY = ["target_date", "run_name", "rule", "label", "finished_at"]
HISTORY_SHOWN = HISTORY_KEY + ["s", "cpu_time", "max_rss", "io_in", "io_out", "mean_load"]
SKIPPED_TSV = ("summary.tsv", "stages.tsv", HISTORY_NAME)
UNITS = {"K": 1 / 1024**2, "M": 1 / 1024, "G": 1.0, "T": 1024.0}  # to GiB


def to_gib(text):
    """'73276504K' -> GiB; '15152M' -> GiB; bare numbers are bytes. None if empty."""
    if not text:
        return None
    m = re.fullmatch(r"([\d.]+)([KMGT]?)", text.strip())
    if not m:
        return None
    value, unit = float(m.group(1)), m.group(2)
    return value * UNITS[unit] if unit else value / 1024**3


def to_seconds(text):
    """'00:19:38' or '1-02:03:04' -> seconds. None if empty."""
    if not text:
        return None
    days, _, rest = text.rpartition("-")
    parts = [float(x) for x in rest.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, sec = parts
    return int(days or 0) * 86400 + h * 3600 + m * 60 + sec


def parse_tres(text):
    """'cpu=00:19:38,fs/disk=123,...' -> dict."""
    return dict(item.split("=", 1) for item in (text or "").split(",") if "=" in item)


LOG_PATH = re.compile(r"rule_([^/\s]+)/(?:(.+)/)?(\d+)\.log$")
SUBMITTED = re.compile(r"SLURM jobid (\d+) \(log: ([^)]+)\)")


def find_jobs(slurm_logs, snakemake_logs):
    """[(rule, label, job_id)] from the Slurm plugin's log files and Snakemake's run logs."""
    found = {}
    for log in Path(slurm_logs).glob("rule_*/**/*.log"):
        m = LOG_PATH.search(log.relative_to(slurm_logs).as_posix())
        if m and log.stem.isdigit():
            found[m.group(3)] = (m.group(1), m.group(2) or "-", m.group(3))
    for run_log in Path(snakemake_logs).glob("*.snakemake.log"):
        for jid, path in SUBMITTED.findall(run_log.read_text(errors="replace")):
            m = LOG_PATH.search(path)
            if m and m.group(3) == jid:
                found.setdefault(jid, (m.group(1), m.group(2) or "-", jid))
    return sorted(found.values(), key=lambda j: (j[0], j[1], int(j[2])))


def run_sacct(job_ids):
    try:
        out = subprocess.run(["sacct", "-j", ",".join(job_ids), "-P", "-n", f"--format={SACCT_FIELDS}"],
                             capture_output=True, text=True, check=True).stdout
    except FileNotFoundError:
        return None
    except subprocess.CalledProcessError as e:
        sys.exit(f"sacct failed: {e.stderr.strip()}")
    return out


def parse_sacct(text):
    """{job_id: stats} from `sacct -P -n` output with SACCT_FIELDS."""
    jobs, steps = {}, {}
    for line in text.splitlines():
        f = line.split("|")
        if len(f) < 8:
            continue
        jid, partition, state, elapsed, maxrss, tres, _exitcode, start = f[:8]
        base = jid.split(".")[0]
        if "." not in jid:
            jobs[base] = {"state": state.split()[0] if state else "?", "partition": partition, "start": start,
                          "elapsed_s": int(elapsed) if elapsed.isdigit() else None}
        elif not jid.endswith(".extern"):
            steps.setdefault(base, []).append((to_gib(maxrss) or 0.0, maxrss, tres))
    for base, job in jobs.items():
        # the step that did the work is the one with the largest memory
        _, maxrss, tres = max(steps.get(base, [(0.0, "", "")]), key=lambda s: s[0])
        t = parse_tres(tres)
        job.update(host_mem_gib=to_gib(maxrss) or to_gib(t.get("mem")), cpu_s=to_seconds(t.get("cpu")),
                   disk_gib=to_gib(t.get("fs/disk")))
    return jobs


def rule_label(path, bench_dir):
    """(rule, label) of <bench_dir>/<rule>[/<label>].tsv; '-' when the rule has no label."""
    parts = Path(path).relative_to(bench_dir).with_suffix("").parts
    return (parts[0], "_".join(parts[1:])) if len(parts) > 1 else (parts[0], "-")


def benchmark_files(bench_dir):
    return [p for p in sorted(Path(bench_dir).glob("**/*.tsv")) if p.name not in SKIPPED_TSV]


def read_benchmarks(bench_dir):
    """{(rule, label): row of the benchmark file} for every <rule>[/<label>].tsv under bench_dir."""
    found = {}
    for path in benchmark_files(bench_dir):
        rows = list(csv.DictReader(path.open(), delimiter="\t"))
        if rows:
            found[rule_label(path, bench_dir)] = rows[0]
    return found


def archive_benchmarks(bench_dir, target_date="", run_name=""):
    """Append every benchmark file under bench_dir that history.tsv doesn't have yet; returns how many.

    Snakemake overwrites a benchmark file when its job runs again, so this keeps each run.
    A run is identified by its rule, label and the time its file was written, which makes
    this safe to call as often as you like, from the Snakefile and from the command line.
    """
    bench_dir = Path(bench_dir)
    if not bench_dir.is_dir():
        return 0
    history = bench_dir / HISTORY_NAME
    old = list(csv.DictReader(history.open(), delimiter="\t")) if history.is_file() else []
    columns = list(old[0].keys()) if old else []
    have = {tuple(r.get(k) for k in HISTORY_KEY[2:]) for r in old}
    new = []
    for path in benchmark_files(bench_dir):
        rule, label = rule_label(path, bench_dir)
        finished = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        rows = list(csv.DictReader(path.open(), delimiter="\t"))
        if rows and (rule, label, finished) not in have:
            new.append({"target_date": target_date, "run_name": run_name or target_date, "rule": rule,
                        "label": label, "finished_at": finished, **rows[0]})
    if not new:
        return 0
    for c in HISTORY_KEY + [c for r in new for c in r]:  # Snakemake's columns can differ between versions
        if c not in columns:
            columns.append(c)
    tmp = history.with_suffix(".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, delimiter="\t", restval="NA")
        w.writeheader()
        w.writerows(old + new)
    tmp.replace(history)
    return len(new)


def history_rows(bench_dir, rules=None):
    history = Path(bench_dir) / HISTORY_NAME
    if not history.is_file():
        return []
    rows = [r for r in csv.DictReader(history.open(), delimiter="\t") if not rules or r["rule"] in rules]
    out = [{c: ("-" if r.get(c) in (None, "", "NA") else (fmt(r[c], 1) if c in ("s", "cpu_time") else r[c]))
            for c in HISTORY_SHOWN} for r in rows]
    return sorted(out, key=lambda r: (r["rule"], r["label"], r["finished_at"]))


def fmt(value, digits=1):
    if value is None or value == "NA" or value == "":
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except ValueError:
        return str(value)


def bench_cells(row):
    row = row or {}
    return {"bench_wall_s": fmt(row.get("s"), 0), "bench_cpu_s": fmt(row.get("cpu_time"), 0),
            "bench_max_rss_mb": fmt(row.get("max_rss"), 0), "bench_mean_load": fmt(row.get("mean_load"), 0)}


def print_table(rows, columns):
    widths = {c: max(len(c), *(len(str(r.get(c, "-"))) for r in rows)) for c in columns}
    print("  ".join(c.ljust(widths[c]) for c in columns))
    for r in rows:
        print("  ".join(str(r.get(c, "-")).ljust(widths[c]) for c in columns))


def write_tsv(path, rows, columns):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"written to {path}", file=sys.stderr)


def job_rows(args, bench_dir, benchmarks):
    """Rows for the jobs table: Slurm jobs joined with benchmark files, else the benchmark files alone."""
    wanted = lambda rule: not args.rules or rule in args.rules  # noqa: E731
    jobs = [j for j in find_jobs(args.slurm_logs, args.snakemake_logs) if wanted(j[0])]
    text = None
    if args.sacct_file:
        text = Path(args.sacct_file).read_text()
    elif jobs:
        text = run_sacct([j[2] for j in jobs])
        if text is None:
            print("sacct not found (not a Delta login node?): showing the benchmark files only.", file=sys.stderr)
    out = []
    if text is not None:
        acct = parse_sacct(text)
        newest_done = {}
        for rule, label, jid in jobs:
            a = acct.get(jid)
            if a and a["state"] == "COMPLETED" and (not args.since or a["start"] >= args.since):
                newest_done[(rule, label)] = max(newest_done.get((rule, label), 0), int(jid))
        for rule, label, jid in jobs:
            a = acct.get(jid)
            if a is None or (args.since and a["start"] < args.since):
                continue  # not in sacct (too old, or not this account's), or before --since
            if args.completed and a["state"] != "COMPLETED":
                continue
            bench = benchmarks.get((rule, label)) if newest_done.get((rule, label)) == int(jid) else None
            out.append({"rule": rule, "label": label, "job_id": jid, "state": a["state"],
                        "partition": a["partition"], "elapsed_s": fmt(a["elapsed_s"], 0),
                        "host_mem_gib": fmt(a["host_mem_gib"], 2), "cpu_s": fmt(a["cpu_s"], 0),
                        "disk_gib": fmt(a["disk_gib"], 2), **bench_cells(bench)})
    # Benchmark files with no Slurm job: everything of a local run, and the rules that run on
    # the login node (build_environment, wait_for_region_merges) in a Slurm one.
    submitted = {(rule, label) for rule, label, _ in jobs} if text is not None else set()
    for (rule, label), row in sorted(benchmarks.items()):
        if wanted(rule) and (rule, label) not in submitted:
            out.append({"rule": rule, "label": label, "job_id": "-", "state": "-", "partition": "-",
                        "elapsed_s": "-", "host_mem_gib": "-", "cpu_s": "-", "disk_gib": "-",
                        **bench_cells(row)})
    return sorted(out, key=lambda r: (r["rule"], r["label"], r["job_id"].zfill(12)))


def stage_rows(work):
    """One row per line of nrt_pipeline_log.jsonl (runs before per-stage timing have no figures)."""
    log = Path(work) / "nrt_pipeline_log.jsonl"
    if not log.is_file():
        return []
    rows = []
    for line in log.read_text().splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        rows.append({c: fmt(rec.get(c), 1) if c in ("elapsed_s", "cpu_s") else
                     ("-" if rec.get(c) in (None, "") else rec.get(c)) for c in STAGE_COLUMNS})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", help="the run's work directory, <paths.work_dir>/<target_date>: holds benchmarks/ "
                   "and nrt_pipeline_log.jsonl")
    p.add_argument("--benchmarks", help="the benchmarks folder (default <work>/benchmarks)")
    p.add_argument("--rules", nargs="*", help="only these rules (default: all)")
    p.add_argument("--completed", action="store_true",
                   help="only Slurm jobs that finished (state COMPLETED); by default failed and cancelled "
                   "attempts are listed too")
    p.add_argument("--since", help="only Slurm jobs that started on or after this date, YYYY-MM-DD[THH:MM:SS]: the "
                   "checkout's Slurm logs hold the jobs of every month")
    p.add_argument("--history", action="store_true",
                   help="also print the history: the benchmark of every run so far, not just the newest")
    p.add_argument("--target-date", help="target_date to record for benchmark files this run adds to the "
                   "history (default: the work directory's name)")
    p.add_argument("--slurm-logs", default=".snakemake/slurm_logs", help="where the Slurm plugin keeps job logs")
    p.add_argument("--snakemake-logs", default=".snakemake/log", help="Snakemake's own run logs")
    p.add_argument("--out", help="where to write the jobs table (TSV); default <benchmarks folder>/summary.tsv")
    p.add_argument("--sacct-file", help="read this saved `sacct -j ... -P -n --format=" + SACCT_FIELDS + "` output "
                   "instead of running sacct (for testing)")
    args = p.parse_args()
    if not (args.work or args.benchmarks):
        p.error("give --work (the run's work directory) or --benchmarks")

    bench_dir = args.benchmarks or str(Path(args.work) / "benchmarks")
    try:
        archive_benchmarks(bench_dir, args.target_date or (Path(args.work).name if args.work else ""))
    except OSError as e:  # e.g. a benchmarks folder we may only read
        print(f"could not update {HISTORY_NAME}: {e}", file=sys.stderr)
    benchmarks = read_benchmarks(bench_dir) if Path(bench_dir).is_dir() else {}

    jobs = job_rows(args, bench_dir, benchmarks)
    if jobs:
        print_table(jobs, JOB_COLUMNS)
        write_tsv(args.out or Path(bench_dir) / "summary.tsv", jobs, JOB_COLUMNS)
    else:
        print(f"No jobs to show (benchmarks read from {bench_dir}).", file=sys.stderr)

    stages = stage_rows(args.work) if args.work else []
    if stages:
        print()
        print_table(stages, STAGE_COLUMNS)
        write_tsv(Path(bench_dir) / "stages.tsv", stages, STAGE_COLUMNS)

    if args.history:
        history = history_rows(bench_dir, args.rules)
        if history:
            print()
            print_table(history, HISTORY_SHOWN)
        else:
            print(f"No {HISTORY_NAME} in {bench_dir} yet.", file=sys.stderr)


if __name__ == "__main__":
    main()
