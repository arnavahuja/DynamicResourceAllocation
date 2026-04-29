"""Best-effort downloader for public cluster traces.

Both Google Cluster Data v3 and Alibaba 2018 are large multi-GB datasets
hosted in cloud storage. We don't fetch the full thing — we pull a small
*sample shard*, parse it to a canonical Parquet schema, and write the
result under data/processed/.

Sources:
    Google v3:  gs://clusterdata_2019_a (instance_events-* CSV/JSON shards)
    Alibaba 18: https://github.com/alibaba/clusterdata (batch_task.csv)

If `gsutil` / network access isn't available, the script prints the exact
manual steps so you can drop the files in `data/raw/` and re-run with
`--skip-download`.

Usage:
    python -m data.download_traces --source google_v3 --sample 5000
    python -m data.download_traces --source alibaba   --sample 5000
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW = REPO_ROOT / "data" / "raw"
PROCESSED = REPO_ROOT / "data" / "processed"


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _try_run(args: list[str]) -> bool:
    print(f"$ {' '.join(args)}")
    try:
        subprocess.run(args, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  failed: {e}")
        return False


def download_google_v3(skip: bool) -> Path:
    target = RAW / "google_v3"
    target.mkdir(parents=True, exist_ok=True)
    if skip:
        return target
    if not _have("gsutil") and not _have("gcloud"):
        print(
            "\n[!] Neither `gsutil` nor `gcloud` is installed.\n"
            "    Install the Google Cloud SDK or fetch a sample manually:\n"
            "      https://github.com/google/cluster-data\n"
            f"    Drop a CSV/JSON file into {target} and re-run with --skip-download."
        )
        return target

    cmd = "gsutil" if _have("gsutil") else "gcloud"
    if cmd == "gsutil":
        ok = _try_run([
            "gsutil", "-m", "cp",
            "gs://clusterdata_2019_a/instance_events-000000000000.csv.gz",
            str(target),
        ])
    else:
        ok = _try_run([
            "gcloud", "storage", "cp",
            "gs://clusterdata_2019_a/instance_events-000000000000.csv.gz",
            str(target),
        ])
    if not ok:
        print(f"\n[!] Download failed. Place a sample CSV/JSON in {target} manually.")
    return target


def download_alibaba(skip: bool) -> Path:
    target = RAW / "alibaba"
    target.mkdir(parents=True, exist_ok=True)
    if skip:
        return target
    if not _have("curl") and not _have("wget"):
        print("\n[!] Need curl or wget to fetch Alibaba traces.")
        return target
    url = (
        "https://github.com/alibaba/clusterdata/raw/master/cluster-trace-v2018/"
        "trace_2018.zip"
    )
    out = target / "trace_2018.zip"
    if not out.exists():
        cmd = ["curl", "-L", "-o", str(out), url] if _have("curl") else ["wget", "-O", str(out), url]
        ok = _try_run(cmd)
        if not ok:
            print(
                f"\n[!] Could not fetch {url}.\n"
                f"    Download the Alibaba 2018 trace manually and put batch_task.csv in {target}."
            )
            return target
    if _have("unzip"):
        _try_run(["unzip", "-n", "-o", str(out), "-d", str(target)])
    return target


def parse_to_parquet(source: str, raw_dir: Path, sample: int | None) -> Path | None:
    """Run the matching trace parser, dump a canonical Parquet sample."""
    PROCESSED.mkdir(parents=True, exist_ok=True)

    if source == "google_v3":
        from environment.workload.google_v3 import GoogleV3WorkloadGenerator
        try:
            gen = GoogleV3WorkloadGenerator(trace_dir=raw_dir, max_jobs=sample)
        except FileNotFoundError as e:
            print(f"[!] {e}"); return None
    elif source == "alibaba":
        from environment.workload.alibaba import AlibabaWorkloadGenerator
        candidate = next(raw_dir.glob("batch_task*.csv"), None)
        if candidate is None:
            print(f"[!] no batch_task*.csv under {raw_dir}"); return None
        gen = AlibabaWorkloadGenerator(trace_path=candidate, max_jobs=sample)
    else:
        raise SystemExit(source)

    gen.reset()
    rows = []
    # Drain the generator across timesteps until exhausted or sample-met
    t = 0
    while not gen.is_exhausted() and (sample is None or len(rows) < sample):
        for j in gen.get_next_jobs(t):
            rows.append({
                "timestamp": j.arrival_time,
                "job_id": j.job_id,
                "cpu_request": j.cpu_request,
                "memory_request": j.mem_request,
                "duration_estimate": j.duration,
                "sla_deadline": j.sla_deadline,
            })
        t += 1
        if t > 100_000:    # safety
            break

    if not rows:
        print(f"[!] parser produced 0 jobs from {raw_dir}"); return None

    import pandas as pd
    df = pd.DataFrame(rows)
    out = PROCESSED / f"{source}.parquet"
    df.to_parquet(out, index=False)
    print(f"✔ wrote {len(df):,} jobs to {out} ({out.stat().st_size/1e6:.2f} MB)")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["google_v3", "alibaba"], required=True)
    p.add_argument("--sample", type=int, default=10_000,
                   help="Cap on parsed jobs (None to take all).")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip the network step; just parse what is already in data/raw/.")
    args = p.parse_args()

    if args.source == "google_v3":
        raw = download_google_v3(args.skip_download)
    else:
        raw = download_alibaba(args.skip_download)

    if not any(raw.iterdir()):
        print(f"\n[!] No trace files in {raw}. Aborting parse.")
        return 1

    out = parse_to_parquet(args.source, raw, args.sample)
    return 0 if out is not None else 2


if __name__ == "__main__":
    sys.exit(main())
