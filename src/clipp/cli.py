"""Import-safe command line: fit, verify, doctor."""

import argparse
import json
import shutil
import subprocess
import sys

from . import __version__
from .config import FitConfig


def _parser():
    parser = argparse.ArgumentParser(
        prog="clipp", description="Single-sample conditional fixed-chain CliPP1.5"
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="Fit into a new, external output directory")
    fit.add_argument("snv_input")
    fit.add_argument("cn_input")
    fit.add_argument("purity_input")
    fit.add_argument("--output", required=True)
    fit.add_argument("-i", "--sample-id", "--sample_id", default="sample")
    fit.add_argument(
        "--assembly",
        default="unspecified",
        help="Assembly label; inputs must share an assembly (no conversion)",
    )
    capacities = fit.add_mutually_exclusive_group()
    capacities.add_argument("--clusters", "--K", type=int)
    capacities.add_argument("--max-clusters", type=int, default=10)
    fit.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    fit.add_argument("--center-refit", choices=["conditional"], default="conditional")
    fit.add_argument(
        "-b", "--subsampling", action="store_true", help="Legacy alias; requires --subsample-size"
    )
    fit.add_argument("-s", "--subsample-size", "--subsample_size", type=int)
    fit.add_argument("-n", "--replicates", "--rep_num", type=int, default=1)
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("-w", "--window-size", "--window_size", type=float, default=0.05)
    fit.add_argument("--overlap", "--overlap_size", type=float, default=0.0)
    verify = commands.add_parser("verify", help="Independently verify a complete published run")
    verify.add_argument("directory")
    commands.add_parser("doctor", help="Report native identity and external dependency availability")
    return parser


def doctor():
    from .native import load_native

    report = {
        "software_version": __version__,
        "native": None,
        "rscript": shutil.which("Rscript"),
        "cuda_qualification": "unqualified in this revision",
        "container_qualification": "unqualified",
    }
    try:
        _, report["native"] = load_native()
    except (OSError, RuntimeError) as error:
        report["native_error"] = str(error)
    if report["rscript"]:
        checked = subprocess.run(
            [report["rscript"], "-e", 'cat(as.character(packageVersion("data.table")))'],
            text=True,
            capture_output=True,
            check=False,
        )
        report["data_table"] = checked.stdout.strip() if checked.returncode == 0 else None
    report["ready_cpu"] = report["native"] is not None and bool(report.get("data_table"))
    return report


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "fit":
            from .api import fit

            if args.subsampling and args.subsample_size is None:
                parser.error("--subsampling requires --subsample-size")
            config = FitConfig(
                sample_id=args.sample_id,
                device=args.device,
                clusters=args.clusters,
                max_clusters=args.max_clusters,
                subsample_size=args.subsample_size,
                replicates=args.replicates,
                seed=args.seed,
                window_size=args.window_size,
                overlap=args.overlap,
                assembly=args.assembly,
            )
            result = fit(args.snv_input, args.cn_input, args.purity_input, args.output, config=config)
        elif args.command == "verify":
            from .verify import verify_run

            result = verify_run(args.directory)
        else:
            result = doctor()
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if args.command != "doctor" or result["ready_cpu"] else 1
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"clipp: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
