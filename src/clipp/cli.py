"""Import-safe command line: fit, verify, validate, doctor."""

import argparse
from dataclasses import asdict, fields
import json
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
    fit.add_argument("--sample-id")
    fit.add_argument(
        "--assembly",
        help="Assembly label; inputs must share an assembly (no conversion)",
    )
    capacities = fit.add_mutually_exclusive_group()
    capacities.add_argument("--clusters", type=int)
    capacities.add_argument(
        "--max-clusters", type=int,
        help=f"Maximum capacity to compare (default: {FitConfig().max_clusters})",
    )
    fit.add_argument("--device", choices=["auto", "cpu", "cuda"])
    fit.add_argument("--subsample-size", type=int)
    fit.add_argument("--replicates", type=int)
    fit.add_argument("--seed", type=int)
    fit.add_argument("--window-size", type=float)
    fit.add_argument("--overlap", type=float)
    fit.set_defaults(**asdict(FitConfig()))
    verify = commands.add_parser("verify", help="Independently verify a complete published run")
    verify.add_argument("directory")
    validate = commands.add_parser("validate", help="Preflight inputs and resource estimates without fitting")
    for name in ("snv_input", "cn_input", "purity_input"):
        validate.add_argument(name)
    commands.add_parser("doctor", help="Report native identity and external dependency availability")
    return parser


def doctor():
    from .native import load_native

    report = {
        "software_version": __version__,
        "native": None,
        "cuda_qualification": "unqualified in this revision",
        "container_qualification": "unqualified",
    }
    try:
        _, report["native"] = load_native()
    except (OSError, RuntimeError) as error:
        report["native_error"] = str(error)
    report["ready_cpu"] = report["native"] is not None
    return report


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "fit":
            from .api import fit

            config = FitConfig(**{field.name: getattr(args, field.name) for field in fields(FitConfig)})
            result = fit(args.snv_input, args.cn_input, args.purity_input, args.output, config=config)
        elif args.command == "verify":
            from .verify import verify_run

            result = verify_run(args.directory)
        elif args.command == "validate":
            from .preprocessing import validate_inputs

            result = validate_inputs(args.snv_input, args.cn_input, args.purity_input)
        else:
            result = doctor()
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if args.command != "doctor" or result["ready_cpu"] else 1
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"clipp: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
