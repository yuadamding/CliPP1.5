"""The public interface exposes one fitting pipeline."""

import argparse
import sys

from . import __version__
from .api import fit


def main(argv=None):
    parser = argparse.ArgumentParser(prog="clipp1d", description="Single-sample adaptive-chain CCF clustering")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("fit", help="Fit one tumor sample")
    command.add_argument("--input-file", required=True)
    command.add_argument("--outdir", required=True)
    command.add_argument("--max-major-cn", type=int, default=4)
    command.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = fit(args.input_file, args.outdir, max_major_cn=args.max_major_cn, verbose=args.verbose)
    except (ValueError, OSError) as exc:
        print(f"clipp1d: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"Fitted {len(result.raw_phi)} mutations in {len(result.cluster_centers)} clusters; "
          f"search: {result.search_status}; outputs: {args.outdir}")
    return 0
