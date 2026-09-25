"""One CUDA inference command; no automatic CPU fallback or alternate graph."""
import argparse
from .api import fit


def main(argv=None):
    parser = argparse.ArgumentParser(prog="clipp1d")
    commands = parser.add_subparsers(dest="command", required=True)
    fitting = commands.add_parser("fit")
    fitting.add_argument("--input-file", required=True)
    fitting.add_argument("--outdir", required=True)
    fitting.add_argument("--max-major-cn", type=int, default=4)
    fitting.add_argument("--device", default="cuda:0")
    fitting.add_argument("--verbose", action="store_true")
    fitting.add_argument("--partition-search", action="store_true",
                         help="Also publish a separately qualified development partition estimate")
    fitting.add_argument("--generic-partition-grouping", action="store_true",
                         help="Ablation: remove exact-one separation for new partition candidates")
    args = parser.parse_args(argv)
    try:
        result = fit(args.input_file, args.outdir, max_major_cn=args.max_major_cn,
                     verbose=args.verbose, device=args.device, partition_search=args.partition_search,
                     generic_partition_grouping=args.generic_partition_grouping)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"clipp1d: {error}\n")
    print(f"status=success search_status={result.search_status}")
    return 0
