"""Compatibility entry point; install the package first. No import side effects.

The old -i path and implicit checkout output are retired. Supply --output DIR;
-i/--sample-id is metadata only. See docs/migration.md.
"""

import sys


def main(argv=None):
    from clipp.cli import main as cli_main

    return cli_main(["fit", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(main())
