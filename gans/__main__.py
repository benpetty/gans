#!/usr/bin/env python
"""The main entry point. Invoke as `gans' or `python -m gans'.
"""
import sys


def main():
    try:

        from gans.core.cli import main

        exit_status = main()

    except KeyboardInterrupt:

        from gans.core.exit_status import ExitStatus

        exit_status = ExitStatus.ERROR_CTRL_C

    sys.exit(exit_status.value)


if __name__ == "__main__":
    main()
