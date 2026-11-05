"""Command-line entry point: ``scholar-mcp`` or ``python -m scholar_mcp``."""

import argparse
import logging
import sys

from .server import mcp


def main() -> None:
    parser = argparse.ArgumentParser(prog="scholar-mcp", description=__doc__)
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    # stdout belongs to the MCP protocol on stdio: all logging must go to stderr.
    logging.basicConfig(
        level=args.log_level.upper(),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.transport == "stdio":
        mcp.run()
    else:
        try:
            mcp.run("streamable-http", host=args.host, port=args.port)  # mcp >= 2
        except TypeError:  # mcp 1.x configures host/port via settings
            mcp.settings.host, mcp.settings.port = args.host, args.port
            mcp.run("streamable-http")


if __name__ == "__main__":
    main()
