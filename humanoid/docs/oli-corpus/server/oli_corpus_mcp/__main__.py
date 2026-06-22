from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from . import tools


def build_server() -> FastMCP:
    mcp = FastMCP("oli-corpus-mcp")

    mcp.tool()(tools.list_docs)
    mcp.tool()(tools.search)
    mcp.tool()(tools.get_section)
    mcp.tool()(tools.cite)
    return mcp


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
