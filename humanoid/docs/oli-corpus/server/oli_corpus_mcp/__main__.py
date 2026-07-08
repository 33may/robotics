from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from . import tools


def build_server() -> FastMCP:
    mcp = FastMCP("oli-corpus-mcp")

    # Free-text tools (existing)
    mcp.tool()(tools.list_docs)
    mcp.tool()(tools.search)
    mcp.tool()(tools.get_section)
    mcp.tool()(tools.cite)

    # Structured-corpus tools (URDF / typed-tables backed)
    mcp.tool()(tools.robots)
    mcp.tool()(tools.joints)
    mcp.tool()(tools.links)
    mcp.tool()(tools.raw_file)
    mcp.tool()(tools.pkg_info)
    mcp.tool()(tools.nodes)
    mcp.tool()(tools.topics)
    mcp.tool()(tools.find_symbol)
    mcp.tool()(tools.sdk_joint_order)
    return mcp


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
