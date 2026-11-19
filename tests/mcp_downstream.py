"""A tiny downstream MCP server (stdio) for the proxy's process test. Not part of the library."""

from mcp.server.mcpserver import MCPServer

server = MCPServer("downstream")


@server.tool()
def check_vendor(trade: str) -> str:
    """Look up vendors."""
    return f"vendors for {trade}"


@server.tool()
def create_order(vendor: str, cost: int) -> str:
    """Book a vendor."""
    return f"booked {vendor} for {cost}"


@server.tool()
def export_all() -> str:
    """Export everything."""
    return "everything"


if __name__ == "__main__":
    server.run()
