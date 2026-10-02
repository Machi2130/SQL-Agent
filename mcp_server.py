"""SQL-Agent launcher.

    python mcp_server.py                      # stdio MCP server (what `claude mcp add ... -- python mcp_server.py` runs)
    python mcp_server.py --http 0.0.0.0:8000  # hosted mode, bearer-token protected

The implementation lives in the sql_agent package; this file only exists so the server
can be started by path from any working directory.
"""
from sql_agent.server import cli

if __name__ == "__main__":
    cli()
