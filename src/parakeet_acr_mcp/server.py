#!/usr/bin/env python3
"""
Parakeet ACR MCP Server

A Model Context Protocol server that provides a dummy tool returning 42.
Uses HTTP streamable transport for communication.
"""

import argparse
from mcp.server.fastmcp import FastMCP


# Create the MCP server instance
mcp = FastMCP("Parakeet ACR MCP Server")


@mcp.tool()
def get_answer() -> int:
    """
    A dummy tool that returns the answer to everything.
    
    Returns:
        int: The answer to the ultimate question of life, the universe, and everything (42)
    """
    return 42


def main():
    """Main entry point for the server."""
    parser = argparse.ArgumentParser(description="Parakeet ACR MCP Server")
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port to run the server on (default: 8000)"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="localhost", 
        help="Host to bind the server to (default: localhost)"
    )
    
    args = parser.parse_args()
    
    print(f"Starting Parakeet ACR MCP Server on http://{args.host}:{args.port}")
    print(f"MCP endpoint will be available at: http://{args.host}:{args.port}/mcp")
    
    # Run the server with streamable HTTP transport
    # FastMCP.run() uses uvicorn internally, but we need to pass host/port via uvicorn directly
    import uvicorn
    
    # Create the streamable HTTP app
    app = mcp.streamable_http_app()
    
    # Run with uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
